"""
DL Script 2e — BiLSTM Ronda 4 (final): combinación de las 3 mejoras que
funcionaron por separado en Ronda 3
==========================================================================
Solo wav2vec2-large-robust.

RESULTADOS RONDA 3 (referencia)
---------------------------------
Top configs (todas sin PCA — PCA fue claramente peor, 0.68-0.71 AUC,
descartado definitivamente):

  dropout05_ensemble3        (hidden=192, dropout=0.5, gamma=2.0, ens=3) → AUC 0.7920
  gamma3_ensemble5           (hidden=192, dropout=0.4, gamma=3.0, ens=5) → AUC 0.7916
  dropout06_ensemble3        (hidden=192, dropout=0.6, gamma=2.0, ens=3) → AUC 0.7909
  wd5e4_ensemble3            (hidden=192, dropout=0.4, wd=5e-4,   ens=3) → AUC 0.7859
  ensemble5 "puro"           (hidden=192, dropout=0.4, gamma=2.0, ens=5) → AUC 0.7859

Tres ejes distintos (dropout=0.5, gamma=3.0, weight_decay=5e-4) mejoraron
cada uno por separado sobre el ensemble5 puro, pero NUNCA se combinaron
entre sí. Esta ronda cierra esa combinatoria pendiente:

RONDA 4 — experimentos (3, todos hidden=192, sin PCA, ensemble5):
  1. r4_dropout05_gamma3_ensemble5      — combina las dos mejoras más fuertes
  2. r4_dropout05_wd5e4_ensemble5       — combina las dos regularizaciones
  3. r4_dropout055_gamma3_ensemble5     — dropout intermedio (0.5 y 0.6
     dieron resultados casi idénticos en R3, sugiriendo que el óptimo
     está en esa zona; se prueba 0.55 con la mejora de gamma incluida)

Si ninguna de estas supera claramente ~0.795-0.80, es una señal razonable
de que se llegó al techo que da este dataset con n=79 pacientes vía
tuning de hiperparámetros de una sola arquitectura — con std_auc≈0.11-0.12
consistente en TODAS las configs de las 4 rondas, la varianza entre folds
ya domina sobre las diferencias entre configs.

No se guarda ningún modelo/checkpoint en disco. Solo métricas por fold.

Uso:
  python dl_script2e_bilstm_round4.py
"""

import json
import logging
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, precision_recall_curve,
)

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DL_S2e_BiLSTM_R4")

# ─── Semillas ──────────────────────────────────────────────────────────────
SEED = 42

def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_global_seed(SEED)

# ─── Configuración general ────────────────────────────────────────────────
MODELS_TO_TEST = [
    "wav2vec2-large-robust",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CV_SPLITS   = 5
CV_REPEATS  = 10
EPOCHS      = 60
BATCH_SIZE  = 16
PATIENCE    = 8
PCA_VAR     = 0.95
MIXUP_ALPHA = 0.4
WARMUP_EPOCHS = 5
CALIB_FRAC  = 0.2   # fracción de train reservada para calibrar el threshold


# =============================================================================
# 1. CARGA DE SECUENCIAS (idéntico a scripts anteriores)
# =============================================================================

def load_sequence_embeddings(embeddings_root: str, model_key: str):
    from collections import defaultdict
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}'.")

    logger.info("ℹ️ Reconstruyendo secuencias cronológicas (Formato v2)...")

    patient_data_raw = defaultdict(list)
    for entry in entries:
        pid = entry["audio_id"]
        patient_data_raw[pid].append({
            "segment_id": int(entry["segment_id"]),
            "file": entry["file"],
            "label": int(entry["class_id"])
        })

    patient_data = {}
    for pid, segs in patient_data_raw.items():
        segs_sorted = sorted(segs, key=lambda x: x["segment_id"])
        seq = []
        for s in segs_sorted:
            with open(s["file"]) as f:
                record = json.load(f)
            seq.append(record["embedding"])

        patient_data[pid] = {
            "seq": seq,
            "label": segs[0]["label"]
        }

    max_T     = max(len(v["seq"]) for v in patient_data.values())
    embed_dim = len(next(iter(patient_data.values()))["seq"][0])

    X, y, ids, lengths = [], [], [], []
    for pid, data in sorted(patient_data.items()):
        seq = np.array(data["seq"], dtype=np.float32)
        real_len = len(seq)
        if real_len < max_T:
            pad = np.zeros((max_T - real_len, embed_dim), dtype=np.float32)
            seq = np.vstack([seq, pad])
        X.append(seq)
        y.append(data["label"])
        ids.append(pid)
        lengths.append(real_len)

    X = np.array(X)
    y = np.array(y, dtype=np.int64)
    logger.info(
        f"   Cargados: {len(y)} pacientes | shape={X.shape} | "
        f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}"
    )
    return X, y, ids, lengths


# =============================================================================
# 2. PREPROCESAMIENTO (fit solo en timesteps reales de FIT, nunca en padding,
#    calib o val)
# =============================================================================

def scale_sequences(X_fit, X_calib, X_val, lengths_fit):
    B_fit, T, D = X_fit.shape
    real_vecs = np.vstack([X_fit[i, :lengths_fit[i], :] for i in range(B_fit)])
    scaler = StandardScaler()
    scaler.fit(real_vecs)

    def _apply(X):
        B = X.shape[0]
        return scaler.transform(X.reshape(-1, D)).reshape(B, T, D)

    return _apply(X_fit), _apply(X_calib), _apply(X_val)


def pca_sequences(X_fit, X_calib, X_val, lengths_fit, variance=PCA_VAR):
    B_fit, T, D = X_fit.shape
    real_vecs = np.vstack([X_fit[i, :lengths_fit[i], :] for i in range(B_fit)])
    pca = PCA(n_components=variance, svd_solver="full", random_state=SEED)
    pca.fit(real_vecs)

    def _apply(X):
        B = X.shape[0]
        return pca.transform(X.reshape(-1, D)).reshape(B, T, -1)

    X_fit_p, X_calib_p, X_val_p = _apply(X_fit), _apply(X_calib), _apply(X_val)
    return X_fit_p, X_calib_p, X_val_p, X_fit_p.shape[2]


def split_fit_calib(X_train, y_train, lengths_train, frac=CALIB_FRAC, seed=SEED):
    """
    FIX-E: separa train en FIT (para gradiente) y CALIB (para threshold),
    de forma estratificada. CALIB queda intacto (sin mixup/time-mask).
    Si no se puede estratificar (muy pocos positivos en el fold), cae a
    fallback: todo es FIT y no hay CALIB real (se usará threshold=0.5).
    """
    idx = np.arange(len(y_train))
    try:
        idx_fit, idx_calib = train_test_split(
            idx, test_size=frac, stratify=y_train, random_state=seed
        )
        if len(np.unique(y_train[idx_calib])) < 2 or len(idx_calib) < 4:
            raise ValueError("calib set insuficiente")
        fallback = False
    except Exception:
        idx_fit, idx_calib = idx, np.array([], dtype=int)
        fallback = True

    X_fit, y_fit = X_train[idx_fit], y_train[idx_fit]
    l_fit = [lengths_train[i] for i in idx_fit]

    if fallback:
        X_calib, y_calib, l_calib = None, None, None
    else:
        X_calib, y_calib = X_train[idx_calib], y_train[idx_calib]
        l_calib = [lengths_train[i] for i in idx_calib]

    return X_fit, y_fit, l_fit, X_calib, y_calib, l_calib, fallback


def mixup_sequences_fixed(X_fit, y_fit, lengths_fit, alpha=MIXUP_ALPHA, seed=SEED):
    """Mixup con longitud sintética = max(len_i, len_j) (FIX-C). Solo toca FIT."""
    idx_pos = np.where(y_fit == 1)[0]
    idx_neg = np.where(y_fit == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    if n_pos < 2:
        return X_fit, y_fit, lengths_fit

    n_syn = n_neg - n_pos
    if n_syn <= 0:
        return X_fit, y_fit, lengths_fit

    rng = np.random.default_rng(seed)
    X_syn_list, len_syn_list = [], []
    for _ in range(n_syn):
        i, j = rng.choice(idx_pos, size=2, replace=False)
        lam = rng.beta(alpha, alpha)
        x_new = lam * X_fit[i] + (1.0 - lam) * X_fit[j]
        X_syn_list.append(x_new)
        len_syn_list.append(max(lengths_fit[i], lengths_fit[j]))

    X_syn = np.array(X_syn_list, dtype=np.float32)
    y_syn = np.ones(n_syn, dtype=np.int64)

    X_out = np.concatenate([X_fit, X_syn], axis=0)
    y_out = np.concatenate([y_fit, y_syn])
    len_out = list(lengths_fit) + len_syn_list
    return X_out, y_out, len_out


def time_mask_sequences(X_fit, lengths_fit, max_masks=2, seed=SEED):
    """Enmascara 1-2 segmentos reales por secuencia, solo en FIT."""
    rng = np.random.default_rng(seed)
    X_out = X_fit.copy()
    for i in range(X_out.shape[0]):
        real_len = lengths_fit[i]
        if real_len <= 2:
            continue
        n_masks = rng.integers(1, max_masks + 1)
        mask_idx = rng.choice(real_len, size=min(n_masks, real_len - 1), replace=False)
        X_out[i, mask_idx, :] = 0.0
    return X_out


# =============================================================================
# 3. DATASET
# =============================================================================

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, lengths: list):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.FloatTensor(y.astype(np.float32))
        self.lengths = lengths

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


# =============================================================================
# 4. BiLSTM CON POOLING CONFIGURABLE
# =============================================================================

class BiLSTMFlexible(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 1,
                 pooling: str = "last", dropout_p: float = 0.4):
        super().__init__()
        self.pooling = pooling
        self.hidden = hidden
        self.lstm = nn.LSTM(
            input_dim, hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(0.3 if n_layers > 1 else 0.0),
        )

        if pooling == "last":
            feat_dim = hidden * 2
        elif pooling == "meanmax":
            feat_dim = hidden * 2 * 3
        elif pooling == "attention":
            feat_dim = hidden * 2
            self.attn = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
        else:
            raise ValueError(f"pooling desconocido: {pooling}")

        self.norm    = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.head    = nn.Linear(feat_dim, 1)

    def _make_mask(self, B, T, lengths, device):
        ar = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        len_t = torch.tensor(lengths, device=device).unsqueeze(1)
        return ar < len_t

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        if lengths is not None:
            lens_cpu = torch.clamp(torch.tensor(lengths), min=1).cpu()
            x_packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
            outputs_packed, (h_n, _) = self.lstm(x_packed)
            outputs, _ = pad_packed_sequence(outputs_packed, batch_first=True, total_length=T)
        else:
            outputs, (h_n, _) = self.lstm(x)
            lengths = [T] * B

        h_last = torch.cat([h_n[0], h_n[1]], dim=-1)

        if self.pooling == "last":
            feat = h_last

        elif self.pooling == "meanmax":
            mask = self._make_mask(B, T, lengths, x.device).unsqueeze(-1).float()
            summed = (outputs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            mean_pool = summed / counts
            masked_out = outputs.masked_fill(mask == 0, float("-inf"))
            max_pool, _ = masked_out.max(dim=1)
            max_pool = torch.nan_to_num(max_pool, neginf=0.0)
            feat = torch.cat([mean_pool, max_pool, h_last], dim=-1)

        elif self.pooling == "attention":
            mask = self._make_mask(B, T, lengths, x.device)
            scores = self.attn(outputs).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            feat = (outputs * weights).sum(dim=1)

        return self.head(self.dropout(self.norm(feat)))


# =============================================================================
# 5. FOCAL LOSS (gamma configurable)
# =============================================================================

class FocalLossLogits(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# =============================================================================
# 6. SCHEDULER CON WARMUP OPCIONAL
# =============================================================================

def make_scheduler(optimizer, use_warmup: bool, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS):
    if not use_warmup:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def find_best_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thresholds[np.argmax(f1[:-1])])


# =============================================================================
# 7. ENTRENAMIENTO DE UN SOLO SEED (una corrida completa dentro de un fold)
# =============================================================================

def train_single_seed(
    X_fit, y_fit, l_fit,
    X_calib, y_calib, l_calib,     # pueden ser None (fallback)
    X_val, y_val, l_val,
    hidden, pooling, loss_type, focal_gamma, use_warmup, lr, seed,
    dropout_p=0.4, weight_decay=1e-4,
):
    set_global_seed(seed)
    input_dim = X_fit.shape[2]
    model = BiLSTMFlexible(input_dim, hidden=hidden, pooling=pooling, dropout_p=dropout_p).to(DEVICE)

    ds_fit = SequenceDataset(X_fit, y_fit, l_fit)
    ds_val = SequenceDataset(X_val, y_val, l_val)
    loader_fit = DataLoader(ds_fit, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=(len(ds_fit) % BATCH_SIZE == 1))
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    has_calib = X_calib is not None and len(X_calib) > 0
    if has_calib:
        ds_calib = SequenceDataset(X_calib, y_calib, l_calib)
        loader_calib = DataLoader(ds_calib, batch_size=BATCH_SIZE, shuffle=False)

    n_neg = np.sum(y_fit == 0)
    n_pos = np.sum(y_fit == 1)

    if loss_type == "focal":
        alpha = 1.0 - float(n_pos) / float(n_pos + n_neg + 1e-6)
        criterion = FocalLossLogits(alpha=alpha, gamma=focal_gamma)
    else:
        pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, use_warmup)

    best_auc = -1.0
    best_val_probs = best_val_labels = None
    best_calib_probs = best_calib_labels = None
    no_improve = 0

    def _eval(loader):
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for Xb, yb, lb in loader:
                p = torch.sigmoid(model(Xb.to(DEVICE), lb).squeeze(-1))
                probs.extend(p.cpu().numpy().tolist())
                labels.extend(yb.cpu().numpy().tolist())
        return np.array(probs), np.array(labels)

    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb, lb in loader_fit:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb, lb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_probs, val_labels = _eval(loader_val)
        auc = (roc_auc_score(val_labels, val_probs)
               if len(np.unique(val_labels)) > 1 else 0.0)

        if auc > best_auc:
            best_auc = auc
            best_val_probs, best_val_labels = val_probs, val_labels
            if has_calib:
                best_calib_probs, best_calib_labels = _eval(loader_calib)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return {
        "val_probs": best_val_probs, "val_labels": best_val_labels,
        "calib_probs": best_calib_probs, "calib_labels": best_calib_labels,
        "auc": best_auc, "has_calib": has_calib,
    }


# =============================================================================
# 8. ENSEMBLE DE N SEEDS PARA UN FOLD (usado desde run_bilstm_experiment,
#    que ya calculó el split FIT/CALIB antes del preprocesamiento — ver
#    _run_ensemble_with_precomputed_split más abajo)
# =============================================================================


# =============================================================================
# 9. LOOP DE CROSS-VALIDATION PARA UNA CONFIG
# =============================================================================

def run_bilstm_experiment(X, y, lengths, exp_config: dict, emb_model: str, results_path: Path):
    exp_id   = exp_config["id"]
    hidden   = exp_config["hidden"]
    pooling  = exp_config["pooling"]
    loss_t   = exp_config["loss"]
    gamma    = exp_config.get("focal_gamma", 2.0)
    use_pca  = exp_config["pca"]
    mixup_t  = exp_config["mixup"]      # "none" | "fix" | "fix_tmask" | "tmask_only"
    use_wu   = exp_config["warmup"]
    lr       = exp_config.get("lr", 1e-3)
    n_seeds  = exp_config.get("n_seeds", 1)
    dropout_p     = exp_config.get("dropout_p", 0.4)
    weight_decay  = exp_config.get("weight_decay", 1e-4)

    set_global_seed(SEED)
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        rep_num  = (fold_idx - 1) // CV_SPLITS + 1
        fold_num = (fold_idx - 1) %  CV_SPLITS + 1

        X_tr, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr, y_vl = y[train_idx].copy(), y[val_idx].copy()
        l_tr = [lengths[i] for i in train_idx]
        l_vl = [lengths[i] for i in val_idx]

        # split FIT/CALIB (FIX-E) — se hace ANTES de scaler/PCA para que el
        # scaler/PCA se ajusten sobre FIT+CALIB reales tal como antes
        # (el split de calibración es solo para el threshold, no cambia
        # qué datos ve el preprocesamiento no-supervisado)
        X_fit_raw, y_fit, l_fit, X_calib_raw, y_calib, l_calib, fallback = split_fit_calib(
            X_tr, y_tr, l_tr, seed=SEED + fold_idx
        )

        # ── Scaler (fit sobre FIT real) ────────────────────────────────
        empty_calib = X_fit_raw[:0]
        X_fit_s, X_calib_s_tmp, X_vl_s = scale_sequences(
            X_fit_raw, X_calib_raw if X_calib_raw is not None else empty_calib, X_vl, l_fit
        )
        X_calib_s = X_calib_s_tmp if X_calib_raw is not None else None

        n_dims = X_fit_s.shape[2]
        if use_pca:
            empty_calib_s = X_fit_s[:0]
            X_fit_s, X_calib_s_tmp, X_vl_s, n_dims = pca_sequences(
                X_fit_s, X_calib_s if X_calib_s is not None else empty_calib_s, X_vl_s, l_fit
            )
            X_calib_s = X_calib_s_tmp if X_calib_raw is not None else None

        # ── Aumentos: SOLO sobre FIT, nunca sobre CALIB ni val ────────────
        X_fit_aug, y_fit_aug, l_fit_aug = X_fit_s, y_fit, l_fit
        if mixup_t in ("fix", "fix_tmask"):
            X_fit_aug, y_fit_aug, l_fit_aug = mixup_sequences_fixed(
                X_fit_s, y_fit, l_fit, seed=SEED + fold_idx
            )
        if mixup_t in ("fix_tmask", "tmask_only"):
            X_fit_aug = time_mask_sequences(X_fit_aug, l_fit_aug, max_masks=2, seed=SEED + fold_idx)

        # reconstruir X_train "lógico" (fit+calib) para pasar al ensemble,
        # que internamente vuelve a hacer el split — para mantener la firma
        # de train_fold_bilstm_ensemble simple, le pasamos directamente
        # fit/calib ya preparados vía un pequeño wrapper:
        metrics = _run_ensemble_with_precomputed_split(
            X_fit_aug, y_fit_aug, l_fit_aug,
            X_calib_s, y_calib, l_calib,
            X_vl_s, y_vl, l_vl,
            hidden, pooling, loss_t, gamma, use_wu, lr, n_seeds,
            fold_seed_base=SEED + fold_idx,
            dropout_p=dropout_p, weight_decay=weight_decay,
        )

        row = {
            "emb_model": emb_model, "architecture": "BiLSTM", "experiment": exp_id,
            "hidden": hidden, "pooling": pooling, "loss": loss_t, "focal_gamma": gamma,
            "pca": use_pca, "mixup": mixup_t, "warmup": use_wu, "lr": lr, "n_seeds": n_seeds,
            "dropout_p": dropout_p, "weight_decay": weight_decay,
            "repeat": rep_num, "fold": fold_num,
            "n_train_real": len(train_idx), "n_fit": len(X_fit_aug),
            "n_calib": (0 if X_calib_s is None else len(X_calib_s)),
            "n_val": len(val_idx), "n_dims": n_dims,
            **metrics,
        }
        rows.append(row)

        pd.DataFrame([row]).to_csv(
            results_path, mode="a",
            header=not results_path.exists() or results_path.stat().st_size == 0,
            index=False,
        )

        if fold_idx % CV_SPLITS == 0:
            recent = pd.DataFrame(rows[-CV_SPLITS:])
            logger.info(
                f"      Rep{rep_num:>2} | AUC={recent['auc'].mean():.3f} | "
                f"F1-macro={recent['f1_macro'].mean():.3f} | dims={n_dims}"
            )

    df = pd.DataFrame(rows)
    logger.info(
        f"    ✅ [{exp_id}] AUC={df['auc'].mean():.3f}±{df['auc'].std():.3f} | "
        f"F1={df['f1_macro'].mean():.3f} | F1-min={df['f1_minority'].mean():.3f} | "
        f"calib_fallback_rate={df['calib_fallback'].mean():.2f}"
    )
    return df


def _run_ensemble_with_precomputed_split(
    X_fit, y_fit, l_fit, X_calib, y_calib, l_calib, X_val, y_val, l_val,
    hidden, pooling, loss_type, focal_gamma, use_warmup, lr, n_seeds, fold_seed_base,
    dropout_p=0.4, weight_decay=1e-4,
):
    """Variante de train_fold_bilstm_ensemble que recibe el split FIT/CALIB
    ya calculado (porque en run_bilstm_experiment el split se hizo antes
    del preprocesamiento). Evita duplicar el split."""
    val_probs_runs, calib_probs_runs = [], []
    val_labels_ref = calib_labels_ref = None
    has_calib = X_calib is not None and len(X_calib) > 0

    for s in range(n_seeds):
        seed_s = fold_seed_base * 1000 + s
        out = train_single_seed(
            X_fit, y_fit, l_fit,
            X_calib if has_calib else None, y_calib if has_calib else None, l_calib if has_calib else None,
            X_val, y_val, l_val,
            hidden, pooling, loss_type, focal_gamma, use_warmup, lr, seed_s,
            dropout_p=dropout_p, weight_decay=weight_decay,
        )
        val_probs_runs.append(out["val_probs"])
        val_labels_ref = out["val_labels"]
        if out["has_calib"]:
            calib_probs_runs.append(out["calib_probs"])
            calib_labels_ref = out["calib_labels"]

    ensemble_val_probs = np.mean(np.stack(val_probs_runs, axis=0), axis=0)
    auc = (roc_auc_score(val_labels_ref, ensemble_val_probs)
           if len(np.unique(val_labels_ref)) > 1 else 0.0)

    if calib_probs_runs:
        ensemble_calib_probs = np.mean(np.stack(calib_probs_runs, axis=0), axis=0)
        thr = find_best_threshold(calib_labels_ref, ensemble_calib_probs)
        calib_fallback = False
    else:
        thr = 0.5
        calib_fallback = True

    val_pred_thrcalib = (ensemble_val_probs >= thr).astype(int)
    val_pred_thr05    = (ensemble_val_probs >= 0.5).astype(int)

    return {
        "auc": auc,
        "f1_macro":    f1_score(val_labels_ref, val_pred_thrcalib, average="macro",  zero_division=0),
        "f1_minority": f1_score(val_labels_ref, val_pred_thrcalib, average="binary", zero_division=0),
        "acc":         accuracy_score(val_labels_ref, val_pred_thrcalib),
        "threshold":   thr,
        "calib_fallback": calib_fallback,
        "f1_macro_thr05":    f1_score(val_labels_ref, val_pred_thr05, average="macro",  zero_division=0),
        "f1_minority_thr05": f1_score(val_labels_ref, val_pred_thr05, average="binary", zero_division=0),
        "acc_thr05":         accuracy_score(val_labels_ref, val_pred_thr05),
    }


# =============================================================================
# 10. DEFINICIÓN DE EXPERIMENTOS — RONDA 4 (final, solo wav2vec2-large-robust)
# =============================================================================
# Ganador de Ronda 1: meanmax + focal(gamma=2) + warmup + tmask, hidden=128
WINNER = dict(hidden=128, pooling="meanmax", loss="focal", focal_gamma=2.0,
              pca=False, mixup="fix_tmask", warmup=True, lr=1e-3, n_seeds=1,
              dropout_p=0.4, weight_decay=1e-4)

def cfg(id_, **overrides):
    c = dict(WINNER)
    c.update(overrides)
    c["id"] = id_
    return c

ALL_EXPERIMENTS = [
    # Combina las dos mejoras más fuertes de R3 (dropout=0.5 y gamma=3.0),
    # que nunca se probaron juntas
    cfg("r4_dropout05_gamma3_ensemble5",
        hidden=192, dropout_p=0.5, focal_gamma=3.0, n_seeds=5),

    # Combina las dos regularizaciones que ayudaron por separado
    cfg("r4_dropout05_wd5e4_ensemble5",
        hidden=192, dropout_p=0.5, weight_decay=5e-4, n_seeds=5),

    # Dropout intermedio (0.5 y 0.6 dieron resultados casi idénticos en R3)
    # + la mejora de gamma
    cfg("r4_dropout055_gamma3_ensemble5",
        hidden=192, dropout_p=0.55, focal_gamma=3.0, n_seeds=5),
]


# =============================================================================
# 11. PIPELINE PRINCIPAL
# =============================================================================

def run_all(embeddings_root: str, output_dir: str):
    output_dir = Path(output_dir) / "dl_script2e_bilstm_round4"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = output_dir / "raw_folds_bilstm_r4.csv"
    summary_path = output_dir / "summary_bilstm_r4.csv"

    if raw_path.exists():
        logger.warning(f"⚠️  {raw_path} ya existe. Se añaden resultados al final.")

    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 Embeddings:   {embeddings_root}")
    logger.info(f"📂 Salida:       {output_dir}")
    logger.info(f"🧪 Total configs Ronda 4 (final): {len(ALL_EXPERIMENTS)} (solo wav2vec2-large-robust)")
    for c in ALL_EXPERIMENTS:
        logger.info(f"    · {c['id']}  (n_seeds={c['n_seeds']})")
    logger.info("")

    t0_global = time.time()

    for emb_model in MODELS_TO_TEST:
        logger.info(f"{'='*65}")
        logger.info(f"📦 Modelo de embeddings: {emb_model}")
        logger.info(f"{'='*65}")

        try:
            X, y, _ids, lengths = load_sequence_embeddings(embeddings_root, emb_model)
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar {emb_model}: {e}")
            continue

        if len(np.unique(y)) < 2:
            logger.warning("   Solo una clase. Saltando.")
            continue

        for exp_config in ALL_EXPERIMENTS:
            logger.info(f"\n  🔬 Experimento: {exp_config['id']}")
            t0 = time.time()
            run_bilstm_experiment(X, y, lengths, exp_config, emb_model, raw_path)
            logger.info(f"       ⏱️  {(time.time()-t0)/60:.1f} min")

    if raw_path.exists():
        df_all = pd.read_csv(raw_path)
        summary = (
            df_all
            .groupby(["emb_model", "experiment", "hidden", "pooling", "loss",
                      "focal_gamma", "mixup", "warmup", "lr", "n_seeds",
                      "dropout_p", "weight_decay", "pca"])
            .agg(
                mean_auc              =("auc",              "mean"),
                std_auc               =("auc",               "std"),
                mean_f1_macro         =("f1_macro",          "mean"),
                std_f1_macro          =("f1_macro",           "std"),
                mean_f1_minority      =("f1_minority",       "mean"),
                std_f1_minority       =("f1_minority",        "std"),
                mean_acc              =("acc",               "mean"),
                mean_f1_macro_thr05   =("f1_macro_thr05",    "mean"),
                mean_f1_minority_thr05=("f1_minority_thr05", "mean"),
                calib_fallback_rate   =("calib_fallback",    "mean"),
                n_folds               =("auc",               "count"),
            )
            .reset_index()
            .sort_values("mean_auc", ascending=False)
        )
        summary.to_csv(summary_path, index=False)

        logger.info("\n" + "="*65)
        logger.info("📊 CONFIGURACIONES RONDA 4 — FINAL (por AUC, wav2vec2-large-robust)")
        logger.info("="*65)
        cols = ["experiment","hidden","pca","focal_gamma","dropout_p","weight_decay","n_seeds",
                "mean_auc","std_auc","mean_f1_macro","mean_f1_minority",
                "mean_f1_macro_thr05","calib_fallback_rate"]
        logger.info("\n" + summary[cols].head(20).to_string(index=False))
        logger.info(f"\n📄 Summary → {summary_path}")
        logger.info(f"📄 Raw folds → {raw_path}")

    total_h = (time.time() - t0_global) / 3600
    logger.info(f"\n⏱️  Tiempo total: {total_h:.1f} h")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    EMBEDDINGS_ROOT = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/"
        "Depresión/embeddings_v2/"
    )
    OUTPUT_DIR = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Wav2Vec/results_DL/embeddings_nuevos"
    )
    run_all(EMBEDDINGS_ROOT, OUTPUT_DIR)
