"""
DL Script BiLSTM Final — Early stopping por val LOSS + threshold desde TRAIN
=============================================================================
Protocolo de evaluación limpio para paper científico:

  Early stopping:
    · Se monitorea la LOSS en val en cada época (no AUC, no F1)
    · Se guarda el checkpoint de menor val loss
    · El AUC reportado es distinto de la métrica monitoreada
      → el sesgo de selección múltiple no contamina directamente el AUC
    · Declarable en paper como: "early stopping based on validation loss"

  Threshold:
    · Se busca el umbral óptimo sobre predicciones de TRAIN
    · Val nunca participa en la búsqueda del umbral
    · Se reporta también F1 con threshold=0.5 fijo como referencia

  Scaler / PCA:
    · Ajustados SOLO sobre timesteps reales de train (excluye padding y val)

  Augmentación:
    · Mixup y time-masking SOLO en train, nunca en val

  Métricas reportadas por fold:
    · AUC  — métrica principal (threshold-independent)
    · F1-macro / F1-minority con threshold de TRAIN
    · F1-macro / F1-minority con threshold=0.5 (referencia)
    · Val loss del mejor checkpoint (para comparar configs)

Arquitectura: BiLSTM con pooling configurable (last / meanmax / attention)

Experimentos: ablaciones + grid de hidden + combinaciones
  (mismos que Script 2b, ahora con el protocolo de evaluación correcto)

Uso:
  python dl_script_bilstm_final.py

Ajusta EMBEDDINGS_ROOT y OUTPUT_DIR al final del archivo.
"""

import copy
import json
import logging
import random
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BiLSTM_FINAL")

# ─── Semillas ─────────────────────────────────────────────────────────────────
SEED = 42

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_global_seed(SEED)

# ─── Configuración general ────────────────────────────────────────────────────
MODELS_TO_TEST = [
    "xlsr-300m",
    "xlsr-53",
    "whisper-large-encoder",
    "wav2vec2-large-robust",
    "wavlm-large",
    "hubert-large",
]

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CV_SPLITS     = 5
CV_REPEATS    = 10
EPOCHS        = 60
BATCH_SIZE    = 16
PATIENCE      = 8
PCA_VAR       = 0.95
MIXUP_ALPHA   = 0.4
WARMUP_EPOCHS = 5
BASE_LR       = 1e-3


# =============================================================================
# 1. CARGA DE SECUENCIAS
# =============================================================================

def load_sequence_embeddings(embeddings_root: str, model_key: str):
    """
    Carga secuencias de embeddings por paciente en orden cronológico.
    Retorna X (n, T, D), y (n,), ids, lengths.
    """
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}'.")

    patient_data_raw = defaultdict(list)
    for entry in entries:
        pid = entry["audio_id"]
        patient_data_raw[pid].append({
            "segment_id": int(entry.get("segment_id", 0)),
            "file":       entry["file"],
            "label":      int(entry["class_id"]),
        })

    patient_data = {}
    for pid, segs in patient_data_raw.items():
        segs_sorted = sorted(segs, key=lambda x: x["segment_id"])
        seq = []
        for s in segs_sorted:
            with open(s["file"]) as f:
                rec = json.load(f)
            emb = rec if isinstance(rec, list) else rec["embedding"]
            seq.append(emb)
        patient_data[pid] = {"seq": seq, "label": segs[0]["label"]}

    max_T     = max(len(v["seq"]) for v in patient_data.values())
    embed_dim = len(next(iter(patient_data.values()))["seq"][0])

    X, y, ids, lengths = [], [], [], []
    for pid, data in sorted(patient_data.items()):
        seq      = np.array(data["seq"], dtype=np.float32)
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
# 2. PREPROCESAMIENTO (ajustado SOLO sobre timesteps reales de train)
# =============================================================================

def scale_sequences(X_train, X_val, lengths_train):
    """
    StandardScaler fiteado solo sobre los timesteps REALES de train,
    excluyendo los vectores de ceros de padding.
    """
    B_tr, T, D = X_train.shape
    B_vl       = X_val.shape[0]
    real_vecs  = np.vstack([
        X_train[i, :lengths_train[i], :] for i in range(B_tr)
    ])
    scaler = StandardScaler()
    scaler.fit(real_vecs)
    X_tr_s = scaler.transform(X_train.reshape(-1, D)).reshape(B_tr, T, D)
    X_vl_s = scaler.transform(X_val.reshape(-1, D)).reshape(B_vl, T, D)
    return X_tr_s, X_vl_s


def pca_sequences(X_train, X_val, lengths_train, variance=PCA_VAR):
    """
    PCA fiteada solo sobre los timesteps REALES de train.
    La dimensión temporal T se preserva.
    """
    B_tr, T, D = X_train.shape
    B_vl       = X_val.shape[0]
    real_vecs  = np.vstack([
        X_train[i, :lengths_train[i], :] for i in range(B_tr)
    ])
    pca = PCA(n_components=variance, svd_solver="full", random_state=SEED)
    pca.fit(real_vecs)
    X_tr_p = pca.transform(X_train.reshape(-1, D)).reshape(B_tr, T, -1)
    X_vl_p = pca.transform(X_val.reshape(-1, D)).reshape(B_vl, T, -1)
    new_D  = X_tr_p.shape[2]
    logger.debug(f"    PCA: D {D} → {new_D} (fit sobre {len(real_vecs)} timesteps reales)")
    return X_tr_p, X_vl_p, new_D


# =============================================================================
# 3. AUMENTACIÓN (solo sobre train, nunca sobre val)
# =============================================================================

def mixup_sequences(X_train, y_train, lengths_train, alpha=MIXUP_ALPHA, seed=SEED):
    """
    Mixup intra-clase minoritaria.
    FIX: longitud sintética = max(len_i, len_j), no max_T.
    """
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == 0)[0]
    n_syn   = len(idx_neg) - len(idx_pos)

    if len(idx_pos) < 2 or n_syn <= 0:
        return X_train, y_train, lengths_train

    rng = np.random.default_rng(seed)
    X_syn, len_syn = [], []
    for _ in range(n_syn):
        i, j = rng.choice(idx_pos, size=2, replace=False)
        lam  = rng.beta(alpha, alpha)
        X_syn.append(lam * X_train[i] + (1 - lam) * X_train[j])
        len_syn.append(max(lengths_train[i], lengths_train[j]))

    X_out   = np.concatenate([X_train, np.array(X_syn, dtype=np.float32)], axis=0)
    y_out   = np.concatenate([y_train, np.ones(n_syn, dtype=np.int64)])
    len_out = list(lengths_train) + len_syn
    return X_out, y_out, len_out


def time_mask_sequences(X_train, lengths_train, max_masks=2, seed=SEED):
    """
    Enmascara 1-2 segmentos reales por secuencia (SpecAugment sobre train).
    """
    rng   = np.random.default_rng(seed)
    X_out = X_train.copy()
    for i in range(X_out.shape[0]):
        real_len = lengths_train[i]
        if real_len <= 2:
            continue
        n_m      = rng.integers(1, max_masks + 1)
        mask_idx = rng.choice(real_len, size=min(n_m, real_len - 1), replace=False)
        X_out[i, mask_idx, :] = 0.0
    return X_out


# =============================================================================
# 4. DATASET PYTORCH
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
# 5. BiLSTM CON POOLING CONFIGURABLE
# =============================================================================

class BiLSTMFlexible(nn.Module):
    """
    BiLSTM con tres modos de pooling:
      · last      — hidden state final (forward + backward)
      · meanmax   — mean-pool + max-pool + last (sobre timesteps reales)
      · attention — atención aditiva con máscara de padding
    """
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 1,
                 pooling: str = "last", dropout_p: float = 0.4):
        super().__init__()
        self.pooling = pooling
        self.hidden  = hidden

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
            feat_dim = hidden * 6          # mean(2H) + max(2H) + last(2H)
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

    def _mask(self, B, T, lengths, device):
        ar    = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        len_t = torch.tensor(lengths, device=device).unsqueeze(1)
        return ar < len_t                              # (B, T) bool

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        if lengths is not None:
            lens_cpu  = torch.clamp(torch.tensor(lengths), min=1).cpu()
            x_packed  = pack_padded_sequence(x, lens_cpu, batch_first=True,
                                             enforce_sorted=False)
            out_pack, (h_n, _) = self.lstm(x_packed)
            outputs, _ = pad_packed_sequence(out_pack, batch_first=True, total_length=T)
        else:
            outputs, (h_n, _) = self.lstm(x)
            lengths = [T] * B

        h_last = torch.cat([h_n[0], h_n[1]], dim=-1)   # (B, 2H)

        if self.pooling == "last":
            feat = h_last

        elif self.pooling == "meanmax":
            mask      = self._mask(B, T, lengths, x.device).unsqueeze(-1).float()
            mean_pool = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            max_pool, _ = outputs.masked_fill(mask == 0, float("-inf")).max(dim=1)
            max_pool  = torch.nan_to_num(max_pool, neginf=0.0)
            feat      = torch.cat([mean_pool, max_pool, h_last], dim=-1)

        elif self.pooling == "attention":
            mask    = self._mask(B, T, lengths, x.device)
            scores  = self.attn(outputs).squeeze(-1)
            scores  = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            feat    = (outputs * weights).sum(dim=1)

        return self.head(self.dropout(self.norm(feat)))


# =============================================================================
# 6. FOCAL LOSS
# =============================================================================

class FocalLossLogits(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce    = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1 - probs) * (1 - targets)
        a_t    = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


# =============================================================================
# 7. SCHEDULER
# =============================================================================

def make_scheduler(optimizer, use_warmup: bool, epochs=EPOCHS):
    if not use_warmup:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# 8. UTILIDADES DE EVALUACIÓN
# =============================================================================

def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Umbral que maximiza F1 sobre y_true/y_proba dados."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thrs[np.argmax(f1[:-1])])


def eval_loader(model, loader, criterion=None, device=DEVICE):
    """
    Evalúa el modelo sobre un DataLoader.
    Retorna (probs, labels) y opcionalmente la loss media.
    """
    model.eval()
    probs_l, labels_l = [], []
    loss_sum, n_total = 0.0, 0

    with torch.no_grad():
        for Xb, yb, lb in loader:
            Xb_d, yb_d = Xb.to(device), yb.to(device)
            logits = model(Xb_d, lb).squeeze(-1)
            if criterion is not None:
                loss_sum += criterion(logits, yb_d).item() * len(yb_d)
                n_total  += len(yb_d)
            probs_l.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels_l.extend(yb.cpu().numpy().tolist())

    probs  = np.array(probs_l)
    labels = np.array(labels_l)
    loss   = loss_sum / max(n_total, 1) if criterion is not None else None
    return probs, labels, loss


# =============================================================================
# 9. ENTRENAMIENTO DE UN FOLD
#    Early stopping: val LOSS  (no val AUC, no val F1)
#    Threshold:      predicciones de TRAIN (no val)
# =============================================================================

def train_fold(
    X_train, y_train, lengths_train,
    X_val,   y_val,   lengths_val,
    hidden, pooling, loss_type, use_warmup,
    lr=BASE_LR, dropout_p=0.4, fold_seed=SEED,
):
    """
    Entrena el BiLSTM en un fold con protocolo limpio para paper.

    Early stopping:
      · Monitorea la LOSS en val cada época
      · Guarda el checkpoint de MENOR val loss
      · Val no participa en ninguna otra decisión

    Threshold para F1:
      · Se busca el umbral óptimo sobre predicciones de TRAIN
      · Val nunca ve el umbral

    AUC:
      · Calculado en val una sola vez, con el checkpoint ya fijado
      · La métrica monitoreada (val loss) ≠ métrica reportada (AUC)
        → no hay direct leakage del AUC reportado
    """
    set_global_seed(fold_seed)
    input_dim = X_train.shape[2]

    model = BiLSTMFlexible(
        input_dim, hidden=hidden, pooling=pooling, dropout_p=dropout_p
    ).to(DEVICE)

    ds_train    = SequenceDataset(X_train, y_train, lengths_train)
    ds_val      = SequenceDataset(X_val,   y_val,   lengths_val)
    # loader_train_nograd: mismo train pero sin shuffle, para calcular
    # threshold sobre sus predicciones al final (sin re-entrenar nada)
    loader_tr   = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=(len(ds_train) % BATCH_SIZE == 1))
    loader_tr_e = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
    loader_val  = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Función de pérdida ─────────────────────────────────────────────────
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))

    if loss_type == "focal":
        alpha     = 1.0 - n_pos / (n_pos + n_neg + 1e-6)
        criterion = FocalLossLogits(alpha=alpha, gamma=2.0)
    else:
        pos_w     = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, use_warmup)

    # ── Loop de entrenamiento ──────────────────────────────────────────────
    best_val_loss = np.inf
    best_state    = None
    no_improve    = 0

    for epoch in range(EPOCHS):
        # Entrenamiento
        model.train()
        for Xb, yb, lb in loader_tr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb, lb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ── Early stopping por VAL LOSS ────────────────────────────────────
        # Solo se calcula la loss, NO el AUC ni el F1 en val.
        # Esto evita que el AUC reportado sea el máximo de 60 evaluaciones.
        _, _, val_loss = eval_loader(model, loader_val, criterion=criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.debug(f"    Early stop época {epoch+1}")
                break

    # ── Cargar mejor checkpoint (elegido por val loss, no por AUC) ─────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Threshold desde TRAIN (val no participa en esta decisión) ──────────
    tr_probs, tr_labels, _ = eval_loader(model, loader_tr_e)
    thr_from_train = find_best_threshold(tr_labels, tr_probs)

    # ── Evaluación final en VAL — UNA SOLA VEZ ────────────────────────────
    val_probs, val_labels, final_val_loss = eval_loader(
        model, loader_val, criterion=criterion
    )

    auc = (roc_auc_score(val_labels, val_probs)
           if len(np.unique(val_labels)) > 1 else 0.0)

    # Métricas con threshold de train
    val_pred_tr   = (val_probs >= thr_from_train).astype(int)
    f1_macro      = f1_score(val_labels, val_pred_tr, average="macro",  zero_division=0)
    f1_minority   = f1_score(val_labels, val_pred_tr, average="binary", zero_division=0)
    acc           = accuracy_score(val_labels, val_pred_tr)

    # Métricas con threshold fijo 0.5 (referencia adicional)
    val_pred_05   = (val_probs >= 0.5).astype(int)
    f1_mac_05     = f1_score(val_labels, val_pred_05, average="macro",  zero_division=0)
    f1_min_05     = f1_score(val_labels, val_pred_05, average="binary", zero_division=0)
    acc_05        = accuracy_score(val_labels, val_pred_05)

    return {
        "auc":               auc,
        "f1_macro":          f1_macro,
        "f1_minority":       f1_minority,
        "acc":               acc,
        "threshold_train":   thr_from_train,
        "f1_macro_thr05":    f1_mac_05,
        "f1_minority_thr05": f1_min_05,
        "acc_thr05":         acc_05,
        "val_loss_best":     best_val_loss,
        "input_dim":         input_dim,
    }


# =============================================================================
# 10. LOOP DE CROSS-VALIDACIÓN
# =============================================================================

def run_bilstm_experiment(
    X: np.ndarray, y: np.ndarray, lengths: list,
    exp_config: dict, emb_model: str,
    results_path: Path,
):
    exp_id    = exp_config["id"]
    hidden    = exp_config["hidden"]
    pooling   = exp_config["pooling"]
    loss_t    = exp_config["loss"]
    use_pca   = exp_config["pca"]
    mixup_t   = exp_config["mixup"]     # None | "fix" | "fix_tmask"
    use_wu    = exp_config["warmup"]
    lr        = exp_config.get("lr", BASE_LR)
    dropout_p = exp_config.get("dropout_p", 0.4)

    set_global_seed(SEED)
    cv    = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                                    random_state=SEED)
    rows  = []
    total = CV_SPLITS * CV_REPEATS

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        rep_num  = (fold_idx - 1) // CV_SPLITS + 1
        fold_num = (fold_idx - 1) %  CV_SPLITS + 1

        X_tr, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr, y_vl = y[train_idx].copy(), y[val_idx].copy()
        l_tr = [lengths[i] for i in train_idx]
        l_vl = [lengths[i] for i in val_idx]

        # ── Preprocesamiento (fit SOLO en timesteps reales de train) ───────
        X_tr, X_vl = scale_sequences(X_tr, X_vl, l_tr)

        n_dims = X_tr.shape[2]
        if use_pca:
            X_tr, X_vl, n_dims = pca_sequences(X_tr, X_vl, l_tr)

        # ── Aumentación (SOLO train, val no se toca) ───────────────────────
        if mixup_t in ("fix", "fix_tmask"):
            X_tr, y_tr, l_tr = mixup_sequences(
                X_tr, y_tr, l_tr, seed=SEED + fold_idx
            )
        if mixup_t == "fix_tmask":
            X_tr = time_mask_sequences(X_tr, l_tr, max_masks=2, seed=SEED + fold_idx)

        # ── Entrenar y evaluar ─────────────────────────────────────────────
        metrics = train_fold(
            X_tr, y_tr, l_tr,
            X_vl, y_vl, l_vl,
            hidden=hidden, pooling=pooling, loss_type=loss_t,
            use_warmup=use_wu, lr=lr, dropout_p=dropout_p,
            fold_seed=SEED + fold_idx,
        )

        row = {
            "emb_model":    emb_model,
            "architecture": "BiLSTM",
            "experiment":   exp_id,
            "hidden":       hidden,
            "pooling":      pooling,
            "loss":         loss_t,
            "pca":          use_pca,
            "mixup":        str(mixup_t),
            "warmup":       use_wu,
            "lr":           lr,
            "dropout_p":    dropout_p,
            "repeat":       rep_num,
            "fold":         fold_num,
            "n_train_real": len(train_idx),
            "n_train_aug":  len(X_tr),
            "n_val":        len(val_idx),
            "n_dims":       n_dims,
            **metrics,
        }
        rows.append(row)

        # Guardar incrementalmente
        pd.DataFrame([row]).to_csv(
            results_path, mode="a",
            header=not results_path.exists() or results_path.stat().st_size == 0,
            index=False,
        )

        if fold_idx % CV_SPLITS == 0:
            recent = pd.DataFrame(rows[-CV_SPLITS:])
            logger.info(
                f"      Rep{rep_num:>2}/{CV_REPEATS} | "
                f"AUC={recent['auc'].mean():.3f} | "
                f"F1-macro={recent['f1_macro'].mean():.3f} | "
                f"val_loss={recent['val_loss_best'].mean():.4f}"
            )

    df = pd.DataFrame(rows)
    logger.info(
        f"    ✅ [{exp_id}] "
        f"AUC={df['auc'].mean():.3f}±{df['auc'].std():.3f} | "
        f"F1={df['f1_macro'].mean():.3f} | "
        f"F1-min={df['f1_minority'].mean():.3f} | "
        f"F1-mac@0.5={df['f1_macro_thr05'].mean():.3f}"
    )
    return df


# =============================================================================
# 11. DEFINICIÓN DE EXPERIMENTOS
# =============================================================================

BASELINE = dict(
    hidden=128, pooling="last", loss="bce_posw",
    pca=False, mixup="fix", warmup=False, lr=BASE_LR, dropout_p=0.4,
)

def cfg(id_, **overrides):
    c = dict(BASELINE)
    c.update(overrides)
    c["id"] = id_
    return c

# Fase 1 — Ablaciones: una variable a la vez
ABLATIONS = [
    cfg("baseline"),
    cfg("pooling_meanmax",   pooling="meanmax"),
    cfg("pooling_attention", pooling="attention"),
    cfg("loss_focal",        loss="focal"),
    cfg("warmup",            warmup=True),
    cfg("aug_tmask",         mixup="fix_tmask"),
    cfg("pca_on",            pca=True),
    cfg("dropout_05",        dropout_p=0.5),
]

# Fase 2 — Grid de hidden size
HIDDEN_GRID = [
    cfg(f"hidden_{h}", hidden=h) for h in [64, 96, 128, 192]
]

# Fase 3 — Combinaciones prometedoras
COMBINED = [
    cfg("comb_meanmax_focal_warmup",
        pooling="meanmax", loss="focal", warmup=True),
    cfg("comb_meanmax_focal_warmup_tmask",
        pooling="meanmax", loss="focal", warmup=True, mixup="fix_tmask"),
    cfg("comb_attention_focal_warmup",
        pooling="attention", loss="focal", warmup=True),
    cfg("comb_meanmax_focal_warmup_h192",
        pooling="meanmax", loss="focal", warmup=True, hidden=192),
    cfg("comb_meanmax_focal_warmup_h192_drop05",
        pooling="meanmax", loss="focal", warmup=True, hidden=192, dropout_p=0.5),
]

ALL_EXPERIMENTS = ABLATIONS + HIDDEN_GRID + COMBINED


# =============================================================================
# 12. PIPELINE PRINCIPAL
# =============================================================================

def run_all(embeddings_root: str, output_dir: str):
    output_dir = Path(output_dir) / "dl_bilstm_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = output_dir / "raw_folds.csv"
    summary_path = output_dir / "summary.csv"

    if raw_path.exists():
        logger.warning(f"⚠️  {raw_path} ya existe — se añaden resultados al final.")

    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 Embeddings:   {embeddings_root}")
    logger.info(f"📂 Salida:       {output_dir}")
    logger.info(
        f"🔁 {CV_REPEATS} repeats × {CV_SPLITS} folds = "
        f"{CV_REPEATS * CV_SPLITS} evaluaciones por config"
    )
    logger.info(f"🧪 Configs: {len(ALL_EXPERIMENTS)}")
    logger.info(
        "📋 Protocolo: early stopping por VAL LOSS | "
        "threshold desde TRAIN | AUC métrica principal"
    )
    logger.info("")

    t0_global = time.time()

    for emb_model in MODELS_TO_TEST:
        logger.info(f"{'='*65}")
        logger.info(f"📦 Modelo de embeddings: {emb_model}")
        logger.info(f"{'='*65}")

        try:
            X, y, _ids, lengths = load_sequence_embeddings(
                embeddings_root, emb_model
            )
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar {emb_model}: {e}")
            continue

        if len(np.unique(y)) < 2:
            logger.warning("   Solo una clase. Saltando.")
            continue

        for exp_config in ALL_EXPERIMENTS:
            logger.info(f"\n  🔬 {exp_config['id']}")
            t0 = time.time()
            run_bilstm_experiment(X, y, lengths, exp_config, emb_model, raw_path)
            logger.info(f"       ⏱️  {(time.time()-t0)/60:.1f} min")

    # ── Resumen final ──────────────────────────────────────────────────────
    if raw_path.exists():
        df_all = pd.read_csv(raw_path)
        agg_cols = ["emb_model", "experiment", "hidden", "pooling",
                    "loss", "mixup", "warmup", "dropout_p", "pca"]
        summary = (
            df_all
            .groupby(agg_cols)
            .agg(
                mean_auc              =("auc",               "mean"),
                std_auc               =("auc",                "std"),
                mean_f1_macro         =("f1_macro",           "mean"),
                std_f1_macro          =("f1_macro",            "std"),
                mean_f1_minority      =("f1_minority",        "mean"),
                std_f1_minority       =("f1_minority",         "std"),
                mean_acc              =("acc",                "mean"),
                mean_f1_mac_thr05     =("f1_macro_thr05",     "mean"),
                mean_f1_min_thr05     =("f1_minority_thr05",  "mean"),
                mean_val_loss         =("val_loss_best",      "mean"),
                n_folds               =("auc",                "count"),
            )
            .reset_index()
            .sort_values("mean_auc", ascending=False)
        )
        summary.to_csv(summary_path, index=False)

        logger.info("\n" + "="*65)
        logger.info("📊 TOP 20 CONFIGS (ordenadas por AUC medio)")
        logger.info("="*65)
        show_cols = [
            "emb_model", "experiment", "hidden", "pooling", "loss",
            "mixup", "warmup", "dropout_p",
            "mean_auc", "std_auc", "mean_f1_macro", "mean_f1_minority",
            "mean_f1_mac_thr05",
        ]
        logger.info("\n" + summary[show_cols].head(20).to_string(index=False))
        logger.info(f"\n📄 Summary → {summary_path}")
        logger.info(f"📄 Raw folds → {raw_path}")

    total_h = (time.time() - t0_global) / 3600
    logger.info(f"\n⏱️  Tiempo total: {total_h:.1f} h")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # ── Ajusta estas rutas ─────────────────────────────────────────────────
    EMBEDDINGS_ROOT = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/"
        "Ansiedad/embeddings_v2/"
    )
    OUTPUT_DIR = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Ansiedad/Wav2Vec/results_DL/"
        "embeddings_nuevos"
    )
    # ───────────────────────────────────────────────────────────────────────

    run_all(EMBEDDINGS_ROOT, OUTPUT_DIR)
