"""
DL Script 2b — Optimización experimental de BiLSTM sobre secuencias temporales
================================================================================
Objetivo: mejorar AUC (métrica prioritaria) del BiLSTM manteniendo el
protocolo del Script 2 (RepeatedStratifiedKFold 5x10, CV agrupado por
paciente porque cada fila de X ya es un paciente, preprocesamiento
ajustado solo en train, VAL siempre real).

Cambios metodológicos respecto al script original (ver revisión de código):
  FIX-A: el checkpoint/época "mejor" de cada fold se elige por AUC,
         no por F1-macro (antes sesgaba la selección hacia F1).
  FIX-B: el threshold para F1/acc YA NO se busca en val (eso era leakage:
         el umbral "veía" las etiquetas del propio val antes de evaluarlo).
         Ahora el threshold se calcula con la curva precision-recall SOLO
         sobre las predicciones de TRAIN, y ese umbral fijo se aplica a
         val sin más ajuste. Además se reporta F1/acc con threshold=0.5
         fijo como referencia adicional (columnas *_thr05), para que
         puedas comparar ambos criterios sin ambigüedad. AUC sigue siendo
         la métrica de referencia principal porque no depende de ningún
         umbral.
  FIX-C: Mixup ahora asigna longitud sintética = max(len_i, len_j) en vez
         de max_T, evitando que el modelo "vea" como reales colas de
         padding mezclado.
  FIX-D: semillas fijadas (torch, numpy, random) para reducir varianza
         espuria entre configs — así las diferencias que veamos se deben
         más a la arquitectura/hparams que al azar de inicialización.

No se guarda ningún modelo/checkpoint en disco. Solo métricas por fold.

Ejes de experimentación para BiLSTM (curados, no producto cartesiano
completo para mantener el costo de cómputo razonable):

  POOLING:
    · last        — baseline actual (concat de h_n forward/backward)
    · meanmax     — concat de mean-pool + max-pool sobre timesteps reales
                     + h_n final (más barato que atención, buena mejora típica)
    · attention   — atención aditiva sobre outputs enmascarando padding

  LOSS:
    · bce_posw    — baseline actual (BCEWithLogitsLoss + pos_weight)
    · focal       — Focal Loss (gamma=2), suele calibrar mejor con desbalance

  HIDDEN SIZE:
    · 64 / 96 / 128 / 192

  AUMENTOS:
    · mixup_fix   — Mixup con longitud sintética corregida (FIX-C)
    · mixup_tmask — Mixup + time-masking adicional (enmascara 1-2 segmentos
                     reales aleatorios por secuencia de train)

  SCHEDULER:
    · cosine      — baseline (CosineAnnealingLR)
    · warmup_cos  — 5 épocas de warmup lineal + coseno

La lista EXPERIMENTS al final combina:
  1) Ablaciones aisladas (una variable a la vez, resto = baseline) para
     saber qué aporta cada pieza.
  2) Un grid pequeño de hidden_size con la mejor config encontrada en (1).
  3) Una config "combinada" final con las mejoras que más ayuden.

Como no sabemos de antemano cuál es "la mejor" antes de correr (1),
el diseño es en 2 fases: correr ABLATIONS primero, revisar resultados,
y luego decidir manualmente la combinación final (o dejar prendida la
bandera RUN_COMBINED_GUESS que arma una combinación razonable por
defecto: meanmax + focal + hidden=128 + mixup_fix + warmup_cos).

Uso:
  python dl_script2b_bilstm_optim.py
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

from sklearn.model_selection import RepeatedStratifiedKFold
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
logger = logging.getLogger("DL_S2b_BiLSTM")

# ─── Semillas (FIX-D) ─────────────────────────────────────────────────────
SEED = 42

def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_global_seed(SEED)

# ─── Configuración general ────────────────────────────────────────────────
MODELS_TO_TEST = [
    "xlsr-300m",
    "xlsr-53",
    "whisper-large-encoder",
    "wav2vec2-large-robust",
    "wavlm-large",
    "hubert-large",
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

BASE_LR = 1e-3   # lr base para BiLSTM (igual que en script original)


# =============================================================================
# 1. CARGA DE SECUENCIAS (idéntico al script original)
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
# 2. PREPROCESAMIENTO (idéntico al script original: fit solo en timesteps
#    reales de train, nunca sobre padding ni sobre val)
# =============================================================================

def scale_sequences(X_train, X_val, lengths_train):
    B_tr, T, D = X_train.shape
    B_vl       = X_val.shape[0]
    real_vecs  = np.vstack([X_train[i, :lengths_train[i], :] for i in range(B_tr)])
    scaler     = StandardScaler()
    scaler.fit(real_vecs)
    X_tr_s = scaler.transform(X_train.reshape(-1, D)).reshape(B_tr, T, D)
    X_vl_s = scaler.transform(X_val.reshape(-1, D)).reshape(B_vl, T, D)
    return X_tr_s, X_vl_s


def pca_sequences(X_train, X_val, lengths_train, variance=PCA_VAR):
    B_tr, T, D = X_train.shape
    B_vl       = X_val.shape[0]
    real_vecs  = np.vstack([X_train[i, :lengths_train[i], :] for i in range(B_tr)])
    pca = PCA(n_components=variance, svd_solver="full", random_state=SEED)
    pca.fit(real_vecs)
    X_tr_p = pca.transform(X_train.reshape(-1, D)).reshape(B_tr, T, -1)
    X_vl_p = pca.transform(X_val.reshape(-1, D)).reshape(B_vl, T, -1)
    new_D  = X_tr_p.shape[2]
    return X_tr_p, X_vl_p, new_D


def mixup_sequences_fixed(X_train, y_train, lengths_train, alpha=MIXUP_ALPHA):
    """
    FIX-C: la secuencia sintética recibe length = max(len_i, len_j),
    no max_T. Evita que colas de padding-mezclado se traten como reales.
    """
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    if n_pos < 2:
        return X_train, y_train, lengths_train

    n_syn = n_neg - n_pos
    if n_syn <= 0:
        return X_train, y_train, lengths_train

    rng = np.random.default_rng(SEED)
    X_syn_list = []
    len_syn_list = []
    for _ in range(n_syn):
        i, j  = rng.choice(idx_pos, size=2, replace=False)
        lam   = rng.beta(alpha, alpha)
        x_new = lam * X_train[i] + (1.0 - lam) * X_train[j]
        X_syn_list.append(x_new)
        len_syn_list.append(max(lengths_train[i], lengths_train[j]))

    X_syn = np.array(X_syn_list, dtype=np.float32)
    y_syn = np.ones(n_syn, dtype=np.int64)

    X_out = np.concatenate([X_train, X_syn], axis=0)
    y_out = np.concatenate([y_train, y_syn])
    len_out = list(lengths_train) + len_syn_list
    return X_out, y_out, len_out


def time_mask_sequences(X_train, lengths_train, max_masks=2, rng=None):
    """
    Aumento adicional tipo SpecAugment: enmascara (pone a cero) 1-2
    segmentos REALES elegidos al azar por secuencia, solo en train.
    No cambia 'lengths' (el segmento sigue existiendo temporalmente,
    solo se apaga su contenido) ni toca val.
    """
    if rng is None:
        rng = np.random.default_rng(SEED + 1)
    X_out = X_train.copy()
    for i in range(X_out.shape[0]):
        real_len = lengths_train[i]
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
    """
    BiLSTM con 3 modos de pooling configurables:
      - "last":     h_n final (forward + backward), igual al script original
      - "meanmax":  concat de mean-pool + max-pool (sobre timesteps reales)
                    + h_n final
      - "attention": atención aditiva sobre outputs, enmascarando padding
    """
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
            feat_dim = hidden * 2 * 3  # mean + max + last
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
        return ar < len_t  # (B,T) bool

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

        h_last = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, 2H)

        if self.pooling == "last":
            feat = h_last

        elif self.pooling == "meanmax":
            mask = self._make_mask(B, T, lengths, x.device).unsqueeze(-1).float()  # (B,T,1)
            summed = (outputs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            mean_pool = summed / counts
            masked_out = outputs.masked_fill(mask == 0, float("-inf"))
            max_pool, _ = masked_out.max(dim=1)
            max_pool = torch.nan_to_num(max_pool, neginf=0.0)
            feat = torch.cat([mean_pool, max_pool, h_last], dim=-1)

        elif self.pooling == "attention":
            mask = self._make_mask(B, T, lengths, x.device)  # (B,T) bool
            scores = self.attn(outputs).squeeze(-1)          # (B,T)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
            feat = (outputs * weights).sum(dim=1)

        return self.head(self.dropout(self.norm(feat)))


# =============================================================================
# 5. FOCAL LOSS
# =============================================================================

class FocalLossLogits(nn.Module):
    """
    Focal loss binaria sobre logits, con pos_weight-like balance implícito
    vía alpha. gamma controla cuánto se "enfoca" en ejemplos difíciles.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
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


# =============================================================================
# 7. THRESHOLD Y ENTRENAMIENTO DE UN FOLD
# =============================================================================

def find_best_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thresholds[np.argmax(f1[:-1])])


def train_fold_bilstm(
    X_train, y_train, lengths_train,
    X_val, y_val, lengths_val,
    hidden, pooling, loss_type, use_warmup,
):
    input_dim = X_train.shape[2]
    model = BiLSTMFlexible(input_dim, hidden=hidden, pooling=pooling).to(DEVICE)

    ds_train = SequenceDataset(X_train, y_train, lengths_train)
    ds_val   = SequenceDataset(X_val,   y_val,   lengths_val)
    loader_tr  = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=(len(ds_train) % BATCH_SIZE == 1))
    # loader de train SIN shuffle, para calcular el threshold sobre
    # las predicciones del propio train (nunca sobre val)
    loader_tr_eval = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
    loader_val = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)

    if loss_type == "focal":
        alpha = float(n_pos) / float(n_pos + n_neg + 1e-6)
        alpha = 1.0 - alpha  # más peso a la clase minoritaria
        criterion = FocalLossLogits(alpha=alpha, gamma=2.0)
    else:  # bce_posw
        pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, use_warmup)

    best_auc = -1.0
    best_metrics = {}
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb, lb in loader_tr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb, lb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for Xb, yb, lb in loader_val:
                p = torch.sigmoid(model(Xb.to(DEVICE), lb).squeeze(-1))
                probs_list.extend(p.cpu().numpy().tolist())
                labels_list.extend(yb.cpu().numpy().tolist())

        val_probs  = np.array(probs_list)
        val_labels = np.array(labels_list)

        auc = (roc_auc_score(val_labels, val_probs)
               if len(np.unique(val_labels)) > 1 else 0.0)

        # FIX-A: seleccionamos el checkpoint (en memoria, sin guardar a
        # disco) por AUC, que es la métrica prioritaria y no depende
        # de ningún threshold.
        if auc > best_auc:
            best_auc = auc

            # FIX-B: el threshold para F1/acc se calcula con las
            # predicciones del propio TRAIN (nunca con etiquetas de val).
            # Esto elimina el leakage de threshold-selection que tenía
            # el script original (buscar el umbral óptimo directamente
            # sobre val antes de reportar F1 de ese mismo val).
            model.eval()
            tr_probs_list, tr_labels_list = [], []
            with torch.no_grad():
                for Xb, yb, lb in loader_tr_eval:
                    p = torch.sigmoid(model(Xb.to(DEVICE), lb).squeeze(-1))
                    tr_probs_list.extend(p.cpu().numpy().tolist())
                    tr_labels_list.extend(yb.cpu().numpy().tolist())
            tr_probs  = np.array(tr_probs_list)
            tr_labels = np.array(tr_labels_list)
            thr_from_train = find_best_threshold(tr_labels, tr_probs)

            val_pred_thrtrain = (val_probs >= thr_from_train).astype(int)
            val_pred_thr05    = (val_probs >= 0.5).astype(int)

            best_metrics = {
                "auc":         auc,
                # Métricas con threshold aprendido en TRAIN (sin leakage)
                "f1_macro":    f1_score(val_labels, val_pred_thrtrain, average="macro",  zero_division=0),
                "f1_minority": f1_score(val_labels, val_pred_thrtrain, average="binary", zero_division=0),
                "acc":         accuracy_score(val_labels, val_pred_thrtrain),
                "threshold":   thr_from_train,
                # Métricas con threshold fijo 0.5, como referencia adicional
                "f1_macro_thr05":    f1_score(val_labels, val_pred_thr05, average="macro",  zero_division=0),
                "f1_minority_thr05": f1_score(val_labels, val_pred_thr05, average="binary", zero_division=0),
                "acc_thr05":         accuracy_score(val_labels, val_pred_thr05),
                "best_epoch":  epoch + 1,
                "input_dim":   input_dim,
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return best_metrics


# =============================================================================
# 8. LOOP DE CROSS-VALIDATION PARA UNA CONFIG DE BiLSTM
# =============================================================================

def run_bilstm_experiment(
    X, y, lengths,
    exp_config: dict, emb_model: str,
    results_path: Path,
):
    exp_id   = exp_config["id"]
    hidden   = exp_config["hidden"]
    pooling  = exp_config["pooling"]
    loss_t   = exp_config["loss"]
    use_pca  = exp_config["pca"]
    mixup_t  = exp_config["mixup"]      # None | "fix" | "fix_tmask"
    use_wu   = exp_config["warmup"]

    set_global_seed(SEED)  # misma semilla al empezar cada config
    cv    = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    rows  = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        rep_num  = (fold_idx - 1) // CV_SPLITS + 1
        fold_num = (fold_idx - 1) %  CV_SPLITS + 1

        X_tr, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr, y_vl = y[train_idx].copy(), y[val_idx].copy()
        l_tr = [lengths[i] for i in train_idx]
        l_vl = [lengths[i] for i in val_idx]

        X_tr, X_vl = scale_sequences(X_tr, X_vl, l_tr)

        n_dims = X_tr.shape[2]
        if use_pca:
            X_tr, X_vl, n_dims = pca_sequences(X_tr, X_vl, l_tr)

        if mixup_t in ("fix", "fix_tmask"):
            X_tr, y_tr, l_tr = mixup_sequences_fixed(X_tr, y_tr, l_tr)

        if mixup_t == "fix_tmask":
            rng = np.random.default_rng(SEED + fold_idx)
            X_tr = time_mask_sequences(X_tr, l_tr, max_masks=2, rng=rng)

        metrics = train_fold_bilstm(
            X_tr, y_tr, l_tr, X_vl, y_vl, l_vl,
            hidden=hidden, pooling=pooling, loss_type=loss_t, use_warmup=use_wu,
        )

        row = {
            "emb_model":    emb_model,
            "architecture": "BiLSTM",
            "experiment":   exp_id,
            "hidden":       hidden,
            "pooling":      pooling,
            "loss":         loss_t,
            "pca":          use_pca,
            "mixup":        mixup_t,
            "warmup":       use_wu,
            "repeat":       rep_num,
            "fold":         fold_num,
            "n_train_real": len(train_idx),
            "n_train_aug":  len(X_tr),
            "n_val":        len(val_idx),
            "n_dims":       n_dims,
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
                f"      Rep{rep_num:>2} | "
                f"AUC={recent['auc'].mean():.3f} | "
                f"F1-macro={recent['f1_macro'].mean():.3f} | dims={n_dims}"
            )

    df = pd.DataFrame(rows)
    logger.info(
        f"    ✅ [{exp_id}] "
        f"AUC={df['auc'].mean():.3f}±{df['auc'].std():.3f} | "
        f"F1={df['f1_macro'].mean():.3f} | F1-min={df['f1_minority'].mean():.3f}"
    )
    return df


# =============================================================================
# 9. DEFINICIÓN DE EXPERIMENTOS (BiLSTM únicamente)
# =============================================================================

BASELINE = dict(hidden=128, pooling="last", loss="bce_posw", pca=False, mixup="fix", warmup=False)

def cfg(id_, **overrides):
    c = dict(BASELINE)
    c.update(overrides)
    c["id"] = id_
    return c

# --- Fase 1: ablaciones aisladas (cada una cambia UNA variable) ---------
ABLATIONS = [
    cfg("baseline_mixupfix"),                              # baseline + solo el fix de longitud (FIX-C)
    cfg("pooling_meanmax",   pooling="meanmax"),
    cfg("pooling_attention", pooling="attention"),
    cfg("loss_focal",        loss="focal"),
    cfg("warmup_cosine",     warmup=True),
    cfg("aug_tmask",         mixup="fix_tmask"),
    cfg("pca_on",            pca=True),
]

# --- Fase 2: grid de hidden_size (se corre para el pooling baseline) ----
HIDDEN_GRID = [
    cfg(f"hidden_{h}", hidden=h) for h in [64, 96, 128, 192]
]

# --- Fase 3: combinación razonable de las mejoras esperadas -------------
# (ajusta manualmente después de ver resultados de Fase 1 si quieres
#  otra combinación distinta; esta es una apuesta razonable de partida)
COMBINED_GUESS = [
    cfg("combined_meanmax_focal_warmup",
        pooling="meanmax", loss="focal", warmup=True, mixup="fix"),
    cfg("combined_attention_focal_warmup",
        pooling="attention", loss="focal", warmup=True, mixup="fix"),
    cfg("combined_meanmax_focal_warmup_tmask",
        pooling="meanmax", loss="focal", warmup=True, mixup="fix_tmask"),
]

ALL_EXPERIMENTS = ABLATIONS + HIDDEN_GRID + COMBINED_GUESS


# =============================================================================
# 10. PIPELINE PRINCIPAL
# =============================================================================

def run_all(embeddings_root: str, output_dir: str):
    output_dir = Path(output_dir) / "dl_script2b_bilstm_optim"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = output_dir / "raw_folds_bilstm.csv"
    summary_path = output_dir / "summary_bilstm.csv"

    if raw_path.exists():
        logger.warning(f"⚠️  {raw_path} ya existe. Se añaden resultados al final.")

    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 Embeddings:   {embeddings_root}")
    logger.info(f"📂 Salida:       {output_dir}")
    logger.info(f"🔁 CV: {CV_REPEATS} repeats × {CV_SPLITS} folds = {CV_REPEATS*CV_SPLITS} evals/config")
    logger.info(f"🧪 Total configs BiLSTM a correr: {len(ALL_EXPERIMENTS)}")
    for c in ALL_EXPERIMENTS:
        logger.info(f"    · {c['id']}")
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

    # ── Resumen final ──────────────────────────────────────────────────────
    if raw_path.exists():
        df_all = pd.read_csv(raw_path)
        summary = (
            df_all
            .groupby(["emb_model", "experiment", "hidden", "pooling", "loss", "mixup", "warmup"])
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
                mean_acc_thr05        =("acc_thr05",         "mean"),
                n_folds               =("auc",               "count"),
            )
            .reset_index()
            .sort_values("mean_auc", ascending=False)
        )
        summary.to_csv(summary_path, index=False)

        logger.info("\n" + "="*65)
        logger.info("📊 TOP 20 CONFIGURACIONES BiLSTM (por AUC)")
        logger.info("="*65)
        cols = ["emb_model","experiment","hidden","pooling","loss","mixup","warmup",
                "mean_auc","std_auc","mean_f1_macro","mean_f1_minority",
                "mean_f1_macro_thr05","mean_f1_minority_thr05"]
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
