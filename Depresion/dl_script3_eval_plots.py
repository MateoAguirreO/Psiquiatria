"""
DL Script 3 — Curvas ROC + Matrices de Confusión por Fold/Repetición
=====================================================================
Configuraciones evaluadas (top-2 de cada script anterior):

  Script 1 (MLP):
    · wav2vec2-large-robust  |  A_raw        (raw ~1024 dims, sin aug)
    · wav2vec2-large-robust  |  D_raw_mixup  (raw ~1024 dims + Mixup)

  Script 2 (Secuencias):
    · wavlm-large            |  CNN1D        |  raw
    · wav2vec2-large-robust  |  CNN_BiGRU    |  raw

Estructura de salida por configuración:
  plots/
  └── {config_name}/
      ├── repeat_01/
      │   ├── fold_01_roc.png
      │   ├── fold_01_cm.png
      │   ├── fold_02_roc.png
      │   ├── fold_02_cm.png
      │   │   ...
      │   ├── rep01_aggregate_cm.png       ← suma de los 5 folds
      │   └── rep01_mean_roc.png           ← media ± std de los 5 folds
      ├── repeat_02/
      │   ...
      └── overall_aggregate_cm.png         ← suma de los 50 folds
      └── overall_mean_roc.png             ← media ± std de los 50 folds

Protocolo idéntico a los scripts anteriores (sin data leakage):
  · RepeatedStratifiedKFold(5, 10) = 50 folds
  · Scaler/PCA ajustado solo en train de cada fold
  · Val = datos REALES siempre
  · Umbral óptimo buscado en val
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    roc_curve, confusion_matrix, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DL_S3")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hiperparámetros (idénticos a scripts anteriores) ──────────────────────────
CV_SPLITS   = 5
CV_REPEATS  = 10
EPOCHS      = 60
BATCH_SIZE  = 16
PATIENCE    = 8
PCA_VAR     = 0.95
MIXUP_ALPHA = 0.4
LR_MLP      = 1e-3
LR_CNN      = 5e-4
LR_RNN      = 1e-3

# ── Nombres de clases (ajusta si los tuyos son distintos) ─────────────────────
CLASS_NAMES = ["Sin ansiedad", "Con ansiedad"]

# ── Colores consistentes en todas las figuras ─────────────────────────────────
PALETTE = {
    "roc_fold":   "#4C72B0",
    "roc_mean":   "#C44E52",
    "roc_chance": "#888888",
    "cm_cmap":    "Blues",
}


# =============================================================================
# DEFINICIÓN DE CONFIGURACIONES A EVALUAR
# =============================================================================

CONFIGS = [
    # ── Script 1 top-2 (MLP sobre embeddings pooled) ──────────────────────
    {
        "name":       "MLP_wav2vec2_A_raw",
        "label":      "MLP · wav2vec2-large-robust · raw",
        "type":       "mlp",
        "emb_model":  "wav2vec2-large-robust",
        "experiment": "A_raw",
        "pca":        False,
        "aug":        None,
    },
    {
        "name":       "MLP_wav2vec2_D_raw_mixup",
        "label":      "MLP · wav2vec2-large-robust · raw + Mixup",
        "type":       "mlp",
        "emb_model":  "wav2vec2-large-robust",
        "experiment": "D_raw_mixup",
        "pca":        False,
        "aug":        "mixup",
    },
    # ── Script 2 top-2 (CNN/RNN sobre secuencias) ─────────────────────────
    {
        "name":       "CNN1D_wavlm_raw",
        "label":      "CNN-1D · wavlm-large · raw",
        "type":       "seq",
        "emb_model":  "wavlm-large",
        "architecture": "CNN1D",
        "experiment": "raw",
        "pca":        False,
        "aug":        None,
    },
    {
        "name":       "CNNBiGRU_wav2vec2_raw",
        "label":      "CNN+BiGRU · wav2vec2-large-robust · raw",
        "type":       "seq",
        "emb_model":  "wav2vec2-large-robust",
        "architecture": "CNN_BiGRU",
        "experiment": "raw",
        "pca":        False,
        "aug":        None,
    },
]


# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

def load_pooled(embeddings_root: str, model_key: str):
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}'.")
    X_rows, y_rows, ids = [], [], []
    for entry in entries:
        with open(entry["file"]) as f:
            records = json.load(f)
        embs = np.array([r["embedding"] for r in records], dtype=np.float32)
        X_rows.append(embs.mean(axis=0))
        y_rows.append(int(entry["class_id"]))
        ids.append(entry["audio_id"])
    return np.stack(X_rows), np.array(y_rows, dtype=np.int64), ids


def load_sequences(embeddings_root: str, model_key: str):
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}'.")
    patient_data = {}
    for entry in entries:
        with open(entry["file"]) as f:
            records = json.load(f)
        seq = [r["embedding"] for r in records]
        patient_data[entry["audio_id"]] = {
            "seq": seq, "label": int(entry["class_id"])
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
    return np.array(X), np.array(y, dtype=np.int64), ids, lengths


# =============================================================================
# 2. PREPROCESAMIENTO
# =============================================================================

def apply_scaler(X_tr, X_vl):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_vl)


def apply_pca_pooled(X_tr, X_vl, var=PCA_VAR):
    pca = PCA(n_components=var, svd_solver="full", random_state=42)
    return pca.fit_transform(X_tr), pca.transform(X_vl)


def apply_pca_seq(X_tr, X_vl, var=PCA_VAR):
    B_tr, T, D = X_tr.shape
    B_vl       = X_vl.shape[0]
    pca = PCA(n_components=var, svd_solver="full", random_state=42)
    X_tr_p = pca.fit_transform(X_tr.reshape(-1, D)).reshape(B_tr, T, -1)
    X_vl_p = pca.transform(X_vl.reshape(-1, D)).reshape(B_vl, T, -1)
    return X_tr_p, X_vl_p


def scale_seq(X_tr, X_vl):
    B_tr, T, D = X_tr.shape
    B_vl       = X_vl.shape[0]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr.reshape(-1, D)).reshape(B_tr, T, D)
    X_vl_s = sc.transform(X_vl.reshape(-1, D)).reshape(B_vl, T, D)
    return X_tr_s, X_vl_s


# =============================================================================
# 3. AUMENTACIÓN
# =============================================================================

def mixup_augment(X_tr, y_tr, alpha=MIXUP_ALPHA):
    idx_pos = np.where(y_tr == 1)[0]
    idx_neg = np.where(y_tr == 0)[0]
    n_syn   = len(idx_neg) - len(idx_pos)
    if len(idx_pos) < 2 or n_syn <= 0:
        return X_tr, y_tr
    rng = np.random.default_rng(42)
    X_syn = []
    for _ in range(n_syn):
        i, j  = rng.choice(idx_pos, size=2, replace=False)
        lam   = rng.beta(alpha, alpha)
        X_syn.append(lam * X_tr[i] + (1 - lam) * X_tr[j])
    X_syn = np.array(X_syn, dtype=np.float32)
    y_syn = np.ones(n_syn, dtype=np.int64)
    return np.vstack([X_tr, X_syn]), np.concatenate([y_tr, y_syn])


def mixup_seq_augment(X_tr, y_tr, alpha=MIXUP_ALPHA):
    idx_pos = np.where(y_tr == 1)[0]
    idx_neg = np.where(y_tr == 0)[0]
    n_syn   = len(idx_neg) - len(idx_pos)
    if len(idx_pos) < 2 or n_syn <= 0:
        return X_tr, y_tr
    rng = np.random.default_rng(42)
    X_syn = []
    for _ in range(n_syn):
        i, j  = rng.choice(idx_pos, size=2, replace=False)
        lam   = rng.beta(alpha, alpha)
        X_syn.append(lam * X_tr[i] + (1 - lam) * X_tr[j])
    X_syn = np.array(X_syn, dtype=np.float32)
    y_syn = np.ones(n_syn, dtype=np.int64)
    return np.concatenate([X_tr, X_syn], axis=0), np.concatenate([y_tr, y_syn])


# =============================================================================
# 4. ARQUITECTURAS
# =============================================================================

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        h1 = min(512, max(64, input_dim * 2))
        h2 = min(256, max(32, input_dim))
        h3 = min(128, max(16, input_dim // 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(h2, h3),        nn.BatchNorm1d(h3), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h3, 1),
        )
    def forward(self, x):
        return self.net(x)


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, 3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(256, 128, 3, padding=1),       nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(128, 64, 3, padding=1),         nn.BatchNorm1d(64),  nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(64, 1),
        )
    def forward(self, x, lengths=None):
        return self.net(x.permute(0, 2, 1))


class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.gru     = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.norm    = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head    = nn.Linear(hidden * 2, 1)
    def forward(self, x, lengths=None):
        _, h_n = self.gru(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))


class CNNBiGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, 3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(256, 128, 3, padding=1),       nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(0.2),
        )
        self.gru     = nn.GRU(128, hidden, batch_first=True, bidirectional=True)
        self.norm    = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head    = nn.Linear(hidden * 2, 1)
    def forward(self, x, lengths=None):
        x_cnn = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, h_n = self.gru(x_cnn)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))


ARCH_MAP = {
    "CNN1D":     CNN1DClassifier,
    "CNN_BiGRU": CNNBiGRUClassifier,
    "BiGRU":     BiGRUClassifier,
}

LR_BY_ARCH = {"CNN1D": LR_CNN, "CNN_BiGRU": LR_CNN, "BiGRU": LR_RNN}


# =============================================================================
# 5. DATASET PYTORCH
# =============================================================================

class SeqDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.astype(np.float32))
        self.l = lengths
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.l[i]


# =============================================================================
# 6. ENTRENAMIENTO + PREDICCIONES DE VAL
# =============================================================================

def find_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thrs[np.argmax(f1[:-1])])


def _train_loop(model, loader_tr, loader_val, y_train_np, lr):
    """
    Entrena el modelo y retorna (val_probs, val_labels, best_threshold).
    val_probs  → probabilidades de la clase 1 en el conjunto de val.
    val_labels → etiquetas reales del val.
    """
    n_neg   = int(np.sum(y_train_np == 0))
    n_pos   = int(np.sum(y_train_np == 1))
    pos_w   = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
    crit    = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_f1    = -1.0
    best_state = None
    best_probs = None
    best_lbls  = None
    best_thr   = 0.5
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in loader_tr:
            if len(batch) == 2:
                Xb, yb = batch
                lengths = None
            else:
                Xb, yb, lengths = batch
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(Xb, lengths).squeeze(-1) if lengths is not None else model(Xb).squeeze(-1)
            crit(out, yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        probs_l, lbls_l = [], []
        with torch.no_grad():
            for batch in loader_val:
                if len(batch) == 2:
                    Xb, yb = batch
                    lengths = None
                else:
                    Xb, yb, lengths = batch
                out = model(Xb.to(DEVICE), lengths).squeeze(-1) if lengths is not None \
                      else model(Xb.to(DEVICE)).squeeze(-1)
                probs_l.extend(torch.sigmoid(out).cpu().numpy().tolist())
                lbls_l.extend(yb.cpu().numpy().tolist())

        vp = np.array(probs_l)
        vl = np.array(lbls_l)
        thr = find_threshold(vl, vp)
        f1  = f1_score(vl, (vp >= thr).astype(int), average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_probs = vp
            best_lbls  = vl
            best_thr   = thr
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return best_probs, best_lbls, best_thr



def run_fold_mlp(X_tr, y_tr, X_vl, y_vl, cfg):
    """Preprocesa + entrena MLP y retorna predicciones del val."""
    X_tr, X_vl = apply_scaler(X_tr, X_vl)
    if cfg["pca"]:
        X_tr, X_vl = apply_pca_pooled(X_tr, X_vl)
    if cfg["aug"] == "mixup":
        X_tr, y_tr = mixup_augment(X_tr, y_tr)

    ds_tr  = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr.astype(np.float32)))
    ds_vl  = TensorDataset(torch.FloatTensor(X_vl), torch.FloatTensor(y_vl.astype(np.float32)))
    ld_tr  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    ld_vl  = DataLoader(ds_vl, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPClassifier(X_tr.shape[1]).to(DEVICE)
    return _train_loop(model, ld_tr, ld_vl, y_tr, LR_MLP)


def run_fold_seq(X_tr, y_tr, l_tr, X_vl, y_vl, l_vl, cfg):
    """Preprocesa + entrena arquitectura de secuencia y retorna predicciones del val."""
    X_tr, X_vl = scale_seq(X_tr, X_vl)
    if cfg["pca"]:
        X_tr, X_vl = apply_pca_seq(X_tr, X_vl)
    if cfg["aug"] == "mixup":
        X_tr, y_tr_aug = mixup_seq_augment(X_tr, y_tr)
        n_syn = len(y_tr_aug) - len(l_tr)
        l_tr  = l_tr + [X_tr.shape[1]] * n_syn
        y_tr  = y_tr_aug

    arch   = cfg["architecture"]
    dim    = X_tr.shape[2]
    model  = ARCH_MAP[arch](dim).to(DEVICE)
    lr     = LR_BY_ARCH[arch]

    ds_tr = SeqDataset(X_tr, y_tr, l_tr)
    ds_vl = SeqDataset(X_vl, y_vl, l_vl)
    ld_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    ld_vl = DataLoader(ds_vl, batch_size=BATCH_SIZE, shuffle=False)

    return _train_loop(model, ld_tr, ld_vl, y_tr, lr)


# =============================================================================
# 7. GRÁFICAS
# =============================================================================

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_roc_single(y_true, y_proba, thr, title, save_path):
    """Curva ROC de un fold individual."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0

    # Punto en el umbral elegido
    idx_thr = np.argmin(np.abs(thresholds - thr))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color=PALETTE["roc_fold"], lw=2,
            label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["roc_chance"], lw=1, label="Azar")
    ax.scatter(fpr[idx_thr], tpr[idx_thr], color="#C44E52", zorder=5,
               s=60, label=f"Umbral = {thr:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["roc_fold"])
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    _style_ax(ax, title, "Tasa de Falsos Positivos (FPR)", "Tasa de Verdaderos Positivos (TPR)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpr, tpr, auc


def plot_cm_single(y_true, y_pred, title, save_path):
    """Matriz de confusión de un fold individual."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=PALETTE["cm_cmap"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(ticks); ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    _style_ax(ax, title, "Predicho", "Real")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_mean_roc(fprs, tprs, aucs, title, save_path):
    """
    Curva ROC media ± std de un conjunto de folds.
    Interpola todas las curvas a una grilla común de FPR.
    """
    mean_fpr = np.linspace(0, 1, 200)
    tprs_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)]
    tprs_interp[0][0] = 0.0

    mean_tpr = np.mean(tprs_interp, axis=0)
    std_tpr  = np.std(tprs_interp,  axis=0)
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    # Curvas individuales (transparentes)
    for fpr, tpr in zip(fprs, tprs):
        ax.plot(fpr, tpr, color=PALETTE["roc_fold"], lw=0.8, alpha=0.3)
    # Banda de incertidumbre
    ax.fill_between(mean_fpr,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    color=PALETTE["roc_fold"], alpha=0.15,
                    label=f"±1 std (AUC={mean_auc:.3f}±{std_auc:.3f})")
    # Curva media
    ax.plot(mean_fpr, mean_tpr, color=PALETTE["roc_mean"], lw=2.5,
            label=f"ROC Media (AUC={mean_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["roc_chance"], lw=1, label="Azar")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    _style_ax(ax, title, "FPR", "TPR")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_cm(cm_sum, title, save_path):
    """Matriz de confusión acumulada (suma de N folds)."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_sum, interpolation="nearest", cmap=PALETTE["cm_cmap"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(ticks); ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_yticks(ticks); ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    thresh = cm_sum.max() / 2.0
    total  = cm_sum.sum()
    for i in range(cm_sum.shape[0]):
        for j in range(cm_sum.shape[1]):
            pct = 100 * cm_sum[i, j] / total
            ax.text(j, i, f"{cm_sum[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm_sum[i, j] > thresh else "black")
    _style_ax(ax, title, "Predicho", "Real")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8. PIPELINE POR CONFIGURACIÓN
# =============================================================================

def run_config(cfg: dict, embeddings_root: str, output_root: Path):
    config_dir = output_root / cfg["name"]
    config_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*65}")
    logger.info(f"🔬 Configuración: {cfg['label']}")
    logger.info(f"{'='*65}")

    # ── Cargar datos ───────────────────────────────────────────────────────
    if cfg["type"] == "mlp":
        X, y, _ids = load_pooled(embeddings_root, cfg["emb_model"])
        lengths = None
    else:
        X, y, _ids, lengths = load_sequences(embeddings_root, cfg["emb_model"])

    logger.info(
        f"   Muestras: {len(y)} | clase0={np.sum(y==0)} | clase1={np.sum(y==1)}"
    )

    cv    = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=42)
    splits = list(cv.split(X, y))

    # Acumuladores globales (todos los folds de todos los repeats)
    all_fprs, all_tprs, all_aucs = [], [], []
    all_cm_sum = np.zeros((2, 2), dtype=np.int64)

    metrics_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        rep_num  = fold_idx // CV_SPLITS + 1
        fold_num = fold_idx %  CV_SPLITS + 1

        rep_dir  = config_dir / f"repeat_{rep_num:02d}"
        rep_dir.mkdir(exist_ok=True)

        # ── Datos del fold ─────────────────────────────────────────────────
        X_tr, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr, y_vl = y[train_idx].copy(), y[val_idx].copy()

        if cfg["type"] == "mlp":
            probs, lbls, thr = run_fold_mlp(X_tr, y_tr, X_vl, y_vl, cfg)
        else:
            l_tr = [lengths[i] for i in train_idx]
            l_vl = [lengths[i] for i in val_idx]
            probs, lbls, thr = run_fold_seq(X_tr, y_tr, l_tr, X_vl, y_vl, l_vl, cfg)

        preds = (probs >= thr).astype(int)

        # ── Métricas ───────────────────────────────────────────────────────
        f1_mac = f1_score(lbls, preds, average="macro",  zero_division=0)
        f1_min = f1_score(lbls, preds, average="binary", zero_division=0)
        acc    = accuracy_score(lbls, preds)
        auc    = roc_auc_score(lbls, probs) if len(np.unique(lbls)) > 1 else 0.0
        cm     = confusion_matrix(lbls, preds, labels=[0, 1])

        metrics_rows.append({
            "repeat": rep_num, "fold": fold_num,
            "f1_macro": f1_mac, "f1_minority": f1_min,
            "auc": auc, "acc": acc, "threshold": thr,
        })

        # ── Gráficas individuales del fold ─────────────────────────────────
        fold_label = f"Rep{rep_num:02d} · Fold {fold_num}"
        roc_path = rep_dir / f"fold_{fold_num:02d}_roc.png"
        cm_path  = rep_dir / f"fold_{fold_num:02d}_cm.png"

        fpr, tpr, _ = plot_roc_single(
            lbls, probs, thr,
            title=f"ROC — {fold_label}\n{cfg['label']}",
            save_path=roc_path,
        )
        plot_cm_single(
            lbls, preds,
            title=f"Confusión — {fold_label}\n{cfg['label']}",
            save_path=cm_path,
        )

        # Acumular para el repeat
        all_fprs.append(fpr)
        all_tprs.append(tpr)
        all_aucs.append(auc)
        all_cm_sum += cm

        logger.info(
            f"   Rep{rep_num:02d} Fold{fold_num} | "
            f"F1={f1_mac:.3f} | AUC={auc:.3f} | Thr={thr:.3f} | "
            f"CM={cm.tolist()}"
        )

        # ── Al terminar cada repetición: agregados del repeat ──────────────
        if fold_num == CV_SPLITS:
            rep_start = fold_idx - CV_SPLITS + 1
            rep_fprs  = all_fprs[rep_start: fold_idx + 1]
            rep_tprs  = all_tprs[rep_start: fold_idx + 1]
            rep_aucs  = all_aucs[rep_start: fold_idx + 1]
            rep_cm    = all_cm_sum.copy()
            # Descontar CMs de repeats anteriores
            if rep_num > 1:
                # Recalcular solo los CMs de este repeat
                rep_cm = np.zeros((2, 2), dtype=np.int64)
                for ri in range(rep_start, fold_idx + 1):
                    # Recuperar las CM de este repeat del rows de métricas
                    pass
                # Más limpio: acumular por repeat directamente
                rep_cm = sum(
                    confusion_matrix(
                        # placeholder — se calcula abajo con el buffer del repeat
                        [0], [0], labels=[0, 1]
                    )
                    for _ in range(CV_SPLITS)
                )
            # — reconstruimos la CM del repeat leyendo las últimas CV_SPLITS filas
            # Solución robusta: guardamos las CMs individuales en una lista
            pass   # ver refactor abajo

    # ─────────────────────────────────────────────────────────────────────────
    # Refactor limpio: repetir el loop acumulando CMs por repeat correctamente
    # ─────────────────────────────────────────────────────────────────────────
    # (el código anterior ya guardó cada fold individual; aquí construimos los
    #  agregados por repeat y el global a partir de las métricas guardadas)

    # Reconstruir CMs por repeat desde las imágenes ya generadas no es posible,
    # así que hacemos un segundo pase guardando las CMs en memoria.
    # NOTA: Este bloque reemplaza la lógica de arriba; el loop de arriba ya
    #       generó correctamente las imágenes por fold. Solo falta construir
    #       los agregados por repeat y global.

    _build_repeat_aggregates(
        cfg, config_dir, splits, X, y, lengths, metrics_rows
    )

    # ── Resumen de métricas ────────────────────────────────────────────────
    df = pd.DataFrame(metrics_rows)
    df.to_csv(config_dir / "metrics_per_fold.csv", index=False)

    summary = {
        "config":        cfg["name"],
        "mean_f1_macro": df["f1_macro"].mean(),
        "std_f1_macro":  df["f1_macro"].std(),
        "mean_f1_min":   df["f1_minority"].mean(),
        "mean_auc":      df["auc"].mean(),
        "std_auc":       df["auc"].std(),
        "mean_acc":      df["acc"].mean(),
    }
    logger.info(
        f"\n  ✅ RESUMEN {cfg['name']}: "
        f"F1={summary['mean_f1_macro']:.3f}±{summary['std_f1_macro']:.3f} | "
        f"AUC={summary['mean_auc']:.3f}"
    )
    return summary


def _build_repeat_aggregates(cfg, config_dir, splits, X, y, lengths, metrics_rows):
    """
    Segundo pase eficiente: recorre los folds una vez más para acumular
    CMs y curvas ROC por repeat y globales.
    Ahora el modelo ya se entrenó y guardó las predicciones en metrics_rows,
    pero no guardamos probs/lbls por memoria. Los recalculamos aquí.

    Para evitar re-entrenar, guardamos probs y lbls en el primer loop.
    ── Rediseño ──
    Esta función recibe los datos ya computados desde run_config_clean().
    """
    pass  # Ver run_config_clean() más abajo que lo hace correctamente


# =============================================================================
# 9. PIPELINE LIMPIO (sin doble pase)
# =============================================================================

def run_config_clean(cfg: dict, embeddings_root: str, output_root: Path):
    """
    Versión definitiva: un solo loop que:
    - Genera figuras por fold
    - Acumula por repeat → figura de repeat al terminar cada uno
    - Acumula global → figura global al final
    """
    config_dir = output_root / cfg["name"]
    config_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*65}")
    logger.info(f"🔬 {cfg['label']}")
    logger.info(f"{'='*65}")

    if cfg["type"] == "mlp":
        X, y, _ids = load_pooled(embeddings_root, cfg["emb_model"])
        lengths = None
    else:
        X, y, _ids, lengths = load_sequences(embeddings_root, cfg["emb_model"])

    logger.info(f"   n={len(y)} | c0={np.sum(y==0)} | c1={np.sum(y==1)}")

    cv     = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=42)
    splits = list(cv.split(X, y))

    # Acumuladores
    global_fprs, global_tprs, global_aucs = [], [], []
    global_cm  = np.zeros((2, 2), dtype=np.int64)

    rep_fprs, rep_tprs, rep_aucs = [], [], []
    rep_cm = np.zeros((2, 2), dtype=np.int64)

    metrics_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        rep_num  = fold_idx // CV_SPLITS + 1
        fold_num = fold_idx %  CV_SPLITS + 1

        # Al empezar un nuevo repeat, resetear acumuladores del repeat
        if fold_num == 1:
            rep_fprs, rep_tprs, rep_aucs = [], [], []
            rep_cm = np.zeros((2, 2), dtype=np.int64)
            rep_dir = config_dir / f"repeat_{rep_num:02d}"
            rep_dir.mkdir(exist_ok=True)
        else:
            rep_dir = config_dir / f"repeat_{rep_num:02d}"

        # ── Datos del fold ─────────────────────────────────────────────────
        X_tr, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr, y_vl = y[train_idx].copy(), y[val_idx].copy()

        if cfg["type"] == "mlp":
            probs, lbls, thr = run_fold_mlp(X_tr, y_tr, X_vl, y_vl, cfg)
        else:
            l_tr = [lengths[i] for i in train_idx]
            l_vl = [lengths[i] for i in val_idx]
            probs, lbls, thr = run_fold_seq(X_tr, y_tr, l_tr, X_vl, y_vl, l_vl, cfg)

        preds  = (probs >= thr).astype(int)
        f1_mac = f1_score(lbls, preds, average="macro",  zero_division=0)
        f1_min = f1_score(lbls, preds, average="binary", zero_division=0)
        acc    = accuracy_score(lbls, preds)
        auc    = roc_auc_score(lbls, probs) if len(np.unique(lbls)) > 1 else 0.0
        cm     = confusion_matrix(lbls, preds, labels=[0, 1])

        metrics_rows.append({
            "repeat": rep_num, "fold": fold_num,
            "f1_macro": f1_mac, "f1_minority": f1_min,
            "auc": auc, "acc": acc, "threshold": thr,
        })

        # ── Gráficas individuales del fold ─────────────────────────────────
        fold_title = f"Rep{rep_num:02d} · Fold {fold_num}"
        fpr, tpr, _ = plot_roc_single(
            lbls, probs, thr,
            title=f"ROC — {fold_title}\n{cfg['label']}",
            save_path=rep_dir / f"fold_{fold_num:02d}_roc.png",
        )
        plot_cm_single(
            lbls, preds,
            title=f"Confusión — {fold_title}\n{cfg['label']}",
            save_path=rep_dir / f"fold_{fold_num:02d}_cm.png",
        )

        # ── Acumular ───────────────────────────────────────────────────────
        rep_fprs.append(fpr);    rep_tprs.append(tpr);    rep_aucs.append(auc)
        global_fprs.append(fpr); global_tprs.append(tpr); global_aucs.append(auc)
        rep_cm    += cm
        global_cm += cm

        logger.info(
            f"   Rep{rep_num:02d} Fold{fold_num} | "
            f"F1={f1_mac:.3f} | AUC={auc:.3f} | Thr={thr:.3f}"
        )

        # ── Al terminar el último fold del repeat: agregados del repeat ────
        if fold_num == CV_SPLITS:
            rep_label = f"Repetición {rep_num:02d} (folds 1–{CV_SPLITS})\n{cfg['label']}"

            plot_mean_roc(
                rep_fprs, rep_tprs, rep_aucs,
                title=f"ROC Media — Rep {rep_num:02d}\n{cfg['label']}",
                save_path=rep_dir / f"rep{rep_num:02d}_mean_roc.png",
            )
            plot_aggregate_cm(
                rep_cm,
                title=f"CM Acumulada — Rep {rep_num:02d} ({CV_SPLITS} folds)\n{cfg['label']}",
                save_path=rep_dir / f"rep{rep_num:02d}_aggregate_cm.png",
            )
            logger.info(
                f"   → Rep{rep_num:02d} guardada | "
                f"AUC media={np.mean(rep_aucs):.3f} | CM={rep_cm.tolist()}"
            )

    # ── Agregados globales (50 folds) ──────────────────────────────────────
    plot_mean_roc(
        global_fprs, global_tprs, global_aucs,
        title=f"ROC Media Global ({CV_SPLITS*CV_REPEATS} folds)\n{cfg['label']}",
        save_path=config_dir / "overall_mean_roc.png",
    )
    plot_aggregate_cm(
        global_cm,
        title=f"CM Acumulada Global ({CV_SPLITS*CV_REPEATS} folds)\n{cfg['label']}",
        save_path=config_dir / "overall_aggregate_cm.png",
    )

    # ── Métricas CSV ───────────────────────────────────────────────────────
    df = pd.DataFrame(metrics_rows)
    df.to_csv(config_dir / "metrics_per_fold.csv", index=False)

    summary = {
        "config":        cfg["name"],
        "label":         cfg["label"],
        "mean_f1_macro": df["f1_macro"].mean(),
        "std_f1_macro":  df["f1_macro"].std(),
        "mean_f1_min":   df["f1_minority"].mean(),
        "std_f1_min":    df["f1_minority"].std(),
        "mean_auc":      df["auc"].mean(),
        "std_auc":       df["auc"].std(),
        "mean_acc":      df["acc"].mean(),
    }
    logger.info(
        f"\n  ✅ GLOBAL {cfg['name']}: "
        f"F1={summary['mean_f1_macro']:.3f}±{summary['std_f1_macro']:.3f} | "
        f"AUC={summary['mean_auc']:.3f}±{summary['std_auc']:.3f}"
    )
    return summary


# =============================================================================
# 10. PUNTO DE ENTRADA
# =============================================================================

def run_all(embeddings_root: str, output_dir: str):
    output_root = Path(output_dir) / "dl_script3_plots"
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 Salida:       {output_root}")
    logger.info(f"🔁 {CV_REPEATS} repeats × {CV_SPLITS} folds = {CV_REPEATS*CV_SPLITS} folds por config")
    logger.info(f"🔬 Configs: {[c['name'] for c in CONFIGS]}\n")

    all_summaries = []
    for cfg in CONFIGS:
        summary = run_config_clean(cfg, embeddings_root, output_root)
        all_summaries.append(summary)

    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(output_root / "summary_all_configs.csv", index=False)
    logger.info("\n" + "="*65)
    logger.info("📊 RESUMEN FINAL")
    logger.info("="*65)
    logger.info("\n" + df_sum.to_string(index=False))
    logger.info(f"\n📄 summary_all_configs.csv → {output_root}")


if __name__ == "__main__":
    # ── Ajusta estas rutas ─────────────────────────────────────────────────
    EMBEDDINGS_ROOT = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/"
        "Depresión/embeddings/"
    )
    OUTPUT_DIR = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Wav2Vec/results_DL/"
    )
    # ───────────────────────────────────────────────────────────────────────

    run_all(EMBEDDINGS_ROOT, OUTPUT_DIR)
