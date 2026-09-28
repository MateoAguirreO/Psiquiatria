"""
DL Script 2 — Barrido de Arquitecturas sobre Secuencias Temporales
==================================================================
Arquitecturas:
  · CNN-1D       — patrones locales en la secuencia de segmentos
  · BiLSTM       — dependencias temporales bidireccionales (LSTM)
  · BiGRU        — dependencias temporales bidireccionales (GRU, más ligero)
  · CNN + BiGRU  — CNN extrae features locales, GRU integra la secuencia

Entrada:
  Secuencias de embeddings por paciente: (n_patients, max_segments, embed_dim)
  Zero-padding dinámico al máximo número de segmentos.

Reducción de features aplicada a la DIMENSIÓN de embedding (no temporal):
  · raw  — embedding completo por segmento
  · pca  — PCA aplicada a la dimensión D de cada segmento
             Se ajusta sobre X_train aplanado (n_train * T, D) para no
             filtrar información del fold.

Protocolo idéntico al Script 1:
  · RepeatedStratifiedKFold(5, 10) = 50 evaluaciones por config
  · Todo preprocesamiento ajustado solo en train del fold
  · VAL = datos REALES del paciente, nunca sintéticos

Nota sobre datos sintéticos y secuencias:
  Mixup puede aplicarse a secuencias completas (interpolación de (T,D)×(T,D)).
  SMOTE sobre secuencias requiere aplanar y repadear, lo que puede distorsionar
  la estructura temporal. Por eso aquí solo se usa Mixup + pos_weight.

Uso:
  python dl_script2_architectures.py

Ajusta EMBEDDINGS_ROOT y OUTPUT_DIR al final del archivo.
"""

import json
import logging
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
logger = logging.getLogger("DL_S2")

# ─── Configuración ────────────────────────────────────────────────────────────
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

# Combinaciones de feature-space × augmentación a evaluar
EXPERIMENTS = [
    {"id": "raw",       "pca": False, "mixup": False},
    {"id": "pca",       "pca": True,  "mixup": False},
    {"id": "raw_mixup", "pca": False, "mixup": True},
    {"id": "pca_mixup", "pca": True,  "mixup": True},
]


# =============================================================================
# 1. CARGA DE SECUENCIAS
# =============================================================================

def load_sequence_embeddings(embeddings_root: str, model_key: str):
    """
    Carga embeddings versión v2: junta los segmentos sueltos de cada paciente,
    los ordena cronológicamente y arma la secuencia con zero-padding.
    """
    from collections import defaultdict
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}'.")

    logger.info("ℹ️ Reconstruyendo secuencias cronológicas (Formato v2)...")

    # 1. Agrupar segmentos por paciente guardando su número de segmento
    patient_data_raw = defaultdict(list)
    for entry in entries:
        pid = entry["audio_id"]
        patient_data_raw[pid].append({
            "segment_id": int(entry["segment_id"]),
            "file": entry["file"],
            "label": int(entry["class_id"])
        })
    
    # 2. Ordenar los segmentos en el tiempo (del 0 al último) y cargar sus vectores
    patient_data = {}
    for pid, segs in patient_data_raw.items():
        segs_sorted = sorted(segs, key=lambda x: x["segment_id"]) # Orden temporal crucial
        seq = []
        for s in segs_sorted:
            with open(s["file"]) as f:
                record = json.load(f)
            seq.append(record["embedding"])
        
        patient_data[pid] = {
            "seq": seq,
            "label": segs[0]["label"]
        }

    # 3. Aplicar el padding dinámico como antes
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
        f"   Cargados NUEVOS: {len(y)} pacientes | shape={X.shape} | "
        f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}"
    )
    return X, y, ids, lengths


# =============================================================================
# 2. PREPROCESAMIENTO DE SECUENCIAS (ajustado en train)
# =============================================================================

def scale_sequences(X_train, X_val, lengths_train):
    """
    StandardScaler sobre la dimensión de embedding.

    FIX (Problema 2): fittea el scaler SOLO sobre los timesteps reales,
    excluyendo los ceros de padding que distorsionan media y varianza.
    """
    B_tr, T, D = X_train.shape
    B_vl       = X_val.shape[0]
    real_vecs  = np.vstack([X_train[i, :lengths_train[i], :] for i in range(B_tr)])
    scaler     = StandardScaler()
    scaler.fit(real_vecs)
    X_tr_s = scaler.transform(X_train.reshape(-1, D)).reshape(B_tr, T, D)
    X_vl_s = scaler.transform(X_val.reshape(-1, D)).reshape(B_vl, T, D)
    return X_tr_s, X_vl_s


def pca_sequences(X_train, X_val, lengths_train, variance=PCA_VAR):
    """
    PCA sobre la dimensión de embedding D de cada segmento.

    FIX (Problema 2): fittea la PCA SOLO sobre los timesteps reales,
    excluyendo los ceros de padding que distorsionan la covarianza.
    """
    B_tr, T, D = X_train.shape
    B_vl       = X_val.shape[0]
    real_vecs  = np.vstack([X_train[i, :lengths_train[i], :] for i in range(B_tr)])
    pca = PCA(n_components=variance, svd_solver="full", random_state=42)
    pca.fit(real_vecs)
    X_tr_p = pca.transform(X_train.reshape(-1, D)).reshape(B_tr, T, -1)
    X_vl_p = pca.transform(X_val.reshape(-1, D)).reshape(B_vl, T, -1)
    new_D  = X_tr_p.shape[2]
    logger.debug(f"    PCA secuencias: D {D} → {new_D} (fit sobre {len(real_vecs)} timesteps reales)")
    return X_tr_p, X_vl_p, new_D


def mixup_sequences(X_train, y_train, alpha=MIXUP_ALPHA):
    """
    Mixup intra-clase minoritaria sobre secuencias completas (T×D).
    Preserva la estructura temporal interpolando secuencias enteras.
    Solo toca X_train.
    """
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    if n_pos < 2:
        return X_train, y_train

    n_syn = n_neg - n_pos
    if n_syn <= 0:
        return X_train, y_train

    rng = np.random.default_rng(42)
    X_syn_list = []
    for _ in range(n_syn):
        i, j  = rng.choice(idx_pos, size=2, replace=False)
        lam   = rng.beta(alpha, alpha)
        x_new = lam * X_train[i] + (1.0 - lam) * X_train[j]
        X_syn_list.append(x_new)

    X_syn = np.array(X_syn_list, dtype=np.float32)
    y_syn = np.ones(n_syn, dtype=np.int64)

    logger.debug(f"    Mixup secuencias: +{n_syn} secuencias sintéticas")
    return np.concatenate([X_train, X_syn], axis=0), np.concatenate([y_train, y_syn])


# =============================================================================
# 3. DATASET PYTORCH
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
# 4. ARQUITECTURAS
# =============================================================================

class CNN1DClassifier(nn.Module):
    """
    CNN 1D sobre secuencias temporales de segmentos.
    input: (B, T, D) → permuta → (B, D, T) → conv → pool → linear
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.25),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.25),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths=None):
        return self.net(x.permute(0, 2, 1))


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM: captura dependencias temporales largas en ambas direcciones.
    Diseñado para datasets pequeños: 1 capa, hidden pequeño.
    """
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # dropout solo funciona con n_layers > 1
        )
        self.norm    = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head    = nn.Linear(hidden * 2, 1)

    def forward(self, x, lengths=None):
        # x: (B, T, D)
        # FIX (Problema 3): usar pack_padded_sequence para que el LSTM
        # ignore los timesteps de padding y el hidden state final
        # corresponda al último segmento REAL del paciente.
        if lengths is not None:
            lens_cpu = torch.clamp(torch.tensor(lengths), min=1).cpu()
            x_packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(x_packed)
        else:
            _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))


class BiGRUClassifier(nn.Module):
    """
    BiGRU: más ligero que BiLSTM, converge más rápido con n=79.
    Suele igualar o superar a BiLSTM en datasets pequeños.
    """
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.norm    = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head    = nn.Linear(hidden * 2, 1)

    def forward(self, x, lengths=None):
        # FIX (Problema 3): pack_padded_sequence garantiza que el GRU
        # se detiene en el último timestep real de cada paciente.
        if lengths is not None:
            lens_cpu = torch.clamp(torch.tensor(lengths), min=1).cpu()
            x_packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(x_packed)
        else:
            _, h_n = self.gru(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))


class CNNBiGRUClassifier(nn.Module):
    """
    CNN + BiGRU híbrido:
      1. CNN 1D extrae features locales de la secuencia.
      2. BiGRU integra el contexto temporal sobre esas features.
    Combina ventajas de ambas arquitecturas.
    """
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        # CNN: reduce dimensión temporal y extrae patrones locales
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.2),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
        )
        # GRU: integra la secuencia de features CNN
        self.gru = nn.GRU(
            128, hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm    = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head    = nn.Linear(hidden * 2, 1)

    def forward(self, x, lengths=None):
        # x: (B, T, D) → (B, D, T) para Conv1d
        x_cnn = self.cnn(x.permute(0, 2, 1))   # (B, 128, T')
        x_gru = x_cnn.permute(0, 2, 1)          # (B, T', 128)
        # FIX (Problema 3): ajustar lengths a T' reducida por MaxPool(2)
        # La CNN tiene un MaxPool1d(2), entonces T' = floor(T / 2).
        if lengths is not None:
            T_prime   = x_gru.shape[1]
            lens_adj  = torch.clamp(
                torch.tensor([min(l // 2, T_prime) for l in lengths], dtype=torch.long),
                min=1
            ).cpu()
            x_packed  = pack_padded_sequence(x_gru, lens_adj, batch_first=True, enforce_sorted=False)
            _, h_n    = self.gru(x_packed)
        else:
            _, h_n = self.gru(x_gru)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))


# Mapa de arquitecturas
ARCHITECTURES = {
    "CNN1D":      CNN1DClassifier,
    "BiLSTM":     BiLSTMClassifier,
    "BiGRU":      BiGRUClassifier,
    "CNN_BiGRU":  CNNBiGRUClassifier,
}

LR_BY_ARCH = {
    "CNN1D":     5e-4,
    "BiLSTM":    1e-3,
    "BiGRU":     1e-3,
    "CNN_BiGRU": 5e-4,
}


# =============================================================================
# 5. ENTRENAMIENTO DE UN FOLD
# =============================================================================

def find_best_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thresholds[np.argmax(f1[:-1])])


def train_fold_sequence(
    X_train, y_train, lengths_train,
    X_val, y_val, lengths_val,
    arch_name: str,
):
    """
    Entrena una arquitectura de secuencia en un fold.
    X_val es siempre datos REALES.
    """
    input_dim = X_train.shape[2]
    ModelClass = ARCHITECTURES[arch_name]
    model = ModelClass(input_dim).to(DEVICE)
    lr    = LR_BY_ARCH[arch_name]

    ds_train = SequenceDataset(X_train, y_train, lengths_train)
    ds_val   = SequenceDataset(X_val,   y_val,   lengths_val)
    loader_tr  = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                           drop_last=(len(ds_train) % BATCH_SIZE == 1))
    loader_val = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = -1.0
    best_state = None
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
        thr        = find_best_threshold(val_labels, val_probs)
        val_pred   = (val_probs >= thr).astype(int)

        f1_macro = f1_score(val_labels, val_pred, average="macro",  zero_division=0)
        f1_min   = f1_score(val_labels, val_pred, average="binary", zero_division=0)
        acc      = accuracy_score(val_labels, val_pred)
        auc      = (roc_auc_score(val_labels, val_probs)
                    if len(np.unique(val_labels)) > 1 else 0.0)

        if f1_macro > best_f1:
            best_f1    = f1_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "f1_macro":    f1_macro,
                "f1_minority": f1_min,
                "auc":         auc,
                "acc":         acc,
                "threshold":   thr,
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
# 6. LOOP DE CROSS-VALIDATION
# =============================================================================

def run_experiment_seq(
    X, y, lengths,
    exp_config, arch_name, emb_model,
    results_path: Path,
):
    exp_id    = exp_config["id"]
    use_pca   = exp_config["pca"]
    use_mixup = exp_config["mixup"]

    cv    = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=42)
    total = CV_SPLITS * CV_REPEATS
    rows  = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        rep_num  = (fold_idx - 1) // CV_SPLITS + 1
        fold_num = (fold_idx - 1) %  CV_SPLITS + 1

        X_tr, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr, y_vl = y[train_idx].copy(), y[val_idx].copy()
        l_tr = [lengths[i] for i in train_idx]
        l_vl = [lengths[i] for i in val_idx]

        # ── Scaler ────────────────────────────────────────────────────────
        X_tr, X_vl = scale_sequences(X_tr, X_vl, l_tr)

        # ── PCA sobre dim de embedding (opcional) ─────────────────────────
        n_dims = X_tr.shape[2]
        if use_pca:
            X_tr, X_vl, n_dims = pca_sequences(X_tr, X_vl, l_tr)

        # ── Mixup en train (val NO se toca) ───────────────────────────────
        if use_mixup:
            X_tr, y_tr_aug = mixup_sequences(X_tr, y_tr)
            # Lengths sintéticos: longitud máxima (secuencias completas)
            n_syn = len(y_tr_aug) - len(l_tr)
            max_T = X_tr.shape[1]
            l_tr  = l_tr + [max_T] * n_syn
            y_tr  = y_tr_aug

        # ── Entrenar y evaluar ────────────────────────────────────────────
        metrics = train_fold_sequence(X_tr, y_tr, l_tr, X_vl, y_vl, l_vl, arch_name)

        row = {
            "emb_model":    emb_model,
            "architecture": arch_name,
            "experiment":   exp_id,
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
                f"F1-macro={recent['f1_macro'].mean():.3f} | "
                f"AUC={recent['auc'].mean():.3f} | dims={n_dims}"
            )

    df = pd.DataFrame(rows)
    logger.info(
        f"    ✅ [{arch_name}|{exp_id}] "
        f"F1={df['f1_macro'].mean():.3f}±{df['f1_macro'].std():.3f} | "
        f"F1-min={df['f1_minority'].mean():.3f} | "
        f"AUC={df['auc'].mean():.3f}"
    )
    return df


# =============================================================================
# 7. PIPELINE PRINCIPAL
# =============================================================================

def run_all(embeddings_root: str, output_dir: str):
    output_dir = Path(output_dir) / "dl_script2"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = output_dir / "raw_folds.csv"
    summary_path = output_dir / "summary.csv"

    if raw_path.exists():
        logger.warning(f"⚠️  {raw_path} ya existe. Se añaden resultados al final.")

    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 Embeddings:   {embeddings_root}")
    logger.info(f"📂 Salida:       {output_dir}")
    logger.info(
        f"🔁 CV: {CV_REPEATS} repeats × {CV_SPLITS} folds = "
        f"{CV_REPEATS * CV_SPLITS} evaluaciones por config"
    )
    archs = list(ARCHITECTURES.keys())
    exps  = [e["id"] for e in EXPERIMENTS]
    logger.info(f"🧱 Arquitecturas: {archs}")
    logger.info(f"🧪 Experimentos:  {exps}\n")

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

        for arch_name in ARCHITECTURES:
            logger.info(f"\n  🔷 Arquitectura: {arch_name}")
            for exp_config in EXPERIMENTS:
                logger.info(f"    🔬 Experimento: {exp_config['id']}")
                t0 = time.time()
                run_experiment_seq(
                    X, y, lengths,
                    exp_config, arch_name, emb_model,
                    raw_path,
                )
                logger.info(f"       ⏱️  {(time.time()-t0)/60:.1f} min")

    # ── Resumen final ──────────────────────────────────────────────────────
    if raw_path.exists():
        df_all = pd.read_csv(raw_path)
        summary = (
            df_all
            .groupby(["emb_model", "architecture", "experiment"])
            .agg(
                mean_f1_macro    =("f1_macro",    "mean"),
                std_f1_macro     =("f1_macro",    "std"),
                mean_f1_minority =("f1_minority", "mean"),
                std_f1_minority  =("f1_minority", "std"),
                mean_auc         =("auc",         "mean"),
                std_auc          =("auc",         "std"),
                mean_acc         =("acc",         "mean"),
                mean_dims        =("n_dims",       "mean"),
                n_folds          =("f1_macro",    "count"),
            )
            .reset_index()
            .sort_values("mean_f1_macro", ascending=False)
        )
        summary.to_csv(summary_path, index=False)

        logger.info("\n" + "="*65)
        logger.info("📊 TOP 20 CONFIGURACIONES (por F1-macro)")
        logger.info("="*65)
        cols = ["emb_model","architecture","experiment",
                "mean_f1_macro","std_f1_macro","mean_f1_minority","mean_auc"]
        logger.info("\n" + summary[cols].head(20).to_string(index=False))
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
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Ansiedad/Wav2Vec/results_DL/embeddings_nuevos"
    )
    # ───────────────────────────────────────────────────────────────────────

    run_all(EMBEDDINGS_ROOT, OUTPUT_DIR)
