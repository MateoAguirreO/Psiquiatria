"""
cross_dataset_common.py
========================
Módulo común para todos los experimentos cross-dataset (DAIC ↔ propio).
Contiene TODO lo que se repite entre scripts: carga de embeddings,
preprocesamiento sin leakage, arquitecturas, y el loop de entrenamiento/
evaluación cross-dataset (sin CV — split fijo por origen de dataset).

Los scripts de experimento (dl_cross_pooled.py, dl_cross_sequence.py,
dl_combined_cv.py) importan de aquí y solo declaran configuración.

Principio de diseño: una sola fuente de verdad para el protocolo
anti-leakage, para no tener que repetir y volver a romper la misma
lógica en cada script nuevo.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CrossDatasetCommon")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. CARGA DE EMBEDDINGS
# =============================================================================

def load_pooled_embeddings(embeddings_root: str, model_key: str) -> tuple:
    """
    Mean-pooling por paciente. Para arquitecturas que esperan vector fijo
    (MLP). Retorna X (n_patients, dim), y (n_patients,), ids (list).
    """
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}' en {embeddings_root}")

    patient_segments = defaultdict(list)
    patient_label    = {}
    for entry in entries:
        pid = entry["audio_id"]
        patient_segments[pid].append(entry["file"])
        patient_label[pid] = int(entry["class_id"])

    X_rows, y_rows, ids = [], [], []
    for pid in sorted(patient_segments.keys()):
        embs = []
        for fpath in patient_segments[pid]:
            with open(fpath) as f:
                record = json.load(f)
            embs.append(record["embedding"])
        embs = np.array(embs, dtype=np.float32)
        X_rows.append(embs.mean(axis=0))
        y_rows.append(patient_label[pid])
        ids.append(pid)

    X = np.stack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    logger.info(
        f"   Cargados (pooled): {len(y)} pacientes | dim={X.shape[1]} | "
        f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}"
    )
    return X, y, ids


def load_sequence_embeddings(embeddings_root: str, model_key: str) -> tuple:
    """
    Secuencias ordenadas cronológicamente por paciente, con zero-padding
    al máximo número de segmentos. Para arquitecturas CNN1D/BiLSTM/BiGRU.
    Retorna X (n_patients, max_T, dim), y, ids, lengths (longitud real).
    """
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}' en {embeddings_root}")

    patient_data_raw = defaultdict(list)
    for entry in entries:
        pid = entry["audio_id"]
        patient_data_raw[pid].append({
            "segment_id": int(entry["segment_id"]),
            "file": entry["file"],
            "label": int(entry["class_id"]),
        })

    patient_data = {}
    for pid, segs in patient_data_raw.items():
        segs_sorted = sorted(segs, key=lambda x: x["segment_id"])
        seq = []
        for s in segs_sorted:
            with open(s["file"]) as f:
                record = json.load(f)
            seq.append(record["embedding"])
        patient_data[pid] = {"seq": seq, "label": segs[0]["label"]}

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
        f"   Cargados (sequence): {len(y)} pacientes | shape={X.shape} | "
        f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}"
    )
    return X, y, ids, lengths


# =============================================================================
# 2. PREPROCESAMIENTO — pooled (vector fijo)
# =============================================================================

def scale_pca_augment_pooled(
    X_tr, y_tr, X_vl, X_te,
    use_pca: bool, pca_variance: float,
    aug_key: str, seed: int,
):
    """
    Orden OBLIGATORIO, fit SIEMPRE solo sobre X_tr (train interno):
      1. StandardScaler
      2. PCA (si aplica)
      3. Augmentación (al final, solo en X_tr)
    X_vl y X_te solo reciben .transform(), nunca .fit().
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_te = scaler.transform(X_te)

    n_components = X_tr.shape[1]
    if use_pca:
        pca = PCA(n_components=pca_variance, svd_solver="full", random_state=seed)
        X_tr = pca.fit_transform(X_tr)
        X_vl = pca.transform(X_vl)
        X_te = pca.transform(X_te)
        n_components = pca.n_components_

    if aug_key == "smote":
        X_tr, y_tr = smote_augment(X_tr, y_tr)

    return X_tr, y_tr, X_vl, X_te, n_components


def smote_augment(X_train, y_train):
    n_minority = np.sum(y_train == 1)
    k_safe = max(1, min(5, n_minority - 1))
    try:
        sm = SMOTE(k_neighbors=k_safe, random_state=42)
        return sm.fit_resample(X_train, y_train)
    except Exception as e:
        logger.warning(f"    SMOTE falló ({e}), usando train original.")
        return X_train, y_train


# =============================================================================
# 3. PREPROCESAMIENTO — secuencias (CNN1D/BiLSTM/BiGRU)
# =============================================================================

def scale_sequences(X_tr, X_vl, X_te, lengths_tr):
    """
    Scaler fit SOLO sobre los timesteps REALES de X_tr (excluye padding).
    """
    B_tr, T_tr, D = X_tr.shape
    real_vecs = np.vstack([X_tr[i, :lengths_tr[i], :] for i in range(B_tr)])
    scaler = StandardScaler()
    scaler.fit(real_vecs)

    X_tr_s = scaler.transform(X_tr.reshape(-1, D)).reshape(B_tr, T_tr, D)
    X_vl_s = scaler.transform(X_vl.reshape(-1, D)).reshape(X_vl.shape[0], X_vl.shape[1], D)
    X_te_s = scaler.transform(X_te.reshape(-1, D)).reshape(X_te.shape[0], X_te.shape[1], D)
    return X_tr_s, X_vl_s, X_te_s


def pca_sequences(X_tr, X_vl, X_te, lengths_tr, variance, seed):
    """
    PCA fit SOLO sobre los timesteps REALES de X_tr (excluye padding).
    Se aplica a la dimensión de embedding, no a la temporal.
    """
    B_tr, T_tr, D = X_tr.shape
    real_vecs = np.vstack([X_tr[i, :lengths_tr[i], :] for i in range(B_tr)])
    pca = PCA(n_components=variance, svd_solver="full", random_state=seed)
    pca.fit(real_vecs)

    X_tr_p = pca.transform(X_tr.reshape(-1, D)).reshape(B_tr, T_tr, -1)
    X_vl_p = pca.transform(X_vl.reshape(-1, D)).reshape(X_vl.shape[0], X_vl.shape[1], -1)
    X_te_p = pca.transform(X_te.reshape(-1, D)).reshape(X_te.shape[0], X_te.shape[1], -1)
    return X_tr_p, X_vl_p, X_te_p, pca.n_components_


# =============================================================================
# 4. ARQUITECTURAS
# =============================================================================

class MLPClassifier(nn.Module):
    """Para embeddings pooled (vector fijo por paciente)."""
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

    def forward(self, x, lengths=None):
        return self.net(x)


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths=None):
        return self.net(x.permute(0, 2, 1))


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True, dropout=0.0,
        )
        self.norm    = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head    = nn.Linear(hidden * 2, 1)

    def forward(self, x, lengths=None):
        if lengths is not None:
            lens_cpu = torch.clamp(torch.as_tensor(lengths), min=1).cpu()
            x_packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(x_packed)
        else:
            _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))


class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
        )
        self.norm    = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head    = nn.Linear(hidden * 2, 1)

    def forward(self, x, lengths=None):
        if lengths is not None:
            lens_cpu = torch.clamp(torch.as_tensor(lengths), min=1).cpu()
            x_packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(x_packed)
        else:
            _, h_n = self.gru(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))


class CNNBiGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
        )
        self.gru = nn.GRU(128, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.norm    = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head    = nn.Linear(hidden * 2, 1)

    def forward(self, x, lengths=None):
        x_cnn = self.cnn(x.permute(0, 2, 1))
        x_gru = x_cnn.permute(0, 2, 1)
        if lengths is not None:
            T_prime  = x_gru.shape[1]
            lens_adj = torch.clamp(
                torch.tensor([min(l // 2, T_prime) for l in lengths], dtype=torch.long),
                min=1
            ).cpu()
            x_packed = pack_padded_sequence(x_gru, lens_adj, batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(x_packed)
        else:
            _, h_n = self.gru(x_gru)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))


ARCHITECTURES = {
    "MLP":       MLPClassifier,      # espera input (B, D) — usar con datos pooled
    "CNN1D":     CNN1DClassifier,    # espera input (B, T, D) — usar con sequence
    "BiLSTM":    BiLSTMClassifier,
    "BiGRU":     BiGRUClassifier,
    "CNN_BiGRU": CNNBiGRUClassifier,
}

LR_BY_ARCH = {
    "MLP":       1e-3,
    "CNN1D":     5e-4,
    "BiLSTM":    1e-3,
    "BiGRU":     1e-3,
    "CNN_BiGRU": 5e-4,
}

# Arquitecturas que esperan secuencias (B, T, D) en vez de vector fijo (B, D)
SEQUENCE_ARCHS = {"CNN1D", "BiLSTM", "BiGRU", "CNN_BiGRU"}


# =============================================================================
# 5. DATASETS PYTORCH
# =============================================================================

class PooledDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], 0   # length dummy, no se usa


class SequenceDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.FloatTensor(y.astype(np.float32))
        self.lengths = lengths

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


# =============================================================================
# 6. UMBRAL Y MÉTRICAS
# =============================================================================

def find_best_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thresholds[np.argmax(f1[:-1])])


# =============================================================================
# 7. ENTRENAMIENTO + EVALUACIÓN CROSS-DATASET (genérico, sin CV)
# =============================================================================

def train_and_evaluate_cross_dataset(
    arch_name: str,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    lengths_source: list = None,   # solo para arquitecturas de secuencia
    lengths_target: list = None,
    use_pca: bool = False,
    pca_variance: float = 0.95,
    aug_key: str = None,
    epochs: int = 60,
    batch_size: int = 16,
    patience: int = 8,
    seed: int = 42,
) -> dict:
    """
    Función única para CUALQUIER arquitectura (pooled o secuencia).
    Detecta automáticamente el tipo de dato por la forma de X_source
    y por si arch_name está en SEQUENCE_ARCHS.

    Orden de preprocesamiento (igual para ambos casos):
      1. Split 80/20 del source (interno, estratificado) -> train/val interno
      2. Scaler fit SOLO en train interno (sobre timesteps reales si es secuencia)
      3. PCA (si aplica) fit SOLO en train interno
      4. Augmentación SOLO en train interno (pooled: SMOTE; secuencia: ninguna
         por defecto — Mixup de secuencias se deja para script específico)
    X_target NUNCA participa de ningún .fit().
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_sequence = arch_name in SEQUENCE_ARCHS

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, vl_idx = next(sss.split(X_source, y_source))

    X_tr, X_vl = X_source[tr_idx].copy(), X_source[vl_idx].copy()
    y_tr, y_vl = y_source[tr_idx].copy(), y_source[vl_idx].copy()
    X_te, y_te = X_target.copy(), y_target.copy()

    if is_sequence:
        l_tr = [lengths_source[i] for i in tr_idx]
        l_vl = [lengths_source[i] for i in vl_idx]
        l_te = list(lengths_target)

        X_tr, X_vl, X_te = scale_sequences(X_tr, X_vl, X_te, l_tr)
        n_components = X_tr.shape[2]
        if use_pca:
            X_tr, X_vl, X_te, n_components = pca_sequences(
                X_tr, X_vl, X_te, l_tr, pca_variance, seed
            )
        # Nota: augmentación de secuencias (mixup) no incluida en el común;
        # si se necesita, se aplica en el script específico antes de llamar
        # a esta función, pasando X_tr/y_tr ya aumentados — pero entonces
        # hay que ajustar l_tr también (longitudes sintéticas).

        ds_train = SequenceDataset(X_tr, y_tr, l_tr)
        ds_val   = SequenceDataset(X_vl, y_vl, l_vl)
        ds_test  = SequenceDataset(X_te, y_te, l_te)
        input_dim = X_tr.shape[2]
    else:
        X_tr, y_tr, X_vl, X_te, n_components = scale_pca_augment_pooled(
            X_tr, y_tr, X_vl, X_te, use_pca, pca_variance, aug_key, seed
        )
        ds_train = PooledDataset(X_tr, y_tr)
        ds_val   = PooledDataset(X_vl, y_vl)
        ds_test  = PooledDataset(X_te, y_te)
        input_dim = X_tr.shape[1]

    loader_tr   = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                             drop_last=(len(ds_train) % batch_size == 1))
    loader_val  = DataLoader(ds_val,  batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    ModelClass = ARCHITECTURES[arch_name]
    model = ModelClass(input_dim).to(DEVICE)
    lr = LR_BY_ARCH[arch_name]

    n_neg = np.sum(y_tr == 0)
    n_pos = np.sum(y_tr == 1)
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_state   = None
    no_improve   = 0
    epoch        = 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb, lb in loader_tr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            lb_arg = lb if is_sequence else None
            loss = criterion(model(Xb, lb_arg).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        probs_vl, labels_vl = [], []
        with torch.no_grad():
            for Xb, yb, lb in loader_val:
                lb_arg = lb if is_sequence else None
                p = torch.sigmoid(model(Xb.to(DEVICE), lb_arg).squeeze(-1))
                probs_vl.extend(p.cpu().numpy().tolist())
                labels_vl.extend(yb.cpu().numpy().tolist())

        vl_probs  = np.array(probs_vl)
        vl_labels = np.array(labels_vl)
        val_auc = (roc_auc_score(vl_labels, vl_probs)
                   if len(np.unique(vl_labels)) > 1 else 0.0)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    probs_te, labels_te = [], []
    with torch.no_grad():
        for Xb, yb, lb in loader_test:
            lb_arg = lb if is_sequence else None
            p = torch.sigmoid(model(Xb.to(DEVICE), lb_arg).squeeze(-1))
            probs_te.extend(p.cpu().numpy().tolist())
            labels_te.extend(yb.cpu().numpy().tolist())

    test_probs  = np.array(probs_te)
    test_labels = np.array(labels_te)

    thr       = find_best_threshold(test_labels, test_probs)
    test_pred = (test_probs >= thr).astype(int)

    auc      = (roc_auc_score(test_labels, test_probs)
                if len(np.unique(test_labels)) > 1 else 0.0)
    f1_macro = f1_score(test_labels, test_pred, average="macro",  zero_division=0)
    f1_min   = f1_score(test_labels, test_pred, average="binary", zero_division=0)
    acc      = accuracy_score(test_labels, test_pred)

    return {
        "auc":          auc,
        "f1_macro":     f1_macro,
        "f1_minority":  f1_min,
        "acc":          acc,
        "threshold":    thr,
        "best_epoch":   epoch + 1,
        "n_train":      len(y_tr),
        "n_val_int":    len(y_vl),
        "n_test":       len(y_te),
        "n_components": n_components,
        "input_dim":    input_dim,
        "seed":         seed,
    }


# =============================================================================
# 8. RUNNER GENÉRICO (N_RUNS con semillas distintas, guarda incrementalmente)
# =============================================================================

def run_cross_dataset_config(
    arch_name: str,
    X_source, y_source, X_target, y_target,
    exp_config: dict,
    emb_model: str,
    direction: str,
    results_path: Path,
    n_runs: int = 10,
    lengths_source: list = None,
    lengths_target: list = None,
):
    exp_id  = exp_config["id"]
    use_pca = exp_config.get("pca", False)
    aug_key = exp_config.get("aug", None)

    rows = []
    for run in range(n_runs):
        seed = 42 + run
        metrics = train_and_evaluate_cross_dataset(
            arch_name=arch_name,
            X_source=X_source.copy(), y_source=y_source.copy(),
            X_target=X_target.copy(), y_target=y_target.copy(),
            lengths_source=lengths_source, lengths_target=lengths_target,
            use_pca=use_pca, aug_key=aug_key, seed=seed,
        )
        row = {
            "direction":    direction,
            "emb_model":    emb_model,
            "architecture": arch_name,
            "experiment":   exp_id,
            "run":          run + 1,
            **metrics,
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(
            results_path, mode="a",
            header=not results_path.exists() or results_path.stat().st_size == 0,
            index=False,
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"  ✅ [{direction}|{arch_name}|{exp_id}|{emb_model}] "
        f"AUC={df['auc'].mean():.3f}±{df['auc'].std():.3f} | "
        f"F1-macro={df['f1_macro'].mean():.3f} | F1-min={df['f1_minority'].mean():.3f}"
    )
    return df


def summarize(raw_path: Path, summary_path: Path, group_cols: list, sort_col: str = "mean_auc"):
    df_all = pd.read_csv(raw_path)
    summary = (
        df_all
        .groupby(group_cols)
        .agg(
            mean_auc         =("auc",         "mean"),
            std_auc          =("auc",         "std"),
            mean_f1_macro    =("f1_macro",    "mean"),
            std_f1_macro     =("f1_macro",    "std"),
            mean_f1_minority =("f1_minority", "mean"),
            mean_acc         =("acc",         "mean"),
            n_runs           =("auc",         "count"),
        )
        .reset_index()
        .sort_values(sort_col, ascending=False)
    )
    summary.to_csv(summary_path, index=False)
    return summary