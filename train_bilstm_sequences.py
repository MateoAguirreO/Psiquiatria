"""
train_bilstm_sequences.py
==========================

Consume las secuencias frame-a-frame generadas por `extract_features.py`
(--save_sequences, MediaPipe) o `extract_au_features_pyfeat.py`
(--save_sequences, Action Units), y entrena y compara 5 arquitecturas con
masking para longitudes variables — el mismo menú que ya usaste en tu
pipeline de audio, para que los resultados sean comparables entre modalidades:

  - bilstm        : LSTM bidireccional sobre la secuencia de features.
  - bigru         : igual pero con GRU (menos parámetros).
  - cnn1d         : CNN-1D sobre el eje temporal + mean-pooling enmascarado.
  - cnn_bigru     : CNN-1D (patrones locales) + BiGRU (dependencia temporal larga).
  - mlp_meanpool  : ignora el orden temporal (promedia los frames, como tu CSV
                    agregado) + MLP chico. Sirve de control: si esto empata o
                    gana contra las arquitecturas recurrentes, es señal de que
                    la dinámica temporal no está aportando nada extra con este N.

Sirve para los dos casos (MediaPipe / AU): solo cambia --sequences_dir y --labels_csv.

ADVERTENCIA HONESTA sobre el tamaño de la muestra:
------------------------------------------------------------------------------
Con 80 sujetos (21 positivos en depresión) entrenar una red desde cero es de
alto riesgo de overfitting, sin importar qué tan bien se regularice. Este script:
  - Usa RepeatedStratifiedKFold (n_splits x n_repeats configurables) sobre los
    MISMOS folds para las 5 arquitecturas, así la comparación es justa.
  - Reduce dimensionalidad de entrada (solo columnas numéricas con suficiente
    cobertura) y usa capas chicas (16-32 unidades) + dropout fuerte.
  - Reporta AUC promedio ± std entre folds, comparable directo contra tu tabla
    de LazyPredict del baseline tabular.
  - Trátalo como un torneo de experimentos a comparar entre sí y contra el
    baseline tabular, no como el pipeline definitivo. Es un resultado válido
    (y publicable) que ninguna arquitectura le gane al azar con este N.

Requisitos:
    pip install torch scikit-learn pandas numpy pyarrow tqdm

Uso (recomendado, corre las 5 arquitecturas sobre los mismos folds):
    python train_bilstm_sequences.py \
        --sequences_dir sequences_mediapipe \
        --labels_csv dataset_facial_features.csv \
        --target target_depresion \
        --max_seq_len 600 \
        --architecture all \
        --output_csv resultados_video_arquitecturas.csv

Uso (una sola arquitectura):
    python train_bilstm_sequences.py --sequences_dir sequences_au \
        --labels_csv dataset_au_features.csv --target target_depresion \
        --architecture cnn_bigru
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def set_global_seed(seed: int) -> None:
    """Fija TODAS las fuentes de aleatoriedad relevantes (Python, numpy, torch CPU/GPU)
    para que el mismo fold + misma arquitectura + misma seed reproduzca el mismo
    resultado en cualquier máquina. Sin esto, los folds eran reproducibles (ya
    tenían random_state fijo) pero el entrenamiento en sí no lo era: init de
    pesos, orden de shuffle del DataLoader y dropout quedaban libres.

    Nota honesta: con CUDA, algunas operaciones (p.ej. las de cuDNN para RNNs)
    no tienen una implementación 100% determinista incluso con esto activado.
    El resultado va a ser MUY cercano entre corridas (mismo orden de magnitud,
    misma conclusión sobre qué arquitectura gana), pero no se garantiza
    bit-a-bit idéntico en GPU. En CPU sí debería ser exactamente igual.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 1. CARGA DE SECUENCIAS
# ==============================================================================

NON_FEATURE_COLS = {"video_id", "frame_order", "timestamp_ms", "frame", "approx_time", "input"}


def _normalize_video_id(vid: str) -> str:
    """Normaliza un video_id para que '002', '2', y '2.0' se traten como el mismo video.
    Necesario porque pandas a veces interpreta video_id como numérico al leer/escribir CSV
    y le come los ceros a la izquierda que sí tienen los nombres de archivo (002.mp4 -> 002.parquet)."""
    vid = str(vid).strip()
    try:
        return str(int(float(vid)))
    except (ValueError, TypeError):
        return vid


def load_sequences(sequences_dir: str) -> dict[str, pd.DataFrame]:
    """Carga cada archivo (parquet o csv) de sequences_dir como un DataFrame; retorna {video_id: df}.
    Las claves quedan normalizadas (ver _normalize_video_id) para que el cruce contra labels_csv
    no falle por diferencias de formato como ceros a la izquierda."""
    files = sorted(glob.glob(os.path.join(sequences_dir, "*.parquet")) + glob.glob(os.path.join(sequences_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No se encontraron secuencias en '{sequences_dir}'.")
    sequences = {}
    for f in files:
        video_id = _normalize_video_id(os.path.splitext(os.path.basename(f))[0])
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        sequences[video_id] = df
    return sequences


def list_candidate_columns(sequences: dict[str, pd.DataFrame]) -> list[str]:
    """Devuelve las columnas candidatas a ser features: descarta solo las columnas
    no numéricas o auxiliares (video_id, frame_order, etc). Esto es puramente
    estructural (qué columnas EXISTEN y qué tipo de dato tienen), no una decisión
    estadística sobre los datos, así que es seguro calcularlo sobre todos los
    videos sin que cuente como leakage (no usa cuánta información falta, ni las
    etiquetas). El filtro real de "¿tiene demasiados NaN?" se hace fold por fold,
    usando solo los videos de train de ese fold — ver `filter_high_nan_columns`.
    """
    all_df = pd.concat(sequences.values(), ignore_index=True)
    candidate_cols = [c for c in all_df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [c for c in candidate_cols if pd.to_numeric(all_df[c], errors="coerce").notna().mean() > 0]
    if not numeric_cols:
        raise ValueError("Ninguna columna numérica encontrada; revisa las secuencias generadas.")
    return numeric_cols


def filter_high_nan_columns(train_concat: pd.DataFrame, candidate_cols: list[str], max_nan_frac: float = 0.5) -> list[str]:
    """Filtra columnas casi-todo-NaN usando SOLO los frames de los videos de
    train del fold actual (nunca los de validation), para no dejar que la
    validación influya, ni siquiera indirectamente, en qué features se usan."""
    keep = [c for c in candidate_cols if train_concat[c].isna().mean() <= max_nan_frac]
    if not keep:
        raise ValueError("Ninguna columna pasó el filtro de NaNs en este fold; revisa las secuencias generadas o sube max_nan_frac.")
    return keep


# ==============================================================================
# 2. DATASET CON PADDING + MÁSCARA
# ==============================================================================

class SequenceDataset(Dataset):
    def __init__(self, video_ids: list[str], sequences: dict[str, pd.DataFrame], feature_cols: list[str], labels: dict[str, int], max_seq_len: int):
        self.video_ids = video_ids
        self.sequences = sequences
        self.feature_cols = feature_cols
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        # Nota: ya no se rellenan NaN acá. La imputación se hace una sola vez en
        # run_cv, usando SOLO estadísticas del train de cada fold (ver más abajo),
        # antes de escalar. Si llegara algún NaN residual (no debería pasar), se
        # cae a 0.0 explícitamente para que un bug de imputación no quede oculto.
        df = self.sequences[vid][self.feature_cols]
        arr = df.to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        if len(arr) > self.max_seq_len:
            # Muestreo uniforme a lo largo del video (conserva la forma general de la señal)
            idxs = np.linspace(0, len(arr) - 1, self.max_seq_len).astype(int)
            arr = arr[idxs]
        return torch.from_numpy(arr), self.labels[vid]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True)  # (B, T_max, F)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, labels


# ==============================================================================
# 3. MODELOS: mismo menú de arquitecturas que usaste en el pipeline de audio
#    (MLP, CNN-1D, BiLSTM, BiGRU, CNN+BiGRU), para que los resultados de video
#    sean directamente comparables contra los de audio en el paper.
# ==============================================================================

class BiLSTMClassifier(nn.Module):
    """RNN bidireccional simple sobre la secuencia de features por frame."""

    def __init__(self, n_features: int, hidden_size: int = 24, dropout: float = 0.5):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.rnn(packed)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        h_cat = self.dropout(h_cat)
        return self.head(h_cat).squeeze(-1)


class BiGRUClassifier(nn.Module):
    """Igual que BiLSTMClassifier pero con GRU (menos parámetros, a veces más estable con N chico)."""

    def __init__(self, n_features: int, hidden_size: int = 24, dropout: float = 0.5):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        h_cat = self.dropout(h_cat)
        return self.head(h_cat).squeeze(-1)


class CNN1DClassifier(nn.Module):
    """CNN-1D sobre el eje temporal (equivalente a tu CNN-1D de audio, pero sobre la secuencia de features faciales).
    Usa masking manual antes del pooling para no dejar que el padding contamine el promedio."""

    def __init__(self, n_features: int, n_filters: int = 24, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(n_filters, 1)

    def forward(self, x, lengths):
        # x: (B, T, F) -> (B, F, T) para Conv1d
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None].to(x.device)  # (B, T)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = x * mask.unsqueeze(-1)  # anula el padding
        pooled = x.sum(dim=1) / lengths.to(x.device).unsqueeze(-1).clamp(min=1)  # mean-pool solo sobre frames reales
        pooled = self.dropout(pooled)
        return self.head(pooled).squeeze(-1)


class CNNBiGRUClassifier(nn.Module):
    """CNN-1D (extrae patrones locales por ventana de frames) seguida de BiGRU (captura dependencia temporal larga).
    Equivalente a tu CNN+BiGRU de audio."""

    def __init__(self, n_features: int, n_filters: int = 24, hidden_size: int = 24, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Conv1d(n_features, n_filters, kernel_size=5, padding=2)
        self.rnn = nn.GRU(n_filters, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = x.transpose(1, 2)  # (B, T, n_filters)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        h_cat = self.dropout(h_cat)
        return self.head(h_cat).squeeze(-1)


class MLPMeanPoolClassifier(nn.Module):
    """Baseline: ignora el orden temporal, promedia los frames (igual que tu CSV agregado) y pasa por un MLP chico.
    Sirve como control: si esto empata o gana contra las arquitecturas recurrentes, es señal de que la dinámica
    temporal no está aportando nada extra sobre el promedio simple, con este N."""

    def __init__(self, n_features: int, hidden_size: int = 24, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None].to(x.device)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(x.device).unsqueeze(-1).clamp(min=1)
        h = torch.relu(self.fc1(pooled))
        h = self.dropout(h)
        return self.head(h).squeeze(-1)


ARCHITECTURES = {
    "bilstm": BiLSTMClassifier,
    "bigru": BiGRUClassifier,
    "cnn1d": CNN1DClassifier,
    "cnn_bigru": CNNBiGRUClassifier,
    "mlp_meanpool": MLPMeanPoolClassifier,
}


def build_model(architecture: str, n_features: int, hidden_size: int, dropout: float = 0.5) -> nn.Module:
    cls = ARCHITECTURES[architecture]
    if architecture == "cnn1d":
        return cls(n_features, n_filters=hidden_size, dropout=dropout)
    if architecture == "cnn_bigru":
        return cls(n_features, n_filters=hidden_size, hidden_size=hidden_size, dropout=dropout)
    return cls(n_features, hidden_size=hidden_size, dropout=dropout)


# ==============================================================================
# 4. ENTRENAMIENTO CON STRATIFIED K-FOLD + EARLY STOPPING
# ==============================================================================

def train_one_fold(model, train_loader, val_loader, device, max_epochs=60, patience=8, lr=1e-3, weight_decay=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses, all_logits, all_labels = [], [], []
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                val_losses.append(criterion(logits, y).item())
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, lengths, y in val_loader:
            x = x.to(device)
            logits = model(x, lengths)
            all_logits.append(torch.sigmoid(logits).cpu())
            all_labels.append(y)
    y_pred = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    return y_true, y_pred, model, best_val_loss, epoch + 1


def permutation_importance_for_fold(model, val_ids, scaled_sequences, feature_cols, label_map, max_seq_len, device, batch_size, y_true, y_pred_baseline):
    """Importancia por permutación: para cada feature, se baraja su serie temporal
    ENTERA entre los videos del set de validación (cada video se queda con la serie
    de otro video para esa sola columna, el resto de columnas intactas), se vuelve
    a predecir, y se mide cuánto cae el AUC. Una caída grande = el modelo dependía
    fuerte de esa feature; una caída ~0 o negativa = no aportaba (o hasta estorbaba).

    Se permuta la serie completa (no frame por frame) para respetar la estructura
    temporal de la señal real de otro sujeto, en vez de romperla con ruido frame a frame.
    """
    if len(np.unique(y_true)) < 2:
        return {}

    baseline_auc = roc_auc_score(y_true, y_pred_baseline)
    importances = {}
    rng = np.random.default_rng(0)

    for feat_idx, feat_name in enumerate(feature_cols):
        shuffled_ids = list(val_ids)
        perm = rng.permutation(len(shuffled_ids))
        donor_ids = [shuffled_ids[p] for p in perm]

        permuted_sequences = {}
        for vid, donor_vid in zip(val_ids, donor_ids):
            df = scaled_sequences[vid].copy()
            df[feat_name] = scaled_sequences[donor_vid][feat_name].to_numpy()[: len(df)] if len(scaled_sequences[donor_vid]) >= len(df) else np.resize(scaled_sequences[donor_vid][feat_name].to_numpy(), len(df))
            permuted_sequences[vid] = df

        ds = SequenceDataset(val_ids, permuted_sequences, feature_cols, label_map, max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model.eval()
        all_logits = []
        with torch.no_grad():
            for x, lengths, y in loader:
                x = x.to(device)
                logits = model(x, lengths)
                all_logits.append(torch.sigmoid(logits).cpu())
        y_pred_perm = torch.cat(all_logits).numpy()
        permuted_auc = roc_auc_score(y_true, y_pred_perm)
        importances[feat_name] = baseline_auc - permuted_auc

    return importances


def run_cv(sequences_dir: str, labels_csv: str, target: str, max_seq_len: int, n_splits: int, n_repeats: int,
           hidden_size: int, batch_size: int, seed: int, architecture: str, patience: int, max_epochs: int,
           precomputed=None, compute_importance: bool = False):
    """Corre RepeatedStratifiedKFold para UNA arquitectura. Si se pasa `precomputed`
    (de otra llamada previa), reutiliza sequences/labels/folds para que la comparación
    entre arquitecturas sea sobre EXACTAMENTE los mismos splits.

    Retorna (fold_aucs, precomputed, detailed_records, importance_records):
      - detailed_records: 1 dict por fold con auc/n_train/n_val/n_features/epochs/best_val_loss
      - importance_records: 1 dict por (fold, feature) con la caída de AUC al permutar esa feature
        (vacío si compute_importance=False)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if precomputed is None:
        sequences = load_sequences(sequences_dir)
        candidate_cols = list_candidate_columns(sequences)
        print(f"{len(candidate_cols)} columnas candidatas (antes de filtrar por NaN, que se hace por fold usando solo train): {candidate_cols}")

        labels_df = pd.read_csv(labels_csv)
        labels_df["video_id"] = labels_df["video_id"].apply(_normalize_video_id)
        labels_df = labels_df.dropna(subset=[target])
        label_map = dict(zip(labels_df["video_id"], labels_df[target].astype(int)))

        video_ids = [vid for vid in sequences.keys() if vid in label_map]
        missing = set(sequences.keys()) - set(label_map.keys())
        if missing:
            print(f"Aviso: {len(missing)} video_id con secuencia pero sin etiqueta en '{labels_csv}', se excluyen: {sorted(missing)}")

        y_all = np.array([label_map[v] for v in video_ids])
        print(f"N={len(video_ids)} videos, distribución de clases: {dict(zip(*np.unique(y_all, return_counts=True)))}")

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        folds = list(rskf.split(video_ids, y_all))
        precomputed = {
            "sequences": sequences, "candidate_cols": candidate_cols, "label_map": label_map,
            "video_ids": video_ids, "y_all": y_all, "folds": folds,
        }

    sequences = precomputed["sequences"]
    candidate_cols = precomputed["candidate_cols"]
    label_map = precomputed["label_map"]
    video_ids = precomputed["video_ids"]
    folds = precomputed["folds"]

    fold_aucs = []
    detailed_records = []
    importance_records = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Seed determinística por fold: distinta entre folds (para no inicializar
        # siempre igual) pero idéntica si se repite exactamente esta misma corrida.
        set_global_seed(seed + fold_idx)

        train_ids = [video_ids[i] for i in train_idx]
        val_ids = [video_ids[i] for i in val_idx]

        # Filtro de columnas casi-todo-NaN Y normalización: ambos se calculan
        # SOLO con los frames de train de este fold (nunca val), para que ni la
        # elección de features ni el escalado tomen prestada información de validation.
        train_concat = pd.concat([sequences[v][candidate_cols].apply(pd.to_numeric, errors="coerce") for v in train_ids], ignore_index=True)
        feature_cols = filter_high_nan_columns(train_concat, candidate_cols)
        if fold_idx == 0:
            dropped = set(candidate_cols) - set(feature_cols)
            if dropped:
                print(f"[{architecture}] Fold 0: se descartan por NaN (calculado solo con train de este fold): {sorted(dropped)}")
        train_concat = train_concat[feature_cols]
        scaler = StandardScaler().fit(train_concat.fillna(train_concat.mean()))

        scaled_sequences = {}
        for v in train_ids + val_ids:
            df = sequences[v][feature_cols].apply(pd.to_numeric, errors="coerce")
            df = df.fillna(train_concat.mean())
            scaled = pd.DataFrame(scaler.transform(df), columns=feature_cols)
            scaled_sequences[v] = scaled

        train_ds = SequenceDataset(train_ids, scaled_sequences, feature_cols, label_map, max_seq_len)
        val_ds = SequenceDataset(val_ids, scaled_sequences, feature_cols, label_map, max_seq_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = build_model(architecture, n_features=len(feature_cols), hidden_size=hidden_size).to(device)
        y_true, y_pred, trained_model, best_val_loss, epochs_used = train_one_fold(
            model, train_loader, val_loader, device, max_epochs=max_epochs, patience=patience
        )

        if len(np.unique(y_true)) < 2:
            print(f"[{architecture}] Fold {fold_idx}: val set con una sola clase, se omite AUC.")
            continue
        auc = roc_auc_score(y_true, y_pred)
        fold_aucs.append(auc)
        print(f"[{architecture}] Fold {fold_idx}: AUC = {auc:.3f} (n_val={len(val_ids)})")

        detailed_records.append({
            "architecture": architecture, "fold": fold_idx, "n_train": len(train_ids), "n_val": len(val_ids),
            "n_features": len(feature_cols), "auc": auc, "best_val_loss": best_val_loss, "epochs_used": epochs_used,
        })

        if compute_importance:
            fold_importances = permutation_importance_for_fold(
                trained_model, val_ids, scaled_sequences, feature_cols, label_map, max_seq_len, device, batch_size, y_true, y_pred
            )
            for feat_name, drop in fold_importances.items():
                importance_records.append({"architecture": architecture, "fold": fold_idx, "feature": feat_name, "auc_drop": drop})

    print(f"[{architecture}] AUC promedio ({n_splits} splits x {n_repeats} repeticiones): {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f} ({len(fold_aucs)} folds efectivos)\n")
    return fold_aucs, precomputed, detailed_records, importance_records


def run_all_architectures(sequences_dir: str, labels_csv: str, target: str, max_seq_len: int, n_splits: int,
                           n_repeats: int, hidden_size: int, batch_size: int, seed: int, patience: int, max_epochs: int,
                           compute_importance: bool = False):
    """Corre TODAS las arquitecturas del menú sobre los MISMOS folds y arma la tabla comparativa,
    en el mismo espíritu de tu tabla de LazyPredict para el baseline tabular."""
    results = {}
    precomputed = None
    all_detailed = []
    all_importance = []
    for arch in ARCHITECTURES:
        fold_aucs, precomputed, detailed_records, importance_records = run_cv(
            sequences_dir, labels_csv, target, max_seq_len, n_splits, n_repeats,
            hidden_size, batch_size, seed, arch, patience, max_epochs, precomputed=precomputed,
            compute_importance=compute_importance,
        )
        results[arch] = fold_aucs
        all_detailed.extend(detailed_records)
        all_importance.extend(importance_records)

    print("=" * 60)
    print(f"RESUMEN COMPARATIVO — target={target}")
    print("=" * 60)
    summary_rows = []
    for arch, aucs in results.items():
        mean_auc = np.mean(aucs) if aucs else float("nan")
        std_auc = np.std(aucs) if aucs else float("nan")
        summary_rows.append((arch, mean_auc, std_auc, len(aucs)))
        print(f"{arch:15s}  AUC = {mean_auc:.3f} +/- {std_auc:.3f}  ({len(aucs)} folds)")

    summary_df = pd.DataFrame(summary_rows, columns=["architecture", "auc_mean", "auc_std", "n_folds"]).sort_values("auc_mean", ascending=False)
    detailed_df = pd.DataFrame(all_detailed)
    importance_df = pd.DataFrame(all_importance)
    return summary_df, detailed_df, importance_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena modelos sobre secuencias frame-a-frame (MediaPipe o AU): BiLSTM, BiGRU, CNN-1D, CNN+BiGRU, o MLP con mean-pooling.")
    parser.add_argument("--sequences_dir", type=str, required=True, help="Carpeta con las secuencias (--save_sequences de extract_features.py o extract_au_features_pyfeat.py)")
    parser.add_argument("--labels_csv", type=str, required=True, help="CSV con video_id y las columnas target_ansiedad/target_depresion (el CSV agregado que ya generas)")
    parser.add_argument("--target", type=str, default="target_depresion", choices=["target_ansiedad", "target_depresion"])
    parser.add_argument("--architecture", type=str, default="all", choices=list(ARCHITECTURES.keys()) + ["all"],
                         help="Arquitectura a entrenar. 'all' corre las 5 sobre los MISMOS folds y muestra una tabla comparativa (recomendado, como tu tabla de LazyPredict en audio).")
    parser.add_argument("--max_seq_len", type=int, default=600, help="Longitud máxima de secuencia (se sub-muestrea uniformemente si el video es más largo)")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=3, help="Repeticiones del K-Fold (como en tu baseline tabular, pero más bajo por costo de entrenar una red por fold; sube a 5-10 si el tiempo en Colab te lo permite)")
    parser.add_argument("--hidden_size", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8, help="Épocas sin mejora en val_loss antes de parar (baja a 5-6 si ves que se toma épocas de más sin mejorar de verdad)")
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default=None, help="Si se da (solo con --architecture all), guarda la tabla comparativa (mean/std por arquitectura) en este CSV")
    parser.add_argument("--detail_csv", type=str, default=None, help="Si se da, guarda el detalle POR FOLD (auc, n_train, n_val, epochs_used, best_val_loss) — útil para pruebas estadísticas pareadas entre arquitecturas")
    parser.add_argument("--compute_importance", action="store_true", help="Calcula importancia por permutación de cada feature (cuánto cae el AUC al barajarla). Añade tiempo de cómputo extra (forward passes sobre validation por cada feature y fold).")
    parser.add_argument("--importance_csv", type=str, default=None, help="Si se da junto con --compute_importance, guarda el detalle de importancia (arquitectura, fold, feature, auc_drop) en este CSV")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.architecture == "all":
        summary_df, detailed_df, importance_df = run_all_architectures(
            sequences_dir=args.sequences_dir, labels_csv=args.labels_csv, target=args.target,
            max_seq_len=args.max_seq_len, n_splits=args.n_splits, n_repeats=args.n_repeats,
            hidden_size=args.hidden_size, batch_size=args.batch_size, seed=args.seed,
            patience=args.patience, max_epochs=args.max_epochs, compute_importance=args.compute_importance,
        )
        print(summary_df.to_string(index=False))
        if args.output_csv:
            summary_df.to_csv(args.output_csv, index=False)
            print(f"\nTabla comparativa guardada en '{args.output_csv}'.")
        if args.detail_csv:
            detailed_df.to_csv(args.detail_csv, index=False)
            print(f"Detalle por fold guardado en '{args.detail_csv}'.")
        if args.compute_importance and args.importance_csv:
            importance_df.to_csv(args.importance_csv, index=False)
            print(f"Importancia por permutación guardada en '{args.importance_csv}'.")
    else:
        _, _, detailed_records, importance_records = run_cv(
            sequences_dir=args.sequences_dir, labels_csv=args.labels_csv, target=args.target,
            max_seq_len=args.max_seq_len, n_splits=args.n_splits, n_repeats=args.n_repeats,
            hidden_size=args.hidden_size, batch_size=args.batch_size, seed=args.seed,
            architecture=args.architecture, patience=args.patience, max_epochs=args.max_epochs,
            compute_importance=args.compute_importance,
        )
        if args.detail_csv:
            pd.DataFrame(detailed_records).to_csv(args.detail_csv, index=False)
            print(f"Detalle por fold guardado en '{args.detail_csv}'.")
        if args.compute_importance and args.importance_csv:
            pd.DataFrame(importance_records).to_csv(args.importance_csv, index=False)
            print(f"Importancia por permutación guardada en '{args.importance_csv}'.")