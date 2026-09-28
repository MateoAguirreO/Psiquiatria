"""
DL Script 1 — Feature Reduction + Datos Sintéticos (MLP)
=========================================================
Experimentos:
  A · raw       — embeddings completos (~1024 dims), sin reducción
  B · pca       — PCA (95% varianza explicada)
  C · sfs       — SFS secuencial forward (top-K features vía proxy LR)
  D · raw_mixup — embeddings completos + Mixup en train
  E · pca_smote — PCA + SMOTE en train
  F · pca_adasyn— PCA + ADASYN en train

Protocolo de validación:
  · RepeatedStratifiedKFold(n_splits=5, n_repeats=10) = 50 evaluaciones por config
  · Cada fila = 1 paciente → no hay data leakage entre pacientes
  · VAL siempre usa datos REALES (nunca sintéticos)
  · Todo preprocesamiento (scaler, PCA, SFS) ajustado SOLO en X_train del fold
  · Augmentación aplicada SOLO a X_train después del preprocesamiento

Arquitectura DL:
  MLP de 3 capas (baseline comparable al pipeline ML del compañero).
  Script 2 (separado) cubrirá CNN1D, BiLSTM, BiGRU, CNN+BiGRU.

Uso:
  python dl_script1_feature_reduction.py

Ajusta EMBEDDINGS_ROOT y OUTPUT_DIR al final del archivo.
"""

import os
import sys
import json
import logging
import warnings
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SequentialFeatureSelector,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, precision_recall_curve,
)

from imblearn.over_sampling import SMOTE, ADASYN

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DL_S1")

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

# Experimentos a ejecutar (puedes comentar los que no quieras)
EXPERIMENTS = [
    {"id": "A_raw",        "pca": False, "sfs": False, "aug": None},
    {"id": "B_pca",        "pca": True,  "sfs": False, "aug": None},
    {"id": "C_sfs",        "pca": False, "sfs": True,  "aug": None},
    {"id": "D_raw_mixup",  "pca": False, "sfs": False, "aug": "mixup"},
    {"id": "E_pca_smote",  "pca": True,  "sfs": False, "aug": "smote"},
    {"id": "F_pca_adasyn", "pca": True,  "sfs": False, "aug": "adasyn"},
]

# Hiperparámetros globales
CV_SPLITS      = 5
CV_REPEATS     = 10
EPOCHS         = 60
LR             = 1e-3
BATCH_SIZE     = 16
PATIENCE       = 8       # early stopping
PCA_VARIANCE   = 0.95    # varianza explicada para PCA
SFS_PREFILT    = 128     # features pre-filtradas por MI antes del SFS
SFS_K_FEATURES = 48      # features finales seleccionadas por SFS
MIXUP_ALPHA    = 0.4     # parámetro Beta para Mixup


# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

def load_pooled_embeddings(embeddings_root: str, model_key: str):
    """
    Carga embeddings version v2 : junta los segmentos sueltos de cada paciente
    y calcula el promedio (mean-pooling).
    """
    from collections import defaultdict
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}' en manifest.")
    
    logger.info("ℹ️ Agrupando segmentos por paciente (Formato v2)...")

    # 1. Agrupar los archivos de segmentos que le pertenecen a cada paciente (audio_id)
    patient_segments = defaultdict(list)
    patient_label = {}
    for entry in entries:
        pid = entry["audio_id"]
        patient_segments[pid].append(entry["file"])
        patient_label[pid] = int(entry["class_id"])

    # 2. Cargar cada segmento y calcular el promedio por paciente
    X_rows, y_rows, ids = [], [], []
    for pid in sorted(patient_segments.keys()):
        segment_files = patient_segments[pid]
        embs = []
        for fpath in segment_files:
            with open(fpath) as f:
                record = json.load(f)
            embs.append(record["embedding"])
        embs = np.array(embs, dtype=np.float32)
        
        X_rows.append(embs.mean(axis=0)) # Promedio del paciente
        y_rows.append(patient_label[pid])
        ids.append(pid)

    X = np.stack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    logger.info(
        f"   Cargados NUEVOS: {len(y)} pacientes | dim={X.shape[1]} | "
        f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}"
    )
    return X, y, ids


# =============================================================================
# 2. PREPROCESAMIENTO (ajustado SIEMPRE solo en train)
# =============================================================================

def apply_scaler(X_train, X_val):
    """StandardScaler ajustado en train, aplicado a ambos."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val)


def apply_pca(X_train, X_val, variance=PCA_VARIANCE):
    """
    PCA ajustada en train.
    Retorna X reducido y n_components elegido automáticamente.
    """
    pca = PCA(n_components=variance, svd_solver="full", random_state=42)
    X_tr = pca.fit_transform(X_train)
    X_vl = pca.transform(X_val)
    logger.debug(f"    PCA: {X_train.shape[1]} → {X_tr.shape[1]} dims")
    return X_tr, X_vl, pca.n_components_


def apply_sfs(X_train, X_val, y_train,
              n_prefilt=SFS_PREFILT, n_final=SFS_K_FEATURES):
    """
    Selección secuencial de features:
      1. Pre-filtro por Información Mutua: top n_prefilt features
         (rápido, O(n_features))
      2. SequentialFeatureSelector forward sobre esas n_prefilt features
         usando LogisticRegression como estimador proxy (rápido)
         hasta seleccionar n_final features.

    Todo ajustado SOLO en X_train.
    """
    n_features = X_train.shape[1]
    n_prefilt  = min(n_prefilt, n_features)
    n_final    = min(n_final, n_prefilt)

    # ── Paso 1: Filtro por MI ─────────────────────────────────────────────
    mi_scores  = mutual_info_classif(X_train, y_train, random_state=42)
    top_idx    = np.argsort(mi_scores)[::-1][:n_prefilt]
    X_tr_pre   = X_train[:, top_idx]
    X_vl_pre   = X_val[:, top_idx]

    # ── Paso 2: SFS forward ───────────────────────────────────────────────
    proxy = LogisticRegression(
        C=1.0, max_iter=300, class_weight="balanced",
        solver="liblinear", random_state=42
    )
    sfs = SequentialFeatureSelector(
        proxy,
        n_features_to_select=n_final,
        direction="forward",
        scoring="f1_macro",
        cv=3,               # CV interno ligero para el proxy
        n_jobs=-1,
    )
    sfs.fit(X_tr_pre, y_train)
    X_tr_sfs = sfs.transform(X_tr_pre)
    X_vl_sfs = sfs.transform(X_vl_pre)

    logger.debug(
        f"    SFS: {n_features} → MI top {n_prefilt} → SFS {X_tr_sfs.shape[1]} dims"
    )
    return X_tr_sfs, X_vl_sfs


# =============================================================================
# 3. DATOS SINTÉTICOS (solo se aplican a X_train)
# =============================================================================

def mixup_augment(X_train, y_train, alpha=MIXUP_ALPHA):
    """
    Mixup intra-clase minoritaria:
      - Interpola pares de samples de la clase 1 con λ ~ Beta(alpha, alpha)
      - Genera suficientes samples sintéticos para equilibrar las clases
      - Los datos de VALIDACIÓN nunca se tocan

    Retorna X_aug, y_aug con clases equilibradas.
    """
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    if n_pos < 2:
        logger.warning("    Mixup: menos de 2 samples positivos, se omite.")
        return X_train, y_train

    n_synthetic = n_neg - n_pos       # cuántos necesitamos para equilibrar
    if n_synthetic <= 0:
        return X_train, y_train

    rng = np.random.default_rng(seed=42)
    synthetic_X = []
    for _ in range(n_synthetic):
        i, j   = rng.choice(idx_pos, size=2, replace=False)
        lam    = rng.beta(alpha, alpha)
        x_new  = lam * X_train[i] + (1.0 - lam) * X_train[j]
        synthetic_X.append(x_new)

    X_syn = np.array(synthetic_X, dtype=np.float32)
    y_syn = np.ones(n_synthetic, dtype=np.int64)

    X_aug = np.vstack([X_train, X_syn])
    y_aug = np.concatenate([y_train, y_syn])
    logger.debug(f"    Mixup: +{n_synthetic} samples → total {len(y_aug)}")
    return X_aug, y_aug


def smote_augment(X_train, y_train):
    """
    SMOTE sobre X_train.
    k_neighbors ajustado automáticamente para evitar error con n minoritaria pequeña.
    """
    n_minority = np.sum(y_train == 1)
    k_safe = max(1, min(5, n_minority - 1))
    try:
        sm = SMOTE(k_neighbors=k_safe, random_state=42)
        X_aug, y_aug = sm.fit_resample(X_train, y_train)
        n_new = len(y_aug) - len(y_train)
        logger.debug(f"    SMOTE: +{n_new} samples → total {len(y_aug)}")
        return X_aug, y_aug
    except Exception as e:
        logger.warning(f"    SMOTE falló ({e}), se usa X_train original.")
        return X_train, y_train


def adasyn_augment(X_train, y_train):
    """ADASYN sobre X_train con k_neighbors seguro."""
    n_minority = np.sum(y_train == 1)
    k_safe = max(1, min(5, n_minority - 1))
    try:
        ada = ADASYN(n_neighbors=k_safe, random_state=42)
        X_aug, y_aug = ada.fit_resample(X_train, y_train)
        n_new = len(y_aug) - len(y_train)
        logger.debug(f"    ADASYN: +{n_new} samples → total {len(y_aug)}")
        return X_aug, y_aug
    except Exception as e:
        logger.warning(f"    ADASYN falló ({e}), se usa X_train original.")
        return X_train, y_train


AUG_FUNCTIONS = {
    "mixup":  mixup_augment,
    "smote":  smote_augment,
    "adasyn": adasyn_augment,
    None:     lambda X, y: (X, y),
}


# =============================================================================
# 4. ARQUITECTURA MLP
# =============================================================================

class MLPClassifier(nn.Module):
    """
    MLP para embeddings pooled.
    Se adapta automáticamente al input_dim según el experimento
    (raw ~1024, PCA ~50-200, SFS ~48).
    """
    def __init__(self, input_dim: int):
        super().__init__()

        # Tamaño de capas adaptativo según input
        h1 = min(512, max(64, input_dim * 2))
        h2 = min(256, max(32, input_dim))
        h3 = min(128, max(16, input_dim // 2))

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(h3, 1),  # logit → BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 5. ENTRENAMIENTO DE UN FOLD
# =============================================================================

def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Umbral óptimo para F1-macro buscado en VAL (nunca en train).
    Con ≤ 2 clases únicas en val devuelve 0.5 por seguridad.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = np.argmax(f1[:-1])
    return float(thresholds[best])


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Entrena el MLP en un fold y retorna métricas en validación.
    X_train puede contener datos sintéticos; X_val es SIEMPRE real.
    """
    input_dim = X_train.shape[1]

    # ── Dataset y Loaders ─────────────────────────────────────────────────
    ds_train = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train.astype(np.float32))
    )
    ds_val = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val.astype(np.float32))
    )
    # Fix 1: La parte del droplast true
    loader_tr  = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=(len(ds_train) % BATCH_SIZE == 1))
    loader_val = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Modelo ────────────────────────────────────────────────────────────
    model = MLPClassifier(input_dim).to(DEVICE)

    # pos_weight calculado con y_train (SOLO train, nunca val)
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Loop de entrenamiento con early stopping ───────────────────────────
    best_f1_macro = -1.0
    best_state    = None
    best_metrics  = {}
    no_improve    = 0

    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in loader_tr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ── Evaluación en VAL (datos reales) ──────────────────────────────
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for Xb, yb in loader_val:
                p = torch.sigmoid(model(Xb.to(DEVICE)).squeeze(-1))
                probs_list.extend(p.cpu().numpy().tolist())
                labels_list.extend(yb.cpu().numpy().tolist())

        val_probs  = np.array(probs_list)
        val_labels = np.array(labels_list)

        # Umbral adaptativo buscado en val
        thr      = find_best_threshold(val_labels, val_probs)
        val_pred = (val_probs >= thr).astype(int)

        f1_macro = f1_score(val_labels, val_pred, average="macro",  zero_division=0)
        f1_min   = f1_score(val_labels, val_pred, average="binary", zero_division=0)
        acc      = accuracy_score(val_labels, val_pred)
        auc      = (roc_auc_score(val_labels, val_probs)
                    if len(np.unique(val_labels)) > 1 else 0.0)

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics  = {
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

def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
    exp_config: dict,
    emb_model: str,
    results_path: Path,
):
    """
    Ejecuta RepeatedStratifiedKFold para un experimento y modelo de embedding.

    Flujo dentro de cada fold:
      1. Partir train/val (val = datos REALES del paciente)
      2. StandardScaler fit en train → transform ambos
      3. [Opcional] PCA fit en train scaled → transform ambos
      4. [Opcional] SFS fit en train scaled → transform ambos
      5. [Opcional] Augmentación en X_train_proc SOLAMENTE
      6. Entrenar MLP, evaluar en X_val_proc (datos reales)
      7. Guardar fila de resultados incrementalmente
    """
    exp_id  = exp_config["id"]
    use_pca = exp_config["pca"]
    use_sfs = exp_config["sfs"]
    aug_key = exp_config["aug"]
    aug_fn  = AUG_FUNCTIONS[aug_key]

    cv    = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=42)
    total = CV_SPLITS * CV_REPEATS
    rows  = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        rep_num  = (fold_idx - 1) // CV_SPLITS + 1
        fold_num = (fold_idx - 1) %  CV_SPLITS + 1

        X_tr, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr, y_vl = y[train_idx].copy(), y[val_idx].copy()

        # ── 1. Escalar (fit solo en train) ────────────────────────────────
        X_tr, X_vl = apply_scaler(X_tr, X_vl)

        # ── 2. Reducción de features ──────────────────────────────────────
        n_components = X_tr.shape[1]   # default: sin reducción
        if use_pca:
            X_tr, X_vl, n_components = apply_pca(X_tr, X_vl)
        elif use_sfs:
            X_tr, X_vl = apply_sfs(X_tr, X_vl, y_tr)
            n_components = X_tr.shape[1]

        # ── 3. Augmentación SOLO en X_tr ──────────────────────────────────
        # X_vl NO se toca en ningún caso.
        if aug_key is not None:
            X_tr, y_tr = aug_fn(X_tr, y_tr)

        # ── 4. Entrenar y evaluar ─────────────────────────────────────────
        metrics = train_fold(X_tr, y_tr, X_vl, y_vl)

        row = {
            "emb_model":    emb_model,
            "experiment":   exp_id,
            "aug":          str(aug_key),
            "repeat":       rep_num,
            "fold":         fold_num,
            "n_train_real": len(train_idx),       # samples reales en train
            "n_train_aug":  len(X_tr),            # samples tras augmentación
            "n_val":        len(val_idx),          # val = siempre real
            "n_components": n_components,
            **metrics,
        }
        rows.append(row)

        # Guardar incrementalmente (por si el proceso se interrumpe)
        pd.DataFrame([row]).to_csv(
            results_path, mode="a",
            header=not results_path.exists() or results_path.stat().st_size == 0,
            index=False,
        )

        if fold_idx % CV_SPLITS == 0:
            recent = pd.DataFrame(rows[-CV_SPLITS:])
            mean_f1  = recent["f1_macro"].mean()
            mean_auc = recent["auc"].mean()
            logger.info(
                f"    Rep{rep_num:>2} | "
                f"F1-macro={mean_f1:.3f} | AUC={mean_auc:.3f} | "
                f"dims={n_components}"
            )

    df = pd.DataFrame(rows)
    logger.info(
        f"  ✅ [{exp_id}] {emb_model} | "
        f"F1-macro={df['f1_macro'].mean():.3f}±{df['f1_macro'].std():.3f} | "
        f"F1-min={df['f1_minority'].mean():.3f} | "
        f"AUC={df['auc'].mean():.3f}"
    )
    return df


# =============================================================================
# 7. PIPELINE PRINCIPAL
# =============================================================================

def run_all(embeddings_root: str, output_dir: str):
    output_dir = Path(output_dir) / "dl_script1"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = output_dir / "raw_folds.csv"
    summary_path = output_dir / "summary.csv"

    # Limpiar raw_folds si existe de una corrida anterior parcial
    if raw_path.exists():
        logger.warning(f"⚠️  {raw_path} ya existe. Se añadirán resultados al final.")

    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 Embeddings:   {embeddings_root}")
    logger.info(f"📂 Salida:       {output_dir}")
    logger.info(
        f"🔁 CV: {CV_REPEATS} repeats × {CV_SPLITS} folds = "
        f"{CV_REPEATS * CV_SPLITS} evaluaciones por config"
    )
    logger.info(f"🧪 Experimentos: {[e['id'] for e in EXPERIMENTS]}\n")

    t0_global = time.time()

    for emb_model in MODELS_TO_TEST:
        logger.info(f"{'='*65}")
        logger.info(f"📦 Modelo de embeddings: {emb_model}")
        logger.info(f"{'='*65}")

        try:
            X, y, _ids = load_pooled_embeddings(embeddings_root, emb_model)
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar {emb_model}: {e}")
            continue

        if len(np.unique(y)) < 2:
            logger.warning("   Solo una clase. Saltando.")
            continue

        for exp_config in EXPERIMENTS:
            exp_id = exp_config["id"]
            logger.info(f"\n  🔬 Experimento {exp_id}")
            t0 = time.time()
            run_experiment(X, y, exp_config, emb_model, raw_path)
            elapsed = time.time() - t0
            logger.info(f"     ⏱️  {elapsed/60:.1f} min")

    # ── Resumen final ──────────────────────────────────────────────────────
    if raw_path.exists():
        df_all = pd.read_csv(raw_path)
        summary = (
            df_all
            .groupby(["emb_model", "experiment"])
            .agg(
                mean_f1_macro    =("f1_macro",    "mean"),
                std_f1_macro     =("f1_macro",    "std"),
                mean_f1_minority =("f1_minority", "mean"),
                std_f1_minority  =("f1_minority", "std"),
                mean_auc         =("auc",         "mean"),
                std_auc          =("auc",         "std"),
                mean_acc         =("acc",         "mean"),
                mean_dim         =("n_components","mean"),
                n_folds          =("f1_macro",    "count"),
            )
            .reset_index()
            .sort_values("mean_f1_macro", ascending=False)
        )
        summary.to_csv(summary_path, index=False)

        logger.info("\n" + "="*65)
        logger.info("📊 RESUMEN FINAL (ordenado por F1-macro)")
        logger.info("="*65)
        logger.info("\n" + summary.to_string(index=False))
        logger.info(f"\n📄 Summary → {summary_path}")
        logger.info(f"📄 Raw folds → {raw_path}")

    total_min = (time.time() - t0_global) / 60
    logger.info(f"\n⏱️  Tiempo total: {total_min:.1f} min ({total_min/60:.1f} h)")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # ── Ajusta estas rutas ─────────────────────────────────────────────────
    EMBEDDINGS_ROOT = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/"
        "Depresión/embeddings_v2/"
    )
    OUTPUT_DIR = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Wav2Vec/results_DL/embeddings_nuevos"
    )
    # ───────────────────────────────────────────────────────────────────────

    run_all(EMBEDDINGS_ROOT, OUTPUT_DIR)
