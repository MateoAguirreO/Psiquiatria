"""
Cross-Dataset Experiment — Depresión
======================================
Dos experimentos en un mismo script:

  Experimento A: Entrenar con DAIC (inglés) → Evaluar con datos propios (español)
  Experimento B: Entrenar con datos propios (español) → Evaluar con DAIC (inglés)

Sin CV: la partición train/test está definida por el origen del dataset.
No hay riesgo de data leakage entre datasets porque nunca se mezclan.

Orden de preprocesamiento (OBLIGATORIO, no cambiar):
  1. StandardScaler   — fit SOLO en X_source (todo el dataset de entrenamiento)
  2. PCA              — fit SOLO en X_source YA escalado
  3. Split interno 80/20 del source (para early stopping)
  4. Augmentación      — SOLO sobre el 80% interno de train, después del split
  5. X_test (dataset destino) se transforma con scaler/PCA del source,
     pero JAMÁS se usa para fit de nada ni para early stopping.

Uso:
  Ajusta las rutas al final del archivo y ejecuta:
  python dl_script_cross_dataset.py
"""

import json
import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CrossDataset")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

MODELS_TO_TEST = [
    "xlsr-300m",
    "wav2vec2-large-robust",
]

EXPERIMENTS = [
    {"id": "raw",       "pca": False, "aug": None},
    {"id": "pca",       "pca": True,  "aug": None},
    {"id": "pca_smote", "pca": True,  "aug": "smote"},
]

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS       = 60
LR           = 1e-3
BATCH_SIZE   = 16
PATIENCE     = 8
PCA_VARIANCE = 0.95
N_RUNS       = 10

# =============================================================================
# 1. CARGA DE EMBEDDINGS
# =============================================================================

def load_pooled_embeddings(embeddings_root: str, model_key: str) -> tuple:
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
        f"   Cargados: {len(y)} pacientes | dim={X.shape[1]} | "
        f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}"
    )
    return X, y, ids


# =============================================================================
# 2. AUGMENTACIÓN
# =============================================================================


def smote_augment(X_train, y_train):
    n_minority = np.sum(y_train == 1)
    k_safe = max(1, min(5, n_minority - 1))
    try:
        sm = SMOTE(k_neighbors=k_safe, random_state=42)
        X_aug, y_aug = sm.fit_resample(X_train, y_train)
        return X_aug, y_aug
    except Exception as e:
        logger.warning(f"    SMOTE falló ({e}), usando train original.")
        return X_train, y_train


AUG_FUNCTIONS = {
    "smote": smote_augment,
    None:    lambda X, y: (X, y),
}


# =============================================================================
# 3. ARQUITECTURA MLP
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


# =============================================================================
# 4. ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

def find_best_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thresholds[np.argmax(f1[:-1])])


def train_and_evaluate(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    use_pca: bool = False,
    aug_key: str = None,
    seed: int = 42,
) -> dict:
    """
    X_source, X_target llegan SIN procesar (raw embeddings, sin escalar).
    Orden de preprocesamiento, en este orden exacto:
      1. Split 80/20 del source (interno, estratificado)
      2. Scaler: fit SOLO en el 80% de train interno -> transform val interno y target
      3. PCA (si use_pca=True): fit SOLO en train interno YA escalado -> transform val interno y target
      4. Augmentación: SOLO en train interno, después de scaler+PCA
    X_target (test real) solo recibe `.transform()` en cada paso, nunca `.fit()`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 1. Split interno 80/20 del source ──────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, vl_idx = next(sss.split(X_source, y_source))

    X_tr, X_vl = X_source[tr_idx].copy(), X_source[vl_idx].copy()
    y_tr, y_vl = y_source[tr_idx].copy(), y_source[vl_idx].copy()
    X_te = X_target.copy()
    y_te = y_target.copy()

    # ── 2. Scaler: fit SOLO en X_tr (80% interno) ──────────────────────────
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_te = scaler.transform(X_te)

    # ── 3. PCA: fit SOLO en X_tr YA escalado ───────────────────────────────
    n_components = X_tr.shape[1]
    if use_pca:
        pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=seed)
        X_tr = pca.fit_transform(X_tr)
        X_vl = pca.transform(X_vl)
        X_te = pca.transform(X_te)
        n_components = pca.n_components_

    # ── 4. Augmentación SOLO en train interno (después de scaler+PCA) ─────
    if aug_key is not None:
        X_tr, y_tr = AUG_FUNCTIONS[aug_key](X_tr, y_tr)

    input_dim = X_tr.shape[1]

    ds_train = TensorDataset(
        torch.FloatTensor(X_tr), torch.FloatTensor(y_tr.astype(np.float32))
    )
    ds_val = TensorDataset(
        torch.FloatTensor(X_vl), torch.FloatTensor(y_vl.astype(np.float32))
    )
    ds_test = TensorDataset(
        torch.FloatTensor(X_te), torch.FloatTensor(y_te.astype(np.float32))
    )

    loader_tr   = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=(len(ds_train) % BATCH_SIZE == 1))
    loader_val  = DataLoader(ds_val,  batch_size=BATCH_SIZE, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPClassifier(input_dim).to(DEVICE)

    n_neg = np.sum(y_tr == 0)
    n_pos = np.sum(y_tr == 1)
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = -1.0
    best_state   = None
    no_improve   = 0
    epoch        = 0

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

        model.eval()
        probs_vl, labels_vl = [], []
        with torch.no_grad():
            for Xb, yb in loader_val:
                p = torch.sigmoid(model(Xb.to(DEVICE)).squeeze(-1))
                probs_vl.extend(p.cpu().numpy().tolist())
                labels_vl.extend(yb.cpu().numpy().tolist())

        vl_probs  = np.array(probs_vl)
        vl_labels = np.array(labels_vl)
        val_auc   = (roc_auc_score(vl_labels, vl_probs)
                     if len(np.unique(vl_labels)) > 1 else 0.0)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    probs_te, labels_te = [], []
    with torch.no_grad():
        for Xb, yb in loader_test:
            p = torch.sigmoid(model(Xb.to(DEVICE)).squeeze(-1))
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
        "n_train":      len(X_tr),
        "n_val_int":    len(X_vl),
        "n_test":       len(X_te),
        "n_components": n_components,
        "input_dim":    input_dim,
        "seed":         seed,
    }


# =============================================================================
# 5. PIPELINE CROSS-DATASET
# =============================================================================

def run_cross_dataset(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    exp_config: dict,
    emb_model: str,
    direction: str,
    results_path: Path,
):
    """
    Pasa los embeddings crudos a train_and_evaluate, que se encarga de
    todo el preprocesamiento (scaler -> PCA -> augmentación) ajustado
    SIEMPRE sobre el 80% interno de X_source. X_target nunca participa
    de ningún .fit().
    """
    exp_id  = exp_config["id"]
    use_pca = exp_config["pca"]
    aug_key = exp_config["aug"]

    rows = []

    for run in range(N_RUNS):
        seed = 42 + run

        metrics = train_and_evaluate(
            X_source.copy(), y_source.copy(),
            X_target.copy(), y_target.copy(),
            use_pca=use_pca, aug_key=aug_key, seed=seed,
        )

        row = {
            "direction":   direction,
            "emb_model":   emb_model,
            "experiment":  exp_id,
            "run":         run + 1,
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
        f"  ✅ [{direction}|{exp_id}|{emb_model}] "
        f"AUC={df['auc'].mean():.3f}±{df['auc'].std():.3f} | "
        f"F1-macro={df['f1_macro'].mean():.3f} | "
        f"F1-min={df['f1_minority'].mean():.3f}"
    )
    return df


# =============================================================================
# 6. PIPELINE PRINCIPAL
# =============================================================================

def run_all(embeddings_daic: str, embeddings_propio: str, output_dir: str):
    output_dir = Path(output_dir) / "cross_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = output_dir / "raw_runs.csv"
    summary_path = output_dir / "summary.csv"

    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 DAIC:    {embeddings_daic}")
    logger.info(f"📂 Propio:  {embeddings_propio}")
    logger.info(f"📂 Salida:  {output_dir}")
    logger.info(f"🔁 Runs por config: {N_RUNS} (semillas distintas)\n")

    t0_global = time.time()

    for emb_model in MODELS_TO_TEST:
        logger.info(f"{'='*65}")
        logger.info(f"📦 Modelo: {emb_model}")
        logger.info(f"{'='*65}")

        try:
            logger.info("  Cargando DAIC...")
            X_daic, y_daic, _ = load_pooled_embeddings(embeddings_daic, emb_model)
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar DAIC/{emb_model}: {e}")
            continue

        try:
            logger.info("  Cargando datos propios...")
            X_propio, y_propio, _ = load_pooled_embeddings(embeddings_propio, emb_model)
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar propio/{emb_model}: {e}")
            continue

        for exp_config in EXPERIMENTS:
            exp_id = exp_config["id"]

            logger.info(f"\n  🔬 [{exp_id}] DAIC (train) → Propio (test)")
            t0 = time.time()
            run_cross_dataset(
                X_source=X_daic,   y_source=y_daic,
                X_target=X_propio, y_target=y_propio,
                exp_config=exp_config, emb_model=emb_model,
                direction="daic→propio", results_path=raw_path,
            )
            logger.info(f"     ⏱️  {(time.time()-t0)/60:.1f} min")

            logger.info(f"\n  🔬 [{exp_id}] Propio (train) → DAIC (test)")
            t0 = time.time()
            run_cross_dataset(
                X_source=X_propio, y_source=y_propio,
                X_target=X_daic,   y_target=y_daic,
                exp_config=exp_config, emb_model=emb_model,
                direction="propio→daic", results_path=raw_path,
            )
            logger.info(f"     ⏱️  {(time.time()-t0)/60:.1f} min")

    if raw_path.exists():
        df_all = pd.read_csv(raw_path)
        summary = (
            df_all
            .groupby(["direction", "emb_model", "experiment"])
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
            .sort_values("mean_auc", ascending=False)
        )
        summary.to_csv(summary_path, index=False)

        logger.info("\n" + "="*65)
        logger.info("📊 RESUMEN CROSS-DATASET (ordenado por AUC)")
        logger.info("="*65)
        logger.info("\n" + summary.to_string(index=False))
        logger.info(f"\n📄 Summary → {summary_path}")
        logger.info(f"📄 Raw runs → {raw_path}")

    total_min = (time.time() - t0_global) / 60
    logger.info(f"\n⏱️  Tiempo total: {total_min:.1f} min")


# =============================================================================
# PUNTO DE ENTRADA — ajusta estas rutas
# =============================================================================

if __name__ == "__main__":
    EMBEDDINGS_DAIC = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/"
        "Depresión/embeddings_daic/"
    )
    EMBEDDINGS_PROPIO = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/"
        "Depresión/embeddings_v2/"
    )
    OUTPUT_DIR = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Wav2Vec/results_DL/"
    )

    run_all(EMBEDDINGS_DAIC, EMBEDDINGS_PROPIO, OUTPUT_DIR)