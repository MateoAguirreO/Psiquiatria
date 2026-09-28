"""
DL Script 1 — Feature Reduction + Datos Sintéticos (MLP) — VERSIÓN CORREGIDA
FIX-3WAY: División de 3 vías (FIT/CALIB/VAL) dentro de cada fold para eliminar data leakage.
- FIT: Para entrenar (backpropagation)
- CALIB: Para early stopping y búsqueda de threshold
- VAL: Para evaluación final e imparcial (NUNCA se usa para decisiones)

Experimentos:
A · raw       — embeddings completos (~1024 dims), sin reducción
B · pca       — PCA (95% varianza explicada)
C · sfs       — SFS secuencial forward (top-K features vía proxy LR)
D · raw_mixup — embeddings completos + Mixup en FIT
E · pca_smote — PCA + SMOTE en FIT
F · pca_adasyn— PCA + ADASYN en FIT

Protocolo:
· RepeatedStratifiedKFold(n_splits=5, n_repeats=10) = 50 evaluaciones por config
· Dentro de cada fold: 75% FIT / 25% CALIB / 20% VAL externo
· Todo preprocesamiento (scaler, PCA, SFS) ajustado SOLO en FIT
· Augmentación aplicada SOLO a FIT después del preprocesamiento
· Early stopping monitorea CALIB (nunca VAL)
· Threshold buscado en CALIB (nunca VAL)
· AUC/F1 reportados solo en VAL (una sola vez)
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
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_curve
from imblearn.over_sampling import SMOTE, ADASYN

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("DL_S1_FIXED")

# ─── Configuración ─────────────────────────────────────────────────────────
MODELS_TO_TEST = [
    "xlsr-300m", "xlsr-53", "whisper-large-encoder",
    "wav2vec2-large-robust", "wavlm-large", "hubert-large",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = [
    {"id": "A_raw",        "pca": False, "sfs": False, "aug": None},
    {"id": "B_pca",        "pca": True,  "sfs": False, "aug": None},
    #{"id": "C_sfs",        "pca": False, "sfs": True,  "aug": None},
    {"id": "D_raw_mixup",  "pca": False, "sfs": False, "aug": "mixup"},
    {"id": "E_pca_smote",  "pca": True,  "sfs": False, "aug": "smote"},
    {"id": "F_pca_adasyn", "pca": True,  "sfs": False, "aug": "adasyn"},
]

CV_SPLITS      = 5
CV_REPEATS     = 10
EPOCHS         = 60
LR             = 1e-3
BATCH_SIZE     = 16
PATIENCE       = 8
PCA_VARIANCE   = 0.95
SFS_PREFILT    = 128
SFS_K_FEATURES = 48
MIXUP_ALPHA    = 0.4
CALIB_FRAC     = 0.25  # Fracción del train reservada para CALIB

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

def load_pooled_embeddings(embeddings_root: str, model_key: str):
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}' en manifest.")
    X_rows, y_rows, ids = [], [], []
    patient_segments = {}
    for entry in entries:
        pid = entry["audio_id"]
        if pid not in patient_segments:
            patient_segments[pid] = []
        patient_segments[pid].append(entry["file"])
        if pid not in {i for i, _ in enumerate(ids)}:
            y_rows.append(int(entry["class_id"]))
            ids.append(pid)
    # Reconstruir y única vez
    y_rows = []
    ids_unique = []
    for pid in sorted(patient_segments.keys()):
        embs = []
        for fpath in patient_segments[pid]:
            with open(fpath) as f:
                record = json.load(f)
            embs.append(record["embedding"])
        embs = np.array(embs, dtype=np.float32)
        X_rows.append(embs.mean(axis=0))
        ids_unique.append(pid)
    # Obtener labels del primer segmento de cada paciente
    label_map = {}
    for entry in entries:
        if entry["audio_id"] not in label_map:
            label_map[entry["audio_id"]] = int(entry["class_id"])
    y_rows = [label_map[pid] for pid in ids_unique]
    X = np.stack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    logger.info(f"   Cargados: {len(y)} pacientes | dim={X.shape[1]} | "
                f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}")
    return X, y, ids_unique

# =============================================================================
# 2. PREPROCESAMIENTO (ajustado SIEMPRE solo en FIT)
# =============================================================================

def apply_scaler(X_fit, X_calib, X_val):
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    X_calib_s = scaler.transform(X_calib) if len(X_calib) > 0 else X_calib
    X_val_s = scaler.transform(X_val)
    return X_fit_s, X_calib_s, X_val_s

def apply_pca(X_fit, X_calib, X_val, variance=PCA_VARIANCE):
    pca = PCA(n_components=variance, svd_solver="full", random_state=42)
    X_fit_p = pca.fit_transform(X_fit)
    X_calib_p = pca.transform(X_calib) if len(X_calib) > 0 else X_calib
    X_val_p = pca.transform(X_val)
    return X_fit_p, X_calib_p, X_val_p, pca.n_components_

def apply_sfs(X_fit, X_calib, X_val, y_fit, n_prefilt=SFS_PREFILT, n_final=SFS_K_FEATURES):
    n_features = X_fit.shape[1]
    n_prefilt = min(n_prefilt, n_features)
    n_final = min(n_final, n_prefilt)
    mi_scores = mutual_info_classif(X_fit, y_fit, random_state=42)
    top_idx = np.argsort(mi_scores)[::-1][:n_prefilt]
    X_fit_pre = X_fit[:, top_idx]
    X_calib_pre = X_calib[:, top_idx] if len(X_calib) > 0 else X_calib
    X_val_pre = X_val[:, top_idx]
    proxy = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced",
                                solver="liblinear", random_state=42)
    sfs = SequentialFeatureSelector(proxy, n_features_to_select=n_final,
                                     direction="forward", scoring="f1_macro",
                                     cv=3, n_jobs=-1)
    sfs.fit(X_fit_pre, y_fit)
    X_fit_sfs = sfs.transform(X_fit_pre)
    X_calib_sfs = sfs.transform(X_calib_pre) if len(X_calib) > 0 else X_calib
    X_val_sfs = sfs.transform(X_val_pre)
    logger.debug(f"    SFS: {n_features} → MI top {n_prefilt} → SFS {X_fit_sfs.shape[1]} dims")
    return X_fit_sfs, X_calib_sfs, X_val_sfs

# =============================================================================
# 3. DATOS SINTÉTICOS (solo se aplican a FIT)
# =============================================================================

def mixup_augment(X_fit, y_fit, alpha=MIXUP_ALPHA):
    idx_pos = np.where(y_fit == 1)[0]
    idx_neg = np.where(y_fit == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)
    if n_pos < 2:
        logger.warning("    Mixup: menos de 2 samples positivos, se omite.")
        return X_fit, y_fit
    n_synthetic = n_neg - n_pos
    if n_synthetic <= 0:
        return X_fit, y_fit
    rng = np.random.default_rng(seed=42)
    synthetic_X = []
    for _ in range(n_synthetic):
        i, j = rng.choice(idx_pos, size=2, replace=False)
        lam = rng.beta(alpha, alpha)
        x_new = lam * X_fit[i] + (1.0 - lam) * X_fit[j]
        synthetic_X.append(x_new)
    X_syn = np.array(synthetic_X, dtype=np.float32)
    y_syn = np.ones(n_synthetic, dtype=np.int64)
    X_aug = np.vstack([X_fit, X_syn])
    y_aug = np.concatenate([y_fit, y_syn])
    logger.debug(f"    Mixup: +{n_synthetic} samples → total {len(y_aug)}")
    return X_aug, y_aug

def smote_augment(X_fit, y_fit):
    n_minority = np.sum(y_fit == 1)
    k_safe = max(1, min(5, n_minority - 1))
    try:
        sm = SMOTE(k_neighbors=k_safe, random_state=42)
        X_aug, y_aug = sm.fit_resample(X_fit, y_fit)
        logger.debug(f"    SMOTE: +{len(y_aug) - len(y_fit)} samples → total {len(y_aug)}")
        return X_aug, y_aug
    except Exception as e:
        logger.warning(f"    SMOTE falló ({e}), se usa FIT original.")
        return X_fit, y_fit

def adasyn_augment(X_fit, y_fit):
    n_minority = np.sum(y_fit == 1)
    k_safe = max(1, min(5, n_minority - 1))
    try:
        ada = ADASYN(n_neighbors=k_safe, random_state=42)
        X_aug, y_aug = ada.fit_resample(X_fit, y_fit)
        logger.debug(f"    ADASYN: +{len(y_aug) - len(y_fit)} samples → total {len(y_aug)}")
        return X_aug, y_aug
    except Exception as e:
        logger.warning(f"    ADASYN falló ({e}), se usa FIT original.")
        return X_fit, y_fit

AUG_FUNCTIONS = {"mixup": mixup_augment, "smote": smote_augment,
                 "adasyn": adasyn_augment, None: lambda X, y: (X, y)}

# =============================================================================
# 4. ARQUITECTURA MLP
# =============================================================================

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        h1 = min(512, max(64, input_dim * 2))
        h2 = min(256, max(32, input_dim))
        h3 = min(128, max(16, input_dim // 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(h2, h3), nn.BatchNorm1d(h3), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h3, 1),
        )
    def forward(self, x):
        return self.net(x)

# =============================================================================
# 5. UTILIDADES DE EVALUACIÓN
# =============================================================================

def find_best_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thresholds[np.argmax(f1[:-1])])

# =============================================================================
# 6. ENTRENAMIENTO DE UN FOLD CON DIVISIÓN DE 3 VÍAS
# FIX-3WAY: Early stopping sobre CALIB, threshold buscado en CALIB,
#           evaluación final SOLO en VAL
# =============================================================================

def train_fold(X_fit, y_fit, X_calib, y_calib, X_val, y_val, fold_seed=42):
    """Entrena el MLP con división de 3 vías.
    - FIT: backpropagation
    - CALIB: early stopping + búsqueda de threshold
    - VAL: evaluación final (una sola vez)
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    
    input_dim = X_fit.shape[1]
    ds_fit = TensorDataset(torch.FloatTensor(X_fit), torch.FloatTensor(y_fit.astype(np.float32)))
    ds_val = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val.astype(np.float32)))
    
    loader_fit = DataLoader(ds_fit, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=(len(ds_fit) % BATCH_SIZE == 1))
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    
    has_calib = X_calib is not None and len(X_calib) > 0
    if has_calib:
        ds_calib = TensorDataset(torch.FloatTensor(X_calib),
                                  torch.FloatTensor(y_calib.astype(np.float32)))
        loader_calib = DataLoader(ds_calib, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MLPClassifier(input_dim).to(DEVICE)
    n_neg = np.sum(y_fit == 0)
    n_pos = np.sum(y_fit == 1)
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_calib_loss = float("inf")
    best_state = None
    no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in loader_fit:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        # Early stopping SOLO sobre CALIB (nunca VAL)
        if has_calib:
            model.eval()
            calib_losses = []
            with torch.no_grad():
                for Xb, yb in loader_calib:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    calib_losses.append(criterion(model(Xb).squeeze(-1), yb).item())
            calib_loss = float(np.mean(calib_losses))
            
            if calib_loss < best_calib_loss:
                best_calib_loss = calib_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
    
    if has_calib and best_state is not None:
        model.load_state_dict(best_state)
    
    # Threshold buscado en CALIB (nunca VAL)
    if has_calib:
        model.eval()
        calib_probs_list, calib_labels_list = [], []
        with torch.no_grad():
            for Xb, yb in loader_calib:
                p = torch.sigmoid(model(Xb.to(DEVICE)).squeeze(-1))
                calib_probs_list.extend(p.cpu().numpy().tolist())
                calib_labels_list.extend(yb.cpu().numpy().tolist())
        calib_probs = np.array(calib_probs_list)
        calib_labels = np.array(calib_labels_list)
        thr = find_best_threshold(calib_labels, calib_probs)
    else:
        thr = 0.5  # fallback
    
    # Evaluación final SOLO en VAL (una sola vez, con pesos ya fijados)
    model.eval()
    val_probs_list, val_labels_list = [], []
    with torch.no_grad():
        for Xb, yb in loader_val:
            p = torch.sigmoid(model(Xb.to(DEVICE)).squeeze(-1))
            val_probs_list.extend(p.cpu().numpy().tolist())
            val_labels_list.extend(yb.cpu().numpy().tolist())
    val_probs = np.array(val_probs_list)
    val_labels = np.array(val_labels_list)
    
    auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.0
    val_pred_thr = (val_probs >= thr).astype(int)
    val_pred_05 = (val_probs >= 0.5).astype(int)
    
    return {
        "auc": auc,
        "f1_macro": f1_score(val_labels, val_pred_thr, average="macro", zero_division=0),
        "f1_minority": f1_score(val_labels, val_pred_thr, average="binary", zero_division=0),
        "acc": accuracy_score(val_labels, val_pred_thr),
        "threshold": thr,
        "f1_macro_thr05": f1_score(val_labels, val_pred_05, average="macro", zero_division=0),
        "f1_minority_thr05": f1_score(val_labels, val_pred_05, average="binary", zero_division=0),
        "acc_thr05": accuracy_score(val_labels, val_pred_05),
        "best_calib_loss": best_calib_loss if has_calib else float("nan"),
        "input_dim": input_dim,
        "has_calib": has_calib,
    }

# =============================================================================
# 7. LOOP DE CROSS-VALIDATION CON 3 VÍAS
# =============================================================================

def run_experiment(X, y, exp_config, emb_model, results_path: Path):
    exp_id = exp_config["id"]
    use_pca = exp_config["pca"]
    use_sfs = exp_config["sfs"]
    aug_key = exp_config["aug"]
    aug_fn = AUG_FUNCTIONS[aug_key]
    
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=42)
    rows = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        rep_num = (fold_idx - 1) // CV_SPLITS + 1
        fold_num = (fold_idx - 1) % CV_SPLITS + 1
        
        X_tr_full, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr_full, y_vl = y[train_idx].copy(), y[val_idx].copy()
        
        # FIX-3WAY: Dividir train en FIT (75%) y CALIB (25%)
        try:
            fit_idx, calib_idx = train_test_split(
                np.arange(len(y_tr_full)), test_size=CALIB_FRAC,
                stratify=y_tr_full, random_state=42 + fold_idx
            )
            X_fit_raw, y_fit = X_tr_full[fit_idx], y_tr_full[fit_idx]
            X_calib_raw, y_calib = X_tr_full[calib_idx], y_tr_full[calib_idx]
            if len(np.unique(y_calib)) < 2 or len(calib_idx) < 4:
                raise ValueError("calib insuficiente")
            fallback = False
        except Exception:
            X_fit_raw, y_fit = X_tr_full, y_tr_full
            X_calib_raw, y_calib = np.empty((0, X.shape[1])), np.empty((0,), dtype=np.int64)
            fallback = True
            logger.warning(f"  Fold {fold_idx}: fallback (calib no estratificable)")
        
        # Preprocesamiento: fit SOLO en FIT
        X_fit_s, X_calib_s, X_vl_s = apply_scaler(X_fit_raw, X_calib_raw, X_vl)
        n_components = X_fit_s.shape[1]
        
        if use_pca:
            X_fit_s, X_calib_s, X_vl_s, n_components = apply_pca(X_fit_s, X_calib_s, X_vl_s)
        elif use_sfs:
            X_fit_s, X_calib_s, X_vl_s = apply_sfs(X_fit_s, X_calib_s, X_vl_s, y_fit)
            n_components = X_fit_s.shape[1]
        
        # Augmentación SOLO en FIT
        if aug_key is not None:
            X_fit_s, y_fit = aug_fn(X_fit_s, y_fit)
        
        # Entrenar y evaluar con 3 vías
        metrics = train_fold(X_fit_s, y_fit, X_calib_s, y_calib, X_vl_s, y_vl,
                              fold_seed=42 + fold_idx)
        
        row = {
            "emb_model": emb_model, "experiment": exp_id, "aug": str(aug_key),
            "repeat": rep_num, "fold": fold_num,
            "n_fit": len(X_fit_s), "n_calib": len(X_calib_s), "n_val": len(val_idx),
            "n_components": n_components,
            "calib_fallback": fallback,
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
            logger.info(f"    Rep{rep_num:>2} | AUC={recent['auc'].mean():.3f} | "
                        f"F1={recent['f1_macro'].mean():.3f} | dims={n_components}")
    
    df = pd.DataFrame(rows)
    logger.info(f"  ✅ [{exp_id}] {emb_model} | "
                f"AUC={df['auc'].mean():.3f}±{df['auc'].std():.3f} | "
                f"F1={df['f1_macro'].mean():.3f}±{df['f1_macro'].std():.3f} | "
                f"fallback_rate={df['calib_fallback'].mean():.2f}")
    return df

# =============================================================================
# 8. PIPELINE PRINCIPAL
# =============================================================================

def run_all(embeddings_root: str, output_dir: str):
    output_dir = Path(output_dir) / "dl_script1_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_folds_fixed.csv"
    summary_path = output_dir / "summary_fixed.csv"
    
    if raw_path.exists():
        logger.warning(f"⚠️  {raw_path} ya existe. Se añaden resultados al final.")
    
    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 Embeddings:   {embeddings_root}")
    logger.info(f"📂 Salida:       {output_dir}")
    logger.info(f"🔁 CV: {CV_REPEATS} repeats × {CV_SPLITS} folds = {CV_REPEATS * CV_SPLITS} evaluaciones")
    logger.info(f"🧪 Protocolo: FIT/CALIB/VAL (3 vías) — early stopping y threshold en CALIB, reporte en VAL")
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
            logger.info(f"\n  🔬 Experimento {exp_config['id']}")
            t0 = time.time()
            run_experiment(X, y, exp_config, emb_model, raw_path)
            logger.info(f"     ⏱️  {(time.time()-t0)/60:.1f} min")
    
    if raw_path.exists():
        df_all = pd.read_csv(raw_path)
        summary = (
            df_all.groupby(["emb_model", "experiment"])
            .agg(
                mean_auc=("auc", "mean"), std_auc=("auc", "std"),
                mean_f1_macro=("f1_macro", "mean"), std_f1_macro=("f1_macro", "std"),
                mean_f1_minority=("f1_minority", "mean"), std_f1_minority=("f1_minority", "std"),
                mean_acc=("acc", "mean"), mean_dim=("n_components", "mean"),
                calib_fallback_rate=("calib_fallback", "mean"),
                n_folds=("auc", "count"),
            ).reset_index().sort_values("mean_auc", ascending=False)
        )
        summary.to_csv(summary_path, index=False)
        logger.info("\n" + "="*65)
        logger.info("📊 RESUMEN FINAL (ordenado por AUC) — VERSIÓN CORREGIDA 3-VÍAS")
        logger.info("="*65)
        logger.info("\n" + summary.to_string(index=False))
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
        "Depresión/embeddings/"
    )
    OUTPUT_DIR = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Wav2Vec/results_DL/embeddings_viejos"
    )
    run_all(EMBEDDINGS_ROOT, OUTPUT_DIR)