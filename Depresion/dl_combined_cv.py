"""
dl_combined_cv.py
==================
Experimento combinado: mezcla DAIC-WOZ + datos propios en un solo pool
y corre StratifiedGroupKFold CV con grupos por paciente.

Garantías anti-leakage:
  · Todos los segmentos de un mismo paciente van SIEMPRE al mismo fold
    (gracias a groups=patient_id en StratifiedGroupKFold).
  · Scaler y PCA se fittean solo sobre los segmentos de train del fold.
  · Augmentación solo en train del fold.
  · Para arquitecturas de secuencia: se usa mean-pooling por paciente
    antes del CV, así cada paciente = 1 fila. Esto es lo correcto para
    evitar que el modelo sobreajuste a pacientes con muchos segmentos.

Protocolo CV:
  · StratifiedGroupKFold(n_splits=5) × N_REPEATS repeticiones manuales
    (RepeatedStratifiedKFold no soporta groups).
  · Estratificado por clase → balance de positivos/negativos en cada fold.
  · Agrupado por patient_id → segmentos del mismo paciente nunca se separan.

Uso:
  Ajusta las rutas y N_REPEATS al final del archivo, luego:
  python dl_combined_cv.py
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, roc_auc_score,
)
from torch.utils.data import DataLoader

from cross_dataset_common import (
    DEVICE,
    ARCHITECTURES,
    SEQUENCE_ARCHS,
    LR_BY_ARCH,
    PooledDataset,
    SequenceDataset,
    smote_augment,
    find_best_threshold,
    summarize,
    logger,
)

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# =============================================================================
# CONFIGURACIÓN — ajusta aquí
# =============================================================================

MODELS_TO_TEST = [
    "xlsr-300m",
    "xlsr-53",
    "whisper-large-encoder",
    "wav2vec2-large-robust",
    "wavlm-large",
    "hubert-large",
]

ARCHS_TO_TEST = [
    #"MLP",
    "CNN1D",
    "BiLSTM",
    "BiGRU",
    "CNN_BiGRU",
]

EXPERIMENTS = [
    {"id": "raw",       "pca": False, "aug": None},
    {"id": "pca",       "pca": True,  "aug": None},
    {"id": "pca_smote", "pca": True,  "aug": "smote"},  # smote solo para MLP
]

N_SPLITS  = 5
N_REPEATS = 5      # 5 rep × 5 folds = 25 evaluaciones por config (~2-3h con GPU)
PCA_VAR   = 0.95
EPOCHS    = 60
BATCH     = 16
PATIENCE  = 8


# =============================================================================
# 1. CARGA COMBINADA — pooled (un vector por paciente)
# =============================================================================

def load_combined_pooled(
    embeddings_daic: str,
    embeddings_propio: str,
    model_key: str,
) -> tuple:
    """
    Carga y hace mean-pooling por paciente de ambos datasets.
    Devuelve:
      X        (n_patients, dim)
      y        (n_patients,)   — etiqueta binaria
      groups   (n_patients,)   — ID único "ds0_300" / "ds1_004" para GroupKFold
      origins  (n_patients,)   — "daic" o "propio" (para diagnóstico)
    """
    def _load(root, prefix):
        manifest_path = Path(root) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        entries = [m for m in manifest if m["model"] == model_key]
        if not entries:
            raise ValueError(f"Sin embeddings para '{model_key}' en {root}")

        patient_segs  = defaultdict(list)
        patient_label = {}
        for e in entries:
            pid = e["audio_id"]
            patient_segs[pid].append(e["file"])
            patient_label[pid] = int(e["class_id"])

        rows_X, rows_y, rows_g, rows_o = [], [], [], []
        for pid in sorted(patient_segs.keys()):
            embs = []
            for fpath in patient_segs[pid]:
                with open(fpath) as f:
                    rec = json.load(f)
                embs.append(rec["embedding"])
            embs = np.array(embs, dtype=np.float32)
            rows_X.append(embs.mean(axis=0))
            rows_y.append(patient_label[pid])
            rows_g.append(f"{prefix}_{pid}")
            rows_o.append(prefix)
        return rows_X, rows_y, rows_g, rows_o

    Xd, yd, gd, od = _load(embeddings_daic,   "daic")
    Xp, yp, gp, op = _load(embeddings_propio, "propio")

    X       = np.stack(Xd + Xp).astype(np.float32)
    y       = np.array(yd + yp, dtype=np.int64)
    groups  = np.array(gd + gp)
    origins = np.array(od + op)

    n_d = len(yd); n_p = len(yp)
    logger.info(
        f"   Combinado (pooled): {len(y)} pacientes | dim={X.shape[1]} | "
        f"DAIC={n_d} (dep={sum(yd)}) | Propio={n_p} (dep={sum(yp)}) | "
        f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}"
    )
    return X, y, groups, origins


# =============================================================================
# 2. PREPROCESAMIENTO EN FOLD — pooled
# =============================================================================

def preprocess_fold_pooled(
    X_tr, y_tr, X_vl,
    use_pca: bool, aug_key: str, seed: int,
):
    """Scaler → PCA → aug, todo fit SOLO en X_tr."""
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_vl   = scaler.transform(X_vl)

    n_comp = X_tr.shape[1]
    if use_pca:
        pca   = PCA(n_components=PCA_VAR, svd_solver="full", random_state=seed)
        X_tr  = pca.fit_transform(X_tr)
        X_vl  = pca.transform(X_vl)
        n_comp = pca.n_components_

    if aug_key == "smote":
        X_tr, y_tr = smote_augment(X_tr, y_tr)

    return X_tr, y_tr, X_vl, n_comp


# =============================================================================
# 3. ENTRENAMIENTO DE UN FOLD — genérico (pooled únicamente)
#    Las arquitecturas de secuencia usan los datos pooled también —
#    mean-pooling por paciente antes del CV es la estrategia correcta
#    para datos combinados (evita que pacientes con muchos segmentos
#    dominen el entrenamiento).
# =============================================================================

def train_fold(
    arch_name: str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_vl: np.ndarray, y_vl: np.ndarray,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_tr.shape[1]
    ModelClass = ARCHITECTURES[arch_name]

    # Todas las arquitecturas reciben (B, D) en el combinado (ya pooled)
    # CNN1D/BiLSTM/BiGRU trabajan sobre la dim D como si fuera 1 timestep
    # → para el combinado, MLP es la arquitectura natural; las secuenciales
    #   tienen sentido solo si cargamos las secuencias por separado (futuro)
    model = ModelClass(input_dim).to(DEVICE)
    lr    = LR_BY_ARCH[arch_name]

    ds_tr = PooledDataset(X_tr, y_tr)
    ds_vl = PooledDataset(X_vl, y_vl)
    loader_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                           drop_last=(len(ds_tr) % BATCH == 1))
    loader_vl = DataLoader(ds_vl, batch_size=BATCH, shuffle=False)

    n_neg = np.sum(y_tr == 0)
    n_pos = np.sum(y_tr == 1)
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Las arquitecturas de secuencia (CNN1D/BiLSTM/BiGRU/CNN_BiGRU) esperan
    # (B, T, D). En el combinado los datos son pooled (B, D), así que les
    # pasamos (B, 1, D) — un solo timestep por paciente.
    is_seq = arch_name in SEQUENCE_ARCHS

    def forward(model, Xb):
        if is_seq:
            return model(Xb.unsqueeze(1))   # (B,D) → (B,1,D)
        return model(Xb)

    best_auc   = -1.0
    best_state = None
    no_improve = 0
    epoch      = 0

    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb, _ in loader_tr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(forward(model, Xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        probs_vl, labs_vl = [], []
        with torch.no_grad():
            for Xb, yb, _ in loader_vl:
                p = torch.sigmoid(forward(model, Xb.to(DEVICE)).squeeze(-1))
                probs_vl.extend(p.cpu().numpy().tolist())
                labs_vl.extend(yb.cpu().numpy().tolist())

        vl_probs  = np.array(probs_vl)
        vl_labels = np.array(labs_vl)
        val_auc   = (roc_auc_score(vl_labels, vl_probs)
                     if len(np.unique(vl_labels)) > 1 else 0.0)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # Evaluar con el mejor estado
    model.load_state_dict(best_state)
    model.eval()
    probs_vl, labs_vl = [], []
    with torch.no_grad():
        for Xb, yb, _ in loader_vl:
            p = torch.sigmoid(forward(model, Xb.to(DEVICE)).squeeze(-1))
            probs_vl.extend(p.cpu().numpy().tolist())
            labs_vl.extend(yb.cpu().numpy().tolist())

    vl_probs  = np.array(probs_vl)
    vl_labels = np.array(labs_vl)

    thr      = find_best_threshold(vl_labels, vl_probs)
    vl_pred  = (vl_probs >= thr).astype(int)
    auc_val  = (roc_auc_score(vl_labels, vl_probs)
                if len(np.unique(vl_labels)) > 1 else 0.0)
    f1_macro = f1_score(vl_labels, vl_pred, average="macro",  zero_division=0)
    f1_min   = f1_score(vl_labels, vl_pred, average="binary", zero_division=0)
    acc      = accuracy_score(vl_labels, vl_pred)

    return {
        "auc":          auc_val,
        "f1_macro":     f1_macro,
        "f1_minority":  f1_min,
        "acc":          acc,
        "threshold":    thr,
        "best_epoch":   epoch + 1,
        "n_train":      len(y_tr),
        "n_val":        len(y_vl),
        "n_components": input_dim,
    }


# =============================================================================
# 4. LOOP CV COMBINADO
# =============================================================================

def run_combined_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    exp_config: dict,
    arch_name: str,
    emb_model: str,
    results_path: Path,
):
    exp_id  = exp_config["id"]
    use_pca = exp_config["pca"]
    aug_key = exp_config.get("aug", None)

    # NOTA: En el experimento combinado los datos son pooled (mean por paciente),
    # un solo vector por paciente. Las arquitecturas de secuencia (CNN1D, BiLSTM,
    # BiGRU, CNN_BiGRU) necesitan múltiples timesteps por paciente para funcionar
    # correctamente. Para correrlas en el combinado habría que cargar las
    # secuencias de ambos datasets y alinear sus max_T, lo cual complica el CV.
    # Por ahora solo corren MLP en el combinado; las secuenciales ya tienen sus
    # resultados en dl_cross_dataset_all_archs.py.
    if arch_name in SEQUENCE_ARCHS:
        logger.info(f"    ⏭️  [{arch_name}] omitido en combinado — requiere datos de secuencia. "
                    f"Ver dl_cross_dataset_all_archs.py para resultados con secuencias.")
        return None

    rows = []

    for repeat in range(N_REPEATS):
        seed = 42 + repeat
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        for fold_idx, (tr_idx, vl_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
            X_tr, X_vl = X[tr_idx].copy(), X[vl_idx].copy()
            y_tr, y_vl = y[tr_idx].copy(), y[vl_idx].copy()

            X_tr, y_tr, X_vl, n_comp = preprocess_fold_pooled(
                X_tr, y_tr, X_vl, use_pca, aug_key, seed
            )

            metrics = train_fold(arch_name, X_tr, y_tr, X_vl, y_vl, seed=seed + fold_idx)

            row = {
                "emb_model":    emb_model,
                "architecture": arch_name,
                "experiment":   exp_id,
                "repeat":       repeat + 1,
                "fold":         fold_idx,
                "n_components": n_comp,
                **metrics,
            }
            rows.append(row)

            pd.DataFrame([row]).to_csv(
                results_path, mode="a",
                header=not results_path.exists() or results_path.stat().st_size == 0,
                index=False,
            )

        rep_df = pd.DataFrame(rows[-N_SPLITS:])
        logger.info(
            f"      Rep{repeat+1} | "
            f"AUC={rep_df['auc'].mean():.3f}±{rep_df['auc'].std():.3f} | "
            f"F1={rep_df['f1_macro'].mean():.3f}"
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"    ✅ [{arch_name}|{exp_id}|{emb_model}] "
        f"AUC={df['auc'].mean():.3f}±{df['auc'].std():.3f} | "
        f"F1-macro={df['f1_macro'].mean():.3f} | "
        f"F1-min={df['f1_minority'].mean():.3f}"
    )
    return df


# =============================================================================
# 5. PIPELINE PRINCIPAL
# =============================================================================

def run_all(embeddings_daic: str, embeddings_propio: str, output_dir: str):
    output_dir = Path(output_dir) / "combined_cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = output_dir / "raw_folds.csv"
    summary_path = output_dir / "summary.csv"

    total_configs = (
        len(MODELS_TO_TEST) * len(ARCHS_TO_TEST) * len(EXPERIMENTS)
        * N_REPEATS * N_SPLITS
    )
    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 DAIC:    {embeddings_daic}")
    logger.info(f"📂 Propio:  {embeddings_propio}")
    logger.info(f"📂 Salida:  {output_dir}")
    logger.info(f"🔁 CV: {N_REPEATS} rep × {N_SPLITS} folds = {N_REPEATS*N_SPLITS} eval/config")
    logger.info(f"🧱 Arquitecturas: {ARCHS_TO_TEST}")
    logger.info(f"📊 Total entrenamientos: ~{total_configs}\n")

    t0_global = time.time()

    for emb_model in MODELS_TO_TEST:
        logger.info(f"{'='*65}")
        logger.info(f"📦 Modelo: {emb_model}")
        logger.info(f"{'='*65}")

        try:
            X, y, groups, origins = load_combined_pooled(
                embeddings_daic, embeddings_propio, emb_model
            )
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar {emb_model}: {e}")
            continue

        if len(np.unique(y)) < 2:
            logger.warning("   Solo una clase — saltando.")
            continue

        for arch_name in ARCHS_TO_TEST:
            logger.info(f"\n  🔷 Arquitectura: {arch_name}")
            for exp_config in EXPERIMENTS:
                logger.info(f"    🔬 Experimento: {exp_config['id']}")
                t0 = time.time()
                run_combined_cv(
                    X, y, groups,
                    exp_config=exp_config,
                    arch_name=arch_name,
                    emb_model=emb_model,
                    results_path=raw_path,
                )
                logger.info(f"       ⏱️  {(time.time()-t0)/60:.1f} min")

    if raw_path.exists():
        summary = summarize(
            raw_path, summary_path,
            group_cols=["emb_model", "architecture", "experiment"],
        )
        logger.info("\n" + "="*65)
        logger.info("📊 RESUMEN COMBINADO (ordenado por AUC)")
        logger.info("="*65)
        logger.info("\n" + summary.head(20).to_string(index=False))
        logger.info(f"\n📄 Summary → {summary_path}")
        logger.info(f"📄 Raw folds → {raw_path}")

    total_min = (time.time() - t0_global) / 60
    logger.info(f"\n⏱️  Tiempo total: {total_min:.1f} min ({total_min/60:.1f} h)")


# =============================================================================
# PUNTO DE ENTRADA — ajusta estas rutas
# =============================================================================

if __name__ == "__main__":
    EMBEDDINGS_DAIC = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/embeddings_daic"
    )
    EMBEDDINGS_PROPIO = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/"
        "Depresión/embeddings_v2/"
    )
    OUTPUT_DIR = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Wav2Vec/results_DL/"
    )

    run_all(EMBEDDINGS_DAIC, EMBEDDINGS_PROPIO, OUTPUT_DIR)