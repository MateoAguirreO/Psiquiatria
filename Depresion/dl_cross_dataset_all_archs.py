"""
dl_cross_dataset_all_archs.py
================================
Replica el experimento cross-dataset (DAIC ↔ propio) que ya corriste con
MLP, pero ahora con TODAS las arquitecturas: MLP, CNN1D, BiLSTM, BiGRU,
CNN_BiGRU. Usa cross_dataset_common.py para no repetir código.

Sin CV — split fijo por dataset de origen (ver common para el detalle
del protocolo anti-leakage).

Uso:
  Ajusta las rutas al final del archivo y ejecuta:
  python dl_cross_dataset_all_archs.py
"""

import logging
import time
from pathlib import Path

from cross_dataset_common import (
    DEVICE,
    ARCHITECTURES,
    SEQUENCE_ARCHS,
    load_pooled_embeddings,
    load_sequence_embeddings,
    run_cross_dataset_config,
    summarize,
    logger,
)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

MODELS_TO_TEST = [
    "xlsr-300m",
    "wav2vec2-large-robust",
]

# Arquitecturas a correr (comenta las que no quieras)
ARCHS_TO_TEST = [
    "MLP",
    "CNN1D",
    "BiLSTM",
    "BiGRU",
    "CNN_BiGRU",
]

EXPERIMENTS = [
    {"id": "raw",       "pca": False, "aug": None},
    {"id": "pca",       "pca": True,  "aug": None},
    {"id": "pca_smote", "pca": True,  "aug": "smote"},  # smote solo aplica a MLP (pooled)
]

N_RUNS = 10


def run_all(embeddings_daic: str, embeddings_propio: str, output_dir: str):
    output_dir = Path(output_dir) / "cross_dataset_all_archs"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = output_dir / "raw_runs.csv"
    summary_path = output_dir / "summary.csv"

    logger.info(f"🖥️  Dispositivo: {DEVICE}")
    logger.info(f"📂 DAIC:    {embeddings_daic}")
    logger.info(f"📂 Propio:  {embeddings_propio}")
    logger.info(f"📂 Salida:  {output_dir}")
    logger.info(f"🧱 Arquitecturas: {ARCHS_TO_TEST}")
    logger.info(f"🔁 Runs por config: {N_RUNS}\n")

    t0_global = time.time()

    for emb_model in MODELS_TO_TEST:
        logger.info(f"{'='*65}")
        logger.info(f" Modelo de embeddings: {emb_model}")
        logger.info(f"{'='*65}")

        # ── Cargar ambas representaciones (pooled y sequence) una sola vez ──
        try:
            logger.info("  Cargando DAIC (pooled)...")
            Xp_daic, yp_daic, _ = load_pooled_embeddings(embeddings_daic, emb_model)
            logger.info("  Cargando DAIC (sequence)...")
            Xs_daic, ys_daic, _, l_daic = load_sequence_embeddings(embeddings_daic, emb_model)
        except Exception as e:
            logger.warning(f"  No se pudo cargar DAIC/{emb_model}: {e}")
            continue

        try:
            logger.info("  Cargando Propio (pooled)...")
            Xp_propio, yp_propio, _ = load_pooled_embeddings(embeddings_propio, emb_model)
            logger.info("  Cargando Propio (sequence)...")
            Xs_propio, ys_propio, _, l_propio = load_sequence_embeddings(embeddings_propio, emb_model)
        except Exception as e:
            logger.warning(f"  No se pudo cargar Propio/{emb_model}: {e}")
            continue

        for arch_name in ARCHS_TO_TEST:
            is_seq = arch_name in SEQUENCE_ARCHS
            logger.info(f"\n   Arquitectura: {arch_name} ({'sequence' if is_seq else 'pooled'})")

            for exp_config in EXPERIMENTS:
                exp_id = exp_config["id"]

                # SMOTE solo tiene sentido para datos pooled (vector fijo)
                if exp_config.get("aug") == "smote" and is_seq:
                    logger.info(f"    ⏭  [{exp_id}] omitido — SMOTE no aplica a secuencias")
                    continue

                if is_seq:
                    X_d, y_d, l_d = Xs_daic, ys_daic, l_daic
                    X_p, y_p, l_p = Xs_propio, ys_propio, l_propio
                else:
                    X_d, y_d, l_d = Xp_daic, yp_daic, None
                    X_p, y_p, l_p = Xp_propio, yp_propio, None

                # ── DAIC → Propio ────────────────────────────────────────
                logger.info(f"    🔬 [{exp_id}] DAIC (train) → Propio (test)")
                t0 = time.time()
                run_cross_dataset_config(
                    arch_name=arch_name,
                    X_source=X_d, y_source=y_d,
                    X_target=X_p, y_target=y_p,
                    lengths_source=l_d, lengths_target=l_p,
                    exp_config=exp_config, emb_model=emb_model,
                    direction="daic→propio", results_path=raw_path,
                    n_runs=N_RUNS,
                )
                logger.info(f"       ⏱️  {(time.time()-t0)/60:.1f} min")

                # ── Propio → DAIC ────────────────────────────────────────
                logger.info(f"    🔬 [{exp_id}] Propio (train) → DAIC (test)")
                t0 = time.time()
                run_cross_dataset_config(
                    arch_name=arch_name,
                    X_source=X_p, y_source=y_p,
                    X_target=X_d, y_target=y_d,
                    lengths_source=l_p, lengths_target=l_d,
                    exp_config=exp_config, emb_model=emb_model,
                    direction="propio→daic", results_path=raw_path,
                    n_runs=N_RUNS,
                )
                logger.info(f"       ⏱️  {(time.time()-t0)/60:.1f} min")

    if raw_path.exists():
        summary = summarize(
            raw_path, summary_path,
            group_cols=["direction", "emb_model", "architecture", "experiment"],
        )
        logger.info("\n" + "="*65)
        logger.info("📊 RESUMEN CROSS-DATASET — TODAS LAS ARQUITECTURAS")
        logger.info("="*65)
        logger.info("\n" + summary.to_string(index=False))
        logger.info(f"\n📄 Summary → {summary_path}")
        logger.info(f"📄 Raw runs → {raw_path}")

    total_min = (time.time() - t0_global) / 60
    logger.info(f"\n⏱️  Tiempo total: {total_min:.1f} min ({total_min/60:.1f} h)")


if __name__ == "__main__":
    EMBEDDINGS_DAIC = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/embeddings_daic"
    )
    EMBEDDINGS_PROPIO = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/"
        "Depresión/embeddings_v2/"
    )
    OUTPUT_DIR = (
        "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Wav2Vec/results_DL/nuevos_embeddings"
    )

    run_all(EMBEDDINGS_DAIC, EMBEDDINGS_PROPIO, OUTPUT_DIR)