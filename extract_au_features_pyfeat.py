"""
extract_au_features_pyfeat.py
==============================

Igual que `extract_features.py` (MediaPipe), pero usando py-feat para extraer
Action Units (AU) reales del sistema FACS de Ekman, en vez de blendshapes de
MediaPipe. Reutiliza SIN TOCAR la lógica de inferencia de etiquetas desde
carpetas y el baseline de LazyPredict que ya tienes en `extract_features.py`
(mismo folder), para no duplicar código ni criterios entre los dos pipelines.

Genera dos salidas, igual que tu script de MediaPipe:
  1. Un CSV consolidado (1 fila por sujeto, promedio/std de cada AU + emoción
     + pose por video) — mismo formato que `dataset_facial_features.csv`,
     compatible tal cual con tu notebook de Colab (Sección 1 y 2).
  2. (Opcional, --save_sequences) Un archivo por video con la señal AU
     frame-a-frame, lista para una RNN/BiLSTM.

Por qué py-feat: es un toolbox con modelos YA ENTRENADOS (no hay que entrenar
nada) para detectar AUs (modelo XGBoost sobre HOG, da probabilidad continua
0-1 por AU, no solo presencia/ausencia binaria), emociones (ResMaskNet) y pose
cefálica, todo desde el video directamente con `.detect_video()`.

Instalación:
    pip install py-feat pyarrow

Uso típico (misma estructura de carpetas que ya usas):
    python extract_au_features_pyfeat.py \
        --input_dir "./Clasificación Final" \
        --output dataset_au_features.csv \
        --frame_skip 3 \
        --save_sequences --sequences_dir sequences_au

    # Solo correr el baseline sobre un CSV ya generado:
    python extract_au_features_pyfeat.py --run_baseline_only \
        --output dataset_au_features.csv --target target_depresion

Notas metodológicas:
------------------------------------------------------------------------------
1. Las columnas de AU que reporta py-feat (AU01, AU02, AU04, AU05, AU06, AU07,
   AU09, AU10, AU11, AU12, AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU26,
   AU28, AU43 con el modelo XGBoost por defecto) son PROBABILIDADES continuas
   de activación de cada unidad de acción, no presencia binaria. Se agregan
   con mean/std igual que hiciste con los blendshapes de MediaPipe.
2. `frame_skip` aquí se pasa directo como `skip_frames` a `detector.detect(..., data_type="video")`:
   py-feat hace su propio muestreo interno del video, no necesitas abrir el
   video con OpenCV vos mismo (a diferencia del script de MediaPipe).
3. Si un frame no tiene rostro detectado, py-feat devuelve NaN en esa fila;
   se descartan antes de agregar, igual que MediaPipe.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Reutiliza SIN duplicar: inferencia de etiquetas desde estructura de carpetas,
# merge legacy con CSV de etiquetas, y el baseline de LazyPredict. Debe correr
# en la misma carpeta que extract_features.py.
from extract_features import (
    _build_label_index,
    merge_with_labels,
    run_lazypredict_baseline,
)

logger = logging.getLogger("au_feature_extraction_pyfeat")


# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================

class AUProcessingConfig:
    def __init__(
        self,
        frame_skip: int = 3,
        au_model: str = "xgb",
        emotion_model: str = "resmasknet",
        face_model: str = "retinaface",
        batch_size: int = 5,
        face_detection_threshold: float = 0.9,
        save_sequences: bool = False,
        sequences_dir: str = "sequences_au",
        sequence_format: str = "parquet",
        device: str = "auto",
    ):
        self.frame_skip = frame_skip
        self.au_model = au_model
        self.emotion_model = emotion_model
        self.face_model = face_model
        self.batch_size = batch_size
        self.face_detection_threshold = face_detection_threshold
        self.save_sequences = save_sequences
        self.sequences_dir = sequences_dir
        self.sequence_format = sequence_format
        self.device = device


# ==============================================================================
# 2. PROCESAMIENTO DE UN VIDEO CON PY-FEAT
# ==============================================================================

def _get_detector(config: AUProcessingConfig):
    """Crea (una sola vez por proceso) el Detector de py-feat con los modelos elegidos.

    Nota: en py-feat >= 2.x la clase se llama `Detectorv1` (pipeline modular
    clásico: face + landmarks + AU + emoción por separado, que es lo que
    queremos aquí para tener control fino de cada modelo), no `Detector`
    a secas. `Detectorv2` es la alternativa (un solo modelo multitarea).

    IMPORTANTE: el default interno de py-feat para `device` es "cpu" (NO
    autodetecta GPU). Por eso acá se pasa explícitamente device="auto",
    que sí hace que py-feat elija cuda/mps si están disponibles y si no,
    cae a cpu.
    """
    from feat import Detectorv1
    return Detectorv1(
        face_model=config.face_model,
        au_model=config.au_model,
        emotion_model=config.emotion_model,
        device=config.device,
    )


def process_single_video_au(video_path: str, config: AUProcessingConfig, detector=None) -> tuple[dict, "pd.DataFrame"]:
    """
    Procesa un video con py-feat y retorna:
      - row: dict agregado (1 fila, mean/std por AU/emoción/pose) para el CSV consolidado.
      - fex_df: DataFrame frame-a-frame (la señal cruda, para --save_sequences).
    """
    video_path = Path(video_path)
    subject_id = video_path.stem
    logger.info("Procesando video (py-feat): %s (subject_id=%s)", video_path.name, subject_id)

    if detector is None:
        detector = _get_detector(config)

    try:
        fex = detector.detect(
            str(video_path),
            data_type="video",
            skip_frames=config.frame_skip,
            batch_size=config.batch_size,
            face_detection_threshold=config.face_detection_threshold,
        )
    except Exception:  # noqa: BLE001
        logger.error("Fallo procesando '%s' con py-feat: %s", video_path.name, traceback.format_exc())
        return {"video_id": subject_id, "n_frames_detected": 0, "error": "pyfeat_processing_failed"}, pd.DataFrame()

    fex_df = pd.DataFrame(fex)

    # Descarta frames sin rostro detectado (AUs todas NaN)
    au_cols = list(getattr(fex, "au_columns", [])) or [c for c in fex_df.columns if c.upper().startswith("AU")]
    emotion_cols = list(getattr(fex, "emotion_columns", [])) or [
        c for c in fex_df.columns if c.lower() in
        ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
    ]
    pose_cols = list(getattr(fex, "facepose_columns", [])) or [c for c in fex_df.columns if c.lower() in ("pitch", "roll", "yaw")]

    if au_cols:
        valid_mask = fex_df[au_cols].notna().any(axis=1)
        fex_df = fex_df[valid_mask].reset_index(drop=True)

    row: dict = {"video_id": subject_id, "n_frames_detected": len(fex_df), "error": None}

    if fex_df.empty:
        logger.warning("Video %s: py-feat no detectó rostro en ningún frame procesado.", subject_id)
        row["n_frames_detected"] = 0
        return row, fex_df

    for col in au_cols + emotion_cols + pose_cols:
        vals = pd.to_numeric(fex_df[col], errors="coerce")
        row[f"{col}_mean"] = float(np.nanmean(vals)) if vals.notna().any() else np.nan
        row[f"{col}_std"] = float(np.nanstd(vals)) if vals.notna().any() else np.nan

    return row, fex_df


def save_au_sequence(subject_id: str, fex_df: "pd.DataFrame", config: AUProcessingConfig) -> Optional[str]:
    """Guarda la señal AU frame-a-frame de un video (mismo criterio que el script de MediaPipe)."""
    if fex_df is None or fex_df.empty:
        return None
    os.makedirs(config.sequences_dir, exist_ok=True)

    keep_cols = [c for c in fex_df.columns if c.upper().startswith("AU")]
    keep_cols += [c for c in fex_df.columns if c.lower() in
                  ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral",
                   "pitch", "roll", "yaw", "frame", "approx_time")]
    seq_df = fex_df[keep_cols].copy()
    seq_df.insert(0, "video_id", subject_id)
    seq_df.insert(1, "frame_order", range(len(seq_df)))

    ext = "parquet" if config.sequence_format == "parquet" else "csv"
    out_path = os.path.join(config.sequences_dir, f"{subject_id}.{ext}")
    if config.sequence_format == "parquet":
        seq_df.to_parquet(out_path, index=False)
    else:
        seq_df.to_csv(out_path, index=False)
    return out_path


# ==============================================================================
# 3. ORQUESTACIÓN DEL BATCH (secuencial — py-feat ya usa el GPU/batching interno)
# ==============================================================================

def run_batch_au(input_dir: str, config: AUProcessingConfig) -> pd.DataFrame:
    """
    Recorre input_dir con la MISMA lógica de inferencia de etiquetas que
    extract_features.py (reutilizada por import), y corre py-feat sobre cada
    video_id único.
    """
    all_paths = sorted(p for p in Path(input_dir).rglob("*.mp4") if not p.name.startswith("._"))
    if not all_paths:
        raise FileNotFoundError(f"No se encontraron archivos .mp4 en '{input_dir}'.")

    label_index = _build_label_index(all_paths)
    logger.info(
        "Se encontraron %d archivos .mp4 (%d video_id únicos tras deduplicar) en '%s'.",
        len(all_paths), len(label_index), input_dir,
    )

    detector = _get_detector(config)  # se crea UNA vez y se reutiliza en todos los videos
    logger.info("py-feat corriendo en device='%s'.", getattr(detector, "device", "desconocido"))

    rows: list[dict] = []
    for video_id, entry in label_index.items():
        video_path = entry["canonical_path"]
        try:
            row, fex_df = process_single_video_au(str(video_path), config, detector=detector)
        except Exception:  # noqa: BLE001
            logger.error("Video '%s' falló completamente, se continúa con el siguiente: %s", video_path.name, traceback.format_exc())
            row, fex_df = {"video_id": video_id, "n_frames_detected": 0, "error": "unhandled_exception"}, pd.DataFrame()

        row["target_ansiedad"] = entry["target_ansiedad"]
        row["target_depresion"] = entry["target_depresion"]
        rows.append(row)

        if config.save_sequences:
            try:
                seq_path = save_au_sequence(video_id, fex_df, config)
                if seq_path:
                    logger.info("Secuencia AU de %s guardada en '%s' (%d frames).", video_id, seq_path, len(fex_df))
            except Exception:  # noqa: BLE001
                logger.error("No se pudo guardar la secuencia AU de %s: %s", video_id, traceback.format_exc())

    df = pd.DataFrame(rows)
    ordered_cols = ["video_id"] + sorted(c for c in df.columns if c != "video_id")
    return df[ordered_cols]


# ==============================================================================
# 4. CLI / MAIN
# ==============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extracción de Action Units (FACS) desde videos MP4 con py-feat.")
    parser.add_argument("--input_dir", type=str, default="./videos", help="Carpeta con los .mp4 (misma estructura que extract_features.py)")
    parser.add_argument("--output", type=str, default="dataset_au_features.csv", help="CSV de salida")
    parser.add_argument("--labels_csv", type=str, default=None, help="[OPCIONAL/LEGACY] igual que en extract_features.py")
    parser.add_argument("--frame_skip", type=int, default=3, help="Procesar 1 de cada N frames (se pasa como skip_frames a py-feat)")
    parser.add_argument("--au_model", type=str, default="xgb", choices=["xgb", "svm", "logistic"], help="Modelo de AU de py-feat (xgb da probabilidades continuas, recomendado)")
    parser.add_argument("--emotion_model", type=str, default="resmasknet", choices=["resmasknet", "svm", "rf"], help="Modelo de emociones de py-feat")
    parser.add_argument("--face_model", type=str, default="retinaface", help="Modelo de detección de rostro de py-feat")
    parser.add_argument("--batch_size", type=int, default=5, help="Frames por batch en py-feat (subir si hay GPU con VRAM disponible)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Dispositivo para los modelos de py-feat. 'auto' usa GPU si está disponible (cuda/mps), si no cae a CPU. OJO: el default de py-feat internamente es 'cpu', por eso este script fuerza 'auto'.")
    parser.add_argument("--save_sequences", action="store_true", help="Guarda 1 archivo por video con la señal AU frame-a-frame (para RNN/BiLSTM)")
    parser.add_argument("--sequences_dir", type=str, default="sequences_au", help="Carpeta de salida para las secuencias")
    parser.add_argument("--sequence_format", type=str, default="parquet", choices=["parquet", "csv"])
    parser.add_argument("--log_file", type=str, default="extraction_au.log")
    parser.add_argument("--run_baseline_only", action="store_true", help="Solo ejecuta el baseline LazyPredict sobre --output existente")
    parser.add_argument("--target", type=str, default="target_ansiedad", choices=["target_ansiedad", "target_depresion"])
    return parser


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(args.log_file)

    if args.run_baseline_only:
        run_lazypredict_baseline(args.output, target_col=args.target)
        return

    config = AUProcessingConfig(
        frame_skip=args.frame_skip,
        au_model=args.au_model,
        emotion_model=args.emotion_model,
        face_model=args.face_model,
        batch_size=args.batch_size,
        save_sequences=args.save_sequences,
        sequences_dir=args.sequences_dir,
        sequence_format=args.sequence_format,
        device=args.device,
    )

    start = time.time()
    features_df = run_batch_au(args.input_dir, config)
    merged_df = merge_with_labels(features_df, args.labels_csv)
    merged_df.to_csv(args.output, index=False)
    elapsed = time.time() - start

    logger.info(
        "Listo. %d videos procesados en %.1f s (%.1f s/video). CSV guardado en '%s'.",
        len(merged_df), elapsed, elapsed / max(len(merged_df), 1), args.output,
    )


if __name__ == "__main__":
    main()