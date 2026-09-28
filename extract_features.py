"""
extract_features.py
====================

Pipeline modular, robusto y optimizado para extraer características faciales
agregadas (microexpresiones, dinámica de parpadeo/mirada, pose cefálica) desde
un lote de videos MP4, orientado a un estudio de explicabilidad (XAI) sobre
detección de Ansiedad y Depresión a partir de señales faciales.

Uso típico (estructura de carpetas en el servidor, etiquetas inferidas solas):
    .../Clasificación Final/Ansiedad/mp4/0/002.mp4
    .../Clasificación Final/Ansiedad/mp4/1/004.mp4
    .../Clasificación Final/Depresion/mp4/0/002.mp4
    .../Clasificación Final/Depresion/mp4/1/004.mp4

    python extract_features.py \
        --input_dir "./Clasificación Final" \
        --output dataset_facial_features.csv \
        --frame_skip 3 \
        --delegate CPU \
        --n_jobs 4

No hace falta --labels_csv: target_ansiedad y target_depresion se infieren
directamente de la carpeta '0'/'1' bajo cada árbol Ansiedad/Depresion. Si un
mismo video_id (ej. '002') aparece físicamente duplicado en ambos árboles, se
procesa con MediaPipe UNA sola vez y se le pegan ambas etiquetas en la misma
fila. --labels_csv queda disponible solo como respaldo opcional (ver función
`merge_with_labels`).

Requisitos (requirements.txt sugerido):
    mediapipe>=0.10.14
    opencv-python>=4.9
    numpy>=1.26
    pandas>=2.2
    tqdm>=4.66
    scikit-learn>=1.4
    lazypredict>=0.2.12
    pyarrow>=14.0        # necesario solo si usas --save_sequences con --sequence_format parquet

Notas metodológicas importantes:
------------------------------------------------------------------------------
1. `jawClench` NO existe como blendshape nativo de MediaPipe (los 52
   blendshapes estilo ARKit no incluyen una categoría de "apretar la
   mandíbula"; sí incluyen jawOpen, jawForward, jawLeft, jawRight). Para no
   inventar un dato falso, este script calcula un PROXY GEOMÉTRICO de tensión
   mandibular (`jaw_tension_proxy`) a partir de landmarks del contorno
   maxilar/masetero, normalizado por el ancho interocular. Repórtalo en tu
   paper como proxy geométrico, no como blendshape ARKit.
2. El "vector de mirada" se calcula de forma geométrica usando los landmarks
   del iris (incluidos por defecto en FaceLandmarker, índices 468-477) en
   relación a las comisuras del ojo, no solo con blendshapes eyeLookX, que
   son más gruesos.
3. Con frame skipping alto (N grande) se pierde sensibilidad para detectar
   parpadeos individuales (son eventos rápidos, ~100-400ms). Si el parpadeo
   es una variable central de tu estudio, considera correr con N=1 o N=2 al
   menos para ese cálculo específico. El script deja esto configurable.
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
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ==============================================================================
# 1. CONFIGURACIÓN GLOBAL / CONSTANTES
# ==============================================================================

# Modelo oficial de Google para FaceLandmarker (incluye blendshapes + malla 478 pts)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH_DEFAULT = "face_landmarker.task"

# Blendshapes clave (reales, nativas de MediaPipe) relacionadas con afecto/estrés
KEY_BLENDSHAPES = [
    "browDownLeft",
    "browDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthSmileLeft",
    "mouthSmileRight",
]

# Blendshapes de parpadeo (nativas)
BLINK_BLENDSHAPES = ["eyeBlinkLeft", "eyeBlinkRight"]

# Blendshapes de mirada (nativas, usadas como corroboración secundaria del gaze geométrico)
GAZE_BLENDSHAPES = [
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookDownLeft", "eyeLookDownRight",
]

# Índices de landmarks (malla de 478 puntos, incluye iris) usados para proxies geométricos
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)   # (exterior, interior)
RIGHT_EYE_CORNERS = (362, 263)
# Puntos de referencia del ángulo mandibular / masetero (aprox. contorno de mejilla-mandíbula)
JAW_ANGLE_LEFT = 172
JAW_ANGLE_RIGHT = 397
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263  # usados para normalizar por ancho interocular

# Umbrales para detección de parpadeo (histéresis sobre el score de blendshape)
BLINK_THRESHOLD_HIGH = 0.5
BLINK_THRESHOLD_LOW = 0.3

# Umbral de "cabeza hacia abajo" en grados (pitch negativo = mirando hacia abajo, ver docstring)
HEAD_DOWN_PITCH_THRESHOLD_DEG = -15.0

logger = logging.getLogger("facial_feature_extraction")


# ==============================================================================
# 2. CONFIGURACIÓN DE EJECUCIÓN (dataclass para hiperparámetros)
# ==============================================================================

@dataclass
class ProcessingConfig:
    """Hiperparámetros ajustables del pipeline. Modifica aquí para tunear."""
    frame_skip: int = 3                 # procesar 1 de cada N frames (~10 FPS si video a 30FPS)
    delegate: str = "CPU"               # "CPU" o "GPU" (usa GPU en el supercomputador)
    model_path: str = MODEL_PATH_DEFAULT
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    blink_threshold_high: float = BLINK_THRESHOLD_HIGH
    blink_threshold_low: float = BLINK_THRESHOLD_LOW
    head_down_pitch_threshold_deg: float = HEAD_DOWN_PITCH_THRESHOLD_DEG
    n_jobs: int = 1                     # procesos en paralelo (usar >1 solo con delegate=CPU)
    save_sequences: bool = False        # NUEVO: además del CSV agregado, guarda 1 archivo por video con la señal frame-a-frame (para RNN/BiLSTM)
    sequences_dir: str = "sequences_mediapipe"  # carpeta donde se guardan las secuencias
    sequence_format: str = "parquet"    # "parquet" (recomendado) o "csv"


# ==============================================================================
# 3. UTILIDADES: descarga de modelo, geometría, pose cefálica
# ==============================================================================

def ensure_model_available(model_path: str) -> str:
    """Descarga el modelo .task de MediaPipe si no existe localmente."""
    if os.path.isfile(model_path):
        return model_path
    logger.info("Modelo '%s' no encontrado. Descargando desde %s ...", model_path, MODEL_URL)
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        logger.info("Modelo descargado correctamente en '%s'.", model_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"No se pudo descargar el modelo FaceLandmarker automáticamente ({exc}). "
            f"Descárgalo manualmente desde {MODEL_URL} y colócalo en '{model_path}'."
        ) from exc
    return model_path


def rotation_matrix_to_euler_angles(rot: np.ndarray) -> tuple[float, float, float]:
    """
    Convierte una matriz de rotación 3x3 a ángulos de Euler (pitch, yaw, roll) en grados.

    Convención: pitch = rotación en X (asentir sí/no vertical, negativo = mirar hacia abajo),
    yaw = rotación en Y (girar la cabeza izq/der), roll = rotación en Z (inclinar la cabeza
    lateralmente). Se usa la convención estándar cámara/rostro de MediaPipe (Z hacia el frente).
    """
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        roll = np.arctan2(rot[2, 1], rot[2, 2])
    else:
        # Gimbal lock: fallback a una descomposición degenerada pero estable
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
        roll = np.arctan2(-rot[1, 2], rot[1, 1])

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


def compute_gaze_deviation(landmarks: np.ndarray) -> float:
    """
    Calcula la magnitud promedio de desviación del iris respecto al centro del ojo,
    normalizada por el ancho interocular. Valores altos = mayor evitación de mirada
    directa (el iris se aleja del centro geométrico del ojo).

    landmarks: array (478, 3) con coordenadas normalizadas (x, y, z) de MediaPipe.
    """
    inter_ocular = np.linalg.norm(
        landmarks[LEFT_EYE_OUTER, :2] - landmarks[RIGHT_EYE_OUTER, :2]
    )
    if inter_ocular < 1e-6:
        return np.nan

    def eye_deviation(iris_idx: list[int], corners: tuple[int, int]) -> float:
        iris_center = landmarks[iris_idx, :2].mean(axis=0)
        eye_center = landmarks[list(corners), :2].mean(axis=0)
        return np.linalg.norm(iris_center - eye_center)

    left_dev = eye_deviation(LEFT_IRIS_IDX, LEFT_EYE_CORNERS)
    right_dev = eye_deviation(RIGHT_IRIS_IDX, RIGHT_EYE_CORNERS)
    return float(((left_dev + right_dev) / 2.0) / inter_ocular)


def compute_jaw_tension_proxy(landmarks: np.ndarray) -> float:
    """
    Proxy geométrico de tensión mandibular (sustituto de 'jawClench', que no existe
    como blendshape nativo). Mide la distancia entre los puntos del ángulo mandibular
    (zona del masetero), normalizada por el ancho interocular. En un apretamiento
    mandibular sostenido suele observarse abultamiento del masetero, lo que se traduce
    en variación de esta distancia relativa. Se documenta como PROXY, no como medida
    validada clínicamente.
    """
    inter_ocular = np.linalg.norm(
        landmarks[LEFT_EYE_OUTER, :2] - landmarks[RIGHT_EYE_OUTER, :2]
    )
    if inter_ocular < 1e-6:
        return np.nan
    jaw_dist = np.linalg.norm(
        landmarks[JAW_ANGLE_LEFT, :2] - landmarks[JAW_ANGLE_RIGHT, :2]
    )
    return float(jaw_dist / inter_ocular)


# ==============================================================================
# 4. EXTRACCIÓN POR FRAME (estructura intermedia)
# ==============================================================================

@dataclass
class FrameFeatures:
    """Features crudas extraídas de un único frame con rostro detectado."""
    timestamp_ms: int
    blendshapes: dict = field(default_factory=dict)
    pitch: float = np.nan
    yaw: float = np.nan
    roll: float = np.nan
    gaze_deviation: float = np.nan
    jaw_tension_proxy: float = np.nan


class FaceLandmarkerWrapper:
    """
    Envoltorio del FaceLandmarker de MediaPipe en modo VIDEO (headless), pensado
    para reutilizarse en todos los frames de un video sin recrear el modelo.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        base_options = mp_python.BaseOptions(
            model_asset_path=config.model_path,
            delegate=(
                mp_python.BaseOptions.Delegate.GPU
                if config.delegate.upper() == "GPU"
                else mp_python.BaseOptions.Delegate.CPU
            ),
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[FrameFeatures]:
        """Procesa un frame BGR (OpenCV) y retorna features o None si no hay rostro."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None  # rostro no detectado: se ignora este frame, no rompe la ejecución

        landmarks_norm = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]], dtype=np.float32
        )

        blendshapes = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                blendshapes[category.category_name] = category.score

        pitch = yaw = roll = np.nan
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
            rot = matrix[:3, :3]
            pitch, yaw, roll = rotation_matrix_to_euler_angles(rot)

        gaze_dev = compute_gaze_deviation(landmarks_norm)
        jaw_tension = compute_jaw_tension_proxy(landmarks_norm)

        return FrameFeatures(
            timestamp_ms=timestamp_ms,
            blendshapes=blendshapes,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            gaze_deviation=gaze_dev,
            jaw_tension_proxy=jaw_tension,
        )

    def close(self) -> None:
        self._landmarker.close()


# ==============================================================================
# 4bis. GUARDADO DE LA SEÑAL FRAME-A-FRAME (para RNN / BiLSTM)
# ==============================================================================

# Todas las claves de blendshape que puede reportar MediaPipe FaceLandmarker
# (52 ARKit blendshapes estándar). Se usan para tener columnas consistentes
# entre frames y entre videos, aunque algún frame puntual no las traiga todas.
ALL_BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight",
]


def frame_features_to_dataframe(subject_id: str, frame_features: list[FrameFeatures]) -> pd.DataFrame:
    """
    Convierte la lista de FrameFeatures de UN video en un DataFrame "largo":
    1 fila por frame con rostro detectado, 1 columna por variable. Esto es la
    señal cruda que necesitas para una RNN/BiLSTM (en vez del promedio de
    todo el video que usa `aggregate_video_features`).

    Columnas: video_id, frame_order (0..N-1, en orden temporal, sin huecos —
    útil como índice de secuencia para el modelo), timestamp_ms, pitch, yaw,
    roll, gaze_deviation, jaw_tension_proxy, y una columna por blendshape.
    """
    rows = []
    for order, f in enumerate(frame_features):
        row = {
            "video_id": subject_id,
            "frame_order": order,
            "timestamp_ms": f.timestamp_ms,
            "pitch": f.pitch,
            "yaw": f.yaw,
            "roll": f.roll,
            "gaze_deviation": f.gaze_deviation,
            "jaw_tension_proxy": f.jaw_tension_proxy,
        }
        for shape_name in ALL_BLENDSHAPE_NAMES:
            row[shape_name] = f.blendshapes.get(shape_name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def save_frame_sequence(
    subject_id: str, frame_features: list[FrameFeatures], config: "ProcessingConfig"
) -> Optional[str]:
    """
    Guarda la secuencia frame-a-frame de un video en disco (1 archivo por
    video_id, dentro de config.sequences_dir). Si el video no tuvo ninguna
    detección, no guarda nada y retorna None (mismo criterio que el CSV
    agregado: se deja constancia en el log, no se inventa una fila vacía).
    """
    if not frame_features:
        return None
    os.makedirs(config.sequences_dir, exist_ok=True)
    df = frame_features_to_dataframe(subject_id, frame_features)
    ext = "parquet" if config.sequence_format == "parquet" else "csv"
    out_path = os.path.join(config.sequences_dir, f"{subject_id}.{ext}")
    if config.sequence_format == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


# ==============================================================================
# 5. AGREGACIÓN A NIVEL DE VIDEO (1 fila por sujeto)
# ==============================================================================

def _detect_blink_rate_per_minute(
    blink_scores: np.ndarray,
    timestamps_ms: np.ndarray,
    threshold_high: float,
    threshold_low: float,
) -> float:
    """
    Detecta eventos de parpadeo mediante histéresis sobre el promedio de
    eyeBlinkLeft/eyeBlinkRight y devuelve la tasa por minuto, normalizada por
    la duración real cubierta por los frames analizados (no por el conteo de
    frames, para no verse afectada por el frame skipping).
    """
    if len(blink_scores) < 2:
        return np.nan

    is_closed = False
    blink_count = 0
    for score in blink_scores:
        if not is_closed and score >= threshold_high:
            is_closed = True
            blink_count += 1
        elif is_closed and score <= threshold_low:
            is_closed = False

    duration_minutes = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0 / 60.0
    if duration_minutes <= 0:
        return np.nan
    return blink_count / duration_minutes


def aggregate_video_features(
    subject_id: str, frame_features: list[FrameFeatures], config: ProcessingConfig
) -> dict:
    """Agrega la lista de FrameFeatures de un video en un único diccionario (1 fila)."""
    row: dict = {"video_id": subject_id, "n_frames_detected": len(frame_features)}

    if not frame_features:
        logger.warning("Video %s: no se detectó rostro en ningún frame procesado.", subject_id)
        return row

    timestamps = np.array([f.timestamp_ms for f in frame_features])

    # --- Blendshapes clave: mean / std ---
    for shape_name in KEY_BLENDSHAPES:
        values = np.array(
            [f.blendshapes.get(shape_name, np.nan) for f in frame_features], dtype=np.float64
        )
        row[f"{shape_name}_mean"] = np.nanmean(values) if np.any(~np.isnan(values)) else np.nan
        row[f"{shape_name}_std"] = np.nanstd(values) if np.any(~np.isnan(values)) else np.nan

    # --- Proxy geométrico de tensión mandibular (sustituto de jawClench) ---
    jaw_vals = np.array([f.jaw_tension_proxy for f in frame_features], dtype=np.float64)
    row["jaw_tension_proxy_mean"] = np.nanmean(jaw_vals)
    row["jaw_tension_proxy_std"] = np.nanstd(jaw_vals)

    # --- Parpadeo ---
    blink_scores = np.array(
        [
            np.nanmean(
                [f.blendshapes.get(b, np.nan) for b in BLINK_BLENDSHAPES]
            )
            for f in frame_features
        ],
        dtype=np.float64,
    )
    row["blink_rate_per_minute"] = _detect_blink_rate_per_minute(
        blink_scores, timestamps, config.blink_threshold_high, config.blink_threshold_low
    )

    # --- Mirada / evitación de contacto visual (geométrico, primario) ---
    gaze_vals = np.array([f.gaze_deviation for f in frame_features], dtype=np.float64)
    row["gaze_deviation_mean"] = np.nanmean(gaze_vals)
    row["gaze_deviation_std"] = np.nanstd(gaze_vals)

    # --- Blendshapes de mirada (corroboración secundaria) ---
    for shape_name in GAZE_BLENDSHAPES:
        values = np.array(
            [f.blendshapes.get(shape_name, np.nan) for f in frame_features], dtype=np.float64
        )
        row[f"{shape_name}_mean"] = np.nanmean(values) if np.any(~np.isnan(values)) else np.nan

    # --- Pose cefálica: pitch / yaw / roll ---
    pitch = np.array([f.pitch for f in frame_features], dtype=np.float64)
    yaw = np.array([f.yaw for f in frame_features], dtype=np.float64)
    roll = np.array([f.roll for f in frame_features], dtype=np.float64)

    row["head_pitch_mean"] = np.nanmean(pitch)
    row["head_pitch_std"] = np.nanstd(pitch)
    row["head_yaw_mean"] = np.nanmean(yaw)
    row["head_yaw_std"] = np.nanstd(yaw)
    row["head_roll_mean"] = np.nanmean(roll)
    row["head_roll_std"] = np.nanstd(roll)

    valid_pitch = pitch[~np.isnan(pitch)]
    if len(valid_pitch) > 0:
        row["head_down_ratio"] = float(
            np.mean(valid_pitch < config.head_down_pitch_threshold_deg)
        )
    else:
        row["head_down_ratio"] = np.nan

    # --- Inquietud / inestabilidad cefálica: std del movimiento frame-a-frame ---
    angles = np.stack([pitch, yaw, roll], axis=1)
    valid_mask = ~np.isnan(angles).any(axis=1)
    angles_valid = angles[valid_mask]
    if len(angles_valid) > 2:
        deltas = np.linalg.norm(np.diff(angles_valid, axis=0), axis=1)
        row["head_instability_std"] = float(np.std(deltas))
        row["head_instability_mean"] = float(np.mean(deltas))
    else:
        row["head_instability_std"] = np.nan
        row["head_instability_mean"] = np.nan

    return row


# ==============================================================================
# 6. PROCESAMIENTO DE UN VIDEO COMPLETO (headless)
# ==============================================================================

def extract_subject_id(video_path: Path) -> str:
    """Extrae el id de sujeto/video a partir del nombre del archivo (sin extensión)."""
    return video_path.stem


def process_single_video(video_path: str, config: ProcessingConfig) -> dict:
    """
    Procesa un único video MP4 en modo headless: abre el video, submuestrea frames
    cada `config.frame_skip`, detecta rostro/landmarks/blendshapes por frame y
    agrega todo en un único vector de características.

    Robusto a: video corrupto/no abrible, ausencia de rostro en frames puntuales,
    ausencia total de detecciones en el video.
    """
    video_path = Path(video_path)
    subject_id = extract_subject_id(video_path)
    logger.info("Procesando video: %s (subject_id=%s)", video_path.name, subject_id)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("No se pudo abrir el video '%s'. Se omite.", video_path)
        return {"video_id": subject_id, "n_frames_detected": 0, "error": "cannot_open_video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        logger.warning("FPS inválido para '%s', se asume 30.0 por defecto.", video_path.name)
        fps = 30.0

    landmarker = FaceLandmarkerWrapper(config)
    frame_features: list[FrameFeatures] = []
    frame_idx = 0
    last_timestamp_ms = -1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # fin del video

            if frame_idx % config.frame_skip == 0:
                timestamp_ms = int((frame_idx / fps) * 1000)
                # MediaPipe VIDEO mode exige timestamps estrictamente crecientes
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                try:
                    features = landmarker.process_frame(frame, timestamp_ms)
                    if features is not None:
                        frame_features.append(features)
                    # si features es None: rostro no detectado en este frame, se ignora
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Fallo puntual en frame %d de '%s': %s",
                        frame_idx, video_path.name, traceback.format_exc(),
                    )
                    # error puntual de un frame no debe tumbar el video completo

            frame_idx += 1

    except Exception:  # noqa: BLE001
        logger.error(
            "Error irrecuperable procesando '%s': %s",
            video_path.name, traceback.format_exc(),
        )
        return {"video_id": subject_id, "n_frames_detected": len(frame_features), "error": "video_processing_failed"}
    finally:
        cap.release()
        landmarker.close()

    row = aggregate_video_features(subject_id, frame_features, config)
    row["error"] = None

    if config.save_sequences:
        try:
            seq_path = save_frame_sequence(subject_id, frame_features, config)
            if seq_path:
                logger.info("Secuencia frame-a-frame de %s guardada en '%s' (%d frames).", subject_id, seq_path, len(frame_features))
        except Exception:  # noqa: BLE001
            logger.error("No se pudo guardar la secuencia de %s: %s", subject_id, traceback.format_exc())

    return row


# ==============================================================================
# 7. ORQUESTACIÓN DEL BATCH (secuencial o paralelo)
# ==============================================================================

def _worker_init(model_path: str) -> None:
    """Asegura que el modelo esté descargado antes de que cada proceso worker lo use."""
    ensure_model_available(model_path)


def _process_video_worker(args: tuple[str, ProcessingConfig]) -> dict:
    """Función de nivel de módulo (picklable) para usar con ProcessPoolExecutor."""
    video_path, config = args
    try:
        return process_single_video(video_path, config)
    except Exception:  # noqa: BLE001
        logger.error("Fallo crítico en worker para '%s': %s", video_path, traceback.format_exc())
        return {"video_id": Path(video_path).stem, "n_frames_detected": 0, "error": "worker_crashed"}


def _strip_accents(text: str) -> str:
    """Quita tildes/acentos para comparar nombres de carpeta sin depender del encoding exacto."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _infer_label_from_path(video_path: Path) -> tuple[Optional[str], Optional[int]]:
    """
    Infiere (nombre_columna_target, valor) a partir de una estructura de carpetas
    EXACTA del tipo:

        .../<Ansiedad|Depresion>/mp4/<0|1>/archivo.mp4

    Es decir: el padre inmediato debe llamarse '0' o '1', y el abuelo debe
    llamarse 'mp4', y el bisabuelo debe contener 'ansiedad' o 'depresion'
    (sin distinguir tildes/mayúsculas). Si la ruta no calza exactamente con
    ese patrón, retorna (None, None) en vez de adivinar, para no asignar
    etiquetas incorrectas por accidente.
    """
    label_dir = video_path.parent            # carpeta '0' o '1'
    mp4_dir = label_dir.parent                # carpeta 'mp4'
    task_dir = mp4_dir.parent                 # carpeta 'Ansiedad' o 'Depresion'

    if label_dir.name not in ("0", "1"):
        return None, None
    if mp4_dir.name.lower() != "mp4":
        return None, None

    task_name = _strip_accents(task_dir.name).lower()
    if "ansiedad" in task_name:
        return "target_ansiedad", int(label_dir.name)
    if "depresion" in task_name:
        return "target_depresion", int(label_dir.name)
    return None, None


def _build_label_index(video_paths: list[Path]) -> dict[str, dict]:
    """
    Recorre TODAS las rutas encontradas (incluyendo duplicados del mismo video_id
    bajo el árbol de Ansiedad y bajo el de Depresión) y construye un índice:

        { video_id: {"target_ansiedad": 0/1/None, "target_depresion": 0/1/None,
                      "canonical_path": Path} }

    `canonical_path` es la primera ruta física encontrada para ese video_id; se
    usa para procesar el video UNA sola vez con MediaPipe aunque el archivo esté
    duplicado en ambos árboles de carpetas (Ansiedad/... y Depresion/...).
    Si un mismo video_id trae valores contradictorios para la misma tarea
    (ej. aparece como 0 en un lado y como 1 en otro), se registra un warning y
    se conserva el primero encontrado, dejando rastro en el log para que lo
    revises a mano (es una señal de inconsistencia en el dataset, no algo que
    el script deba decidir silenciosamente).
    """
    index: dict[str, dict] = {}
    for video_path in video_paths:
        video_id = video_path.stem
        entry = index.setdefault(
            video_id,
            {"target_ansiedad": None, "target_depresion": None, "canonical_path": video_path},
        )
        target_col, value = _infer_label_from_path(video_path)
        if target_col is None:
            continue
        existing = entry[target_col]
        if existing is not None and existing != value:
            logger.warning(
                "Inconsistencia de etiquetas para video_id=%s en '%s': ya tenía %s=%s, "
                "se encontró también %s en '%s'. Se conserva el primer valor (%s).",
                video_id, target_col, target_col, existing, value, video_path, existing,
            )
            continue
        entry[target_col] = value
    return index


def run_batch(input_dir: str, config: ProcessingConfig) -> pd.DataFrame:
    """
    Recorre `input_dir` recursivamente, infiere las etiquetas de Ansiedad/Depresión
    a partir de la estructura de carpetas (ver `_infer_label_from_path`), procesa
    cada video_id ÚNICO una sola vez con MediaPipe (evitando reprocesar duplicados
    físicos que existan tanto en el árbol de Ansiedad como en el de Depresión) y
    retorna un único DataFrame con 1 fila por sujeto, incluyendo ambos targets.
    """
    all_paths = sorted(p for p in Path(input_dir).rglob("*.mp4") if not p.name.startswith("._"))
    if not all_paths:
        raise FileNotFoundError(f"No se encontraron archivos .mp4 en '{input_dir}'.")

    label_index = _build_label_index(all_paths)
    n_unique = len(label_index)
    logger.info(
        "Se encontraron %d archivos .mp4 (%d video_id únicos tras deduplicar) en '%s'.",
        len(all_paths), n_unique, input_dir,
    )

    n_with_ansiedad = sum(1 for e in label_index.values() if e["target_ansiedad"] is not None)
    n_with_depresion = sum(1 for e in label_index.values() if e["target_depresion"] is not None)
    n_without_any = sum(
        1 for e in label_index.values()
        if e["target_ansiedad"] is None and e["target_depresion"] is None
    )
    logger.info(
        "Etiquetas inferidas de carpetas: %d con target_ansiedad, %d con target_depresion, "
        "%d sin ninguna etiqueta detectada (revisa la estructura de esos videos).",
        n_with_ansiedad, n_with_depresion, n_without_any,
    )

    ensure_model_available(config.model_path)

    video_items = list(label_index.items())  # [(video_id, entry), ...]
    rows: list[dict] = []

    if config.n_jobs <= 1:
        # --- Modo secuencial (recomendado en laptop / cuando delegate=GPU) ---
        for video_id, entry in video_items:
            video_path = entry["canonical_path"]
            try:
                row = process_single_video(str(video_path), config)
            except Exception:  # noqa: BLE001
                logger.error(
                    "Video '%s' falló completamente, se continúa con el siguiente: %s",
                    video_path.name, traceback.format_exc(),
                )
                row = {"video_id": video_id, "n_frames_detected": 0, "error": "unhandled_exception"}
            row["target_ansiedad"] = entry["target_ansiedad"]
            row["target_depresion"] = entry["target_depresion"]
            rows.append(row)
    else:
        # --- Modo paralelo (recomendado en el supercomputador con delegate=CPU) ---
        logger.info("Procesando en paralelo con n_jobs=%d", config.n_jobs)
        tasks = [(str(entry["canonical_path"]), config) for _, entry in video_items]
        with ProcessPoolExecutor(
            max_workers=config.n_jobs,
            initializer=_worker_init,
            initargs=(config.model_path,),
        ) as executor:
            futures = {executor.submit(_process_video_worker, task): video_id for task, (video_id, _) in zip(tasks, video_items)}
            for future in as_completed(futures):
                video_id = futures[future]
                entry = label_index[video_id]
                try:
                    row = future.result()
                except Exception:  # noqa: BLE001
                    logger.error("Excepción no controlada procesando video_id=%s: %s", video_id, traceback.format_exc())
                    row = {"video_id": video_id, "n_frames_detected": 0, "error": "future_exception"}
                row["target_ansiedad"] = entry["target_ansiedad"]
                row["target_depresion"] = entry["target_depresion"]
                rows.append(row)

    df = pd.DataFrame(rows)
    # Orden de columnas: video_id primero, luego el resto ordenado alfabéticamente
    ordered_cols = ["video_id"] + sorted(c for c in df.columns if c != "video_id")
    return df[ordered_cols]


# ==============================================================================
# 8. MERGE CON ETIQUETAS Y EXPORTACIÓN
# ==============================================================================

def _normalize_id_for_matching(raw_id: str) -> str:
    """
    Normaliza un id para hacer match robusto entre nombre de archivo de video y
    subject_id del CSV de etiquetas, sin importar ceros a la izquierda.

    Ej: '001', '01', '1' -> '1' (si es puramente numérico).
    Si el id no es puramente numérico (contiene letras/guiones), se deja tal cual
    (en minúsculas y sin espacios) para no romper esquemas de naming no numéricos.
    """
    raw_id = str(raw_id).strip()
    if raw_id.isdigit():
        return str(int(raw_id))  # quita ceros a la izquierda: '001' -> '1'
    return raw_id.lower()


def merge_with_labels(features_df: pd.DataFrame, labels_csv: Optional[str]) -> pd.DataFrame:
    """
    [LEGACY / OPCIONAL] Rellena target_ansiedad/target_depresion desde un CSV
    externo (subject_id,target_depresion,target_ansiedad), por subject_id.

    Ya NO es necesario si tus videos están organizados en la estructura de
    carpetas .../<Ansiedad|Depresion>/mp4/<0|1>/video.mp4, porque `run_batch`
    infiere las etiquetas directamente de esa estructura. Esta función solo
    RELLENA valores que hayan quedado en None/NaN (no sobreescribe lo que ya
    vino de las carpetas), útil como respaldo si algunos videos no calzan con
    el patrón de carpetas esperado.

    El match se hace sobre una clave normalizada (sin ceros a la izquierda) para
    tolerar diferencias de formato como video '001.mp4' vs subject_id=1 en el CSV
    de etiquetas.
    """
    if not labels_csv:
        return features_df

    labels_df = pd.read_csv(labels_csv)
    labels_df["subject_id"] = labels_df["subject_id"].astype(str)
    labels_df["_match_key"] = labels_df["subject_id"].map(_normalize_id_for_matching)

    features_df = features_df.copy()
    features_df["video_id"] = features_df["video_id"].astype(str)
    features_df["_match_key"] = features_df["video_id"].map(_normalize_id_for_matching)

    label_cols = [c for c in ("target_depresion", "target_ansiedad") if c in labels_df.columns]
    merged = features_df.merge(
        labels_df[["_match_key", *label_cols]], on="_match_key", how="left", suffixes=("", "_csv")
    )
    for col in label_cols:
        csv_col = f"{col}_csv"
        if csv_col in merged.columns:
            # solo rellena donde el valor original (de carpetas) es NaN/None
            merged[col] = merged[col].where(merged[col].notna(), merged[csv_col])
            merged = merged.drop(columns=[csv_col])
        elif col not in merged.columns:
            pass  # el CSV no traía esa columna, no hay nada que rellenar

    still_missing = merged[label_cols].isna().all(axis=1).sum() if label_cols else 0
    if still_missing > 0:
        unmatched = merged.loc[merged[label_cols].isna().all(axis=1), "video_id"].tolist()
        logger.warning(
            "%d videos siguen sin ninguna etiqueta tras combinar carpetas + CSV: %s. "
            "Revisa manualmente esos ids en '%s'.",
            still_missing, unmatched, labels_csv,
        )
    merged = merged.drop(columns=[c for c in ("_match_key",) if c in merged.columns])
    return merged


# ==============================================================================
# 9. EVALUACIÓN BASELINE CON LAZYPREDICT + RepeatedStratifiedKFold
# ==============================================================================

def run_lazypredict_baseline(
    csv_path: str,
    target_col: str = "target_ansiedad",
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carga el CSV de features generado, y evalúa un baseline de clasificación
    con LazyPredict dentro de un esquema RepeatedStratifiedKFold (n_splits x
    n_repeats), agregando el desempeño promedio por modelo a través de todos
    los folds. Si `output_csv` se especifica, exporta la tabla resumen (mean/std
    por modelo y métrica) a un archivo CSV plano, además de mostrarla en el log.

    Requiere: pip install lazypredict scikit-learn

    Uso:
        python extract_features.py --run_baseline_only \
            --output dataset_facial_features.csv --target target_ansiedad \
            --baseline_output baseline_ansiedad.csv
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from lazypredict.Supervised import LazyClassifier

    df = pd.read_csv(csv_path)

    non_feature_cols = {
        "video_id", "error", "n_frames_detected",
        "target_depresion", "target_ansiedad",
    }
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True))
    y = df_clean[target_col].astype(int)

    # --- Validación del tamaño de clase antes de armar los folds ---
    class_counts = y.value_counts()
    logger.info("Distribución de clases para '%s': %s", target_col, class_counts.to_dict())
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0

    if len(class_counts) < 2:
        raise ValueError(
            f"'{target_col}' solo tiene una clase presente ({class_counts.to_dict()}); "
            "no se puede entrenar un clasificador binario. Revisa el CSV y el filtro de NaN."
        )
    if min_class_count < n_splits:
        adjusted_splits = max(2, min_class_count)
        logger.warning(
            "La clase minoritaria de '%s' tiene solo %d muestras, menor que n_splits=%d. "
            "Se ajusta automáticamente n_splits=%d para poder estratificar. "
            "Con tan pocas muestras, interpreta los resultados con cautela.",
            target_col, min_class_count, n_splits, adjusted_splits,
        )
        n_splits = adjusted_splits

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    all_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        try:
            models_scores, _ = clf.fit(X_train, X_test, y_train, y_test)
        except Exception:  # noqa: BLE001
            logger.warning("Fold %d falló en LazyClassifier: %s", fold_idx, traceback.format_exc())
            continue

        models_scores = models_scores.reset_index().rename(columns={"index": "Model"})
        models_scores["fold"] = fold_idx
        all_results.append(models_scores)

    if not all_results:
        raise RuntimeError("Ningún fold pudo completarse con LazyClassifier.")

    combined = pd.concat(all_results, ignore_index=True)
    metric_cols = [c for c in combined.columns if c not in ("Model", "fold")]
    summary = (
        combined.groupby("Model")[metric_cols]
        .agg(["mean", "std"])
        .sort_values(("Accuracy", "mean"), ascending=False)
    )
    logger.info("Resumen baseline (%s, %d folds efectivos):\n%s", target_col, len(all_results), summary)

    if output_csv:
        # Aplana las columnas multi-nivel (Accuracy/mean, Accuracy/std, ...) a nombres planos
        flat_summary = summary.copy()
        flat_summary.columns = [f"{metric}_{stat}" for metric, stat in flat_summary.columns]
        flat_summary = flat_summary.reset_index()
        flat_summary.insert(1, "target", target_col)
        flat_summary.insert(2, "n_folds", len(all_results))
        flat_summary.to_csv(output_csv, index=False)
        logger.info("Resumen del baseline exportado a '%s'.", output_csv)

    return summary


# ==============================================================================
# 10. CLI / MAIN
# ==============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extracción de características faciales agregadas desde videos MP4 (XAI Ansiedad/Depresión)."
    )
    parser.add_argument("--input_dir", type=str, default="./videos", help="Carpeta con los .mp4")
    parser.add_argument("--output", type=str, default="dataset_facial_features.csv", help="CSV de salida")
    parser.add_argument(
        "--labels_csv", type=str, default=None,
        help=(
            "[OPCIONAL/LEGACY] CSV con subject_id,target_depresion,target_ansiedad. "
            "No es necesario si input_dir sigue la estructura "
            ".../<Ansiedad|Depresion>/mp4/<0|1>/video.mp4 (las etiquetas se infieren solas). "
            "Si se pasa, solo rellena targets que hayan quedado sin detectar."
        ),
    )
    parser.add_argument("--frame_skip", type=int, default=3, help="Procesar 1 de cada N frames")
    parser.add_argument("--delegate", type=str, default="CPU", choices=["CPU", "GPU"], help="CPU (laptop) o GPU (supercomputador)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH_DEFAULT, help="Ruta al .task de FaceLandmarker")
    parser.add_argument("--n_jobs", type=int, default=1, help="Procesos en paralelo (usar solo con --delegate CPU)")
    parser.add_argument("--save_sequences", action="store_true", help="Además del CSV agregado, guarda 1 archivo por video con la señal frame-a-frame (para RNN/BiLSTM)")
    parser.add_argument("--sequences_dir", type=str, default="sequences_mediapipe", help="Carpeta de salida para las secuencias frame-a-frame")
    parser.add_argument("--sequence_format", type=str, default="parquet", choices=["parquet", "csv"], help="Formato de las secuencias (parquet es más liviano y rápido de cargar en Colab)")
    parser.add_argument("--log_file", type=str, default="extraction.log", help="Archivo de log")
    parser.add_argument("--run_baseline_only", action="store_true", help="Solo ejecuta el baseline LazyPredict sobre --output existente")
    parser.add_argument("--target", type=str, default="target_ansiedad", choices=["target_ansiedad", "target_depresion"], help="Target para el baseline")
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

    config = ProcessingConfig(
        frame_skip=args.frame_skip,
        delegate=args.delegate,
        model_path=args.model_path,
        n_jobs=args.n_jobs,
        save_sequences=args.save_sequences,
        sequences_dir=args.sequences_dir,
        sequence_format=args.sequence_format,
    )

    start = time.time()
    features_df = run_batch(args.input_dir, config)
    merged_df = merge_with_labels(features_df, args.labels_csv)
    merged_df.to_csv(args.output, index=False)
    elapsed = time.time() - start

    logger.info(
        "Listo. %d videos procesados en %.1f s (%.1f s/video). CSV guardado en '%s'.",
        len(merged_df), elapsed, elapsed / max(len(merged_df), 1), args.output,
    )


if __name__ == "__main__":
    main()
