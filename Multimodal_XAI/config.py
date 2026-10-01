"""Configuración del pipeline multimodal con explicabilidad por muestra (ansiedad y depresión).

Los datos por participante NO están en este repositorio (dato clínico): se leen del repo
hermano `../Multimodal` (o de la ruta en la variable de entorno MULTIMODAL_DATA).
Las salidas van a `stacking_output/`, que está en el .gitignore: nunca subirlas.
"""
import os
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MULTIMODAL_DATA", AQUI.parents[1] / "Multimodal"))

AU_CSV = DATA / "dataset_au_features.csv"                     # rostro: AU/pose/emoción (py-feat)
VT_CSV = DATA / "features" / "video_features_v2.csv"          # rostro: 256 temporales (MediaPipe)
EGEMAPS = {dx: DATA / f"features_{dx}_egemaps.csv" for dx in ("ansiedad", "depresion")}  # voz, audio editado
R_NPZ = DATA / "features" / "emb_replica_psiquiatria.npz"     # voz: micro-ventanas 0.19 s cada 2.5 s
RD_DIR = DATA / "features" / "emb_microdensas"                 # voz: micro-ventanas densas
TXT_NPZ = DATA / "features" / "texto_features.npz"            # texto: embeddings + léxico (Whisper)

OUT = AQUI / "stacking_output"                                 # gitignored
DXS = ("depresion", "ansiedad")
SEED = 2026
N_SPLITS = 16            # 80 participantes -> entrena con 75, prueba con 5
N_INNER = 5              # CV interna (elección de configuración) dentro de los 75
BG_SHAP = 50             # filas de fondo para SHAP (muestra del train del fold)
LIME_SAMPLES = 1000
