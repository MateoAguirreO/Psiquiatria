"""
==============================================================================
build_stacking_oof_matrix.py
==============================================================================
Genera la matriz Out-Of-Fold (OOF) para fusión multimodal (stacking) de
4 modelos de clasificación de depresión, sobre EXACTAMENTE el mismo conjunto
de pacientes y los mismos folds (Paso 0 y Paso 1, ya resueltos).

Modelos (columnas de la matriz final):
  1. prob_extratrees_embeddings   -> ExtraTreesClassifier sobre embeddings de
                                      audio (wav2vec2-large-robust), agregados
                                      a nivel de paciente.
  2. prob_bilstm_embeddings       -> BiLSTM sobre SECUENCIAS de embeddings de
                                      audio (wav2vec2-large-robust).
  3. prob_destilacion_estudiante  -> Teacher BiLSTM (misma arquitectura que #2)
                                      -> Student AdaBoostRegressor sobre
                                      características acústicas eGeMAPS
                                      (dataset tabular), agregado a nivel de
                                      paciente.
  4. prob_bigru_video             -> BiGRU sobre secuencias de Action Units
                                      (sequences_au/*.parquet).

──────────────────────────────────────────────────────────────────────────
REGLA NO NEGOCIABLE
──────────────────────────────────────────────────────────────────────────
Los 4 modelos leen folds_manifest.csv y usan EXACTAMENTE esos train/val por
(repeat, fold). Ninguno genera folds propios con RepeatedStratifiedKFold
internamente -- eso quedó reservado para folds_manifest.csv (Paso 0/1, ya
generado con RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
estratificado por target_depresion sobre los 79 pacientes).

──────────────────────────────────────────────────────────────────────────
DECISIONES TOMADAS A PARTIR DE TUS ARCHIVOS (documentadas, revísalas)
──────────────────────────────────────────────────────────────────────────
· ExtraTrees:  hiperparámetros tomados literalmente de tu celda de
  "Evaluación Final de ExtraTreesClassifier optimizado":
      class_weight=None, criterion='gini', max_depth=None,
      max_features='log2', min_samples_leaf=1, min_samples_split=2,
      n_estimators=200, random_state=42. Sin SMOTE, sin escalado (igual
      que tu script).

· BiLSTM:      mejor configuración según
  Copia_de_summary_final_BiLSTM_con_val-loss_embeddings.csv, filtrando
  emb_model == 'wav2vec2-large-robust' y tomando el mayor mean_auc
  (experiment='pooling_meanmax', mean_auc=0.7659):
      hidden=128, pooling='meanmax', loss='bce_posw', mixup='fix',
      warmup=False, dropout_p=0.4, lr=1e-3, pca=False.
  Arquitectura y funciones (scale_sequences, mixup_sequences,
  BiLSTMFlexible, early stopping por val-loss, etc.) copiadas de
  dl_script_bilstm_final.py.

· Destilación: en Kd_bilstm_lazypredict.py, el Teacher usa
  EXACTAMENTE la misma arquitectura/hparams que el BiLSTM de arriba
  (_BEST_TEACHER_CFG == fila ganadora del summary). Como además debe
  entrenarse con los mismos folds/pacientes de train, el Teacher OOF del
  Paso de destilación y el modelo #2 (BiLSTM) SON EL MISMO MODELO
  entrenado bajo el mismo protocolo -> se REUTILIZA la salida OOF del
  modelo #2 como p_teacher_OOF (evita re-entrenar 2 veces lo mismo y no
  hay ningún leakage adicional: sigue siendo train-only-con-folds-de-
  manifest). Si prefieres reentrenar el Teacher de forma
  independiente, pon REUSE_BILSTM_AS_TEACHER = False más abajo.

  Student: según kd_lazypredict_summary.csv, la mejor configuración para
  'depresion' es dataset_tabular='egemaps', alpha=0.0 (y_KD = p_teacher_OOF
  puro, sin mezclar con y_real) y modelo_student='AdaBoostRegressor'
  (mean_auc=0.6484, la más alta de las 812 filas del summary para
  depresión). Se usa AdaBoostRegressor(random_state=42) con hiperparámetros
  default de sklearn (igual que LazyPredict). El student trabaja a nivel de
  SEGMENTO (audio_id, seg_idx, features...); se promedia por paciente para
  obtener 1 probabilidad por paciente (igual criterio que los otros 3
  modelos).

· BiGRU video: hiperparámetros = defaults de
  Copia_de_train_bilstm_sequences.py (hidden_size=24, dropout=0.5,
  batch_size=8, patience=8, max_epochs=60, lr=1e-3, weight_decay=1e-3),
  que son los que generaron Copia_de_resultados_au_secuencial.csv
  (bigru, auc_mean=0.7157, n_folds=25 = 5 folds x 5 repeats, exactamente
  la misma estructura que folds_manifest.csv).

──────────────────────────────────────────────────────────────────────────
!! ANTES DE CORRER: TIENES QUE VERIFICAR ESTO !!
──────────────────────────────────────────────────────────────────────────
folds_manifest.csv / pacientes_finales.csv identifican a los pacientes por
`subject_id` (entero 1..80, sin el 66). Pero cada modelo carga sus datos
con SU PROPIO identificador:
  - ExtraTrees / BiLSTM / Teacher  -> `audio_id` (desde manifest.json de
    los embeddings, ej. "002_recortado_denoised")
  - Student (tabular eGeMAPS)      -> `audio_id` (columna del CSV, ej.
    "002_recortado_denoised.wav")
  - BiGRU video                    -> `video_id` (nombre de archivo en
    sequences_au/, ej. "002" o "2")

Yo NO tengo acceso a tus archivos reales (manifest.json, features_egemaps,
sequences_au/) para inferir la convención exacta subject_id -> audio_id /
video_id. Implementé un mapeo por defecto en `subject_id_to_audio_id()` y
`subject_id_to_video_id()` que asume que el ID empieza con el subject_id
como entero con padding de ceros (ej. subject_id=2 -> "002...", subject_id=15
-> "015..."). AJUSTA esas dos funciones a tu convención real antes de correr
-- el script falla ruidosamente (assert) si el mapeo no logra encontrar
los 79 pacientes en alguna fuente, en vez de continuar silenciosamente con
menos pacientes.
==============================================================================
"""

import copy
import json
import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

print(f"[{time.strftime('%H:%M:%S')}] Iniciando imports (torch, sklearn, ...) ...", flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

print(f"[{time.strftime('%H:%M:%S')}] Imports base listos. "
      f"torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S", stream=sys.stdout, force=True)
logger = logging.getLogger("stacking_oof")


def _log(msg):
    """Log + flush inmediato. Úsalo en vez de logger.info dentro de loops
    largos para que el progreso se vea en tiempo real incluso si stdout
    está redirigido a un archivo o a `tee` (buffering por bloques)."""
    logger.info(msg)
    sys.stdout.flush()

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# ──────────────────────────────────────────────────────────────────────────
# PARALELISMO: cada (repeat, fold) es independiente (train/val propios,
# modelo/scaler nuevos), así que se pueden entrenar varios folds a la vez
# sin ningún riesgo de leakage ni de compartir estado entre ellos.
#
#   N_CPU_WORKERS -> folds en paralelo para ExtraTrees y el Student
#                     (AdaBoostRegressor), ambos 100% CPU (sklearn).
#   N_GPU_WORKERS -> folds en paralelo para BiLSTM y BiGRU (PyTorch). Si hay
#                     GPU, varios folds comparten la misma tarjeta (varios
#                     hilos Python lanzando kernels CUDA a la vez -- el driver
#                     los serializa en cómputo puro, pero se solapan la carga
#                     de datos y el resto de Python, dando speedup real). Si
#                     NO hay GPU, esto simplemente paraleliza en CPU también,
#                     mismo mecanismo.
#
# Ajusta estos 2 números a tu máquina real. Si ves out-of-memory en GPU,
# baja N_GPU_WORKERS (empieza en 2-3); si ves que la CPU se satura y todo se
# vuelve más lento, baja N_CPU_WORKERS.
# ──────────────────────────────────────────────────────────────────────────
N_CPU_WORKERS = max(1, (os.cpu_count() or 4) - 1)
N_GPU_WORKERS = max(2, N_GPUS * 4) if N_GPUS > 0 else max(1, (os.cpu_count() or 4) // 2)

# ExtraTrees usa n_jobs interno (paraleliza sus propios árboles). Si además
# paralelizamos folds por fuera con N_CPU_WORKERS, hay que repartir los
# núcleos entre ambos niveles para no sobre-suscribir la CPU.
ET_N_JOBS = max(1, (os.cpu_count() or 4) // N_CPU_WORKERS)

# Igual para los hilos internos de PyTorch cuando se entrena en CPU: si
# N_GPU_WORKERS hilos corren a la vez, cada uno no debe intentar usar TODOS
# los núcleos para sus propias operaciones tensoriales.
if DEVICE.type == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // N_GPU_WORKERS))
else:
    torch.backends.cudnn.benchmark = True

logger_cfg_msg = (
    f"Paralelismo: N_CPU_WORKERS={N_CPU_WORKERS} (ExtraTrees/Student, "
    f"ET_N_JOBS={ET_N_JOBS} por worker) | N_GPU_WORKERS={N_GPU_WORKERS} "
    f"(BiLSTM/BiGRU, device={DEVICE}, GPUs detectadas={N_GPUS})"
)


def run_tasks_parallel(tasks, worker_fn, n_workers, label):
    """Corre worker_fn(repeat, fold) para cada (repeat, fold) en `tasks`,
    en paralelo con n_workers hilos. worker_fn debe devolver una lista de
    dicts (filas); se recolectan todas sin usar una lista compartida entre
    hilos (cada hilo devuelve la suya, el hilo principal las junta), así
    que no hay condiciones de carrera al escribir resultados."""
    all_rows = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker_fn, r, f): (r, f) for r, f in tasks}
        for future in as_completed(futures):
            r, f = futures[future]
            try:
                rows = future.result()
            except Exception:
                logger.error(f"[{label}] FALLÓ repeat={r} fold={f}")
                raise
            all_rows.extend(rows)
            n_done += 1
            _log(f"  [{label}] {n_done}/{len(tasks)} folds listos "
                 f"(último: repeat={r} fold={f}, {len(rows)} pacientes en val)")
    return all_rows

# ==============================================================================
# 0. CONFIG -- ajusta rutas a tu entorno real
# ==============================================================================

PACIENTES_FINALES_CSV = "pacientes_finales.csv"
FOLDS_MANIFEST_CSV    = "folds_manifest_v2.csv"  # esquema de 3 vías: train_fit/val_early/test

# --- Audio: embeddings wav2vec2-large-robust (ExtraTrees, BiLSTM, Teacher) --
EMBEDDINGS_DIR = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Depresión/embeddings_v2/"
EMB_MODEL      = "wav2vec2-large-robust"

# Módulo con load_embeddings_for_classification (idéntico a tu celda de ExtraTrees)
print(f"[{time.strftime('%H:%M:%S')}] Importando embedding_pipeline_5s desde "
      f"/home/ci2dt2-ai/Proyectos/Psiquiatria ...", flush=True)
sys.path.append('/home/ci2dt2-ai/Proyectos/Psiquiatria')
try:
    from embedding_pipeline_5s import load_embeddings_for_classification
    print(f"[{time.strftime('%H:%M:%S')}] embedding_pipeline_5s importado OK.", flush=True)
except ImportError as e:
    print(f"[{time.strftime('%H:%M:%S')}] NO se pudo importar embedding_pipeline_5s: {e}", flush=True)
    load_embeddings_for_classification = None  # se valida más abajo, antes de usarse

# --- Audio: features tabulares eGeMAPS (Student de destilación) -----------
FEATURES_DEPRESION_EGEMAPS = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/MachineLearning/features_depresion_egemaps.csv"
PATIENT_ID_COL = "audio_id"
LABEL_COL      = "label"
DROP_COLS      = ["seg_idx"]

# --- Video: secuencias de Action Units (BiGRU) ------------------------------
SEQUENCES_AU_DIR = "/home/ci2dt2-ai/Proyectos/Psiquiatria/sequences_au"   # carpeta con *.parquet, ajusta si no está aquí

OUTPUT_DIR = Path("stacking_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REUSE_BILSTM_AS_TEACHER = True  # ver nota en el docstring de arriba

# --- Hiperparámetros ganadores (documentados arriba) ------------------------
ET_BEST_PARAMS = dict(
    class_weight=None, criterion="gini", max_depth=None, max_features="log2",
    min_samples_leaf=1, min_samples_split=2, n_estimators=200,
)

BILSTM_BEST_CFG = dict(
    hidden=128, pooling="meanmax", loss="bce_posw", mixup="fix",
    warmup=False, dropout_p=0.4, lr=1e-3, pca=False,
)
BILSTM_EPOCHS, BILSTM_PATIENCE, BILSTM_BATCH = 60, 8, 16

KD_DATASET_NAME = "egemaps"
KD_ALPHA        = 0.0   # y_KD = p_teacher_OOF puro (mejor config para depresión)
KD_STUDENT      = "AdaBoostRegressor"

BIGRU_HIDDEN, BIGRU_DROPOUT = 24, 0.5
BIGRU_BATCH, BIGRU_PATIENCE, BIGRU_MAX_EPOCHS = 8, 8, 60
BIGRU_LR, BIGRU_WD, BIGRU_MAX_SEQ_LEN = 1e-3, 1e-3, 600


def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_SEED_LOCK = threading.Lock()
# set_global_seed() toca estado global de random/numpy/torch. Con varios folds
# entrenando a la vez, dos hilos podrían pisarse la seed entre sí -- eso no
# genera ningún leakage (cada fold sigue usando solo sus propios train/val),
# pero sí rompe la reproducibilidad exacta bit-a-bit entre corridas paralelas.
# _SEED_LOCK serializa SOLO el instante de fijar la seed + crear el modelo,
# no el entrenamiento completo, así que el costo en paralelismo es mínimo.


# ==============================================================================
# 1. PASO 0/1 -- cargar pacientes y folds_manifest tal cual (NO regenerar)
# ==============================================================================

def load_manifest_and_patients():
    pacientes = pd.read_csv(PACIENTES_FINALES_CSV)
    manifest  = pd.read_csv(FOLDS_MANIFEST_CSV)

    assert set(manifest.columns) >= {"repeat", "fold", "subject_id", "split"}, \
        "folds_manifest.csv no tiene las columnas esperadas [repeat, fold, subject_id, split]"
    assert set(manifest.split.unique()) == {"train_fit", "val_early", "test"}, \
        "folds_manifest.csv debe tener el esquema de 3 vías [train_fit, val_early, test] " \
        "(manifest v2 -- si tienes el viejo con [train, val], regenéralo primero)."
    assert len(pacientes) == 79, f"Se esperaban 79 pacientes, hay {len(pacientes)}"
    assert set(manifest.subject_id.unique()) == set(pacientes.subject_id.unique()), \
        "Los subject_id de folds_manifest.csv no coinciden con pacientes_finales.csv"

    n_repeats = manifest.repeat.nunique()
    n_folds   = manifest.fold.nunique()
    for r in manifest.repeat.unique():
        test_ids = manifest[(manifest.repeat == r) & (manifest.split == "test")].subject_id
        assert test_ids.nunique() == 79 and len(test_ids) == 79, \
            f"repeat={r}: cada paciente debe aparecer exactamente 1 vez en test."
    for (r, f), g in manifest.groupby(["repeat", "fold"]):
        test_set = set(g.loc[g.split == "test", "subject_id"])
        other_set = set(g.loc[g.split != "test", "subject_id"])
        assert not (test_set & other_set), \
            f"repeat={r} fold={f}: hay sujetos de test dentro de train_fit/val_early -- fuga."

    logger.info(f"Manifiesto OK: {n_repeats} repeticiones x {n_folds} folds, 79 pacientes.")
    return pacientes, manifest, n_repeats, n_folds


def get_fold_subjects(manifest: pd.DataFrame, repeat: int, fold: int):
    """Devuelve (train_fit_ids, val_early_ids, test_ids) para (repeat, fold).
    test_ids es el único conjunto sobre el que se guarda una probabilidad OOF;
    val_early_ids es solo para early stopping (nunca se reporta su desempeño)."""
    sub = manifest[(manifest.repeat == repeat) & (manifest.fold == fold)]
    train_fit_ids = sub.loc[sub.split == "train_fit", "subject_id"].tolist()
    val_early_ids = sub.loc[sub.split == "val_early", "subject_id"].tolist()
    test_ids      = sub.loc[sub.split == "test",      "subject_id"].tolist()
    return train_fit_ids, val_early_ids, test_ids


def get_fold_subjects_combined(manifest: pd.DataFrame, repeat: int, fold: int):
    """Para modelos SIN early stopping (ExtraTrees, Student AdaBoost): no
    necesitan un val separado para elegir checkpoint, así que usan
    train_fit + val_early juntos como su único 'train', y predicen sobre
    test -- idéntico en espíritu a lo que ya hacían antes."""
    train_fit_ids, val_early_ids, test_ids = get_fold_subjects(manifest, repeat, fold)
    return train_fit_ids + val_early_ids, test_ids


# ==============================================================================
# 2. MAPEO subject_id (int) <-> audio_id / video_id (str)
#    !! AJUSTA ESTO A TU CONVENCIÓN REAL ANTES DE CORRER !!
# ==============================================================================

def subject_id_to_audio_id_prefix(subject_id: int) -> str:
    """Prefijo esperado del audio_id para este subject_id, ej. 2 -> '002'.
    Ajusta el padding (2 vs 3 dígitos) y separador si tu convención es distinta."""
    return f"{subject_id:03d}"


def subject_id_to_video_id(subject_id: int) -> str:
    """video_id esperado (nombre de archivo sin extensión) para este subject_id.
    Ajusta si tus archivos en sequences_au/ no siguen 'subject_id sin padding'."""
    return str(subject_id)


def build_audio_id_map(available_audio_ids, subject_ids):
    """Empareja cada subject_id con el audio_id real disponible que empieza
    con su prefijo. Falla ruidosamente si algún subject_id no matchea
    exactamente 1 audio_id (mejor eso que silenciar un mismatch)."""
    mapping = {}
    for sid in subject_ids:
        prefix = subject_id_to_audio_id_prefix(sid)
        matches = [a for a in available_audio_ids if str(a).startswith(prefix)]
        assert len(matches) == 1, (
            f"subject_id={sid}: se esperaba 1 audio_id con prefijo '{prefix}', "
            f"se encontraron {len(matches)}: {matches}. Ajusta "
            f"subject_id_to_audio_id_prefix()/build_audio_id_map()."
        )
        mapping[sid] = matches[0]
    return mapping


def build_video_id_map(available_video_ids, subject_ids):
    available_norm = {str(v): v for v in available_video_ids}
    mapping = {}
    for sid in subject_ids:
        vid = subject_id_to_video_id(sid)
        assert vid in available_norm, (
            f"subject_id={sid}: no se encontró video_id '{vid}' en "
            f"sequences_au/. Ajusta subject_id_to_video_id()."
        )
        mapping[sid] = available_norm[vid]
    return mapping


# ==============================================================================
# 3. MODELO 1 -- ExtraTreesClassifier sobre embeddings de audio (agregados)
# ==============================================================================

def run_extratrees_oof(pacientes: pd.DataFrame, manifest: pd.DataFrame,
                        n_repeats: int, n_folds: int) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("MODELO 1/4: ExtraTreesClassifier sobre embeddings de audio")
    logger.info("=" * 70)

    if load_embeddings_for_classification is None:
        raise ImportError(
            "No se pudo importar embedding_pipeline_5s.load_embeddings_for_classification. "
            "Ajusta sys.path / EMBEDDINGS_DIR arriba en CONFIG."
        )

    _log(f"Cargando embeddings agregados ({EMB_MODEL}) desde {EMBEDDINGS_DIR} ...")
    X_mat, y_mat, ids = load_embeddings_for_classification(
        EMBEDDINGS_DIR, EMB_MODEL, condition="depresion"
    )
    _log(f"Embeddings cargados: shape={X_mat.shape}")
    X = pd.DataFrame(X_mat)
    y = pd.Series(y_mat)
    ids = list(ids)  # audio_id por fila, mismo orden que X/y

    subject_ids = pacientes.subject_id.tolist()
    audio_map = build_audio_id_map(ids, subject_ids)          # subject_id -> audio_id
    id_to_row = {aid: i for i, aid in enumerate(ids)}          # audio_id -> fila de X

    def _fold_worker(repeat, fold):
        train_sub, test_sub = get_fold_subjects_combined(manifest, repeat, fold)
        train_rows = [id_to_row[audio_map[s]] for s in train_sub]
        test_rows = [id_to_row[audio_map[s]] for s in test_sub]

        clf = ExtraTreesClassifier(random_state=SEED, n_jobs=ET_N_JOBS, **ET_BEST_PARAMS)
        clf.fit(X.iloc[train_rows], y.iloc[train_rows])
        proba = clf.predict_proba(X.iloc[test_rows])[:, 1]

        return [{"subject_id": sid, "repeat": repeat, "fold": fold,
                  "prob_extratrees_embeddings": float(p)}
                for sid, p in zip(test_sub, proba)]

    tasks = [(r, f) for r in range(n_repeats) for f in range(n_folds)]
    rows = run_tasks_parallel(tasks, _fold_worker, N_CPU_WORKERS, "ExtraTrees")

    df = pd.DataFrame(rows)
    logger.info(f"ExtraTrees: {df.subject_id.nunique()} pacientes procesados, "
                f"{n_repeats * n_folds} (repeat, fold) evaluados.")
    return df


# ==============================================================================
# 4. MODELO 2 -- BiLSTM sobre secuencias de embeddings de audio
#    (idéntico a dl_script_bilstm_final.py, adaptado a folds del manifiesto)
# ==============================================================================

def load_sequence_embeddings(embeddings_root: str, model_key: str):
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    from collections import defaultdict
    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}'.")

    patient_data_raw = defaultdict(list)
    for entry in entries:
        pid = entry["audio_id"]
        patient_data_raw[pid].append({
            "segment_id": int(entry.get("segment_id", 0)),
            "file": entry["file"],
            "label": int(entry["class_id"]),
        })

    patient_data = {}
    for pid, segs in patient_data_raw.items():
        segs_sorted = sorted(segs, key=lambda x: x["segment_id"])
        seq = []
        for s in segs_sorted:
            with open(s["file"]) as f:
                rec = json.load(f)
            emb = rec if isinstance(rec, list) else rec["embedding"]
            seq.append(emb)
        patient_data[pid] = {"seq": seq, "label": segs[0]["label"]}

    max_T = max(len(v["seq"]) for v in patient_data.values())
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

    return np.array(X), np.array(y, dtype=np.int64), ids, lengths


def scale_sequences(X_train, X_val, lengths_train):
    B_tr, T, D = X_train.shape
    B_vl = X_val.shape[0]
    real_vecs = np.vstack([X_train[i, :lengths_train[i], :] for i in range(B_tr)])
    scaler = StandardScaler()
    scaler.fit(real_vecs)
    X_tr_s = scaler.transform(X_train.reshape(-1, D)).reshape(B_tr, T, D)
    X_vl_s = scaler.transform(X_val.reshape(-1, D)).reshape(B_vl, T, D)
    return X_tr_s, X_vl_s


def scale_sequences_multi(X_train, lengths_train, *others):
    """Igual que scale_sequences pero para N conjuntos adicionales (val_early,
    test, ...). El scaler se ajusta SOLO con X_train (train_fit) -- ni
    val_early ni test participan del fit, se les aplica el mismo transform."""
    B_tr, T, D = X_train.shape
    real_vecs = np.vstack([X_train[i, :lengths_train[i], :] for i in range(B_tr)])
    scaler = StandardScaler()
    scaler.fit(real_vecs)
    X_tr_s = scaler.transform(X_train.reshape(-1, D)).reshape(B_tr, T, D)
    others_s = tuple(
        scaler.transform(arr.reshape(-1, D)).reshape(arr.shape[0], T, D)
        for arr in others
    )
    return (X_tr_s,) + others_s


def mixup_sequences(X_train, y_train, lengths_train, alpha=0.4, seed=SEED):
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == 0)[0]
    n_syn = len(idx_neg) - len(idx_pos)
    if len(idx_pos) < 2 or n_syn <= 0:
        return X_train, y_train, lengths_train

    rng = np.random.default_rng(seed)
    X_syn, len_syn = [], []
    for _ in range(n_syn):
        i, j = rng.choice(idx_pos, size=2, replace=False)
        lam = rng.beta(alpha, alpha)
        X_syn.append(lam * X_train[i] + (1 - lam) * X_train[j])
        len_syn.append(max(lengths_train[i], lengths_train[j]))

    X_out = np.concatenate([X_train, np.array(X_syn, dtype=np.float32)], axis=0)
    y_out = np.concatenate([y_train, np.ones(n_syn, dtype=np.int64)])
    len_out = list(lengths_train) + len_syn
    return X_out, y_out, len_out


class SequenceDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.astype(np.float32))
        self.lengths = lengths

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


class BiLSTMFlexible(nn.Module):
    def __init__(self, input_dim, hidden=128, n_layers=1, pooling="last", dropout_p=0.4):
        super().__init__()
        self.pooling = pooling
        self.hidden = hidden
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=n_layers, batch_first=True,
                             bidirectional=True, dropout=(0.3 if n_layers > 1 else 0.0))
        if pooling == "last":
            feat_dim = hidden * 2
        elif pooling == "meanmax":
            feat_dim = hidden * 6
        elif pooling == "attention":
            feat_dim = hidden * 2
            self.attn = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        else:
            raise ValueError(pooling)
        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.head = nn.Linear(feat_dim, 1)

    def _mask(self, B, T, lengths, device):
        ar = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        len_t = torch.tensor(lengths, device=device).unsqueeze(1)
        return ar < len_t

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        if lengths is not None:
            lens_cpu = torch.clamp(torch.tensor(lengths), min=1).cpu()
            x_packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
            out_pack, (h_n, _) = self.lstm(x_packed)
            outputs, _ = pad_packed_sequence(out_pack, batch_first=True, total_length=T)
        else:
            outputs, (h_n, _) = self.lstm(x)
            lengths = [T] * B

        h_last = torch.cat([h_n[0], h_n[1]], dim=-1)

        if self.pooling == "last":
            feat = h_last
        elif self.pooling == "meanmax":
            mask = self._mask(B, T, lengths, x.device).unsqueeze(-1).float()
            mean_pool = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            max_pool, _ = outputs.masked_fill(mask == 0, float("-inf")).max(dim=1)
            max_pool = torch.nan_to_num(max_pool, neginf=0.0)
            feat = torch.cat([mean_pool, max_pool, h_last], dim=-1)
        elif self.pooling == "attention":
            mask = self._mask(B, T, lengths, x.device)
            scores = self.attn(outputs).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            feat = (outputs * weights).sum(dim=1)

        return self.head(self.dropout(self.norm(feat)))


def eval_loader(model, loader, criterion=None, device=DEVICE):
    model.eval()
    probs_l, labels_l = [], []
    loss_sum, n_total = 0.0, 0
    with torch.no_grad():
        for Xb, yb, lb in loader:
            Xb_d, yb_d = Xb.to(device), yb.to(device)
            logits = model(Xb_d, lb).squeeze(-1)
            if criterion is not None:
                loss_sum += criterion(logits, yb_d).item() * len(yb_d)
                n_total += len(yb_d)
            probs_l.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels_l.extend(yb.cpu().numpy().tolist())
    probs = np.array(probs_l)
    labels = np.array(labels_l)
    loss = loss_sum / max(n_total, 1) if criterion is not None else None
    return probs, labels, loss


def train_bilstm_fold(X_train, y_train, lengths_train,
                       X_valearly, y_valearly, lengths_valearly,
                       X_test, y_test, lengths_test,
                       cfg: dict, fold_seed: int):
    """Entrena 1 fold del BiLSTM. El early stopping mira val_loss SOLO sobre
    val_early (nunca se reporta su desempeño); la única probabilidad que se
    guarda como OOF es la predicción sobre test, que el modelo nunca vio ni
    en entrenamiento ni en la selección de checkpoint."""
    with _SEED_LOCK:
        set_global_seed(fold_seed)
        input_dim = X_train.shape[2]
        model = BiLSTMFlexible(input_dim, hidden=cfg["hidden"], pooling=cfg["pooling"],
                                dropout_p=cfg["dropout_p"]).to(DEVICE)

    ds_train = SequenceDataset(X_train, y_train, lengths_train)
    ds_valearly = SequenceDataset(X_valearly, y_valearly, lengths_valearly)
    ds_test = SequenceDataset(X_test, y_test, lengths_test)
    loader_tr = DataLoader(ds_train, batch_size=BILSTM_BATCH, shuffle=True,
                            drop_last=(len(ds_train) % BILSTM_BATCH == 1))
    loader_valearly = DataLoader(ds_valearly, batch_size=BILSTM_BATCH, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=BILSTM_BATCH, shuffle=False)

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BILSTM_EPOCHS)

    best_val_loss, best_state, no_improve = np.inf, None, 0
    for epoch in range(BILSTM_EPOCHS):
        model.train()
        for Xb, yb, lb in loader_tr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb, lb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # early stopping SOLO con val_early -- test nunca entra aquí
        _, _, val_loss = eval_loader(model, loader_valearly, criterion=criterion)
        if val_loss < best_val_loss:
            best_val_loss, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= BILSTM_PATIENCE:
                break
        if epoch % 10 == 0:
            _log(f"      época {epoch}/{BILSTM_EPOCHS} val_loss={val_loss:.4f} "
                 f"(mejor={best_val_loss:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    # única predicción que se guarda como OOF: sobre test
    test_probs, test_labels, _ = eval_loader(model, loader_test)
    return test_probs, test_labels


def run_bilstm_oof(pacientes: pd.DataFrame, manifest: pd.DataFrame,
                    n_repeats: int, n_folds: int) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("MODELO 2/4: BiLSTM sobre secuencias de embeddings de audio")
    logger.info("=" * 70)

    _log(f"Cargando secuencias de embeddings ({EMB_MODEL}) desde {EMBEDDINGS_DIR} ... "
         f"(lee 1 archivo JSON por segmento, puede tardar unos minutos)")
    X, y, ids, lengths = load_sequence_embeddings(EMBEDDINGS_DIR, EMB_MODEL)
    _log(f"Secuencias cargadas: X.shape={X.shape}, {len(ids)} pacientes")
    subject_ids = pacientes.subject_id.tolist()
    audio_map = build_audio_id_map(ids, subject_ids)
    id_to_row = {aid: i for i, aid in enumerate(ids)}

    def _fold_worker(repeat, fold):
        train_sub, valearly_sub, test_sub = get_fold_subjects(manifest, repeat, fold)
        train_idx = [id_to_row[audio_map[s]] for s in train_sub]
        valearly_idx = [id_to_row[audio_map[s]] for s in valearly_sub]
        test_idx = [id_to_row[audio_map[s]] for s in test_sub]

        X_tr, X_ve, X_te = X[train_idx].copy(), X[valearly_idx].copy(), X[test_idx].copy()
        y_tr, y_ve, y_te = y[train_idx].copy(), y[valearly_idx].copy(), y[test_idx].copy()
        l_tr = [lengths[i] for i in train_idx]
        l_ve = [lengths[i] for i in valearly_idx]
        l_te = [lengths[i] for i in test_idx]

        # scaler ajustado SOLO con train_fit; val_early y test solo se transforman
        X_tr, X_ve, X_te = scale_sequences_multi(X_tr, l_tr, X_ve, X_te)
        if BILSTM_BEST_CFG["mixup"] == "fix":
            X_tr, y_tr, l_tr = mixup_sequences(X_tr, y_tr, l_tr, seed=SEED + repeat * n_folds + fold)

        test_probs, test_labels = train_bilstm_fold(
            X_tr, y_tr, l_tr, X_ve, y_ve, l_ve, X_te, y_te, l_te,
            cfg=BILSTM_BEST_CFG, fold_seed=SEED + repeat * n_folds + fold,
        )
        return [{"subject_id": sid, "repeat": repeat, "fold": fold,
                  "prob_bilstm_embeddings": float(p)}
                for sid, p in zip(test_sub, test_probs)]

    tasks = [(r, f) for r in range(n_repeats) for f in range(n_folds)]
    rows = run_tasks_parallel(tasks, _fold_worker, N_GPU_WORKERS, "BiLSTM")

    df = pd.DataFrame(rows)
    logger.info(f"BiLSTM: {df.subject_id.nunique()} pacientes procesados, "
                f"{n_repeats * n_folds} (repeat, fold) evaluados.")
    return df


# ==============================================================================
# 5. MODELO 3 -- Destilación: Teacher (BiLSTM, reusa modelo 2) -> Student
#    (AdaBoostRegressor sobre eGeMAPS)
# ==============================================================================

_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")


def _normalize_audio_id(value) -> str:
    s = str(value).strip()
    low = s.lower()
    for ext in _AUDIO_EXTENSIONS:
        if low.endswith(ext):
            s = s[: -len(ext)]
            break
    return s.strip().lower()


def run_destilacion_oof(pacientes: pd.DataFrame, manifest: pd.DataFrame,
                         n_repeats: int, n_folds: int,
                         teacher_oof_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("MODELO 3/4: Destilación (Teacher BiLSTM -> Student AdaBoostRegressor, eGeMAPS)")
    logger.info("=" * 70)

    if not REUSE_BILSTM_AS_TEACHER:
        raise NotImplementedError(
            "REUSE_BILSTM_AS_TEACHER=False: re-implementa aquí un entrenamiento "
            "independiente del Teacher si no quieres reusar run_bilstm_oof(); la "
            "lógica es idéntica a train_bilstm_fold() con BILSTM_BEST_CFG."
        )
    logger.info("Teacher OOF: reutilizando la salida del modelo #2 (misma arquitectura, "
                "mismos folds -> es el mismo modelo).")
    teacher_oof = teacher_oof_df.rename(columns={"prob_bilstm_embeddings": "p_teacher_OOF"})

    df_tab = pd.read_csv(FEATURES_DEPRESION_EGEMAPS)
    assert PATIENT_ID_COL in df_tab.columns and LABEL_COL in df_tab.columns, (
        f"Se esperaban columnas '{PATIENT_ID_COL}' y '{LABEL_COL}' en "
        f"{FEATURES_DEPRESION_EGEMAPS}. Columnas reales: {list(df_tab.columns)}"
    )
    df_tab = df_tab.drop(columns=[LABEL_COL])
    df_tab["_merge_key"] = df_tab[PATIENT_ID_COL].map(_normalize_audio_id)

    # audio_id "real" (con extensión / mayúsculas del CSV) para cada subject_id,
    # usando el mismo mapeo de prefijos que en los modelos 1 y 2
    subject_ids = pacientes.subject_id.tolist()
    tab_audio_ids = df_tab[PATIENT_ID_COL].unique().tolist()
    audio_map = build_audio_id_map(tab_audio_ids, subject_ids)  # subject_id -> audio_id del CSV tabular
    subj_lookup = {aid: sid for sid, aid in audio_map.items()}

    df_tab["subject_id"] = df_tab[PATIENT_ID_COL].map(subj_lookup)
    assert df_tab["subject_id"].notna().all(), (
        "Algunos segmentos del CSV tabular no pudieron mapearse a subject_id -- "
        "revisa build_audio_id_map()."
    )

    feature_cols = [c for c in df_tab.columns
                    if c not in ({PATIENT_ID_COL, "subject_id", "_merge_key"} | set(DROP_COLS))]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_tab[c])]

    rows = []
    df_r_by_repeat = {}
    for repeat in range(n_repeats):
        oof_r = teacher_oof[teacher_oof.repeat == repeat][["subject_id", "p_teacher_OOF"]]
        df_r = df_tab.merge(oof_r, on="subject_id", how="inner")
        df_r["y_KD"] = (KD_ALPHA * pacientes.set_index("subject_id").loc[df_r.subject_id, "target_depresion"].values
                         + (1 - KD_ALPHA) * df_r["p_teacher_OOF"])
        df_r_by_repeat[repeat] = df_r  # read-only por fold worker, seguro entre hilos

    def _fold_worker(repeat, fold):
        df_r = df_r_by_repeat[repeat]
        train_sub, test_sub = get_fold_subjects_combined(manifest, repeat, fold)
        train_mask = df_r.subject_id.isin(train_sub)
        test_mask = df_r.subject_id.isin(test_sub)

        X_train = df_r.loc[train_mask, feature_cols].values
        X_test = df_r.loc[test_mask, feature_cols].values
        y_kd_train = df_r.loc[train_mask, "y_KD"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg = AdaBoostRegressor(random_state=SEED)
        reg.fit(X_train_s, y_kd_train)
        p_student_seg = np.clip(reg.predict(X_test_s), 0, 1)

        # Promedio por paciente (el CSV es por segmento; la matriz final es por paciente)
        df_test = df_r.loc[test_mask, ["subject_id"]].copy()
        df_test["p_student"] = p_student_seg
        per_patient = df_test.groupby("subject_id")["p_student"].mean()

        return [{"subject_id": sid, "repeat": repeat, "fold": fold,
                  "prob_destilacion_estudiante": float(p)}
                for sid, p in per_patient.items()]

    tasks = [(r, f) for r in range(n_repeats) for f in range(n_folds)]
    rows = run_tasks_parallel(tasks, _fold_worker, N_CPU_WORKERS, "Destilación")

    df = pd.DataFrame(rows)
    logger.info(f"Destilación: {df.subject_id.nunique()} pacientes procesados, "
                f"{n_repeats * n_folds} (repeat, fold) evaluados. "
                f"Dataset={KD_DATASET_NAME}, alpha={KD_ALPHA}, student={KD_STUDENT}.")
    return df


# ==============================================================================
# 6. MODELO 4 -- BiGRU sobre secuencias de Action Units de video
# ==============================================================================

import glob


def _normalize_video_id(vid) -> str:
    vid = str(vid).strip()
    try:
        return str(int(float(vid)))
    except (ValueError, TypeError):
        return vid


NON_FEATURE_COLS = {"video_id", "frame_order", "timestamp_ms", "frame", "approx_time", "input"}


def load_au_sequences(sequences_dir: str):
    files = sorted(glob.glob(os.path.join(sequences_dir, "*.parquet")) +
                    glob.glob(os.path.join(sequences_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No se encontraron secuencias en '{sequences_dir}'.")
    sequences = {}
    for f in files:
        vid = _normalize_video_id(os.path.splitext(os.path.basename(f))[0])
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        sequences[vid] = df
    return sequences


def list_candidate_columns(sequences: dict) -> list:
    all_df = pd.concat(sequences.values(), ignore_index=True)
    candidate_cols = [c for c in all_df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [c for c in candidate_cols if pd.to_numeric(all_df[c], errors="coerce").notna().mean() > 0]
    if not numeric_cols:
        raise ValueError("Ninguna columna numérica encontrada en sequences_au/.")
    return numeric_cols


def filter_high_nan_columns(train_concat, candidate_cols, max_nan_frac=0.5):
    keep = [c for c in candidate_cols if train_concat[c].isna().mean() <= max_nan_frac]
    if not keep:
        raise ValueError("Ninguna columna pasó el filtro de NaNs en este fold.")
    return keep


class AUSequenceDataset(Dataset):
    def __init__(self, video_ids, sequences, feature_cols, labels, max_seq_len):
        self.video_ids = video_ids
        self.sequences = sequences
        self.feature_cols = feature_cols
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        df = self.sequences[vid][self.feature_cols]
        arr = df.to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        if len(arr) > self.max_seq_len:
            idxs = np.linspace(0, len(arr) - 1, self.max_seq_len).astype(int)
            arr = arr[idxs]
        return torch.from_numpy(arr), self.labels[vid]


def au_collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, labels


class BiGRUClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=24, dropout=0.5):
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


def train_bigru_fold(model, train_loader, valearly_loader, test_loader, device,
                      max_epochs, patience, lr, weight_decay):
    """Early stopping SOLO con valearly_loader; la única predicción que se
    guarda como OOF es la de test_loader, nunca visto en entrenamiento ni
    en la selección de checkpoint."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss, best_state, epochs_no_improve = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, lengths, y in valearly_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                val_losses.append(criterion(logits, y).item())
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
    all_logits = []
    with torch.no_grad():
        for x, lengths, y in test_loader:
            x = x.to(device)
            all_logits.append(torch.sigmoid(model(x, lengths)).cpu())
    return torch.cat(all_logits).numpy()


def run_bigru_video_oof(pacientes: pd.DataFrame, manifest: pd.DataFrame,
                         n_repeats: int, n_folds: int) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("MODELO 4/4: BiGRU sobre secuencias de Action Units (video)")
    logger.info("=" * 70)

    sequences = load_au_sequences(SEQUENCES_AU_DIR)
    candidate_cols = list_candidate_columns(sequences)

    subject_ids = pacientes.subject_id.tolist()
    video_map = build_video_id_map(sequences.keys(), subject_ids)  # subject_id -> video_id real
    label_map = {video_map[sid]: int(pacientes.set_index("subject_id").loc[sid, "target_depresion"])
                 for sid in subject_ids}

    device = DEVICE

    def _fold_worker(repeat, fold):
        train_sub, valearly_sub, test_sub = get_fold_subjects(manifest, repeat, fold)
        train_ids = [video_map[s] for s in train_sub]
        valearly_ids = [video_map[s] for s in valearly_sub]
        test_ids = [video_map[s] for s in test_sub]

        # el scaler (media/std para imputar NaN y estandarizar) se ajusta
        # SOLO con train_fit -- val_early y test solo se transforman
        train_concat = pd.concat(
            [sequences[v][candidate_cols].apply(pd.to_numeric, errors="coerce") for v in train_ids],
            ignore_index=True,
        )
        feature_cols = filter_high_nan_columns(train_concat, candidate_cols)
        train_concat = train_concat[feature_cols]
        scaler = StandardScaler().fit(train_concat.fillna(train_concat.mean()))

        scaled_sequences = {}
        for v in train_ids + valearly_ids + test_ids:
            df = sequences[v][feature_cols].apply(pd.to_numeric, errors="coerce")
            df = df.fillna(train_concat.mean())
            scaled_sequences[v] = pd.DataFrame(scaler.transform(df), columns=feature_cols)

        train_ds = AUSequenceDataset(train_ids, scaled_sequences, feature_cols, label_map, BIGRU_MAX_SEQ_LEN)
        valearly_ds = AUSequenceDataset(valearly_ids, scaled_sequences, feature_cols, label_map, BIGRU_MAX_SEQ_LEN)
        test_ds = AUSequenceDataset(test_ids, scaled_sequences, feature_cols, label_map, BIGRU_MAX_SEQ_LEN)
        train_loader = DataLoader(train_ds, batch_size=BIGRU_BATCH, shuffle=True, collate_fn=au_collate_fn)
        valearly_loader = DataLoader(valearly_ds, batch_size=BIGRU_BATCH, shuffle=False, collate_fn=au_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=BIGRU_BATCH, shuffle=False, collate_fn=au_collate_fn)

        with _SEED_LOCK:  # set_global_seed toca estado global (torch/np/random);
            set_global_seed(SEED + repeat * n_folds + fold)  # se serializa solo este paso
            model = BiGRUClassifier(len(feature_cols), hidden_size=BIGRU_HIDDEN, dropout=BIGRU_DROPOUT).to(device)

        test_probs = train_bigru_fold(model, train_loader, valearly_loader, test_loader, device,
                                       max_epochs=BIGRU_MAX_EPOCHS, patience=BIGRU_PATIENCE,
                                       lr=BIGRU_LR, weight_decay=BIGRU_WD)

        return [{"subject_id": sid, "repeat": repeat, "fold": fold,
                  "prob_bigru_video": float(p)}
                for sid, vid, p in zip(test_sub, test_ids, test_probs)]

    tasks = [(r, f) for r in range(n_repeats) for f in range(n_folds)]
    rows = run_tasks_parallel(tasks, _fold_worker, N_GPU_WORKERS, "BiGRU video")

    df = pd.DataFrame(rows)
    logger.info(f"BiGRU video: {df.subject_id.nunique()} pacientes procesados, "
                f"{n_repeats * n_folds} (repeat, fold) evaluados.")
    return df


# ==============================================================================
# 7. PASO 3 -- ensamblar la matriz final
# ==============================================================================

def build_final_matrix(pacientes, et_df, bilstm_df, kd_df, bigru_df):
    detail = (
        et_df.merge(bilstm_df, on=["subject_id", "repeat", "fold"], how="outer")
             .merge(kd_df, on=["subject_id", "repeat", "fold"], how="outer")
             .merge(bigru_df, on=["subject_id", "repeat", "fold"], how="outer")
    )
    detail = detail.merge(pacientes[["subject_id", "target_depresion"]], on="subject_id", how="left")
    detail = detail.rename(columns={"target_depresion": "y_true"})

    prob_cols = ["prob_extratrees_embeddings", "prob_bilstm_embeddings",
                 "prob_destilacion_estudiante", "prob_bigru_video"]
    final = detail.groupby("subject_id")[prob_cols].mean().reset_index()
    final = final.merge(pacientes[["subject_id", "target_depresion"]], on="subject_id", how="left")
    final = final.rename(columns={"target_depresion": "y_true"})
    final = final.sort_values("subject_id").reset_index(drop=True)

    return final, detail


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    _log(logger_cfg_msg)
    pacientes, manifest, n_repeats, n_folds = load_manifest_and_patients()

    et_df = run_extratrees_oof(pacientes, manifest, n_repeats, n_folds)
    bilstm_df = run_bilstm_oof(pacientes, manifest, n_repeats, n_folds)
    kd_df = run_destilacion_oof(pacientes, manifest, n_repeats, n_folds, teacher_oof_df=bilstm_df)
    bigru_df = run_bigru_video_oof(pacientes, manifest, n_repeats, n_folds)

    final_matrix, detail = build_final_matrix(pacientes, et_df, bilstm_df, kd_df, bigru_df)

    final_path = OUTPUT_DIR / "oof_matrix_final.csv"
    detail_path = OUTPUT_DIR / "oof_matrix_detail_by_repeat.csv"
    final_matrix.to_csv(final_path, index=False)
    detail.to_csv(detail_path, index=False)

    logger.info("=" * 70)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 70)
    logger.info(f"Matriz final ({len(final_matrix)} pacientes, promedio de "
                f"{n_repeats} repeticiones OOF por modelo) -> {final_path}")
    logger.info(f"Detalle sin promediar (subject_id x repeat x modelo) -> {detail_path}")
    logger.info("\nColumnas y modelo que las generó:")
    logger.info("  prob_extratrees_embeddings  -> ExtraTreesClassifier / embeddings audio wav2vec2-large-robust "
                f"({et_df.subject_id.nunique()} pacientes, {n_repeats*n_folds} evaluaciones)")
    logger.info("  prob_bilstm_embeddings      -> BiLSTM / secuencias embeddings audio wav2vec2-large-robust "
                f"({bilstm_df.subject_id.nunique()} pacientes, {n_repeats*n_folds} evaluaciones)")
    logger.info("  prob_destilacion_estudiante -> Teacher BiLSTM -> Student AdaBoostRegressor / eGeMAPS "
                f"({kd_df.subject_id.nunique()} pacientes, {n_repeats*n_folds} evaluaciones)")
    logger.info("  prob_bigru_video            -> BiGRU / secuencias Action Units video "
                f"({bigru_df.subject_id.nunique()} pacientes, {n_repeats*n_folds} evaluaciones)")
    logger.info(f"\nPacientes en matriz final: {len(final_matrix)} / 79 esperados")
    for c in prob_cols_check(final_matrix):
        n_nan = final_matrix[c].isna().sum()
        if n_nan:
            logger.warning(f"  ⚠ {c}: {n_nan} pacientes sin probabilidad (revisar mapeo de IDs)")


def prob_cols_check(df):
    return ["prob_extratrees_embeddings", "prob_bilstm_embeddings",
            "prob_destilacion_estudiante", "prob_bigru_video"]


if __name__ == "__main__":
    main()