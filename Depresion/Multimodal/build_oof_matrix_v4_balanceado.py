"""
==============================================================================
build_oof_matrix_v4.py
==============================================================================
Genera la matriz Out-Of-Fold (OOF) para fusión multimodal (stacking) de
4 modelos de clasificación de depresión, sobre EXACTAMENTE el mismo conjunto
de pacientes y los mismos folds de folds_manifest_v2.csv (esquema de 3 vías:
train_fit / val_early / test).

Modelos (columnas de la matriz final) -- v4, SIN ML acústico puro (AUC ~0.56,
descartado por bajo desempeño) y SIN BiGRU (reemplazado por ML video en v3):
  1. prob_ml_video               -> ExtraTreesClassifier sobre features
                                     tabulares de Action Units (video), a
                                     nivel de paciente (sin segmentos).
  2. prob_bilstm_embeddings      -> BiLSTM sobre SECUENCIAS de embeddings de
                                     audio (wav2vec2-large-robust). Config
                                     "r4_dropout05_wd5e4_ensemble5" (focal
                                     loss + warmup + time-masking + ensemble
                                     de 5 seeds), confirmada por el usuario.
  3. prob_extratrees_embeddings  -> ExtraTreesClassifier sobre embeddings de
                                     audio (wav2vec2-large-robust), agregados
                                     a nivel de paciente. Sin cambios.
  4. prob_destilacion_estudiante -> Teacher (BiLSTM, REUTILIZA la salida OOF
                                     del modelo 2, no se re-entrena aparte) ->
                                     Student ExtraTreesRegressor sobre eGeMAPS,
                                     alpha=0.0 (y_KD = p_teacher_OOF puro).
                                     Config ganadora según el
                                     kd_lazypredict_summary.csv del usuario.
                                     Se mantiene porque dio mejor desempeño
                                     que el ML acústico puro sobre el mismo
                                     dataset eGeMAPS (confirmado por el
                                     usuario) -- ver nota de circularidad más
                                     abajo, sigue aplicando: esta rama y
                                     prob_bilstm_embeddings comparten la
                                     misma fuente de señal (el Teacher).

Para las 2 ramas tabulares (video, destilación), en la MISMA corrida por
fold en que se genera la predicción OOF se calcula también, sobre ese mismo
modelo y ese mismo test de fold:
  - SHAP OOF   (las 2 ramas usan shap.TreeExplainer -- video sobre el
    clasificador, destilación sobre el regresor [compute_shap_oof_regressor];
    AdaBoostRegressor NO está soportado por TreeExplainer [InvalidModelError,
    confirmado empíricamente], pero ExtraTreesRegressor SÍ, por eso el
    Student de destilación se cambió a ExtraTreesRegressor).
  - LIME OOF   SOLO para video (no para destilación -- destilación usa un
    REGRESOR, no un clasificador; LimeTabularExplainer en modo
    "classification" necesita predict_proba, que un regresor no tiene;
    además ya era el dataset más caro de explicar por ser a nivel de
    segmento, no de paciente).
Nunca se usa un explainer entrenado con datos distintos a los que generaron
la predicción de ese fold. Al final (build_explainability_matrix) se
promedia por subject_id a través de los 10 repeats, igual que con las
probabilidades (build_final_matrix), y se escriben shap_matrix.csv (2
modalidades: au__, destilacion_egemaps__) / lime_matrix.csv (1 modalidad:
au__), cada columna con su atribución promedio.

──────────────────────────────────────────────────────────────────────────
REGLA NO NEGOCIABLE (sin cambios respecto a v2/v3)
──────────────────────────────────────────────────────────────────────────
Los 4 modelos leen folds_manifest_v2.csv y usan EXACTAMENTE esos train/val
por (repeat, fold). Ninguno genera folds propios internamente. Los modelos
sin early stopping (ExtraTrees video, ExtraTrees embeddings, Student
ExtraTreesRegressor) usan train_fit+val_early combinados como su único
"train" y predicen sobre test (get_fold_subjects_combined). El BiLSTM sí usa
las 3 vías por separado (val_early solo para early stopping); el Teacher de
destilación REUTILIZA esa misma salida OOF del BiLSTM, así que hereda la
misma disciplina sin re-entrenar nada. El test nunca se usa para ajustar
nada, en ningún modelo.

──────────────────────────────────────────────────────────────────────────
CAMBIOS EN v4 respecto a v3 (documentados, revísalos)
──────────────────────────────────────────────────────────────────────────
1) SE ELIMINÓ el ML acústico puro (ExtraTreesClassifier sobre eGeMAPS
   directo, sin destilación) -- AUC ~0.56, prácticamente azar, confirmado
   por el usuario. Se eliminaron: run_extratrees_acustico_oof(),
   _init_acustico_worker(), _acustico_fold_worker(), _ACUSTICO_WORKER_STATE,
   ET_BEST_PARAMS_ACUSTICO, y toda referencia a "egemaps" (sin prefijo
   "destilacion_") en shap_matrix.csv. destilación SIGUE usando el mismo
   FEATURES_DEPRESION_EGEMAPS -- ese CSV no se eliminó, solo el modelo que
   lo consumía directamente sin pasar por el Teacher.

2) FIX DE LEAKAGE LEVE: el filtro de columnas AU con demasiado NaN
   (antes MAX_NAN_FRAC_VIDEO calculado UNA VEZ sobre los 79 pacientes
   completos, ANTES de entrar a los folds) ahora se calcula DENTRO de cada
   fold worker, usando SOLO train_fit+val_early de ESE fold -- igual
   disciplina que el resto del pipeline (nunca estadísticas de test, nunca
   estadísticas de "todos los pacientes" que incluyan al test de un fold
   dado). Qué columnas sobreviven puede variar levemente de un fold a otro;
   es el precio correcto de hacerlo bien. Ver filter_high_nan_cols_fold().

3) ET_BEST_PARAMS_VIDEO ahora incluye class_weight="balanced" (antes dict()
   vacío, default de sklearn sin ponderar clases). No es un fix de leakage,
   es una recomendación de calibración: con ~27% de prevalencia, un
   ExtraTrees sin ponderar tiende a comprimir sus probabilidades hacia la
   clase mayoritaria (ver AUCs y rangos de probabilidad discutidos con el
   usuario). Si prefieren reproducir el comportamiento exacto de v3, pongan
   ET_BEST_PARAMS_VIDEO = dict() de nuevo -- es la única línea a revertir.
   NO se tocó ET_BEST_PARAMS (embeddings): esos valores salen de una
   búsqueda de hiperparámetros ya confirmada por el usuario ("Evaluación
   Final de ExtraTreesClassifier optimizado"), incluyendo class_weight=None
   como resultado de esa búsqueda -- no de un default sin examinar. Cambiarlo
   unilateralmente pisaría una decisión ya validada empíricamente.

──────────────────────────────────────────────────────────────────────────
!! ANTES DE CORRER: TIENES QUE VERIFICAR ESTO !!
──────────────────────────────────────────────────────────────────────────
Igual que en v2/v3: folds_manifest_v2.csv / pacientes_finales.csv identifican
a los pacientes por `subject_id`. Cada fuente de datos tiene su propio
identificador (audio_id para eGeMAPS/embeddings/BiLSTM, video_id para AU).
Los mapeos subject_id_to_audio_id_prefix() y subject_id_to_video_id()
deben coincidir con tu convención real -- el script falla ruidosamente
(assert) si no logra mapear los 79 pacientes.
==============================================================================
"""

import copy
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

print(f"[{time.strftime('%H:%M:%S')}] Iniciando imports (torch, sklearn, shap, lime, ...) ...", flush=True)

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
#                     (ExtraTreesRegressor), ambos 100% CPU (sklearn).
#   N_GPU_WORKERS -> folds en paralelo para BiLSTM. Si hay GPU, varios folds
#                     comparten la misma tarjeta (varios hilos Python
#                     lanzando kernels CUDA a la vez -- el driver los
#                     serializa en cómputo puro, pero se solapan la carga de
#                     datos y el resto de Python, dando speedup real). Si NO
#                     hay GPU, esto simplemente paraleliza en CPU también,
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
    f"(BiLSTM, device={DEVICE}, GPUs detectadas={N_GPUS})"
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



def run_tabular_shap_lime_parallel(tasks, worker_fn, n_workers, label,
                                    initializer=None, initargs=()):
    """Como run_tasks_parallel, pero para las ramas tabulares (video, y
    destilación cuando trae SHAP), y con PROCESOS en vez de hilos.

    Por qué procesos y no hilos: LIME (explain_instance) es mayormente
    Python puro (muestreo, discretización, ajuste del modelo lineal local),
    no código C que libere el GIL como sí hacen sklearn.fit() o PyTorch. Con
    ThreadPoolExecutor, los N_CPU_WORKERS "paralelos" terminan compitiendo
    por el GIL y corren casi en serie -- con datasets grandes eso puede ser
    la diferencia entre minutos y horas. ProcessPoolExecutor sí da
    paralelismo real de CPU.

    worker_fn(repeat, fold) debe ser una función de MÓDULO (no una closure
    anidada -- no es picklable) que devuelve (prob_rows, shap_fold_df,
    lime_fold_df); los datos que necesita (manifest, df, feature_cols, ...)
    se le pasan UNA VEZ por worker vía `initializer`/`initargs` (se guardan
    en un dict a nivel de módulo en cada proceso hijo), en vez de volver a
    serializarlos en cada una de las 50 tareas."""
    prob_rows_all, shap_dfs, lime_dfs = [], [], []
    n_done = 0
    mp_ctx = multiprocessing.get_context("fork")  # explícito: así es como corre esto en Linux
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx,
                              initializer=initializer, initargs=initargs) as executor:
        futures = {executor.submit(worker_fn, r, f): (r, f) for r, f in tasks}
        for future in as_completed(futures):
            r, f = futures[future]
            try:
                prob_rows, shap_fold_df, lime_fold_df = future.result()
            except Exception:
                logger.error(f"[{label}] FALLÓ repeat={r} fold={f}")
                raise
            prob_rows_all.extend(prob_rows)
            shap_dfs.append(shap_fold_df)
            lime_dfs.append(lime_fold_df)
            n_done += 1
            _log(f"  [{label}] {n_done}/{len(tasks)} folds listos "
                 f"(último: repeat={r} fold={f}, {len(prob_rows)} pacientes en test)")
    shap_all = pd.concat(shap_dfs, ignore_index=True) if shap_dfs else pd.DataFrame()
    lime_all = pd.concat(lime_dfs, ignore_index=True) if lime_dfs else pd.DataFrame()
    return prob_rows_all, shap_all, lime_all
# ==============================================================================
# 0. CONFIG -- ajusta rutas a tu entorno real
# ==============================================================================

PACIENTES_FINALES_CSV = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/pacientes_finales.csv"
FOLDS_MANIFEST_CSV    = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/folds_manifest_v2.csv"  # esquema de 3 vías: train_fit/val_early/test

# --- Audio: embeddings wav2vec2-large-robust (BiLSTM, ExtraTrees embeddings) -
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

# --- Audio: features tabulares eGeMAPS -- SOLO usado por destilación ahora --
# (el ML acústico directo que también leía este CSV fue eliminado en v4,
# AUC~0.56 confirmado por el usuario; destilación SIGUE necesitando este CSV)
FEATURES_DEPRESION_EGEMAPS = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/MachineLearning/features_depresion_egemaps.csv"
PATIENT_ID_COL_ACUSTICO = "audio_id"   # 1 fila por SEGMENTO (varias filas por paciente)
LABEL_COL_ACUSTICO      = "label"      # se descarta -- el label real se toma de pacientes_finales.csv
DROP_COLS_ACUSTICO      = ["seg_idx"]  # columnas no-feature adicionales a excluir

# --- Video: features tabulares de Action Units (ExtraTrees video, v3) ------
# A diferencia de eGeMAPS, este CSV ya viene 1 fila = 1 paciente (sin segmentos).
FEATURES_VIDEO_AU     = "/home/ci2dt2-ai/Proyectos/Psiquiatria/dataset_au_features.csv"
PATIENT_ID_COL_VIDEO  = "video_id"
LABEL_COL_VIDEO       = "target_depresion"  # se descarta como feature (no se usa como y;
                                             # y sale de pacientes_finales.csv, igual que en
                                             # el resto del pipeline, para tener una sola
                                             # fuente de verdad del label)
# Otras columnas del CSV de video que NO son features AU y deben excluirse
# explícitamente además de LABEL_COL_VIDEO -- en particular 'target_ansiedad'
# (OTRO label clínico, correlacionado 0.73 con target_depresion en esta
# cohorte -- incluirlo como feature es leakage directo, no señal real de AU;
# confirmado empíricamente: con target_ansiedad el AUC del video sube de
# ~0.69 a ~0.89) y 'error' (columna de bookkeeping, vacía en el CSV -- ya
# la descarta el filtro de NaN, pero se excluye aquí también por si acaso).
DROP_COLS_VIDEO = ["target_ansiedad", "error"]

# --- Destilación (Teacher BiLSTM -> Student ExtraTreesRegressor) -----------
# Reutiliza EXACTAMENTE el mismo CSV/columnas que antes usaba el ML acústico
# (eGeMAPS, por segmento). dataset='egemaps', alpha=0.0 (y_KD = p_teacher_OOF
# puro, sin mezclar con la etiqueta real), student='ExtraTreesRegressor'
# (confirmado por el usuario -- NO AdaBoostRegressor) con hiperparámetros
# default de sklearn. REUSE_BILSTM_AS_TEACHER=True: el Teacher NO se
# re-entrena aparte -- se reutiliza tal cual la salida OOF del BiLSTM (ya con
# el fix de Ronda 4/focal aplicado), porque es la misma arquitectura
# entrenada bajo el mismo protocolo de folds; reentrenar sería el mismo
# modelo dos veces sin ganar nada.
#
# NOTA DE CIRCULARIDAD (sigue vigente en v4, ya discutida con el usuario):
# esta rama y prob_bilstm_embeddings NO son señales independientes -- el
# Teacher de esta rama ES prob_bilstm_embeddings. Si el Student aprende bien
# a imitarlo, la fusión tardía recibe la misma información dos veces con
# nombres distintos. Está bien mantenerla si la ablación con/sin destilación
# en la fusión (ver late_fusion_nested.py) muestra una ganancia real de AUC;
# si no, es candidata a salir del ensamble aunque su SHAP/LIME individual se
# vea bien.
REUSE_BILSTM_AS_TEACHER = True
KD_DATASET_NAME = "egemaps"
KD_ALPHA        = 0.0
KD_STUDENT      = "ExtraTreesRegressor"
# SHAP para esta rama: a diferencia de AdaBoostRegressor (confirmado que NO
# está soportado por shap.TreeExplainer -- InvalidModelError), ExtraTrees
# Regressor SÍ lo está (confirmado) -- se usa TreeExplainer normal, igual de
# rápido que en la otra rama ExtraTrees. No se calcula LIME para esta rama
# (regresor, no clasificador -- ver docstring del módulo).

OUTPUT_DIR = Path("stacking_output/v4_sin_acustico")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Hiperparámetros ExtraTrees -----------------------------------------
# Video (AU): antes dict() vacío (default sklearn sin ponderar). En v4 se
# agrega class_weight="balanced" -- ver punto 3 de "CAMBIOS EN v4" arriba.
# Si quieren reproducir el comportamiento EXACTO de v3, vuelvan esto a
# dict() -- es la única línea a revertir para ese propósito.
ET_BEST_PARAMS_VIDEO = dict(class_weight="balanced")
SCALE_VIDEO_ET = True

# Fracción máxima de NaN tolerada por columna en dataset_au_features.csv antes
# de descartarla por completo (mismo criterio/umbral que usaba filter_high_nan_
# columns() en el BiGRU de v2). AHORA se aplica POR FOLD (ver
# filter_high_nan_cols_fold() y _video_fold_worker) -- en v3 se calculaba una
# sola vez sobre los 79 pacientes juntos, lo cual es una fuga leve (decidía
# qué columnas sobreviven usando estadísticas que incluían al test de cada
# fold). El resto de los NaN restantes se imputa por fold con la media de
# train_fit+val_early de ESE fold (sin cambios respecto a v3).
MAX_NAN_FRAC_VIDEO = 0.5

# Embeddings (ExtraTrees, SIN CAMBIOS respecto a v2/v3): hiperparámetros
# tomados literalmente de la celda de "Evaluación Final de
# ExtraTreesClassifier optimizado" -- sin SMOTE, sin escalado, class_weight
# ya evaluado y descartado en esa búsqueda (no es un default sin examinar,
# por eso NO se toca en v4 aunque el criterio de "balancear" del punto 3 de
# arriba pudiera sugerir lo contrario).
ET_BEST_PARAMS = dict(
    class_weight=None, criterion="gini", max_depth=None, max_features="log2",
    min_samples_leaf=1, min_samples_split=2, n_estimators=200,
)

# Config "r4_dropout05_wd5e4_ensemble5" de summary_bilstm_r4_fixed.csv
# (mean_auc=0.7086, la que confirmó el usuario) -- reemplaza a la config
# anterior (bce_posw/pooling_meanmax/AUC=0.7659, que venía de un notebook
# de búsqueda con un bug de leakage en el early stopping, ya corregido en
# el script "fixed"; el pipeline en sí nunca tuvo ese bug -- separa
# val_early (checkpoint) de test (única predicción reportada) desde v2).
# n_seeds=5: se entrenan 5 modelos por fold con seeds distintas y se
# promedian sus probabilidades de test (ensemble) -- ver train_bilstm_fold.
BILSTM_BEST_CFG = dict(
    hidden=192, pooling="meanmax", loss="focal", focal_gamma=2.0,
    mixup="fix_tmask", warmup=True, dropout_p=0.5, lr=1e-3,
    weight_decay=5e-4, n_seeds=5, pca=False,
)
BILSTM_EPOCHS, BILSTM_PATIENCE, BILSTM_BATCH = 60, 8, 16
BILSTM_WARMUP_EPOCHS = 5  # solo aplica si BILSTM_BEST_CFG["warmup"] es True

# --- SHAP / LIME OOF (rama video) ----------------------------------------
# LIME es el cuello de botella de runtime: 1 explain_instance() por fila de
# test, por fold, por repeat. En video es 1 fila = 1 paciente (no por
# segmento, así que es mucho más barato que lo que hubiera sido para
# eGeMAPS). 1000 es un punto razonable; si el runtime es prohibitivo,
# bájalo -- el LIME resultante es más ruidoso por instancia, pero se
# promedia por paciente y luego por los 10 repeats, lo que atenúa el ruido.
LIME_NUM_SAMPLES = 1000


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
    """Para modelos SIN early stopping (ExtraTrees, Student ExtraTreesRegressor):
    no necesitan un val separado para elegir checkpoint, así que usan
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


def _normalize_video_id(vid) -> str:
    """Normaliza un video_id a string (ej. 2.0 / '2' / 2 -> '2'). Mismo
    criterio que usaba la rama BiGRU en v2 para los nombres de archivo de
    sequences_au/ -- aquí se aplica a la columna video_id de
    dataset_au_features.csv."""
    vid = str(vid).strip()
    try:
        return str(int(float(vid)))
    except (ValueError, TypeError):
        return vid


# ==============================================================================
# 3. MODELO -- ExtraTreesClassifier sobre embeddings de audio (agregados)
#    (SIN CAMBIOS respecto a v2/v3)
# ==============================================================================
def run_extratrees_oof(pacientes: pd.DataFrame, manifest: pd.DataFrame,
                        n_repeats: int, n_folds: int) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("MODELO 3/4: ExtraTreesClassifier sobre embeddings de audio")
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
# 4. MODELO -- BiLSTM sobre secuencias de embeddings de audio
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


def time_mask_sequences(X_fit, lengths_fit, max_masks=2, seed=SEED):
    """Enmascara 1-2 timesteps reales por secuencia (a 0), SOLO en train
    (nunca en val_early/test). Idéntico a dl_script2e_bilstm_round4_fixed.py.
    Se aplica DESPUÉS de mixup_sequences cuando mixup='fix_tmask'."""
    rng = np.random.default_rng(seed)
    X_out = X_fit.copy()
    for i in range(X_out.shape[0]):
        real_len = lengths_fit[i]
        if real_len <= 2:
            continue
        n_masks = rng.integers(1, max_masks + 1)
        mask_idx = rng.choice(real_len, size=min(n_masks, real_len - 1), replace=False)
        X_out[i, mask_idx, :] = 0.0
    return X_out


class FocalLossLogits(nn.Module):
    """Idéntico a dl_script2e_bilstm_round4_fixed.py. alpha se calcula en
    train_bilstm_fold a partir del desbalance de clases de ESTE fold
    (1 - proporción de positivos en train), igual que en el script fuente."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def make_scheduler(optimizer, use_warmup: bool, epochs=BILSTM_EPOCHS, warmup_epochs=BILSTM_WARMUP_EPOCHS):
    """Idéntico a dl_script2e_bilstm_round4_fixed.py: coseno simple si no
    hay warmup, o coseno con rampa lineal de warmup_epochs si lo hay."""
    if not use_warmup:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


def _train_bilstm_single_seed(X_train, y_train, lengths_train,
                               X_valearly, y_valearly, lengths_valearly,
                               X_test, y_test, lengths_test,
                               cfg: dict, seed: int):
    """1 corrida (1 seed) de 1 fold. El early stopping mira val_loss SOLO
    sobre val_early (nunca se reporta su desempeño); la única probabilidad
    que se devuelve es la predicción sobre test, que el modelo nunca vio ni
    en entrenamiento ni en la selección de checkpoint. train_bilstm_fold()
    llama esto cfg['n_seeds'] veces y promedia -- ver más abajo."""
    with _SEED_LOCK:
        set_global_seed(seed)
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
    if cfg["loss"] == "focal":
        # alpha igual que en dl_script2e_bilstm_round4_fixed.py: 1 - proporción
        # de positivos en train de ESTE fold (no un valor fijo global)
        alpha = 1.0 - float(n_pos) / float(n_pos + n_neg + 1e-6)
        criterion = FocalLossLogits(alpha=alpha, gamma=cfg.get("focal_gamma", 2.0))
    elif cfg["loss"] == "bce_posw":
        pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        raise ValueError(f"loss desconocida en BILSTM_BEST_CFG: {cfg['loss']!r}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = make_scheduler(optimizer, use_warmup=cfg.get("warmup", False))

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
            _log(f"      [seed={seed}] época {epoch}/{BILSTM_EPOCHS} val_loss={val_loss:.4f} "
                 f"(mejor={best_val_loss:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    # única predicción que se devuelve: sobre test
    test_probs, test_labels, _ = eval_loader(model, loader_test)
    return test_probs, test_labels


def train_bilstm_fold(X_train, y_train, lengths_train,
                       X_valearly, y_valearly, lengths_valearly,
                       X_test, y_test, lengths_test,
                       cfg: dict, fold_seed: int):
    """Entrena cfg['n_seeds'] modelos independientes para este fold (misma
    partición train/val_early/test, distinta seed cada uno -- igual que
    _run_ensemble_with_precomputed_split en dl_script2e_bilstm_round4_fixed.py)
    y promedia sus probabilidades de test. Con n_seeds=1 es exactamente el
    comportamiento anterior (una sola corrida)."""
    n_seeds = cfg.get("n_seeds", 1)
    probs_runs = []
    test_labels_ref = None
    for s in range(n_seeds):
        seed_s = fold_seed * 1000 + s  # mismo esquema de seeds que el script fuente
        probs, labels = _train_bilstm_single_seed(
            X_train, y_train, lengths_train,
            X_valearly, y_valearly, lengths_valearly,
            X_test, y_test, lengths_test,
            cfg=cfg, seed=seed_s,
        )
        probs_runs.append(probs)
        test_labels_ref = labels  # igual en todas las corridas (mismo test)

    ensemble_probs = np.mean(np.stack(probs_runs, axis=0), axis=0)
    return ensemble_probs, test_labels_ref


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
        # aumentos: SOLO sobre train, nunca sobre val_early ni test (igual que
        # dl_script2e_bilstm_round4_fixed.py). "fix_tmask" = mixup + time-masking
        # encima; "fix" = solo mixup (comportamiento anterior); "tmask_only" =
        # solo time-masking, sin mixup.
        mixup_mode = BILSTM_BEST_CFG["mixup"]
        fold_aug_seed = SEED + repeat * n_folds + fold
        if mixup_mode in ("fix", "fix_tmask"):
            X_tr, y_tr, l_tr = mixup_sequences(X_tr, y_tr, l_tr, seed=fold_aug_seed)
        if mixup_mode in ("fix_tmask", "tmask_only"):
            X_tr = time_mask_sequences(X_tr, l_tr, max_masks=2, seed=fold_aug_seed)

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
# 5. SHAP / LIME OOF -- helpers compartidos por la rama ExtraTrees tabular
#    (video AU)
# ==============================================================================

def compute_shap_oof(clf, X_test, feature_cols):
    """SHAP OOF de un fold: el TreeExplainer se crea SIEMPRE a partir del
    `clf` que se acaba de entrenar con train_fit+val_early de ESTE fold, y
    se evalúa únicamente sobre el X_test de ese mismo fold -- nunca se
    reutiliza un explainer entrenado con datos de otro fold. Devuelve la
    atribución de la clase positiva, shape (n_test, n_features)."""
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        # shap "viejo": lista [clase0, clase1]
        sv_pos = np.asarray(sv[1])
    else:
        sv = np.asarray(sv)
        # shap "nuevo": (n_samples, n_features, n_clases) para binario
        sv_pos = sv[:, :, 1] if sv.ndim == 3 else sv
    assert sv_pos.shape == (len(X_test), len(feature_cols)), (
        f"Forma inesperada de shap_values: {sv_pos.shape}, se esperaba "
        f"({len(X_test)}, {len(feature_cols)})."
    )
    return sv_pos


def compute_shap_oof_regressor(reg, X_test, feature_cols):
    """SHAP OOF de un fold para un REGRESOR (ExtraTreesRegressor, soportado
    por shap.TreeExplainer -- a diferencia de AdaBoostRegressor, que
    confirmé que NO lo está [InvalidModelError]). El explainer se crea
    SIEMPRE a partir del `reg` recién entrenado con train_fit+val_early de
    ESTE fold, y se evalúa únicamente sobre el X_test de ese mismo fold.
    Un regresor no tiene dimensión de clase, así que shap_values ya viene
    en la forma (n_test, n_features) directamente."""
    explainer = shap.TreeExplainer(reg)
    sv = np.asarray(explainer.shap_values(X_test))
    assert sv.shape == (len(X_test), len(feature_cols)), (
        f"Forma inesperada de shap_values (regresor): {sv.shape}, se esperaba "
        f"({len(X_test)}, {len(feature_cols)})."
    )
    return sv


def compute_lime_oof(clf, X_train_bg, X_test, feature_cols, seed,
                      num_samples=LIME_NUM_SAMPLES, progress_label=None, progress_every=25):
    """LIME OOF de un fold: el explainer se ajusta (background) con el
    MISMO train_fit+val_early que entrenó `clf` en este fold, y solo
    explica las filas del X_test de ese mismo fold -- mismo pareo
    fold-a-fold que compute_shap_oof. Se usa exp.as_map()[1] (clase
    positiva) para obtener la atribución por índice de columna original,
    en vez de as_list() (que devuelve descripciones de texto tras
    discretizar y no es directamente indexable por feature).

    discretize_continuous=False (en vez del default True de la librería):
    con lime==0.2.0.1 + scipy>=1.11 el muestreo interno del discretizador
    (scipy.stats.truncnorm.rvs) puede romper con un TypeError espurio en
    ciertos folds/columnas -- es un conflicto de versiones de la librería,
    no de este código (reproducido y confirmado al construir este script).
    discretize_continuous=False evita ese muestreo por cuantiles y da
    pesos lineales por feature sobre la escala original; se sigue
    extrayendo igual con as_map()[1]. Si tu entorno no tiene ese conflicto
    y prefieres el comportamiento estándar de LIME (bins discretizados),
    cambia esto a True."""
    explainer = LimeTabularExplainer(
        np.asarray(X_train_bg), feature_names=list(feature_cols),
        class_names=["no_depresion", "depresion"], mode="classification",
        discretize_continuous=False, random_state=seed,
    )
    n_feat = len(feature_cols)
    X_test_arr = np.asarray(X_test)
    out = np.zeros((len(X_test_arr), n_feat), dtype=float)
    for i in range(len(X_test_arr)):
        exp = explainer.explain_instance(
            X_test_arr[i], clf.predict_proba, num_features=n_feat,
            num_samples=num_samples, labels=[1],
        )
        fmap = dict(exp.as_map()[1])
        out[i] = [fmap.get(j, 0.0) for j in range(n_feat)]
        if progress_label and (i + 1) % progress_every == 0:
            _log(f"    [LIME {progress_label}] {i + 1}/{len(X_test_arr)} instancias explicadas")
    return out


def build_shap_lime_oof_rows(values_array, feature_cols, row_subject_ids, repeat, fold):
    """Empaqueta un array (n_test_rows, n_features) -- SHAP o LIME de un
    fold -- en un DataFrame (subject_id, repeat, fold, <features>). Si
    row_subject_ids trae subject_id repetidos (caso segmentado), se
    promedian aquí mismo, con el mismo criterio con el que ya se promedian
    las predicciones por paciente. Si cada subject_id aparece una sola vez
    (caso video, o caso donde feature_cols cambia por fold, ver
    filter_high_nan_cols_fold), el groupby es un no-op."""
    df = pd.DataFrame(np.asarray(values_array), columns=list(feature_cols))
    df.insert(0, "subject_id", list(row_subject_ids))
    agg = df.groupby("subject_id", as_index=False)[list(feature_cols)].mean()
    agg.insert(1, "repeat", repeat)
    agg.insert(2, "fold", fold)
    return agg


def filter_high_nan_cols_fold(X_train_df: pd.DataFrame, candidate_cols: list, max_nan_frac: float):
    """FIX v4 (antes se calculaba UNA VEZ sobre los 79 pacientes juntos,
    ver docstring del módulo, punto 2 de 'CAMBIOS EN v4'): decide qué
    columnas sobreviven usando ÚNICAMENTE train_fit+val_early de ESTE fold
    -- nunca estadísticas de test, nunca estadísticas de "todos los
    pacientes" (que en v3 incluían, para cada fold, al propio test de ese
    fold). No usa la etiqueta y en ningún momento, así que no hay riesgo de
    leakage de la variable objetivo -- es puramente una decisión de calidad
    de dato, pero ahora calculada con el conjunto de datos correcto."""
    nan_frac = X_train_df[candidate_cols].isna().mean()
    dropped = nan_frac[nan_frac > max_nan_frac].index.tolist()
    kept = [c for c in candidate_cols if c not in dropped]
    return kept, dropped


# ==============================================================================
# 6. MODELO -- ExtraTreesClassifier sobre features tabulares de video (AU)
#    + SHAP/LIME OOF (v3, con fix de leakage del filtro NaN en v4)
# ==============================================================================

def run_extratrees_video_oof(pacientes: pd.DataFrame, manifest: pd.DataFrame,
                              n_repeats: int, n_folds: int):
    logger.info("=" * 70)
    logger.info("MODELO 1/4: ExtraTreesClassifier sobre features tabulares de Action Units (video)")
    logger.info("=" * 70)

    df_vid = pd.read_csv(FEATURES_VIDEO_AU)
    assert PATIENT_ID_COL_VIDEO in df_vid.columns and LABEL_COL_VIDEO in df_vid.columns, (
        f"Se esperaban las columnas '{PATIENT_ID_COL_VIDEO}' y '{LABEL_COL_VIDEO}' en "
        f"{FEATURES_VIDEO_AU}. Columnas reales: {list(df_vid.columns)}"
    )
    df_vid = df_vid.copy()
    df_vid["_video_id_norm"] = df_vid[PATIENT_ID_COL_VIDEO].map(_normalize_video_id)

    # mapeo video_id -> subject_id, MISMO criterio que ya usaba BiGRU en v2
    subject_ids = pacientes.subject_id.tolist()
    video_map = build_video_id_map(df_vid["_video_id_norm"].tolist(), subject_ids)  # subject_id -> video_id normalizado
    subj_lookup = {vid: sid for sid, vid in video_map.items()}
    df_vid["subject_id"] = df_vid["_video_id_norm"].map(subj_lookup)

    df_vid = df_vid[df_vid["subject_id"].notna()].reset_index(drop=True)
    assert set(df_vid["subject_id"]) >= set(subject_ids), (
        "No todos los subject_id de pacientes_finales.csv se encontraron en "
        f"{FEATURES_VIDEO_AU} tras el mapeo video_id -> subject_id."
    )
    assert df_vid["subject_id"].is_unique, (
        f"{FEATURES_VIDEO_AU}: se esperaba 1 fila por paciente (sin segmentos), "
        "pero hay subject_id repetidos tras el mapeo -- revisa build_video_id_map()."
    )

    # el label real (y) sale de pacientes_finales.csv -- misma fuente de verdad
    # que el resto del pipeline. Se calcula ANTES de descartar LABEL_COL_VIDEO
    # como feature, y luego se DROPEA esa columna del propio CSV de video para
    # que el merge de abajo no choque de nombre y termine dejando -- por orden
    # de pandas -- el target_depresion del CSV en vez del de pacientes.
    candidate_cols = [c for c in df_vid.columns
                      if c not in ({PATIENT_ID_COL_VIDEO, LABEL_COL_VIDEO, "subject_id", "_video_id_norm"}
                                   | set(DROP_COLS_VIDEO))]
    candidate_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df_vid[c])]
    # NOTA v4: el filtro de NaN por umbral YA NO se calcula aquí (sobre los 79
    # pacientes juntos) -- se movió a filter_high_nan_cols_fold(), llamado
    # DENTRO de cada fold worker con solo train_fit+val_early de ese fold (ver
    # docstring del módulo, punto 2 de "CAMBIOS EN v4"). candidate_cols es el
    # universo completo de columnas candidatas (filtradas solo por tipo, no
    # por NaN); cada fold decide su propio subconjunto.

    df_vid = df_vid.drop(columns=[LABEL_COL_VIDEO])
    df_vid = df_vid.merge(pacientes[["subject_id", "target_depresion"]], on="subject_id", how="left")
    logger.info(f"AU (video): {len(candidate_cols)} features candidatas (antes del filtro de NaN "
                f"por fold), {df_vid.subject_id.nunique()} pacientes (1 fila = 1 paciente, sin segmentos).")

    tasks = [(r, f) for r in range(n_repeats) for f in range(n_folds)]
    prob_rows, shap_detail, lime_detail = run_tabular_shap_lime_parallel(
        tasks, _video_fold_worker, N_CPU_WORKERS, "ExtraTrees video (AU) + SHAP/LIME",
        initializer=_init_video_worker,
        initargs=(manifest, df_vid, candidate_cols, n_folds),
    )

    df = pd.DataFrame(prob_rows)
    logger.info(f"ExtraTrees video (AU): {df.subject_id.nunique()} pacientes procesados, "
                f"{n_repeats * n_folds} (repeat, fold) evaluados.")
    # candidate_cols (el universo completo, sin filtrar por NaN) es lo que se
    # devuelve para armar la matriz de explicabilidad final -- una columna que
    # un fold particular haya descartado por NaN simplemente queda ausente
    # (NaN) en las filas de ESE fold dentro de shap_detail/lime_detail, y el
    # promedio final por paciente (build_explainability_matrix) la promedia
    # solo sobre los folds donde sí estuvo disponible (comportamiento correcto
    # de pandas .mean() con skipna=True por default).
    return df, shap_detail, lime_detail, candidate_cols


# --- worker de proceso (nivel de módulo -- picklable) para la rama de video ---
_VIDEO_WORKER_STATE = {}


def _init_video_worker(manifest, df_vid, candidate_cols, n_folds):
    """Igual que en v3: se corre UNA VEZ por proceso hijo, no en cada una
    de las 50 tareas."""
    _VIDEO_WORKER_STATE["manifest"] = manifest
    _VIDEO_WORKER_STATE["df_vid"] = df_vid
    _VIDEO_WORKER_STATE["candidate_cols"] = candidate_cols
    _VIDEO_WORKER_STATE["n_folds"] = n_folds


def _video_fold_worker(repeat, fold):
    manifest = _VIDEO_WORKER_STATE["manifest"]
    df_vid = _VIDEO_WORKER_STATE["df_vid"]
    candidate_cols = _VIDEO_WORKER_STATE["candidate_cols"]
    n_folds = _VIDEO_WORKER_STATE["n_folds"]

    train_sub, test_sub = get_fold_subjects_combined(manifest, repeat, fold)
    train_mask = df_vid.subject_id.isin(train_sub)
    test_mask = df_vid.subject_id.isin(test_sub)

    # FIX v4: el filtro de NaN por umbral se calcula AQUÍ, con SOLO el
    # train_fit+val_early de este fold -- nunca con test, nunca con los 79
    # pacientes juntos (ver filter_high_nan_cols_fold() y punto 2 de
    # "CAMBIOS EN v4" en el docstring del módulo).
    feature_cols, dropped_cols = filter_high_nan_cols_fold(
        df_vid.loc[train_mask], candidate_cols, MAX_NAN_FRAC_VIDEO
    )
    if dropped_cols:
        _log(f"  [video r{repeat}f{fold}] {len(dropped_cols)} features descartadas por NaN "
             f"(>{MAX_NAN_FRAC_VIDEO:.0%} en train_fit+val_early de este fold)")

    X_train_raw = df_vid.loc[train_mask, feature_cols].values
    y_train = df_vid.loc[train_mask, "target_depresion"].values
    X_test_raw = df_vid.loc[test_mask, feature_cols].values
    test_subject_rows = df_vid.loc[test_mask, "subject_id"].tolist()  # 1 fila = 1 paciente

    # Imputación de NaN restantes -- SOLO con la media de train_fit+val_early
    # de ESTE fold (nunca con test, nunca con la media global). Si por mala
    # suerte una columna quedó 100% NaN en el train de este fold puntual
    # (fold_mean también NaN), se cae a 0.0 como último recurso neutro.
    fold_mean = np.nanmean(X_train_raw, axis=0)
    fold_mean = np.nan_to_num(fold_mean, nan=0.0)
    nan_train = np.isnan(X_train_raw)
    nan_test = np.isnan(X_test_raw)
    if nan_train.any():
        X_train_raw = np.where(nan_train, fold_mean, X_train_raw)
    if nan_test.any():
        X_test_raw = np.where(nan_test, fold_mean, X_test_raw)

    if SCALE_VIDEO_ET:
        # scaler ajustado SOLO con train_fit+val_early de este fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    clf = ExtraTreesClassifier(random_state=SEED, n_jobs=ET_N_JOBS, **ET_BEST_PARAMS_VIDEO)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]

    prob_rows = [{"subject_id": sid, "repeat": repeat, "fold": fold,
                   "prob_ml_video": float(p)} for sid, p in zip(test_subject_rows, proba)]

    # SHAP/LIME OOF -- mismo clf, mismo X_test/X_train (ya escalados si
    # SCALE_VIDEO_ET) de ESTE fold, con el feature_cols específico de ESTE
    # fold (puede tener menos columnas que candidate_cols si algo se
    # descartó por NaN arriba).
    sv_pos = compute_shap_oof(clf, X_test, feature_cols)
    shap_fold_df = build_shap_lime_oof_rows(sv_pos, feature_cols, test_subject_rows, repeat, fold)

    lime_arr = compute_lime_oof(clf, X_train, X_test, feature_cols,
                                 seed=SEED + repeat * n_folds + fold,
                                 progress_label=f"video r{repeat}f{fold}")
    lime_fold_df = build_shap_lime_oof_rows(lime_arr, feature_cols, test_subject_rows, repeat, fold)

    return prob_rows, shap_fold_df, lime_fold_df


# ==============================================================================
# 6b. MODELO -- Destilación: Teacher (BiLSTM, reutiliza modelo de arriba) ->
#     Student (ExtraTreesRegressor sobre eGeMAPS) + SHAP OOF (sin LIME, es
#     un regresor -- ver docstring del módulo)
# ==============================================================================

def run_destilacion_oof(pacientes: pd.DataFrame, manifest: pd.DataFrame,
                         n_repeats: int, n_folds: int,
                         teacher_oof_df: pd.DataFrame):
    logger.info("=" * 70)
    logger.info("MODELO 4/4: Destilación (Teacher BiLSTM -> Student ExtraTreesRegressor, eGeMAPS)")
    logger.info("=" * 70)

    if not REUSE_BILSTM_AS_TEACHER:
        raise NotImplementedError(
            "REUSE_BILSTM_AS_TEACHER=False: re-implementa aquí un entrenamiento "
            "independiente del Teacher; la lógica es idéntica a "
            "_train_bilstm_single_seed() con BILSTM_BEST_CFG."
        )
    logger.info("Teacher OOF: reutilizando la salida del BiLSTM (ya con el fix de Ronda "
                "4/focal aplicado) -- mismos folds, mismo protocolo.")
    teacher_oof = teacher_oof_df.rename(columns={"prob_bilstm_embeddings": "p_teacher_OOF"})

    df_tab = pd.read_csv(FEATURES_DEPRESION_EGEMAPS)
    assert PATIENT_ID_COL_ACUSTICO in df_tab.columns, (
        f"Se esperaba la columna '{PATIENT_ID_COL_ACUSTICO}' en "
        f"{FEATURES_DEPRESION_EGEMAPS}. Columnas reales: {list(df_tab.columns)}"
    )
    if LABEL_COL_ACUSTICO in df_tab.columns:
        df_tab = df_tab.drop(columns=[LABEL_COL_ACUSTICO])

    # mapeo audio_id -> subject_id (mismo criterio de prefijo que embeddings/BiLSTM)
    subject_ids = pacientes.subject_id.tolist()
    tab_audio_ids = df_tab[PATIENT_ID_COL_ACUSTICO].unique().tolist()
    audio_map = build_audio_id_map(tab_audio_ids, subject_ids)
    subj_lookup = {aid: sid for sid, aid in audio_map.items()}
    df_tab["subject_id"] = df_tab[PATIENT_ID_COL_ACUSTICO].map(subj_lookup)
    assert df_tab["subject_id"].notna().all(), (
        "Algunos segmentos de eGeMAPS no pudieron mapearse a subject_id -- "
        "revisa build_audio_id_map()."
    )

    feature_cols = [c for c in df_tab.columns
                    if c not in ({PATIENT_ID_COL_ACUSTICO, "subject_id"} | set(DROP_COLS_ACUSTICO))]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_tab[c])]
    logger.info(f"Destilación (eGeMAPS): {len(feature_cols)} features, "
                f"{df_tab.subject_id.nunique()} pacientes, {len(df_tab)} segmentos totales.")

    # y_KD depende de p_teacher_OOF, que es distinto por repeat -- se arma
    # UNA vez por repeat aquí (antes de los folds) y se pasa read-only a los
    # workers de ese repeat vía el initializer de cada proceso.
    df_tab_by_repeat = {}
    pac_target = pacientes.set_index("subject_id")["target_depresion"]
    for repeat in range(n_repeats):
        oof_r = teacher_oof.loc[teacher_oof.repeat == repeat, ["subject_id", "p_teacher_OOF"]]
        df_r = df_tab.merge(oof_r, on="subject_id", how="inner")
        y_real = pac_target.loc[df_r.subject_id].values
        df_r["y_KD"] = KD_ALPHA * y_real + (1 - KD_ALPHA) * df_r["p_teacher_OOF"].values
        df_tab_by_repeat[repeat] = df_r

    tasks = [(r, f) for r in range(n_repeats) for f in range(n_folds)]
    prob_rows, shap_detail, _lime_unused = run_tabular_shap_lime_parallel(
        tasks, _kd_fold_worker, N_CPU_WORKERS, "Destilación (ExtraTreesRegressor) + SHAP",
        initializer=_init_kd_worker,
        initargs=(manifest, df_tab_by_repeat, feature_cols, n_folds),
    )

    df = pd.DataFrame(prob_rows)
    logger.info(f"Destilación: {df.subject_id.nunique()} pacientes procesados, "
                f"{n_repeats * n_folds} (repeat, fold) evaluados. "
                f"Dataset={KD_DATASET_NAME}, alpha={KD_ALPHA}, student={KD_STUDENT}.")
    return df, shap_detail, feature_cols


# --- worker de proceso (nivel de módulo -- picklable) para destilación ---
_KD_WORKER_STATE = {}


def _init_kd_worker(manifest, df_tab_by_repeat, feature_cols, n_folds):
    _KD_WORKER_STATE["manifest"] = manifest
    _KD_WORKER_STATE["df_tab_by_repeat"] = df_tab_by_repeat
    _KD_WORKER_STATE["feature_cols"] = feature_cols
    _KD_WORKER_STATE["n_folds"] = n_folds


def _kd_fold_worker(repeat, fold):
    manifest = _KD_WORKER_STATE["manifest"]
    df_r = _KD_WORKER_STATE["df_tab_by_repeat"][repeat]
    feature_cols = _KD_WORKER_STATE["feature_cols"]

    train_sub, test_sub = get_fold_subjects_combined(manifest, repeat, fold)
    train_mask = df_r.subject_id.isin(train_sub)
    test_mask = df_r.subject_id.isin(test_sub)

    X_train = df_r.loc[train_mask, feature_cols].values
    y_kd_train = df_r.loc[train_mask, "y_KD"].values
    X_test = df_r.loc[test_mask, feature_cols].values
    test_subject_rows = df_r.loc[test_mask, "subject_id"].tolist()  # puede repetir subject_id (segmentos)

    # sin escalado -- ExtraTrees es invariante a transformaciones monótonas
    # por feature (los splits no cambian), mismo criterio que en v3
    reg = ExtraTreesRegressor(random_state=SEED, n_jobs=ET_N_JOBS)  # hiperparámetros default
    reg.fit(X_train, y_kd_train)
    p_student_seg = np.clip(reg.predict(X_test), 0, 1)

    # promedio por paciente (varios segmentos -> 1 probabilidad por paciente)
    df_test = pd.DataFrame({"subject_id": test_subject_rows, "p": p_student_seg})
    per_patient = df_test.groupby("subject_id")["p"].mean()
    prob_rows = [{"subject_id": sid, "repeat": repeat, "fold": fold,
                   "prob_destilacion_estudiante": float(p)} for sid, p in per_patient.items()]

    # SHAP OOF -- ExtraTreesRegressor SÍ está soportado por TreeExplainer
    # (a diferencia de AdaBoostRegressor), mismo clf/X_test de ESTE fold
    sv = compute_shap_oof_regressor(reg, X_test, feature_cols)
    shap_fold_df = build_shap_lime_oof_rows(sv, feature_cols, test_subject_rows, repeat, fold)

    # sin LIME en esta rama (regresor, no clasificador -- ver docstring del
    # módulo) -- placeholder vacío con el mismo esquema de columnas, para no
    # romper run_tabular_shap_lime_parallel
    lime_fold_df = pd.DataFrame(columns=["subject_id", "repeat", "fold"] + list(feature_cols))

    return prob_rows, shap_fold_df, lime_fold_df


# ==============================================================================
# 7. PASO 3 -- ensamblar la matriz final (4 modelos) y las matrices SHAP/LIME
# ==============================================================================

PROB_COLS = ["prob_ml_video", "prob_bilstm_embeddings",
             "prob_extratrees_embeddings", "prob_destilacion_estudiante"]


def build_final_matrix(pacientes, et_vid_df, bilstm_df, et_emb_df, kd_df):
    detail = (
        et_vid_df.merge(bilstm_df, on=["subject_id", "repeat", "fold"], how="outer")
                 .merge(et_emb_df, on=["subject_id", "repeat", "fold"], how="outer")
                 .merge(kd_df, on=["subject_id", "repeat", "fold"], how="outer")
    )
    detail = detail.merge(pacientes[["subject_id", "target_depresion"]], on="subject_id", how="left")
    detail = detail.rename(columns={"target_depresion": "y_true"})

    final = detail.groupby("subject_id")[PROB_COLS].mean().reset_index()
    final = final.merge(pacientes[["subject_id", "target_depresion"]], on="subject_id", how="left")
    final = final.rename(columns={"target_depresion": "y_true"})
    final = final.sort_values("subject_id").reset_index(drop=True)

    return final, detail


def build_explainability_matrix(parts):
    """Promedia SHAP (o LIME) por subject_id a través de los 10 repeats --
    igual criterio que build_final_matrix() con las probabilidades. `parts`
    es una lista de (detail_df, feature_cols, prefix); cada una se promedia
    por separado y se prefija (p.ej. 'au__', 'destilacion_egemaps__') antes
    de unirlas en un único DataFrame ancho. Con el fix de v4 (filtro de NaN
    por fold), una columna que algún fold haya descartado simplemente
    aparece como NaN en las filas de ese fold en el detail -- .mean() la
    promedia solo sobre los folds donde sí existió (skipna=True, default)."""
    matrix = None
    for detail_df, feature_cols, prefix in parts:
        avg = detail_df.groupby("subject_id")[feature_cols].mean().reset_index()
        avg = avg.rename(columns={c: f"{prefix}__{c}" for c in feature_cols})
        matrix = avg if matrix is None else matrix.merge(avg, on="subject_id", how="outer")
    matrix = matrix.sort_values("subject_id").reset_index(drop=True)
    return matrix


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    _log(logger_cfg_msg)
    pacientes, manifest, n_repeats, n_folds = load_manifest_and_patients()

    et_vid_df, shap_vid_detail, lime_vid_detail, vid_feature_cols = run_extratrees_video_oof(
        pacientes, manifest, n_repeats, n_folds
    )
    bilstm_df = run_bilstm_oof(pacientes, manifest, n_repeats, n_folds)
    et_emb_df = run_extratrees_oof(pacientes, manifest, n_repeats, n_folds)
    kd_df, shap_kd_detail, kd_feature_cols = run_destilacion_oof(
        pacientes, manifest, n_repeats, n_folds, teacher_oof_df=bilstm_df
    )

    final_matrix, detail = build_final_matrix(pacientes, et_vid_df, bilstm_df, et_emb_df, kd_df)
    shap_matrix = build_explainability_matrix([
        (shap_vid_detail, vid_feature_cols, "au"),
        (shap_kd_detail, kd_feature_cols, "destilacion_egemaps"),
    ])
    lime_matrix = build_explainability_matrix([
        (lime_vid_detail, vid_feature_cols, "au"),
    ])  # sin LIME para destilación (regresor, no clasificador)

    final_path = OUTPUT_DIR / "oof_matrix_final.csv"
    detail_path = OUTPUT_DIR / "oof_matrix_detail_by_repeat.csv"
    shap_path = OUTPUT_DIR / "shap_matrix.csv"
    lime_path = OUTPUT_DIR / "lime_matrix.csv"

    final_matrix.to_csv(final_path, index=False)
    detail.to_csv(detail_path, index=False)
    shap_matrix.to_csv(shap_path, index=False)
    lime_matrix.to_csv(lime_path, index=False)

    logger.info("=" * 70)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 70)
    logger.info(f"Matriz final ({len(final_matrix)} pacientes, promedio de "
                f"{n_repeats} repeticiones OOF por modelo) -> {final_path}")
    logger.info(f"Detalle sin promediar (subject_id x repeat x modelo) -> {detail_path}")
    logger.info(f"SHAP promedio por paciente ({len(vid_feature_cols)} features AU candidatas + "
                f"{len(kd_feature_cols)} features eGeMAPS-destilación) -> {shap_path}")
    logger.info(f"LIME promedio por paciente ({len(vid_feature_cols)} features AU candidatas -- "
                f"sin destilación) -> {lime_path}")
    logger.info("\nColumnas y modelo que las generó:")
    logger.info("  prob_ml_video               -> ExtraTreesClassifier / Action Units (tabular, por paciente) "
                f"({et_vid_df.subject_id.nunique()} pacientes, {n_repeats*n_folds} evaluaciones)")
    logger.info("  prob_bilstm_embeddings      -> BiLSTM / secuencias embeddings audio wav2vec2-large-robust "
                f"({bilstm_df.subject_id.nunique()} pacientes, {n_repeats*n_folds} evaluaciones)")
    logger.info("  prob_extratrees_embeddings  -> ExtraTrees / embeddings audio wav2vec2-large-robust "
                f"({et_emb_df.subject_id.nunique()} pacientes, {n_repeats*n_folds} evaluaciones)")
    logger.info("  prob_destilacion_estudiante -> Teacher BiLSTM -> Student ExtraTreesRegressor / eGeMAPS "
                f"({kd_df.subject_id.nunique()} pacientes, {n_repeats*n_folds} evaluaciones)")
    logger.info(f"\nPacientes en matriz final: {len(final_matrix)} / 79 esperados")
    for c in PROB_COLS:
        n_nan = final_matrix[c].isna().sum()
        if n_nan:
            logger.warning(f"  ⚠ {c}: {n_nan} pacientes sin probabilidad (revisar mapeo de IDs)")


if __name__ == "__main__":
    main()