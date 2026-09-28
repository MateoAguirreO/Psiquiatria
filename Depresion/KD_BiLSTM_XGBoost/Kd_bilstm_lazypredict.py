"""
==============================================================================
Destilación de conocimiento: BiLSTM (Teacher, embeddings wav2vec2-large-robust)
                              -> LazyPredict (Student, SOLO modelos de regresión)
==============================================================================

*** ACTUALIZACIÓN ***
  - Protocolo de validación: RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
    exactamente igual al usado en el resto del paper (dl_script_bilstm_final.py:
    CV_SPLITS=5, CV_REPEATS=10 -> 50 evaluaciones por configuración). El Teacher
    se re-entrena en cada uno de los 50 (fold, repeat) para generar OOF honesto,
    y el Student corre con exactamente esos mismos (fold, repeat).
  - El Student usa ÚNICAMENTE LazyRegressor (nada de LazyClassifier). El caso
    alpha=1.0 (y_KD = y_real) actúa como "baseline sin destilación" dentro del
    mismo LazyRegressor, evitando así mezclar dos paradigmas (regresión vs
    clasificación) y siendo consistente con el resto de experimentos del paper.
  - Columnas reales de los CSV tabulares (según muestras enviadas):
        audio_id, seg_idx, label, <features...>
    Por eso PATIENT_ID_COL="audio_id", LABEL_COL="label" y se descarta seg_idx
    del vector de features (es solo un índice de segmento, no una característica
    acústica).

ADVERTENCIA DE CÓMPUTO: 5 folds x 10 repeats x 5 alphas x 2 datasets x 2
condiciones x LazyRegressor (~40 modelos de sklearn) es una carga MUY pesada
(miles de entrenamientos). Contempla correrlo en background / por partes, o
reducir N_REPEATS temporalmente para pruebas rápidas antes de la corrida final.
==============================================================================

Adaptación del protocolo descrito en:
    Guia_Destilacion_BiLSTM_XGBoost_5Fold_con_Anexo_OOF.pdf
al caso real del proyecto (Ansiedad / Depresión), usando:

  Teacher : BiLSTM entrenado sobre secuencias de embeddings por paciente
            (modelo "wav2vec2-large-robust", el mejor obtenido en la
            investigación). Se reutiliza la arquitectura y funciones de
            carga de `dl_script_bilstm_final.py`.

  Student : En vez de un único XGBoost, se corre LazyPredict (LazyRegressor)
            sobre dos datasets tabulares por condición (features_5.csv con
            librosa/openSMILE "clásico" y features_egemaps.csv con eGeMAPS),
            intercambiándolos, y añadiendo como columna extra la
            probabilidad OOF del Teacher (ya sea cruda, o combinada con la
            etiqueta real vía y_KD = alpha*y_real + (1-alpha)*p_teacher_OOF).

------------------------------------------------------------------------------
REGLAS ANTI DATA-LEAKAGE APLICADAS (igual que en la guía / anexo OOF):
------------------------------------------------------------------------------
  1. Todo se particiona a NIVEL DE PACIENTE (patient_id), nunca a nivel de
     segmento/fila. Un paciente entero cae en un único fold.
  2. Los folds del Teacher y del Student son EXACTAMENTE los mismos
     (misma asignación paciente -> fold), generados una sola vez.
  3. La probabilidad del Teacher usada para construir y_KD SIEMPRE es
     Out-of-Fold: para el fold k, el Teacher que generó esa probabilidad
     nunca fue entrenado con pacientes del fold k (se re-entrena el
     BiLSTM 5 veces, una por fold, igual que en la guía FASE 4).
  4. El checkpoint final ya entrenado con TODO el dataset (el "mejor
     modelo" que ya tienen) NO se usa para generar targets de destilación
     -- solo se usa (opcionalmente) como referencia de desempeño / para
     producir la predicción final sobre un test set separado, si aplica.
  5. El Student de cada fold se entrena solo con los folds de train y se
     evalúa contra la etiqueta REAL del fold de validación (nunca contra
     p_teacher ni contra y_KD).
  6. Scaler / normalización de features tabulares se ajusta SOLO con las
     filas de train de cada fold.
  7. alpha se explora en rejilla {0, .25, .50, .75, 1.00} igual que la guía;
     no se fija arbitrariamente.

------------------------------------------------------------------------------
QUÉ DEBES COMPLETAR ANTES DE CORRER (sección CONFIG más abajo):
------------------------------------------------------------------------------
  - EMBEDDINGS_ROOT_ANSIEDAD / EMBEDDINGS_ROOT_DEPRESION:
        ruta a la carpeta que contiene manifest.json + JSON de embeddings
        para el modelo "wav2vec2-large-robust" (según las capturas que
        enviaste, algo como ".../Clasificación Final/<Ansiedad|Depresión>/
        embeddings_v2/").
  - TEACHER_HPARAMS_ANSIEDAD / TEACHER_HPARAMS_DEPRESION:
        hidden / pooling / dropout_p / loss / warmup / lr que dieron el
        mejor resultado en tus experimentos con dl_script_bilstm_final.py
        (revisa tu summary.csv de esa corrida para el modelo
        wav2vec2-large-robust y copia la fila con mayor mean_auc).
  - Rutas de los 4 CSV de features tabulares (ya las diste, quedaron
    puestas abajo).
  - COLUMNA DE ID DE PACIENTE Y DE ETIQUETA en los CSV tabulares: ajusta
    PATIENT_ID_COL / LABEL_COL / DROP_COLS según los nombres reales de
    columnas de tus features_*.csv (dejo nombres de ejemplo razonables,
    pero revisa con `df.columns` antes de correr la corrida completa).

Instalar dependencias que falten:
    pip install lazypredict scikit-learn torch pandas numpy --break-system-packages
==============================================================================
"""

import copy
import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_recall_curve,
    mean_squared_error,
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("KD_BiLSTM_LazyPredict")

try:
    from lazypredict.Supervised import LazyRegressor
except ImportError:
    raise ImportError(
        "Falta lazypredict. Instala con:\n"
        "  pip install lazypredict --break-system-packages\n"
        "(puede requerir además: pip install --upgrade scikit-learn "
        "--break-system-packages, ya que lazypredict suele fijar versiones viejas)."
    )

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
N_REPEATS = 10   # RepeatedStratifiedKFold, igual que en el resto del paper (5x10 = 50 evaluaciones)
ALPHA_GRID = [0.0, 0.25, 0.50, 0.75, 1.00]
TEACHER_MODEL_KEY = "wav2vec2-large-robust"   # embedding usado por el Teacher

# =============================================================================
# CONFIG — AJUSTA ESTO ANTES DE CORRER
# =============================================================================

# --- Rutas embeddings (Teacher) -------------------------------------------
EMBEDDINGS_ROOT_ANSIEDAD  = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Ansiedad/embeddings_v2/"
EMBEDDINGS_ROOT_DEPRESION = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Depresión/embeddings_v2/"

# --- Mejor configuración encontrada para el Teacher, tomada DIRECTAMENTE de tu
# summary.csv real (dl_script_bilstm_final.py), filtrando emb_model ==
# "wav2vec2-large-robust" y quedándonos con el mayor mean_auc.
# Fila ganadora: experiment="pooling_meanmax", mean_auc=0.7659 (la más alta de
# TODO el summary, no solo de wav2vec2-large-robust) ->
#   hidden=128, pooling=meanmax, loss=bce_posw, mixup=fix, warmup=False,
#   dropout_p=0.4, pca=False
# Se usa EXACTAMENTE esta configuración para ambas condiciones, ya que en tu
# summary no se corrieron experimentos separados por condición (ansiedad vs
# depresión) sino uno combinado -- si en tu caso tienes un summary.csv por
# condición, actualiza cada bloque con su propia fila ganadora.
_BEST_TEACHER_CFG = dict(
    hidden=128, pooling="meanmax", loss="bce_posw",
    warmup=False, lr=1e-3, dropout_p=0.4, mixup="fix",
)
TEACHER_HPARAMS_ANSIEDAD  = dict(_BEST_TEACHER_CFG)
TEACHER_HPARAMS_DEPRESION = dict(_BEST_TEACHER_CFG)

TEACHER_EPOCHS   = 60
TEACHER_PATIENCE = 8
TEACHER_BATCH    = 16

# --- Rutas features tabulares (Student) ------------------------------------
FEATURES_DEPRESION_EGEMAPS = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/MachineLearning/features_depresion_egemaps.csv"
FEATURES_DEPRESION_5S      = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/MachineLearning/features_depresion_5.csv"
FEATURES_ANSIEDAD_5S       = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Ansiedad/MachineLearning/features_ansiedad_5.csv"
FEATURES_ANSIEDAD_EGEMAPS  = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Ansiedad/MachineLearning/features_ansiedad_egemaps.csv"

# --- Nombres de columnas en los CSV tabulares (según muestras enviadas) ----
# Encabezado real: audio_id, seg_idx, label, <features...>
PATIENT_ID_COL = "audio_id"   # identifica al paciente/audio (merge con Teacher OOF y agrupación de folds)
LABEL_COL      = "label"      # etiqueta real (0/1)
DROP_COLS      = ["seg_idx"]  # índice de segmento -- no es una característica acústica, se excluye de X

OUTPUT_DIR = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/KD_LazyPredict_Results")

CONDITIONS = {
    "ansiedad": dict(
        embeddings_root=EMBEDDINGS_ROOT_ANSIEDAD,
        teacher_hparams=TEACHER_HPARAMS_ANSIEDAD,
        tabular_datasets={
            "5s_opensmile_librosa": FEATURES_ANSIEDAD_5S,
            "egemaps":              FEATURES_ANSIEDAD_EGEMAPS,
        },
    ),
    "depresion": dict(
        embeddings_root=EMBEDDINGS_ROOT_DEPRESION,
        teacher_hparams=TEACHER_HPARAMS_DEPRESION,
        tabular_datasets={
            "5s_opensmile_librosa": FEATURES_DEPRESION_5S,
            "egemaps":              FEATURES_DEPRESION_EGEMAPS,
        },
    ),
}


# =============================================================================
# 1. CARGA DE EMBEDDINGS POR PACIENTE (idéntico a dl_script_bilstm_final.py)
# =============================================================================

def load_sequence_embeddings(embeddings_root: str, model_key: str):
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}' en {embeddings_root}")

    patient_data_raw = defaultdict(list)
    for entry in entries:
        pid = entry["audio_id"]
        patient_data_raw[pid].append({
            "segment_id": int(entry.get("segment_id", 0)),
            "file":       entry["file"],
            "label":      int(entry["class_id"]),
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

    max_T     = max(len(v["seq"]) for v in patient_data.values())
    embed_dim = len(next(iter(patient_data.values()))["seq"][0])

    X, y, ids, lengths = [], [], [], []
    for pid, data in sorted(patient_data.items()):
        seq      = np.array(data["seq"], dtype=np.float32)
        real_len = len(seq)
        if real_len < max_T:
            pad = np.zeros((max_T - real_len, embed_dim), dtype=np.float32)
            seq = np.vstack([seq, pad])
        X.append(seq)
        y.append(data["label"])
        ids.append(pid)
        lengths.append(real_len)

    X = np.array(X)
    y = np.array(y, dtype=np.int64)
    logger.info(f"   Pacientes cargados: {len(y)} | shape={X.shape} | "
                f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}")
    return X, y, ids, lengths


def scale_sequences(X_train, X_val, lengths_train):
    B_tr, T, D = X_train.shape
    B_vl       = X_val.shape[0]
    real_vecs  = np.vstack([X_train[i, :lengths_train[i], :] for i in range(B_tr)])
    scaler = StandardScaler()
    scaler.fit(real_vecs)
    X_tr_s = scaler.transform(X_train.reshape(-1, D)).reshape(B_tr, T, D)
    X_vl_s = scaler.transform(X_val.reshape(-1, D)).reshape(B_vl, T, D)
    return X_tr_s, X_vl_s


def mixup_sequences(X_train, y_train, lengths_train, alpha=0.4, seed=SEED):
    """
    Mixup intra-clase minoritaria (idéntico a dl_script_bilstm_final.py).
    Balancea la clase 1 generando secuencias sintéticas por combinación
    convexa de pares de la clase minoritaria. Solo se aplica a TRAIN.
    """
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == 0)[0]
    n_syn   = len(idx_neg) - len(idx_pos)

    if len(idx_pos) < 2 or n_syn <= 0:
        return X_train, y_train, lengths_train

    rng = np.random.default_rng(seed)
    X_syn, len_syn = [], []
    for _ in range(n_syn):
        i, j = rng.choice(idx_pos, size=2, replace=False)
        lam  = rng.beta(alpha, alpha)
        X_syn.append(lam * X_train[i] + (1 - lam) * X_train[j])
        len_syn.append(max(lengths_train[i], lengths_train[j]))

    X_out   = np.concatenate([X_train, np.array(X_syn, dtype=np.float32)], axis=0)
    y_out   = np.concatenate([y_train, np.ones(n_syn, dtype=np.int64)])
    len_out = list(lengths_train) + len_syn
    return X_out, y_out, len_out


# =============================================================================
# 2. BiLSTM (idéntico a dl_script_bilstm_final.py)
# =============================================================================

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


def train_teacher_fold(X_train, y_train, lengths_train, X_val, y_val, lengths_val,
                        hparams, fold_seed=SEED):
    """
    Entrena un BiLSTM Teacher SOLO con los pacientes de train de este fold
    y devuelve las probabilidades OOF para los pacientes de val (nunca
    vistos durante el entrenamiento). Early stopping por val loss
    (protocolo limpio, igual que dl_script_bilstm_final.py).
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    input_dim = X_train.shape[2]

    model = BiLSTMFlexible(input_dim, hidden=hparams["hidden"], pooling=hparams["pooling"],
                            dropout_p=hparams["dropout_p"]).to(DEVICE)

    ds_train = SequenceDataset(X_train, y_train, lengths_train)
    ds_val   = SequenceDataset(X_val, y_val, lengths_val)
    loader_tr  = DataLoader(ds_train, batch_size=TEACHER_BATCH, shuffle=True,
                             drop_last=(len(ds_train) % TEACHER_BATCH == 1))
    loader_val = DataLoader(ds_val, batch_size=TEACHER_BATCH, shuffle=False)

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = optim.AdamW(model.parameters(), lr=hparams.get("lr", 1e-3), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS)

    best_val_loss = np.inf
    best_state = None
    no_improve = 0

    for epoch in range(TEACHER_EPOCHS):
        model.train()
        for Xb, yb, lb in loader_tr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb, lb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        _, _, val_loss = eval_loader(model, loader_val, criterion=criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= TEACHER_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs, val_labels, _ = eval_loader(model, loader_val)
    auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else np.nan
    return val_probs, val_labels, auc


# =============================================================================
# 3. GENERAR p_teacher_OOF A NIVEL DE PACIENTE (FASE 4 de la guía)
# =============================================================================

def generate_teacher_oof(condition_name: str, cfg: dict):
    """
    Genera predicciones OOF del Teacher usando RepeatedStratifiedKFold
    (N_FOLDS x N_REPEATS), igual que el resto del paper. Cada repetición es
    una partición independiente de los pacientes en N_FOLDS folds; el Teacher
    se re-entrena en cada uno de los N_FOLDS x N_REPEATS splits.

    Devuelve un DataFrame con una fila por (paciente, repeat) -- es decir,
    cada paciente aparece N_REPEATS veces, cada vez con su fold y su
    p_teacher_OOF correspondientes a esa repetición.
    """
    logger.info(f"[{condition_name}] Cargando embeddings ({TEACHER_MODEL_KEY})...")
    X, y, patient_ids, lengths = load_sequence_embeddings(cfg["embeddings_root"], TEACHER_MODEL_KEY)

    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)

    rows = []
    fold_aucs = []

    for split_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        repeat_num = split_idx // N_FOLDS + 1
        fold_num   = split_idx % N_FOLDS + 1

        X_tr, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr, y_vl = y[train_idx].copy(), y[val_idx].copy()
        l_tr = [lengths[i] for i in train_idx]
        l_vl = [lengths[i] for i in val_idx]

        # Scaler ajustado SOLO con timesteps reales de train de este split
        X_tr, X_vl = scale_sequences(X_tr, X_vl, l_tr)

        # Mixup intra-clase minoritaria (SOLO train) -- forma parte de la
        # config ganadora del Teacher (mixup='fix' en tu summary.csv real)
        if cfg["teacher_hparams"].get("mixup") == "fix":
            X_tr, y_tr, l_tr = mixup_sequences(X_tr, y_tr, l_tr, seed=SEED + split_idx)

        val_probs, val_labels, auc = train_teacher_fold(
            X_tr, y_tr, l_tr, X_vl, y_vl, l_vl,
            hparams=cfg["teacher_hparams"], fold_seed=SEED + split_idx,
        )
        fold_aucs.append(auc)

        for local_i, global_i in enumerate(val_idx):
            rows.append({
                PATIENT_ID_COL: patient_ids[global_i],
                "y_real": int(y[global_i]),
                "repeat": repeat_num,
                "fold": fold_num,
                "p_teacher_OOF": float(val_probs[local_i]),
            })

        if fold_num == N_FOLDS:
            recent = fold_aucs[-N_FOLDS:]
            logger.info(f"   Teacher repeat {repeat_num}/{N_REPEATS} -> "
                        f"AUC medio={np.nanmean(recent):.3f}")

    logger.info(f"[{condition_name}] Teacher OOF AUC global: "
                f"{np.nanmean(fold_aucs):.3f} ± {np.nanstd(fold_aucs):.3f} "
                f"({len(fold_aucs)} evaluaciones = {N_FOLDS} folds x {N_REPEATS} repeats)")

    df_oof = pd.DataFrame(rows)
    return df_oof


# =============================================================================
# 4. CONSTRUIR TARGETS DE DESTILACIÓN Y CORRER STUDENT (LazyPredict)
# =============================================================================

def find_best_threshold(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thrs[np.argmax(f1[:-1])])


_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")


def _normalize_audio_id(value) -> str:
    """
    Normaliza un identificador de audio para poder cruzar el manifest.json
    del Teacher (que puede usar el nombre de carpeta SIN extensión, p.ej.
    '002_recortado_denoised') con el audio_id de los CSV tabulares (que en
    tus muestras SÍ incluye la extensión, p.ej. '002_recortado_denoised.wav').
    Quita extensión de audio conocida, espacios y diferencias de mayúsculas.
    """
    s = str(value).strip()
    low = s.lower()
    for ext in _AUDIO_EXTENSIONS:
        if low.endswith(ext):
            s = s[: -len(ext)]
            break
    return s.strip().lower()


def load_tabular(path: str, df_oof: pd.DataFrame) -> pd.DataFrame:
    """
    Carga un CSV de features tabulares (por segmento) y le añade
    repeat / fold / p_teacher_OOF / y_real haciendo merge por audio_id.

    El merge se hace sobre una versión NORMALIZADA del audio_id (sin
    extensión, sin espacios, en minúsculas) porque el manifest.json del
    Teacher y los CSV tabulares pueden nombrar el mismo audio de forma
    ligeramente distinta (con/sin '.wav').

    df_oof tiene N_REPEATS filas por paciente (una por repetición), así que
    el merge es many-to-many: cada segmento del CSV tabular queda replicado
    N_REPEATS veces, una por cada repetición del Teacher. Esto es intencional
    -- luego se filtra por 'repeat' antes de correr el Student.
    """
    df = pd.read_csv(path)
    if PATIENT_ID_COL not in df.columns:
        raise ValueError(
            f"'{PATIENT_ID_COL}' no está en {path}. Columnas disponibles: "
            f"{list(df.columns)}. Ajusta PATIENT_ID_COL en CONFIG."
        )
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"'{LABEL_COL}' no está en {path}. Columnas disponibles: {list(df.columns)}."
        )
    df = df.drop(columns=[LABEL_COL])  # la etiqueta real se toma de df_oof (y_real), evita duplicados/ambigüedad

    df["_merge_key"] = df[PATIENT_ID_COL].map(_normalize_audio_id)
    df_oof_key = df_oof.copy()
    df_oof_key["_merge_key"] = df_oof_key[PATIENT_ID_COL].map(_normalize_audio_id)
    df_oof_key = df_oof_key.drop(columns=[PATIENT_ID_COL])  # evitar columna duplicada; conservamos el audio_id del CSV tabular

    merged = df.merge(df_oof_key, on="_merge_key", how="inner").drop(columns=["_merge_key"])

    if len(merged) == 0:
        tab_examples = sorted(df[PATIENT_ID_COL].astype(str).unique())[:5]
        oof_examples = sorted(df_oof[PATIENT_ID_COL].astype(str).unique())[:5]
        raise ValueError(
            f"El merge de {path} con el Teacher OOF produjo 0 filas -- los "
            f"audio_id no coinciden ni siquiera tras normalizar (quitar "
            f"extensión/mayúsculas/espacios).\n"
            f"  Ejemplos de audio_id en el CSV tabular: {tab_examples}\n"
            f"  Ejemplos de audio_id del Teacher (manifest.json): {oof_examples}\n"
            f"Revisa manualmente si siguen un patrón distinto (p.ej. prefijos, "
            f"guiones, IDs numéricos) y ajusta _normalize_audio_id() o "
            f"PATIENT_ID_COL en CONFIG."
        )

    n_expected = len(df) * N_REPEATS
    if len(merged) != n_expected:
        logger.warning(f"   {path}: se esperaban {n_expected} filas tras el merge "
                        f"({len(df)} segmentos x {N_REPEATS} repeats), se obtuvieron {len(merged)}. "
                        f"Puede haber audio_id sin embeddings de Teacher (se descartan).")
    return merged


def run_student_experiment(df_tabular_repeat: pd.DataFrame, alpha: float, dataset_name: str,
                            condition_name: str, repeat_num: int, results_accum: list):
    """
    Corre LazyRegressor prediciendo y_KD (target continuo de destilación),
    usando la MISMA partición (fold, repeat) por paciente que usó el Teacher
    para esta repetición. Nunca se evalúa contra p_teacher ni contra y_KD --
    solo contra y_real.

    alpha=1.0  ->  y_KD = y_real  =>  actúa como "baseline sin destilación"
                   (mismo LazyRegressor, sin mezclar con un paradigma de
                   clasificación aparte).
    alpha=0.0  ->  y_KD = p_teacher_OOF  =>  imitación pura del Teacher.

    df_tabular_repeat: filas de UNA sola repetición (ya filtrada por 'repeat').
    """
    feature_cols = [c for c in df_tabular_repeat.columns
                    if c not in ({PATIENT_ID_COL, LABEL_COL, "y_real", "repeat", "fold", "p_teacher_OOF"}
                                 | set(DROP_COLS))]
    feature_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(df_tabular_repeat[c])]

    df_tabular_repeat = df_tabular_repeat.copy()
    df_tabular_repeat["y_KD"] = (alpha * df_tabular_repeat["y_real"]
                                  + (1 - alpha) * df_tabular_repeat["p_teacher_OOF"])
    tipo = "Baseline_regressor" if alpha == 1.0 else "KD_regressor"

    for fold_idx in sorted(df_tabular_repeat["fold"].unique()):
        train_mask = df_tabular_repeat["fold"] != fold_idx
        val_mask   = df_tabular_repeat["fold"] == fold_idx

        X_train = df_tabular_repeat.loc[train_mask, feature_cols].values
        X_val   = df_tabular_repeat.loc[val_mask, feature_cols].values
        y_kd_train = df_tabular_repeat.loc[train_mask, "y_KD"].values
        y_real_val = df_tabular_repeat.loc[val_mask, "y_real"].values

        # Escalado SOLO con train de este fold (sin leakage)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None,
                             predictions=True)
        try:
            models_reg, preds_reg = reg.fit(X_train_s, X_val_s, y_kd_train,
                                             np.zeros(len(X_val_s)))  # y_val dummy (se re-evalúa abajo contra y_real)
        except Exception as e:
            logger.warning(f"   LazyRegressor falló ({dataset_name}, alpha={alpha}, "
                            f"repeat={repeat_num}, fold={fold_idx}): {e}")
            continue

        thr = find_best_threshold(df_tabular_repeat.loc[train_mask, "y_real"].values, y_kd_train)

        for model_name in preds_reg.columns:
            p_student = preds_reg[model_name].values
            p_student_clipped = np.clip(p_student, 0, 1)
            y_pred_class = (p_student_clipped >= thr).astype(int)

            row = {
                "condicion": condition_name,
                "dataset_tabular": dataset_name,
                "alpha": alpha,
                "repeat": repeat_num,
                "fold": fold_idx,
                "modelo_student": model_name,
                "tipo": tipo,
                "n_train": len(X_train_s),
                "n_val": len(X_val_s),
                "auc": (roc_auc_score(y_real_val, p_student_clipped)
                        if len(np.unique(y_real_val)) > 1 else np.nan),
                "f1_thr_train": f1_score(y_real_val, y_pred_class, zero_division=0),
                "acc_thr_train": accuracy_score(y_real_val, y_pred_class),
                "threshold_train": thr,
            }
            results_accum.append(row)

        logger.info(f"   [{condition_name}/{dataset_name}] alpha={alpha} repeat={repeat_num} "
                    f"fold={fold_idx} listo ({len(preds_reg.columns)} modelos regresores)")


# =============================================================================
# 5. PIPELINE PRINCIPAL
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for condition_name, cfg in CONDITIONS.items():
        logger.info("=" * 70)
        logger.info(f"CONDICIÓN: {condition_name.upper()}")
        logger.info("=" * 70)

        # FASE 1: OOF del Teacher a nivel de paciente (una sola vez por condición)
        df_oof = generate_teacher_oof(condition_name, cfg)
        oof_path = OUTPUT_DIR / f"teacher_oof_{condition_name}.csv"
        df_oof.to_csv(oof_path, index=False)
        logger.info(f"   Teacher OOF guardado en {oof_path}")

        # FASE 2: Student (LazyPredict) por dataset tabular x alpha x repeat x fold
        for dataset_name, path in cfg["tabular_datasets"].items():
            logger.info(f"-- Dataset tabular: {dataset_name} ({path})")
            df_tab = load_tabular(path, df_oof)

            for repeat_num in range(1, N_REPEATS + 1):
                df_tab_repeat = df_tab[df_tab["repeat"] == repeat_num]
                for alpha in ALPHA_GRID:
                    run_student_experiment(df_tab_repeat, alpha, dataset_name,
                                            condition_name, repeat_num, all_results)

    df_results = pd.DataFrame(all_results)
    results_path = OUTPUT_DIR / "kd_lazypredict_raw_results.csv"
    df_results.to_csv(results_path, index=False)
    logger.info(f"\nResultados crudos guardados en {results_path}")

    if df_results.empty:
        logger.warning("No se generó ningún resultado del Student (all_results está vacío). "
                        "Revisa los logs anteriores -- probablemente el merge Teacher<->tabular "
                        "falló para todas las condiciones/datasets. No se generará resumen.")
        return

    # Resumen: media por modelo/dataset/alpha/condición a través de los
    # N_FOLDS x N_REPEATS = 50 evaluaciones
    summary = (
        df_results
        .groupby(["condicion", "dataset_tabular", "tipo", "alpha", "modelo_student"], dropna=False)
        .agg(mean_auc=("auc", "mean"), std_auc=("auc", "std"),
             mean_f1=("f1_thr_train", "mean"), mean_acc=("acc_thr_train", "mean"),
             n_evaluaciones=("auc", "count"))
        .reset_index()
        .sort_values("mean_auc", ascending=False)
    )
    summary_path = OUTPUT_DIR / "kd_lazypredict_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Resumen guardado en {summary_path}")
    logger.info("\nTOP 20 configuraciones (por AUC medio):")
    logger.info("\n" + summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()