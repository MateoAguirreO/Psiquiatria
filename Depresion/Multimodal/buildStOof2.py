"""
==============================================================================
build_oof_matrix.py
==============================================================================
Genera la matriz Out-Of-Fold (OOF) para fusión multimodal (stacking) de
4 modelos de clasificación de depresión, sobre EXACTAMENTE el mismo conjunto
de pacientes y los mismos folds definidos en folds_manifest_v2.csv
(5 folds x 10 repeats, esquema de 3 vías train_fit/val_early/test).

Modelos (columnas de la matriz final):
  1. prob_extratrees_embeddings   -> ExtraTreesClassifier sobre embeddings de
                                      audio (wav2vec2-large-robust), agregados
                                      a nivel de paciente.
  2. prob_bilstm_embeddings       -> BiLSTM sobre SECUENCIAS de embeddings de
                                      audio (wav2vec2-large-robust).
  3. prob_destilacion_estudiante  -> Teacher BiLSTM (misma arquitectura que #2)
                                      -> Student AdaBoostRegressor sobre
                                      eGeMAPS. Se puede desactivar con
                                      INCLUDE_DESTILACION = False más abajo.
  4. prob_bigru_video             -> BiGRU sobre secuencias de Action Units.

──────────────────────────────────────────────────────────────────────────
CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR
──────────────────────────────────────────────────────────────────────────
1) BILSTM_BEST_CFG ya NO está hardcodeado. Cada vez que vuelvas a correr tu
   barrido de hiperparámetros (dl_script2e_bilstm_round4_fixed.py o el que
   uses), el "mejor" AUC cambia (la última corrida dio ~0.70 en vez del
   0.7659 anterior) -- así que este script LEE el summary.csv de ese barrido
   y toma la fila de mayor mean_auc en tiempo de ejecución. Ver
   BILSTM_SUMMARY_CSV en CONFIG. Si prefieres fijar un config a mano, pasa
   BILSTM_FORCE_CFG con un dict y se usa ese en vez de leer el csv.

2) INCLUDE_DESTILACION: interruptor único para correr con o sin el modelo
   de destilación. Se recomienda correr el pipeline completo UNA VEZ con
   True y otra con False, y comparar el AUC de fusión resultante en
   late_fusion_nested.py (que ya hace esa comparación automáticamente si
   ambas matrices existen) -- no decidas si "vale la pena" a priori, la
   decisión es empírica y toma 2 corridas.

──────────────────────────────────────────────────────────────────────────
REGLA NO NEGOCIABLE (sigue igual)
──────────────────────────────────────────────────────────────────────────
Los 4 modelos leen folds_manifest_v2.csv y usan EXACTAMENTE esos
train_fit/val_early/test por (repeat, fold). Ninguno genera folds propios.

──────────────────────────────────────────────────────────────────────────
!! ANTES DE CORRER: AJUSTA CONFIG MÁS ABAJO !!
──────────────────────────────────────────────────────────────────────────
- Rutas de embeddings, features eGeMAPS y sequences_au.
- subject_id_to_audio_id_prefix() / subject_id_to_video_id() si tu
  convención de nombres de archivo no es "subject_id con padding de ceros".
El script falla ruidosamente (assert) si el mapeo no encuentra los 79
pacientes en alguna fuente, en vez de continuar silenciosamente con menos.
==============================================================================
"""

import copy
import glob
import json
import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S", stream=sys.stdout, force=True)
logger = logging.getLogger("stacking_oof")


def _log(msg):
    logger.info(msg)
    sys.stdout.flush()


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

N_CPU_WORKERS = max(1, (os.cpu_count() or 4) - 1)
N_GPU_WORKERS = max(2, N_GPUS * 4) if N_GPUS > 0 else max(1, (os.cpu_count() or 4) // 2)
ET_N_JOBS = max(1, (os.cpu_count() or 4) // N_CPU_WORKERS)

if DEVICE.type == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // N_GPU_WORKERS))
else:
    torch.backends.cudnn.benchmark = True


def run_tasks_parallel(tasks, worker_fn, n_workers, label):
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

# Rutas relativas al ARCHIVO, no al directorio desde el que lo ejecutes --
# así "python /ruta/al/script.py" desde cualquier cwd sigue encontrando los
# CSV que viven junto a este script en Depresion/Multimodal/. Si tus CSV
# están en otro lugar, reemplaza estas dos líneas por rutas absolutas.
_SCRIPT_DIR = Path(__file__).resolve().parent
PACIENTES_FINALES_CSV = str(_SCRIPT_DIR / "pacientes_finales.csv")
FOLDS_MANIFEST_CSV = str(_SCRIPT_DIR / "folds_manifest_v2.csv")

EMBEDDINGS_DIR = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Depresión/embeddings_v2/"
EMB_MODEL = "wav2vec2-large-robust"

sys.path.append("/home/ci2dt2-ai/Proyectos/Psiquiatria")
try:
    from embedding_pipeline_5s import load_embeddings_for_classification
except ImportError as e:
    print(f"NO se pudo importar embedding_pipeline_5s: {e}", flush=True)
    load_embeddings_for_classification = None

FEATURES_DEPRESION_EGEMAPS = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/MachineLearning/features_depresion_egemaps.csv"
PATIENT_ID_COL = "audio_id"
LABEL_COL = "label"
DROP_COLS = ["seg_idx"]

SEQUENCES_AU_DIR = "/home/ci2dt2-ai/Proyectos/Psiquiatria/sequences_au"

OUTPUT_DIR = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Multimodal/stacking_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REUSE_BILSTM_AS_TEACHER = True

# --- NUEVO: toggle único para incluir/excluir la destilación ---------------
INCLUDE_DESTILACION = True  # corre el pipeline True y False, compara con
                             # late_fusion_nested.py, y decide con datos.

# --- NUEVO: config del BiLSTM se lee del summary de tu último barrido ------
# Debe tener al menos las columnas: emb_model, hidden, pooling, loss, mixup,
# warmup, dropout_p, lr, mean_auc (el formato que ya generan tus scripts
# dl_script*_bilstm*.py). Si tu columna de loss usó gamma focal en vez de
# bce_posw, se propaga igual (ver loss_type más abajo).
BILSTM_SUMMARY_CSV = (
    "/home/ci2dt2-ai/Proyectos/Psiquiatria/Depresion/Wav2Vec/results_DL/"
    "embeddings_nuevos/dl_script2e_bilstm_round4_fixed/summary_bilstm_r4_fixed.csv"
)
BILSTM_FORCE_CFG = None  # dict opcional para saltarte la lectura del csv

BILSTM_EPOCHS, BILSTM_PATIENCE, BILSTM_BATCH = 60, 8, 16

KD_DATASET_NAME = "egemaps"
KD_ALPHA = 0.0
KD_STUDENT = "AdaBoostRegressor"

BIGRU_HIDDEN, BIGRU_DROPOUT = 24, 0.5
BIGRU_BATCH, BIGRU_PATIENCE, BIGRU_MAX_EPOCHS = 8, 8, 60
BIGRU_LR, BIGRU_WD, BIGRU_MAX_SEQ_LEN = 1e-3, 1e-3, 600


def load_best_bilstm_cfg_from_summary(summary_csv: str, emb_model: str) -> dict:
    """Lee el summary de tu barrido de hiperparámetros y toma la fila con
    mayor mean_auc para emb_model. Evita hardcodear un config que queda
    obsoleto cada vez que vuelves a correr el barrido (como pasó: 0.7659 ->
    ~0.70 en la última corrida)."""
    if BILSTM_FORCE_CFG is not None:
        _log(f"BiLSTM: usando BILSTM_FORCE_CFG fijado a mano: {BILSTM_FORCE_CFG}")
        return BILSTM_FORCE_CFG

    df = pd.read_csv(summary_csv)
    df = df[df["emb_model"] == emb_model] if "emb_model" in df.columns else df
    if df.empty:
        raise ValueError(f"{summary_csv}: no hay filas para emb_model='{emb_model}'.")

    best = df.loc[df["mean_auc"].idxmax()]
    cfg = dict(
        hidden=int(best["hidden"]),
        pooling=str(best["pooling"]),
        loss=str(best["loss"]),
        mixup=str(best["mixup"]),
        warmup=bool(best["warmup"]),
        dropout_p=float(best["dropout_p"]),
        lr=float(best.get("lr", 1e-3)),
    )
    _log(f"BiLSTM: mejor config leído de {summary_csv} -> {cfg} "
         f"(mean_auc={best['mean_auc']:.4f}, std_auc={best.get('std_auc', float('nan')):.4f})")
    return cfg


def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_SEED_LOCK = threading.Lock()


# ==============================================================================
# 1. Cargar pacientes y folds_manifest tal cual (NO regenerar)
# ==============================================================================

def load_manifest_and_patients():
    _log(f"Leyendo pacientes desde: {PACIENTES_FINALES_CSV}")
    _log(f"Leyendo manifiesto desde: {FOLDS_MANIFEST_CSV}")
    pacientes = pd.read_csv(PACIENTES_FINALES_CSV)
    manifest = pd.read_csv(FOLDS_MANIFEST_CSV)

    assert set(manifest.columns) >= {"repeat", "fold", "subject_id", "split"}, (
        f"Columnas encontradas: {list(manifest.columns)}. "
        f"¿Seguro que FOLDS_MANIFEST_CSV apunta al archivo correcto?"
    )
    splits_found = set(manifest.split.unique())
    assert splits_found == {"train_fit", "val_early", "test"}, (
        f"Se esperaban los splits ['train_fit','val_early','test'] pero "
        f"{FOLDS_MANIFEST_CSV} tiene {sorted(splits_found)}. Esto casi "
        f"siempre significa que se está leyendo un manifest.csv viejo/de "
        f"otro experimento (2 vías train/val) o el archivo equivocado -- "
        f"revisa la ruta arriba impresa, no solo el nombre del archivo."
    )
    assert len(pacientes) == 79, f"Se esperaban 79 pacientes, hay {len(pacientes)}"
    assert set(manifest.subject_id.unique()) == set(pacientes.subject_id.unique())

    n_repeats = manifest.repeat.nunique()
    n_folds = manifest.fold.nunique()
    for r in manifest.repeat.unique():
        test_ids = manifest[(manifest.repeat == r) & (manifest.split == "test")].subject_id
        assert test_ids.nunique() == 79 and len(test_ids) == 79, \
            f"repeat={r}: cada paciente debe aparecer exactamente 1 vez en test."
    for (r, f), g in manifest.groupby(["repeat", "fold"]):
        test_set = set(g.loc[g.split == "test", "subject_id"])
        other_set = set(g.loc[g.split != "test", "subject_id"])
        assert not (test_set & other_set), f"repeat={r} fold={f}: fuga train/test."

    logger.info(f"Manifiesto OK: {n_repeats} repeticiones x {n_folds} folds, 79 pacientes.")
    return pacientes, manifest, n_repeats, n_folds


def get_fold_subjects(manifest, repeat, fold):
    sub = manifest[(manifest.repeat == repeat) & (manifest.fold == fold)]
    train_fit_ids = sub.loc[sub.split == "train_fit", "subject_id"].tolist()
    val_early_ids = sub.loc[sub.split == "val_early", "subject_id"].tolist()
    test_ids = sub.loc[sub.split == "test", "subject_id"].tolist()
    return train_fit_ids, val_early_ids, test_ids


def get_fold_subjects_combined(manifest, repeat, fold):
    train_fit_ids, val_early_ids, test_ids = get_fold_subjects(manifest, repeat, fold)
    return train_fit_ids + val_early_ids, test_ids


# ==============================================================================
# 2. MAPEO subject_id (int) <-> audio_id / video_id (str)
#    !! AJUSTA ESTO A TU CONVENCIÓN REAL ANTES DE CORRER !!
# ==============================================================================

def subject_id_to_audio_id_prefix(subject_id: int) -> str:
    return f"{subject_id:03d}"


def subject_id_to_video_id(subject_id: int) -> str:
    return str(subject_id)


def build_audio_id_map(available_audio_ids, subject_ids):
    mapping = {}
    for sid in subject_ids:
        prefix = subject_id_to_audio_id_prefix(sid)
        matches = [a for a in available_audio_ids if str(a).startswith(prefix)]
        assert len(matches) == 1, (
            f"subject_id={sid}: se esperaba 1 audio_id con prefijo '{prefix}', "
            f"se encontraron {len(matches)}: {matches}."
        )
        mapping[sid] = matches[0]
    return mapping


def build_video_id_map(available_video_ids, subject_ids):
    available_norm = {str(v): v for v in available_video_ids}
    mapping = {}
    for sid in subject_ids:
        vid = subject_id_to_video_id(sid)
        assert vid in available_norm, f"subject_id={sid}: no se encontró video_id '{vid}'."
        mapping[sid] = available_norm[vid]
    return mapping


# ==============================================================================
# 3. MODELO 1 -- ExtraTreesClassifier sobre embeddings de audio (agregados)
# ==============================================================================

ET_BEST_PARAMS = dict(
    class_weight=None, criterion="gini", max_depth=None, max_features="log2",
    min_samples_leaf=1, min_samples_split=2, n_estimators=200,
)


def run_extratrees_oof(pacientes, manifest, n_repeats, n_folds) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("MODELO 1/4: ExtraTreesClassifier sobre embeddings de audio")
    logger.info("=" * 70)

    if load_embeddings_for_classification is None:
        raise ImportError("Ajusta sys.path / EMBEDDINGS_DIR en CONFIG.")

    X_mat, y_mat, ids = load_embeddings_for_classification(EMBEDDINGS_DIR, EMB_MODEL, condition="depresion")
    X = pd.DataFrame(X_mat)
    y = pd.Series(y_mat)
    ids = list(ids)

    subject_ids = pacientes.subject_id.tolist()
    audio_map = build_audio_id_map(ids, subject_ids)
    id_to_row = {aid: i for i, aid in enumerate(ids)}

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
    return pd.DataFrame(rows)


# ==============================================================================
# 4. MODELO 2 -- BiLSTM sobre secuencias de embeddings de audio
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


def scale_sequences_multi(X_train, lengths_train, *others):
    B_tr, T, D = X_train.shape
    real_vecs = np.vstack([X_train[i, :lengths_train[i], :] for i in range(B_tr)])
    scaler = StandardScaler()
    scaler.fit(real_vecs)
    X_tr_s = scaler.transform(X_train.reshape(-1, D)).reshape(B_tr, T, D)
    others_s = tuple(
        scaler.transform(arr.reshape(-1, D)).reshape(arr.shape[0], T, D) for arr in others
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


def train_bilstm_fold(X_train, y_train, lengths_train, X_valearly, y_valearly, lengths_valearly,
                       X_test, y_test, lengths_test, cfg: dict, fold_seed: int):
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

        _, _, val_loss = eval_loader(model, loader_valearly, criterion=criterion)
        if val_loss < best_val_loss:
            best_val_loss, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= BILSTM_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_probs, test_labels, _ = eval_loader(model, loader_test)
    return test_probs, test_labels


def run_bilstm_oof(pacientes, manifest, n_repeats, n_folds, bilstm_cfg: dict) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("MODELO 2/4: BiLSTM sobre secuencias de embeddings de audio")
    logger.info(f"Config usado: {bilstm_cfg}")
    logger.info("=" * 70)

    X, y, ids, lengths = load_sequence_embeddings(EMBEDDINGS_DIR, EMB_MODEL)
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

        X_tr, X_ve, X_te = scale_sequences_multi(X_tr, l_tr, X_ve, X_te)
        if bilstm_cfg["mixup"] == "fix":
            X_tr, y_tr, l_tr = mixup_sequences(X_tr, y_tr, l_tr, seed=SEED + repeat * n_folds + fold)

        test_probs, test_labels = train_bilstm_fold(
            X_tr, y_tr, l_tr, X_ve, y_ve, l_ve, X_te, y_te, l_te,
            cfg=bilstm_cfg, fold_seed=SEED + repeat * n_folds + fold,
        )
        return [{"subject_id": sid, "repeat": repeat, "fold": fold,
                  "prob_bilstm_embeddings": float(p)}
                for sid, p in zip(test_sub, test_probs)]

    tasks = [(r, f) for r in range(n_repeats) for f in range(n_folds)]
    rows = run_tasks_parallel(tasks, _fold_worker, N_GPU_WORKERS, "BiLSTM")
    return pd.DataFrame(rows)


# ==============================================================================
# 5. MODELO 3 -- Destilación: Teacher (BiLSTM, reusa modelo 2) -> Student
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


def run_destilacion_oof(pacientes, manifest, n_repeats, n_folds, teacher_oof_df) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("MODELO 3/4: Destilación (Teacher BiLSTM -> Student AdaBoostRegressor, eGeMAPS)")
    logger.info("=" * 70)

    if not REUSE_BILSTM_AS_TEACHER:
        raise NotImplementedError("Implementa aquí un Teacher independiente si lo necesitas.")

    teacher_oof = teacher_oof_df.rename(columns={"prob_bilstm_embeddings": "p_teacher_OOF"})

    df_tab = pd.read_csv(FEATURES_DEPRESION_EGEMAPS)
    assert PATIENT_ID_COL in df_tab.columns and LABEL_COL in df_tab.columns
    df_tab = df_tab.drop(columns=[LABEL_COL])
    df_tab["_merge_key"] = df_tab[PATIENT_ID_COL].map(_normalize_audio_id)

    subject_ids = pacientes.subject_id.tolist()
    tab_audio_ids = df_tab[PATIENT_ID_COL].unique().tolist()
    audio_map = build_audio_id_map(tab_audio_ids, subject_ids)
    subj_lookup = {aid: sid for sid, aid in audio_map.items()}

    df_tab["subject_id"] = df_tab[PATIENT_ID_COL].map(subj_lookup)
    assert df_tab["subject_id"].notna().all()

    feature_cols = [c for c in df_tab.columns
                    if c not in ({PATIENT_ID_COL, "subject_id", "_merge_key"} | set(DROP_COLS))]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_tab[c])]

    df_r_by_repeat = {}
    for repeat in range(n_repeats):
        oof_r = teacher_oof[teacher_oof.repeat == repeat][["subject_id", "p_teacher_OOF"]]
        df_r = df_tab.merge(oof_r, on="subject_id", how="inner")
        df_r["y_KD"] = (KD_ALPHA * pacientes.set_index("subject_id").loc[df_r.subject_id, "target_depresion"].values
                         + (1 - KD_ALPHA) * df_r["p_teacher_OOF"])
        df_r_by_repeat[repeat] = df_r

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

        df_test = df_r.loc[test_mask, ["subject_id"]].copy()
        df_test["p_student"] = p_student_seg
        per_patient = df_test.groupby("subject_id")["p_student"].mean()

        return [{"subject_id": sid, "repeat": repeat, "fold": fold,
                  "prob_destilacion_estudiante": float(p)}
                for sid, p in per_patient.items()]

    tasks = [(r, f) for r in range(n_repeats) for f in range(n_folds)]
    rows = run_tasks_parallel(tasks, _fold_worker, N_CPU_WORKERS, "Destilación")
    return pd.DataFrame(rows)


# ==============================================================================
# 6. MODELO 4 -- BiGRU sobre secuencias de Action Units de video
# ==============================================================================

NON_FEATURE_COLS = {"video_id", "frame_order", "timestamp_ms", "frame", "approx_time", "input"}


def _normalize_video_id(vid) -> str:
    vid = str(vid).strip()
    try:
        return str(int(float(vid)))
    except (ValueError, TypeError):
        return vid


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


def run_bigru_video_oof(pacientes, manifest, n_repeats, n_folds) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("MODELO 4/4: BiGRU sobre secuencias de Action Units (video)")
    logger.info("=" * 70)

    sequences = load_au_sequences(SEQUENCES_AU_DIR)
    candidate_cols = list_candidate_columns(sequences)

    subject_ids = pacientes.subject_id.tolist()
    video_map = build_video_id_map(sequences.keys(), subject_ids)
    label_map = {video_map[sid]: int(pacientes.set_index("subject_id").loc[sid, "target_depresion"])
                 for sid in subject_ids}

    device = DEVICE

    def _fold_worker(repeat, fold):
        train_sub, valearly_sub, test_sub = get_fold_subjects(manifest, repeat, fold)
        train_ids = [video_map[s] for s in train_sub]
        valearly_ids = [video_map[s] for s in valearly_sub]
        test_ids = [video_map[s] for s in test_sub]

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

        with _SEED_LOCK:
            set_global_seed(SEED + repeat * n_folds + fold)
            model = BiGRUClassifier(len(feature_cols), hidden_size=BIGRU_HIDDEN, dropout=BIGRU_DROPOUT).to(device)

        test_probs = train_bigru_fold(model, train_loader, valearly_loader, test_loader, device,
                                       max_epochs=BIGRU_MAX_EPOCHS, patience=BIGRU_PATIENCE,
                                       lr=BIGRU_LR, weight_decay=BIGRU_WD)

        return [{"subject_id": sid, "repeat": repeat, "fold": fold, "prob_bigru_video": float(p)}
                for sid, p in zip(test_sub, test_probs)]

    tasks = [(r, f) for r in range(n_repeats) for f in range(n_folds)]
    rows = run_tasks_parallel(tasks, _fold_worker, N_GPU_WORKERS, "BiGRU video")
    return pd.DataFrame(rows)


# ==============================================================================
# 7. Ensamblar la matriz final
# ==============================================================================

def build_final_matrix(pacientes, et_df, bilstm_df, kd_df, bigru_df, include_destilacion: bool):
    dfs = [et_df, bilstm_df]
    prob_cols = ["prob_extratrees_embeddings", "prob_bilstm_embeddings"]
    if include_destilacion:
        dfs.append(kd_df)
        prob_cols.append("prob_destilacion_estudiante")
    dfs.append(bigru_df)
    prob_cols.append("prob_bigru_video")

    detail = dfs[0]
    for d in dfs[1:]:
        detail = detail.merge(d, on=["subject_id", "repeat", "fold"], how="outer")

    detail = detail.merge(pacientes[["subject_id", "target_depresion"]], on="subject_id", how="left")
    detail = detail.rename(columns={"target_depresion": "y_true"})

    final = detail.groupby("subject_id")[prob_cols].mean().reset_index()
    final = final.merge(pacientes[["subject_id", "target_depresion"]], on="subject_id", how="left")
    final = final.rename(columns={"target_depresion": "y_true"})
    final = final.sort_values("subject_id").reset_index(drop=True)

    return final, detail, prob_cols


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    logger.info(f"INCLUDE_DESTILACION = {INCLUDE_DESTILACION}")
    pacientes, manifest, n_repeats, n_folds = load_manifest_and_patients()

    bilstm_cfg = load_best_bilstm_cfg_from_summary(BILSTM_SUMMARY_CSV, EMB_MODEL)

    et_df = run_extratrees_oof(pacientes, manifest, n_repeats, n_folds)
    bilstm_df = run_bilstm_oof(pacientes, manifest, n_repeats, n_folds, bilstm_cfg)

    kd_df = pd.DataFrame(columns=["subject_id", "repeat", "fold", "prob_destilacion_estudiante"])
    if INCLUDE_DESTILACION:
        kd_df = run_destilacion_oof(pacientes, manifest, n_repeats, n_folds, teacher_oof_df=bilstm_df)

    bigru_df = run_bigru_video_oof(pacientes, manifest, n_repeats, n_folds)

    final_matrix, detail, prob_cols = build_final_matrix(
        pacientes, et_df, bilstm_df, kd_df, bigru_df, INCLUDE_DESTILACION
    )

    suffix = "con_kd" if INCLUDE_DESTILACION else "sin_kd"
    final_path = OUTPUT_DIR / f"oof_matrix_final_{suffix}.csv"
    detail_path = OUTPUT_DIR / f"oof_matrix_detail_by_repeat_{suffix}.csv"
    final_matrix.to_csv(final_path, index=False)
    detail.to_csv(detail_path, index=False)

    logger.info("=" * 70)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 70)
    logger.info(f"Matriz final ({len(final_matrix)} pacientes) -> {final_path}")
    logger.info(f"Detalle por repeat -> {detail_path}")
    for c in prob_cols:
        n_nan = final_matrix[c].isna().sum()
        if n_nan:
            logger.warning(f"  ⚠ {c}: {n_nan} pacientes sin probabilidad (revisar mapeo de IDs)")


if __name__ == "__main__":
    main()