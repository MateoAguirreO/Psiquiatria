"""
DL Script 2 — Barrido de Arquitecturas sobre Secuencias Temporales — VERSIÓN CORREGIDA
FIX-3WAY: División de 3 vías (FIT/CALIB/VAL) dentro de cada fold para eliminar data leakage.
- FIT: Para entrenar (backpropagation)
- CALIB: Para early stopping y búsqueda de threshold
- VAL: Para evaluación final e imparcial (NUNCA se usa para decisiones)

Arquitecturas:
· CNN-1D       — patrones locales en la secuencia de segmentos
· BiLSTM       — dependencias temporales bidireccionales (LSTM)
· BiGRU        — dependencias temporales bidireccionales (GRU, más ligero)
· CNN + BiGRU  — CNN extrae features locales, GRU integra la secuencia

Protocolo:
· RepeatedStratifiedKFold(5, 10) = 50 evaluaciones por config
· Dentro de cada fold: 75% FIT / 25% CALIB / 20% VAL externo
· Todo preprocesamiento (scaler, PCA) ajustado SOLO en FIT
· Mixup aplicado SOLO a FIT
· Early stopping monitorea CALIB (nunca VAL)
· Threshold buscado en CALIB (nunca VAL)
· AUC/F1 reportados solo en VAL (una sola vez)
"""
import json
import logging
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_curve

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("DL_S2_FIXED")

# ─── Configuración ─────────────────────────────────────────────────────────
MODELS_TO_TEST = [
    "xlsr-300m", "xlsr-53", "whisper-large-encoder",
    "wav2vec2-large-robust", "wavlm-large", "hubert-large",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CV_SPLITS   = 5
CV_REPEATS  = 10
EPOCHS      = 60
BATCH_SIZE  = 16
PATIENCE    = 8
PCA_VAR     = 0.95
MIXUP_ALPHA = 0.4
CALIB_FRAC  = 0.25  # Fracción del train reservada para CALIB

EXPERIMENTS = [
    {"id": "raw",        "pca": False, "mixup": False},
    {"id": "pca",        "pca": True,  "mixup": False},
    {"id": "raw_mixup",  "pca": False, "mixup": True},
    {"id": "pca_mixup",  "pca": True,  "mixup": True},
]

# =============================================================================
# 1. CARGA DE SECUENCIAS
# =============================================================================

def load_sequence_embeddings(embeddings_root: str, model_key: str):
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"Sin embeddings para '{model_key}'.")
    
    patient_data = {}
    label_map = {}
    for entry in entries:
        pid = entry["audio_id"]
        if pid not in label_map:
            label_map[pid] = int(entry["class_id"])
        if pid not in patient_data:
            patient_data[pid] = []
        patient_data[pid].append(entry["file"])
    
    max_T = 0
    embed_dim = None
    seqs_by_pid = {}
    for pid in sorted(patient_data.keys()):
        seq = []
        for fpath in patient_data[pid]:
            with open(fpath) as f:
                record = json.load(f)
            emb = record if isinstance(record, list) else record["embedding"]
            seq.append(np.array(emb, dtype=np.float32))
        seqs_by_pid[pid] = seq
        max_T = max(max_T, len(seq))
        if embed_dim is None:
            embed_dim = len(seq[0])
    
    X, y, ids, lengths = [], [], [], []
    for pid in sorted(seqs_by_pid.keys()):
        seq = np.array(seqs_by_pid[pid], dtype=np.float32)
        real_len = len(seq)
        if real_len < max_T:
            pad = np.zeros((max_T - real_len, embed_dim), dtype=np.float32)
            seq = np.vstack([seq, pad])
        X.append(seq)
        y.append(label_map[pid])
        ids.append(pid)
        lengths.append(real_len)
    
    X = np.array(X)
    y = np.array(y, dtype=np.int64)
    logger.info(f"   Cargados: {len(y)} pacientes | shape={X.shape} | "
                f"clase0={np.sum(y==0)} | clase1={np.sum(y==1)}")
    return X, y, ids, lengths

# =============================================================================
# 2. PREPROCESAMIENTO (ajustado SOLO en FIT)
# =============================================================================

def scale_sequences(X_fit, X_calib, X_val, lengths_fit):
    B_fit, T, D = X_fit.shape
    real_vecs = np.vstack([X_fit[i, :lengths_fit[i], :] for i in range(B_fit)])
    scaler = StandardScaler()
    scaler.fit(real_vecs)
    
    def _apply(X):
        B = X.shape[0]
        return scaler.transform(X.reshape(-1, D)).reshape(B, T, D)
    
    return _apply(X_fit), _apply(X_calib), _apply(X_val)

def pca_sequences(X_fit, X_calib, X_val, lengths_fit, variance=PCA_VAR):
    B_fit, T, D = X_fit.shape
    real_vecs = np.vstack([X_fit[i, :lengths_fit[i], :] for i in range(B_fit)])
    pca = PCA(n_components=variance, svd_solver="full", random_state=42)
    pca.fit(real_vecs)
    
    def _apply(X):
        B = X.shape[0]
        return pca.transform(X.reshape(-1, D)).reshape(B, T, -1)
    
    X_fit_p = _apply(X_fit)
    X_calib_p = _apply(X_calib)
    X_val_p = _apply(X_val)
    new_D = X_fit_p.shape[2]
    return X_fit_p, X_calib_p, X_val_p, new_D

def mixup_sequences(X_fit, y_fit, alpha=MIXUP_ALPHA):
    idx_pos = np.where(y_fit == 1)[0]
    idx_neg = np.where(y_fit == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)
    if n_pos < 2:
        return X_fit, y_fit
    n_syn = n_neg - n_pos
    if n_syn <= 0:
        return X_fit, y_fit
    rng = np.random.default_rng(42)
    X_syn_list = []
    for _ in range(n_syn):
        i, j = rng.choice(idx_pos, size=2, replace=False)
        lam = rng.beta(alpha, alpha)
        x_new = lam * X_fit[i] + (1.0 - lam) * X_fit[j]
        X_syn_list.append(x_new)
    X_syn = np.array(X_syn_list, dtype=np.float32)
    y_syn = np.ones(n_syn, dtype=np.int64)
    return np.concatenate([X_fit, X_syn], axis=0), np.concatenate([y_fit, y_syn])

# =============================================================================
# 3. DATASET PYTORCH
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

# =============================================================================
# 4. ARQUITECTURAS
# =============================================================================

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(64, 1),
        )
    def forward(self, x, lengths=None):
        return self.net(x.permute(0, 2, 1))

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=n_layers,
                             batch_first=True, bidirectional=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head = nn.Linear(hidden * 2, 1)
    def forward(self, x, lengths=None):
        if lengths is not None:
            lens_cpu = torch.clamp(torch.tensor(lengths), min=1).cpu()
            x_packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(x_packed)
        else:
            _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))

class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=n_layers,
                           batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head = nn.Linear(hidden * 2, 1)
    def forward(self, x, lengths=None):
        if lengths is not None:
            lens_cpu = torch.clamp(torch.tensor(lengths), min=1).cpu()
            x_packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(x_packed)
        else:
            _, h_n = self.gru(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))

class CNNBiGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
        )
        self.gru = nn.GRU(128, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.head = nn.Linear(hidden * 2, 1)
    def forward(self, x, lengths=None):
        x_cnn = self.cnn(x.permute(0, 2, 1))
        x_gru = x_cnn.permute(0, 2, 1)
        if lengths is not None:
            T_prime = x_gru.shape[1]
            lens_adj = torch.clamp(
                torch.tensor([min(l // 2, T_prime) for l in lengths], dtype=torch.long), min=1
            ).cpu()
            x_packed = pack_padded_sequence(x_gru, lens_adj, batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(x_packed)
        else:
            _, h_n = self.gru(x_gru)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(self.dropout(self.norm(h)))

ARCHITECTURES = {
    "CNN1D": CNN1DClassifier, "BiLSTM": BiLSTMClassifier,
    "BiGRU": BiGRUClassifier, "CNN_BiGRU": CNNBiGRUClassifier,
}
LR_BY_ARCH = {"CNN1D": 5e-4, "BiLSTM": 1e-3, "BiGRU": 1e-3, "CNN_BiGRU": 5e-4}

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
# 6. ENTRENAMIENTO DE UN FOLD CON 3 VÍAS
# =============================================================================

def train_fold_sequence(X_fit, y_fit, lengths_fit,
                         X_calib, y_calib, lengths_calib,
                         X_val, y_val, lengths_val,
                         arch_name: str):
    """Entrena con división de 3 vías.
    - FIT: backpropagation
    - CALIB: early stopping + búsqueda de threshold
    - VAL: evaluación final (una sola vez)
    """
    input_dim = X_fit.shape[2]
    ModelClass = ARCHITECTURES[arch_name]
    model = ModelClass(input_dim).to(DEVICE)
    lr = LR_BY_ARCH[arch_name]
    
    ds_fit = SequenceDataset(X_fit, y_fit, lengths_fit)
    ds_val = SequenceDataset(X_val, y_val, lengths_val)
    loader_fit = DataLoader(ds_fit, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=(len(ds_fit) % BATCH_SIZE == 1))
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    
    has_calib = X_calib is not None and len(X_calib) > 0
    if has_calib:
        ds_calib = SequenceDataset(X_calib, y_calib, lengths_calib)
        loader_calib = DataLoader(ds_calib, batch_size=BATCH_SIZE, shuffle=False)
    
    n_neg = np.sum(y_fit == 0)
    n_pos = np.sum(y_fit == 1)
    pos_w = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_calib_loss = float("inf")
    best_state = None
    no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb, lb in loader_fit:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb, lb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        # Early stopping SOLO sobre CALIB
        if has_calib:
            model.eval()
            calib_losses = []
            with torch.no_grad():
                for Xb, yb, lb in loader_calib:
                    p = torch.sigmoid(model(Xb.to(DEVICE), lb).squeeze(-1))
                    calib_losses.append(criterion(torch.log(p / (1 - p + 1e-8) + 1e-8), yb.to(DEVICE)).item())
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
            for Xb, yb, lb in loader_calib:
                p = torch.sigmoid(model(Xb.to(DEVICE), lb).squeeze(-1))
                calib_probs_list.extend(p.cpu().numpy().tolist())
                calib_labels_list.extend(yb.cpu().numpy().tolist())
        calib_probs = np.array(calib_probs_list)
        calib_labels = np.array(calib_labels_list)
        thr = find_best_threshold(calib_labels, calib_probs)
    else:
        thr = 0.5
    
    # Evaluación final SOLO en VAL
    model.eval()
    val_probs_list, val_labels_list = [], []
    with torch.no_grad():
        for Xb, yb, lb in loader_val:
            p = torch.sigmoid(model(Xb.to(DEVICE), lb).squeeze(-1))
            val_probs_list.extend(p.cpu().numpy().tolist())
            val_labels_list.extend(yb.cpu().numpy().tolist())
    val_probs = np.array(val_probs_list)
    val_labels = np.array(val_labels_list)
    
    auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.0
    val_pred_thr = (val_probs >= thr).astype(int)
    val_pred_05 = (val_probs >= 0.5).astype(int)
    
    return {
        "f1_macro": f1_score(val_labels, val_pred_thr, average="macro", zero_division=0),
        "f1_minority": f1_score(val_labels, val_pred_thr, average="binary", zero_division=0),
        "auc": auc,
        "acc": accuracy_score(val_labels, val_pred_thr),
        "threshold": thr,
        "f1_macro_thr05": f1_score(val_labels, val_pred_05, average="macro", zero_division=0),
        "f1_minority_thr05": f1_score(val_labels, val_pred_05, average="binary", zero_division=0),
        "acc_thr05": accuracy_score(val_labels, val_pred_05),
        "best_calib_loss": best_calib_loss if has_calib else float("nan"),
        "best_epoch": epoch + 1,
        "input_dim": input_dim,
        "has_calib": has_calib,
    }

# =============================================================================
# 7. LOOP DE CROSS-VALIDATION CON 3 VÍAS
# =============================================================================

def run_experiment_seq(X, y, lengths, exp_config, arch_name, emb_model, results_path: Path):
    exp_id = exp_config["id"]
    use_pca = exp_config["pca"]
    use_mixup = exp_config["mixup"]
    
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=42)
    rows = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        rep_num = (fold_idx - 1) // CV_SPLITS + 1
        fold_num = (fold_idx - 1) % CV_SPLITS + 1
        
        X_tr_full, X_vl = X[train_idx].copy(), X[val_idx].copy()
        y_tr_full, y_vl = y[train_idx].copy(), y[val_idx].copy()
        l_tr_full = [lengths[i] for i in train_idx]
        l_vl = [lengths[i] for i in val_idx]
        
        # FIX-3WAY: Dividir train en FIT (75%) y CALIB (25%)
        try:
            fit_idx, calib_idx = train_test_split(
                np.arange(len(y_tr_full)), test_size=CALIB_FRAC,
                stratify=y_tr_full, random_state=42 + fold_idx
            )
            X_fit_raw, y_fit = X_tr_full[fit_idx], y_tr_full[fit_idx]
            X_calib_raw, y_calib = X_tr_full[calib_idx], y_tr_full[calib_idx]
            l_fit = [l_tr_full[i] for i in fit_idx]
            l_calib = [l_tr_full[i] for i in calib_idx]
            if len(np.unique(y_calib)) < 2 or len(calib_idx) < 4:
                raise ValueError("calib insuficiente")
            fallback = False
        except Exception:
            X_fit_raw, y_fit = X_tr_full, y_tr_full
            X_calib_raw, y_calib = np.empty((0, X.shape[1], X.shape[2])), np.empty((0,), dtype=np.int64)
            l_fit = l_tr_full
            l_calib = []
            fallback = True
            logger.warning(f"  Fold {fold_idx}: fallback (calib no estratificable)")
        
        # Scaler fit SOLO en FIT
        X_fit_s, X_calib_s, X_vl_s = scale_sequences(X_fit_raw, X_calib_raw, X_vl, l_fit)
        
        # PCA (opcional) fit SOLO en FIT
        n_dims = X_fit_s.shape[2]
        if use_pca:
            X_fit_s, X_calib_s, X_vl_s, n_dims = pca_sequences(X_fit_s, X_calib_s, X_vl_s, l_fit)
        
        # Mixup SOLO en FIT
        if use_mixup:
            X_fit_aug, y_fit_aug = mixup_sequences(X_fit_s, y_fit)
            n_syn = len(y_fit_aug) - len(y_fit)
            max_T = X_fit_s.shape[1]
            l_fit_aug = l_fit + [max_T] * n_syn
            X_fit_s, y_fit, l_fit = X_fit_aug, y_fit_aug, l_fit_aug
        
        # Entrenar y evaluar con 3 vías
        metrics = train_fold_sequence(
            X_fit_s, y_fit, l_fit,
            X_calib_s, y_calib, l_calib,
            X_vl_s, y_vl, l_vl,
            arch_name,
        )
        
        row = {
            "emb_model": emb_model, "architecture": arch_name, "experiment": exp_id,
            "repeat": rep_num, "fold": fold_num,
            "n_fit": len(X_fit_s), "n_calib": len(X_calib_s), "n_val": len(val_idx),
            "n_dims": n_dims, "calib_fallback": fallback,
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
            logger.info(f"      Rep{rep_num:>2} | AUC={recent['auc'].mean():.3f} | "
                        f"F1={recent['f1_macro'].mean():.3f} | dims={n_dims}")
    
    df = pd.DataFrame(rows)
    logger.info(f"    ✅ [{arch_name}|{exp_id}] "
                f"AUC={df['auc'].mean():.3f}±{df['auc'].std():.3f} | "
                f"F1={df['f1_macro'].mean():.3f}±{df['f1_macro'].std():.3f} | "
                f"fallback_rate={df['calib_fallback'].mean():.2f}")
    return df

# =============================================================================
# 8. PIPELINE PRINCIPAL
# =============================================================================

def run_all(embeddings_root: str, output_dir: str):
    output_dir = Path(output_dir) / "dl_script2_fixed"
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
    archs = list(ARCHITECTURES.keys())
    exps = [e["id"] for e in EXPERIMENTS]
    logger.info(f"🧱 Arquitecturas: {archs}")
    logger.info(f"🧪 Experimentos:  {exps}\n")
    
    t0_global = time.time()
    
    for emb_model in MODELS_TO_TEST:
        logger.info(f"{'='*65}")
        logger.info(f"📦 Modelo de embeddings: {emb_model}")
        logger.info(f"{'='*65}")
        try:
            X, y, _ids, lengths = load_sequence_embeddings(embeddings_root, emb_model)
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar {emb_model}: {e}")
            continue
        if len(np.unique(y)) < 2:
            logger.warning("   Solo una clase. Saltando.")
            continue
        
        for arch_name in ARCHITECTURES:
            logger.info(f"\n  🔷 Arquitectura: {arch_name}")
            for exp_config in EXPERIMENTS:
                logger.info(f"    🔬 Experimento: {exp_config['id']}")
                t0 = time.time()
                run_experiment_seq(X, y, lengths, exp_config, arch_name, emb_model, raw_path)
                logger.info(f"       ⏱️  {(time.time()-t0)/60:.1f} min")
    
    if raw_path.exists():
        df_all = pd.read_csv(raw_path)
        summary = (
            df_all.groupby(["emb_model", "architecture", "experiment"])
            .agg(
                mean_f1_macro=("f1_macro", "mean"), std_f1_macro=("f1_macro", "std"),
                mean_f1_minority=("f1_minority", "mean"), std_f1_minority=("f1_minority", "std"),
                mean_auc=("auc", "mean"), std_auc=("auc", "std"),
                mean_acc=("acc", "mean"), mean_dims=("n_dims", "mean"),
                calib_fallback_rate=("calib_fallback", "mean"),
                n_folds=("auc", "count"),
            ).reset_index().sort_values("mean_auc", ascending=False)
        )
        summary.to_csv(summary_path, index=False)
        logger.info("\n" + "="*65)
        logger.info("📊 TOP 20 CONFIGURACIONES (por AUC) — VERSIÓN CORREGIDA 3-VÍAS")
        logger.info("="*65)
        cols = ["emb_model", "architecture", "experiment",
                "mean_auc", "std_auc", "mean_f1_macro", "mean_f1_minority",
                "calib_fallback_rate"]
        logger.info("\n" + summary[cols].head(20).to_string(index=False))
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