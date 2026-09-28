import os
import gc
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.utils import class_weight

# ══════════════════════════════════════════════════════════════
#  BiLSTM — Raw Waveform  [PyTorch | CUDA]
#  Segmentos de 10 segundos | Sin overlap
#  Estructura: wav/0  y  wav/1
# ══════════════════════════════════════════════════════════════

# --- 1. CONFIGURACIÓN Y RUTAS --------------------------------
BASE_PATH_0 = "../Clasificación Final/Ansiedad/wav/0"
BASE_PATH_1 = "../Clasificación Final/Ansiedad/wav/1"

SAMPLE_RATE         = 22050
SEGMENT_DURATION    = 10
SAMPLES_PER_SEGMENT = int(SEGMENT_DURATION * SAMPLE_RATE)   # 220 500
N_FOLDS             = 5
BATCH_SIZE          = 64
EPOCHS              = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    # Permite que PyTorch elija el algoritmo más rápido para cada tamaño de input
    torch.backends.cudnn.benchmark = True

# --- 2. LISTAS DE ARCHIVOS -----------------------------------
def get_file_lists():
    files_0 = [os.path.join(BASE_PATH_0, f)
               for f in sorted(os.listdir(BASE_PATH_0)) if f.endswith('.wav')]
    files_1 = [os.path.join(BASE_PATH_1, f)
               for f in sorted(os.listdir(BASE_PATH_1)) if f.endswith('.wav')]
    files  = np.array(files_0 + files_1)
    labels = np.array([0]*len(files_0) + [1]*len(files_1), dtype=np.int64)
    return files, labels

all_files, all_labels = get_file_lists()
print(f"\nTotal archivos : {len(all_files)}")
print(f" - Sin Ansiedad (0): {np.sum(all_labels == 0)}")
print(f" - Con Ansiedad (1): {np.sum(all_labels == 1)}")
print(f"Segment: {SEGMENT_DURATION}s | Samples/seg: {SAMPLES_PER_SEGMENT}")

# --- 3. DATASET PYTORCH (lazy) --------------------------------
class WaveformDataset(Dataset):
    """
    Indexa (ruta, offset_segundos) sin cargar audio en RAM.
    Carga solo el segmento exacto en __getitem__ usando librosa.load
    con offset + duration.

    Segmentación: sin overlap (igual que Script 3 original).
    Segmentos < 50% de la duración requerida se descartan.
    Segmentos entre 50-100% se rellenan con ceros (zero-pad).
    """
    def __init__(self, file_list, label_list):
        self.samples = []   # (file_path, offset_segundos, label)
        for fp, lbl in zip(file_list, label_list):
            try:
                total_duration = librosa.get_duration(path=fp)
                start_s = 0.0
                while start_s < total_duration:
                    remaining = total_duration - start_s
                    if remaining < SEGMENT_DURATION / 2:
                        break                        # demasiado corto → descartar
                    self.samples.append((fp, start_s, int(lbl)))
                    start_s += SEGMENT_DURATION
            except Exception as e:
                print(f"[AVISO] No se pudo indexar {fp}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, offset_s, label = self.samples[idx]
        audio, _ = librosa.load(fp, sr=SAMPLE_RATE,
                                offset=offset_s,
                                duration=SEGMENT_DURATION)
        # Zero-pad si el segmento es más corto que lo esperado
        if len(audio) < SAMPLES_PER_SEGMENT:
            audio = np.pad(audio, (0, SAMPLES_PER_SEGMENT - len(audio)))
        audio = audio[:SAMPLES_PER_SEGMENT].astype(np.float32)
        audio /= (np.max(np.abs(audio)) + 1e-8)

        # Conv1d de PyTorch espera (canales, longitud) → (1, 220500)
        return torch.from_numpy(audio).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# --- 4. OVERSAMPLE a nivel de archivos -----------------------
def oversample_files(files, labels):
    f1 = files[labels == 1]
    l1 = labels[labels == 1]
    diff = np.sum(labels == 0) - np.sum(labels == 1)
    if diff > 0:
        idx    = np.random.choice(len(f1), size=diff, replace=True)
        files  = np.concatenate([files,  f1[idx]])
        labels = np.concatenate([labels, l1[idx]])
    return files, labels

# --- 5. MODELO CNN + BiLSTM ----------------------------------
class BiLSTMRawWaveform(nn.Module):
    """
    Equivalente al Script 3 (TensorFlow) pero en PyTorch.

    CNN Subsampler (reduce 220 500 → ~430 timesteps):
      Conv1d(1→64,   k=256, stride=128, pad=128) → ReLU → BN → MaxPool(2) → Drop(0.2)
      Conv1d(64→128, k=9,   stride=1,   pad=4  ) → ReLU → BN → Drop(0.2)

    BiLSTM CuDNN-compatible (sin recurrent_dropout):
      BiLSTM(128 unidades, return_seq=True)  → Drop(0.35)
      BiLSTM(64  unidades, return_seq=False) → Drop(0.35)

    Clasificador:
      Dense(128→64) → ReLU → Drop(0.4) → Dense(64→1) → Sigmoid

    Nota: en PyTorch, BiLSTM(hidden=H) produce salida de tamaño 2*H
    porque concatena forward + backward. Por eso la segunda LSTM
    recibe 256 y su salida es 128.
    """
    def __init__(self, input_length: int):
        super().__init__()

        # ── CNN Subsampler ─────────────────────────────────────
        # Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        # 220500 / 128 = ~1723  →  MaxPool(2)  →  ~861  →  ~430
        self.cnn = nn.Sequential(
            nn.Conv1d(1,   64,  kernel_size=256, stride=128, padding=128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=9,   stride=1,   padding=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
        )

        # Calcular longitud de secuencia tras la CNN (sin hardcodear)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            seq_len = self.cnn(dummy).shape[2]
        print(f"  Longitud de secuencia tras CNN: {seq_len} timesteps")

        # ── BiLSTM ─────────────────────────────────────────────
        # bidirectional=True → output_size = hidden_size * 2
        self.lstm1 = nn.LSTM(
            input_size=128, hidden_size=128,
            batch_first=True, bidirectional=True
        )
        self.drop1 = nn.Dropout(0.35)

        self.lstm2 = nn.LSTM(
            input_size=256, hidden_size=64,
            batch_first=True, bidirectional=True
        )
        self.drop2 = nn.Dropout(0.35)

        # ── Clasificador ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 220500)
        x = self.cnn(x)                  # (batch, 128, ~430)
        x = x.permute(0, 2, 1)          # (batch, ~430, 128)  ← lo que LSTM espera

        x, _ = self.lstm1(x)            # (batch, ~430, 256)
        x = self.drop1(x)

        x, _ = self.lstm2(x)            # (batch, ~430, 128)
        x = self.drop2(x)

        x = x[:, -1, :]                 # último timestep → (batch, 128)
        x = self.classifier(x)          # (batch, 1)
        return x.squeeze(1)            # (batch,)

# --- 6. ENTRENAMIENTO (1 época) ------------------------------
def train_one_epoch(model, loader, optimizer, criterion, cw_tensor):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)   # más eficiente que zero_grad()
        preds = model(xb)

        # Pesos por muestra (equivalente a class_weight en Keras)
        sample_w = cw_tensor[yb.long()]
        loss = (criterion(preds, yb) * sample_w).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# --- 7. INFERENCIA POR ARCHIVO (mini-lotes) ------------------
@torch.no_grad()
def predict_file(model, file_path: str):
    """
    Carga el archivo completo, lo trocea en segmentos de 10s
    y devuelve la probabilidad media sobre todos los segmentos.
    """
    model.eval()
    segs = []
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        for start in range(0, len(audio), SAMPLES_PER_SEGMENT):
            seg = audio[start:start + SAMPLES_PER_SEGMENT]
            if len(seg) < SAMPLES_PER_SEGMENT // 2:
                continue
            if len(seg) < SAMPLES_PER_SEGMENT:
                seg = np.pad(seg, (0, SAMPLES_PER_SEGMENT - len(seg)))
            seg = seg.astype(np.float32)
            seg /= (np.max(np.abs(seg)) + 1e-8)
            segs.append(seg)
    except Exception as e:
        print(f"  Error inferencia {file_path}: {e}")
        return None

    if not segs:
        return None

    probs = []
    for b_start in range(0, len(segs), BATCH_SIZE):
        batch = np.stack(segs[b_start:b_start + BATCH_SIZE])      # (B, 220500)
        batch_t = torch.from_numpy(batch).unsqueeze(1).to(DEVICE) # (B, 1, 220500)
        out = model(batch_t).cpu().numpy()
        probs.extend(out.tolist())

    return float(np.mean(probs))

# --- 8. CROSS-VALIDATION -------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
global_y_true, global_y_pred, global_y_prob = [], [], []

print(f"\nINICIANDO CROSS-VALIDATION BiLSTM PyTorch | "
      f"{SEGMENT_DURATION}s | {N_FOLDS} FOLDS\n")

for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(all_files, all_labels)):
    print(f"\n{'─'*55}")
    print(f"  FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'─'*55}")

    Xtr_f, Xte_f = all_files[tr_idx], all_files[te_idx]
    ytr_f, yte_f = all_labels[tr_idx], all_labels[te_idx]
    Xtr_f, ytr_f = oversample_files(Xtr_f, ytr_f)

    # Dataset y DataLoader
    train_ds = WaveformDataset(Xtr_f, ytr_f)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,          # workers paralelos para I/O
        pin_memory=True,        # copia directa RAM → VRAM
        persistent_workers=True # mantiene workers vivos entre épocas
    )
    print(f"  Segmentos train: {len(train_ds)}")

    # Pesos de clase (tensor en GPU)
    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(ytr_f), y=ytr_f)
    cw_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    # Modelo
    model     = BiLSTMRawWaveform(SAMPLES_PER_SEGMENT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss(reduction='none')    # pesos por muestra a mano
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)

    if fold_idx == 0:
        print(model)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parámetros entrenables: {n_params:,}\n")

    # ── Early Stopping ────────────────────────────────────────
    best_loss    = float('inf')
    patience_cnt = 0
    PATIENCE     = 5
    best_state   = None

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_dl, optimizer, criterion, cw_tensor)
        scheduler.step(loss)

        improved = loss < best_loss
        if improved:
            best_loss    = loss
            patience_cnt = 0
            # Guardar estado en CPU para no ocupar VRAM extra
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        mark = "✓" if improved else f"({patience_cnt}/{PATIENCE})"
        print(f"  Epoch {epoch+1:02d}/{EPOCHS} — loss: {loss:.4f}  "
              f"best: {best_loss:.4f}  {mark}")

        if patience_cnt >= PATIENCE:
            print("  ► Early stopping activado.")
            break

    # Restaurar mejor modelo antes de evaluar
    model.load_state_dict(best_state)
    print("  Entrenamiento completado.")

    # ── Inferencia por archivo ─────────────────────────────────
    for i, fp in enumerate(Xte_f):
        avg_prob = predict_file(model, fp)
        if avg_prob is None:
            continue
        global_y_true.append(int(yte_f[i]))
        global_y_pred.append(1 if avg_prob > 0.33 else 0)
        global_y_prob.append(avg_prob)

    # ── Liberar VRAM y RAM ────────────────────────────────────
    del model, train_ds, train_dl, best_state, cw_tensor
    gc.collect()
    torch.cuda.empty_cache()

# --- 9. REPORTE FINAL ----------------------------------------
print("\n" + "="*60)
print(f"RESULTADOS BiLSTM PyTorch | {SEGMENT_DURATION}s  "
      f"({len(global_y_true)} pacientes evaluados)")
print("="*60)

# Matriz de confusión
cm = confusion_matrix(global_y_true, global_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Sin Ansiedad', 'Con Ansiedad'],
            yticklabels=['Sin Ansiedad', 'Con Ansiedad'])
plt.title(f'Confusión BiLSTM {SEGMENT_DURATION}s ({N_FOLDS}-Fold CV)')
plt.ylabel('Real'); plt.xlabel('Predicción')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Métricas
acc      = accuracy_score(global_y_true, global_y_pred)
prec     = precision_score(global_y_true, global_y_pred, zero_division=0)
rec      = recall_score(global_y_true, global_y_pred, zero_division=0)
f1       = f1_score(global_y_true, global_y_pred, zero_division=0)
f1_macro = f1_score(global_y_true, global_y_pred, average='macro', zero_division=0)
print(f"\nAccuracy :  {acc:.4f}")
print(f"Precision:  {prec:.4f}")
print(f"Recall   :  {rec:.4f}")
print(f"F1-Ans   :  {f1:.4f}  (clase positiva)")
print(f"F1-Macro :  {f1_macro:.4f}")
print()
print(classification_report(global_y_true, global_y_pred,
                             target_names=['Sin Ansiedad', 'Con Ansiedad']))

# Curva ROC
fpr, tpr, _ = roc_curve(global_y_true, global_y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
plt.xlabel('Falsos Positivos'); plt.ylabel('Verdaderos Positivos')
plt.title(f'ROC — BiLSTM PyTorch {SEGMENT_DURATION}s')
plt.legend(loc='lower right'); plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()