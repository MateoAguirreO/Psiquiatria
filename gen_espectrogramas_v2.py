#!/usr/bin/env python3
"""
gen_espectrogramas_v2.py — Generador multi-filtro de espectrogramas
====================================================================
Genera 4 tipos de representación tiempo-frecuencia:
  1. log-mel    (Log-Mel Spectrogram — baseline estándar)
  2. cqt        (Constant-Q Transform — mejor resolución en F0/armónicos)
  3. pcen       (Per-Channel Energy Normalization — robusto a ruido)
  4. gammatone  (Gammatonegram — biológicamente inspirado en la cóclea)

IMPORTANTE: Si una carpeta ya tiene imágenes completas, la SALTA.
            No sobreescribe trabajo ya hecho. Seguro para correr varias veces.

Uso:
    python gen_espectrogramas_v2.py

Estructura de salida:
    Clasificación Final/
      Ansiedad/
        specs/
          log-mel/  2s/0/, 2s/1/, 3s/0/ ... 10s/1/
          cqt/      2s/0/, ...
          pcen/     2s/0/, ...
          gammatone/2s/0/, ...
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — CAMBIAR AQUÍ
# ─────────────────────────────────────────────────────────────────────────────

CONDITION = "Depresión"           # "Ansiedad" | "Depresión"
BASE_DIR  = Path("Clasificación Final")
WAV_ROOT  = BASE_DIR / CONDITION / "wav"
OUT_ROOT  = BASE_DIR / CONDITION / "specs"

SR         = 16000
N_FFT      = 1024
HOP_LENGTH = 512
N_MELS     = 128
F_MIN      = 20
F_MAX      = 8000

# Duraciones y overlaps
SEGMENT_CONFIGS = [
    {"dur": 2,  "overlap": 0.5},
    {"dur": 3,  "overlap": 0.5},
    {"dur": 4,  "overlap": 0.5},
    {"dur": 10, "overlap": 0.0},
]

# Filtros a generar — comentar los que NO quieras
FILTERS = ["log-mel", "cqt", "pcen", "gammatone"]

IMG_DPI = 100
CMAP    = "inferno"

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCIAS OPCIONALES
# ─────────────────────────────────────────────────────────────────────────────
try:
    from spafe.features.gfcc import gfcc
    from spafe.utils.preprocessing import SlidingWindow
    import spafe.utils.preprocessing as prep
    SPAFE_OK = True
except ImportError:
    SPAFE_OK = False
    if "gammatone" in FILTERS:
        print("[AVISO] spafe no instalado. Gammatone usará aproximación con librosa.")
        print("        Para instalarlo: pip install spafe")

# ─────────────────────────────────────────────────────────────────────────────
# INVENTARIO DE ARCHIVOS
# ─────────────────────────────────────────────────────────────────────────────
def list_wavs(root):
    result = {}
    for cls in ["0", "1"]:
        d = root / cls
        if not d.exists():
            print(f"  [AVISO] No existe: {d}")
            result[cls] = []
            continue
        wavs = sorted(d.glob("*_recortado_denoised.wav"))
        result[cls] = wavs
        print(f"  Clase {cls}: {len(wavs)} archivos")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN — ¿ya está completa esta carpeta?
# ─────────────────────────────────────────────────────────────────────────────
def is_complete(out_dir, wav_files, dur, overlap):
    """
    Retorna True si ya existen imágenes para todos los archivos wav de esta clase.
    Estimamos el mínimo esperado como n_wavs * 1 segmento (puede haber más).
    Si hay al menos 1 imagen por wav, consideramos completo.
    """
    if not out_dir.exists():
        return False
    existing = list(out_dir.glob("*.png"))
    if len(existing) == 0:
        return False
    # Verificar que hay al menos una imagen por cada wav
    stems_with_imgs = set(p.stem.rsplit("_seg", 1)[0] for p in existing)
    wav_stems = set(p.stem for p in wav_files)
    return wav_stems.issubset(stems_with_imgs)

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN POR FILTRO
# ─────────────────────────────────────────────────────────────────────────────
def extract_log_mel(audio):
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
    return S_norm   # (N_MELS, T)


def extract_cqt(audio):
    """
    Constant-Q Transform.
    - Resolución logarítmica en frecuencia (igual número de bins por octava)
    - Mucho mejor resolución en bajas frecuencias (F0, armónicos)
    - Paper: "Non-linear frequency warping using CQT for SER" (HAL 2021)
              "Analysis of CQT filterbank for SER" (HAL 2022)
    """
    # n_bins = 7 octavas × bins_per_octave → cubre F_MIN a F_MAX
    n_bins = 7 * 24   # 24 bins/octava = buena resolución (~quarter-tone)
    C = np.abs(librosa.cqt(
        audio, sr=SR, hop_length=HOP_LENGTH,
        fmin=F_MIN, n_bins=n_bins, bins_per_octave=24))
    C_dB = librosa.amplitude_to_db(C, ref=np.max)
    C_norm = (C_dB - C_dB.min()) / (C_dB.max() - C_dB.min() + 1e-8)
    return C_norm   # (n_bins, T)


def extract_pcen(audio):
    """
    Per-Channel Energy Normalization sobre banco Mel.
    - Reemplaza la compresión logarítmica por una compresión dinámica adaptativa
    - Mucho más robusto a ruido de fondo variable (grabaciones clínicas)
    - Paper: "Adaptive PCEN Front-end for Robust Audio Signal Processing" (arXiv 2025)
    """
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=1.0)  # power=1 para PCEN
    P = librosa.pcen(S * (2**31),   # PCEN espera valores sin normalizar
                     sr=SR, hop_length=HOP_LENGTH,
                     gain=0.98, bias=2, power=0.5,
                     time_constant=0.4, eps=1e-6)
    P_norm = (P - P.min()) / (P.max() - P.min() + 1e-8)
    return P_norm   # (N_MELS, T)


def extract_gammatone(audio):
    """
    Gammatonegram — banco de filtros biológicamente inspirado en la cóclea.
    - Mejor modelo del sistema auditivo humano que la escala Mel
    - Especialmente bueno en bajas frecuencias (F0, energía vocal)
    - Paper: "Comparative analysis of CQT, gammatonegram, and Mel-spectrogram" (ScienceDirect 2025)
              "PEmoNet: fusion of MTMFS, Gammatonegram, CQTS" (EMODB/RAVDESS, 97% acc)

    Si spafe está disponible lo usa. Si no, aproxima con un banco de filtros
    ERB (Equivalent Rectangular Bandwidth) via librosa + filtros personalizados.
    """
    if SPAFE_OK:
        # Usar spafe para filtros Gammatone auténticos
        try:
            # gfcc devuelve coeficientes; necesitamos el energigrama (filterbank)
            # Computamos el banco de filtros directamente
            from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
            # Llamar a spafe
            spafe_out = gammatone_filter_banks(
                nfilts=N_MELS, nfft=N_FFT, fs=SR,
                low_freq=F_MIN, high_freq=F_MAX)
            fbank = spafe_out[0] if isinstance(spafe_out, tuple) else spafe_out

            # STFT y aplicar banco
            D = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))**2
            G = np.dot(fbank, D[:N_FFT//2+1])  # (N_MELS, T)
            G_dB = librosa.power_to_db(G + 1e-10, ref=np.max)
            G_norm = (G_dB - G_dB.min()) / (G_dB.max() - G_dB.min() + 1e-8)
            return G_norm
        except Exception:
            print(f"  [ERROR] Falló spafe en Gammatone: {Exception}")

    # Fallback: aproximación ERB via scipy
    from scipy.signal import gammatone as scipy_gammatone
    # Generar banco de N_MELS filtros Gammatone en escala ERB
    freqs = np.array([
        F_MIN * (F_MAX / F_MIN) ** (i / (N_MELS - 1))
        for i in range(N_MELS)
    ])
    G_bank = np.zeros((N_MELS, len(audio) // HOP_LENGTH + 1))
    for i, fc in enumerate(freqs):
        try:
            b, a = scipy_gammatone(fc, 'iir', fs=SR)
            from scipy.signal import lfilter
            filtered = lfilter(b, a, audio)
            # Energía por frame
            n_frames = len(audio) // HOP_LENGTH + 1
            for j in range(n_frames):
                start = j * HOP_LENGTH
                frame = filtered[start:start + N_FFT]
                G_bank[i, j] = np.mean(frame ** 2) if len(frame) > 0 else 0
        except Exception:
            print(f"  [ERROR] Falló Scipy Gammatone en frec {fc}: {Exception}")
            break  # Si falla una vez, probablemente fallará siempre

    G_dB = librosa.power_to_db(G_bank + 1e-10, ref=np.max)
    G_norm = (G_dB - G_dB.min()) / (G_dB.max() - G_dB.min() + 1e-8)
    return G_norm


EXTRACTORS = {
    "log-mel":   extract_log_mel,
    "cqt":       extract_cqt,
    "pcen":      extract_pcen,
    "gammatone": extract_gammatone,
}

# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR IMAGEN
# ─────────────────────────────────────────────────────────────────────────────
def save_spectrogram(S_norm, out_path):
    h, w = S_norm.shape
    figw = w / IMG_DPI
    figh = h / IMG_DPI
    fig = plt.figure(figsize=(figw, figh), dpi=IMG_DPI)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.imshow(S_norm, aspect='auto', origin='lower',
              cmap=CMAP, interpolation='none', vmin=0, vmax=1)
    ax.axis('off')
    fig.savefig(out_path, dpi=IMG_DPI, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAR UN ARCHIVO
# ─────────────────────────────────────────────────────────────────────────────
def process_file(wav_path, dur, overlap, out_dir, extractor):
    seg_samples = int(dur * SR)
    hop_samples = int(seg_samples * (1 - overlap)) if overlap > 0 else seg_samples

    try:
        audio, _ = librosa.load(wav_path, sr=SR, mono=True)
    except Exception as e:
        print(f"  [ERROR] {wav_path.name}: {e}")
        return 0

    audio, _ = librosa.effects.trim(audio, top_db=20)
    if len(audio) < seg_samples:
        audio = np.pad(audio, (0, seg_samples - len(audio)))

    stem    = wav_path.stem
    n_saved = 0

    for i, start in enumerate(range(0, len(audio) - seg_samples + 1, hop_samples)):
        seg = audio[start:start + seg_samples]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms < 1e-4:
            continue
        S_norm   = extractor(seg)
        out_path = out_dir / f"{stem}_seg{i:04d}.png"
        save_spectrogram(S_norm, out_path)
        n_saved += 1

    return n_saved


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"  GENERADOR MULTI-FILTRO DE ESPECTROGRAMAS — {CONDITION.upper()}")
print(f"  Filtros: {FILTERS}")
print(f"  SR: {SR} Hz | N_FFT: {N_FFT} | Hop: {HOP_LENGTH} | Mels: {N_MELS}")
print(f"  Rango: {F_MIN}–{F_MAX} Hz")
print(f"  Skip automático si carpeta ya está completa ✓")
print("=" * 65)

print("\nInventario de archivos WAV:")
wav_files = list_wavs(WAV_ROOT)
total_wavs = sum(len(v) for v in wav_files.values())
print(f"  Total: {total_wavs} archivos")
assert total_wavs > 0, f"No se encontraron .wav en {WAV_ROOT}"

# Resumen de lo que se va a hacer vs saltar
print("\nPlan de ejecución:")
skipped_total = 0
pending_total = 0
for filt in FILTERS:
    for cfg in SEGMENT_CONFIGS:
        dur = cfg['dur']
        key = f"{dur}s"
        for cls_str, wavs in wav_files.items():
            out_dir = OUT_ROOT / filt / key / cls_str
            if is_complete(out_dir, wavs, dur, cfg['overlap']):
                existing = len(list(out_dir.glob("*.png")))
                print(f"  SKIP  {filt}/{key}/cls{cls_str}  "
                      f"({existing} imágenes ya existen)")
                skipped_total += 1
            else:
                existing = len(list(out_dir.glob("*.png"))) if out_dir.exists() else 0
                print(f"  GENERAR {filt}/{key}/cls{cls_str}  "
                      f"({existing} existentes, completando...)")
                pending_total += 1

print(f"\n  → {skipped_total} carpetas completas (se saltarán)")
print(f"  → {pending_total} carpetas a generar/completar")

if pending_total == 0:
    print("\n✓ Todo ya está generado. Nada que hacer.")
    sys.exit(0)

print("\nIniciando generación...\n")

# ─── LOOP PRINCIPAL ────────────────────────────────────────────────────────
grand_stats = {}

for filt in FILTERS:
    extractor = EXTRACTORS[filt]
    print(f"\n{'═'*65}")
    print(f"  FILTRO: {filt.upper()}")
    print(f"{'═'*65}")

    for cfg in SEGMENT_CONFIGS:
        dur     = cfg['dur']
        overlap = cfg['overlap']
        key     = f"{dur}s"

        for cls_str, wavs in wav_files.items():
            out_dir = OUT_ROOT / filt / key / cls_str

            if is_complete(out_dir, wavs, dur, overlap):
                existing = len(list(out_dir.glob("*.png")))
                print(f"  ✓ SKIP {filt}/{key}/cls{cls_str} — {existing} imgs")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            label = ("Sin " if cls_str == "0" else "Con ") + CONDITION
            print(f"\n  {filt}/{key}/cls{cls_str} ({label}) — {len(wavs)} archivos:")

            total_segs = 0
            for wav_path in tqdm(wavs, desc=f"    {filt}/{key}/cls{cls_str}", ncols=72):
                # Solo procesar si no tiene imágenes este wav
                stem = wav_path.stem
                existing_for_wav = list(out_dir.glob(f"{stem}_seg*.png"))
                if len(existing_for_wav) > 0:
                    total_segs += len(existing_for_wav)
                    continue
                n = process_file(wav_path, dur, overlap, out_dir, extractor)
                total_segs += n

            total_imgs = len(list(out_dir.glob("*.png")))
            print(f"    → {total_imgs} imágenes en {out_dir}")
            grand_stats[f"{filt}/{key}/{cls_str}"] = total_imgs

# ─── RESUMEN FINAL ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RESUMEN FINAL")
print("="*65)

for filt in FILTERS:
    print(f"\n  {filt.upper()}:")
    for cfg in SEGMENT_CONFIGS:
        key = f"{cfg['dur']}s"
        n0 = len(list((OUT_ROOT / filt / key / "0").glob("*.png"))) \
             if (OUT_ROOT / filt / key / "0").exists() else 0
        n1 = len(list((OUT_ROOT / filt / key / "1").glob("*.png"))) \
             if (OUT_ROOT / filt / key / "1").exists() else 0
        ratio = max(n0, n1) / max(min(n0, n1), 1)
        print(f"    {key:4s} → cls0: {n0:6,} | cls1: {n1:6,} | "
              f"total: {n0+n1:6,} | ratio: {ratio:.2f}x")

import subprocess
result = subprocess.run(['du', '-sh', str(OUT_ROOT)], capture_output=True, text=True)
if result.returncode == 0:
    print(f"\n  Espacio total en disco: {result.stdout.strip().split()[0]}")

print("\n✓ Listo. Espectrogramas disponibles en:")
for filt in FILTERS:
    print(f"  {OUT_ROOT / filt}")
