"""
=============================================================
PIPELINE DE EXTRACCIÓN DE CARACTERÍSTICAS ACÚSTICAS
Trastorno objetivo: DEPRESIÓN (MDD)
=============================================================
Características basadas en revisión de literatura (~10 últimos años):
  - MFCCs (1-13) + Deltas + Delta-Delta
  - LPCCs (1-13)
  - Pitch / F0 (media, std, rango, pendientes, contorno)
  - Formantes F1–F3 (via Parselmouth)
  - Jitter y Shimmer (via Parselmouth)
  - HNR – Harmonic-to-Noise Ratio (via Parselmouth)
  - CPP – Cepstral Peak Prominence (indicador breathiness)
  - TEO – Teager Energy Operator
  - Voice Breaks (%)
  - Intensidad / RMS
  - Spectral Tilt, centroid, rolloff, bandwidth, flatness
  - Tiempos: duración habla, silencio total, # pausas, longitud media de pausa
  - Tasa de articulación / Velocidad de habla
  - Medidas de prosodia monótona (desviación F0, rango inter-cuartil)

Rutas de entrada:
    /home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Depresión/wav/0   → sin depresión (label=0)
    /home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Depresión/wav/1   → con depresión  (label=1)

Salida:
  features_depresion.csv

Dependencias principales: librosa, numpy, scipy, pandas
Opcionales (más precisas): praat-parselmouth

Instalación:
    pip install librosa numpy scipy pandas
    pip install praat-parselmouth   # opcional pero recomendado
=============================================================
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
import librosa
import librosa.effects
from scipy.stats import iqr as scipy_iqr
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Intento de importar Parselmouth (opcional) ──────────────────────────────
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("[INFO] parselmouth no disponible. Jitter, Shimmer, HNR, CPP y Formantes "
          "se calcularán con aproximaciones o se omitirán.")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — RUTAS Y PARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════
BASE_PATH_0 = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Depresión/wav/0"   # sin depresión
BASE_PATH_1 = "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Depresión/wav/1"   # con depresión
OUTPUT_CSV  = "features_depresion_5.csv"

SR_TARGET    = 16_000
PRE_EMPHASIS = 0.97
FRAME_MS     = 25
HOP_MS       = 10
N_MFCC       = 13
VAD_TOP_DB   = 20
F0_FMIN      = 75
F0_FMAX      = 600
ROLLOFF_PCT  = 0.85
SEG_DURATION = 5
OVERLAP      = 0.50
MIN_SEG_FRAC = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# PASO 1 — INVENTARIO DE AUDIOS
# ═══════════════════════════════════════════════════════════════════════════
def collect_inventory() -> pd.DataFrame:
    rows = []
    for label, base_path in [(0, BASE_PATH_0), (1, BASE_PATH_1)]:
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Carpeta no encontrada: {base_path}")
        for fname in sorted(os.listdir(base_path)):
            if fname.lower().endswith("_recortado_denoised.wav"):
                rows.append({
                    "audio_id": fname,
                    "path":     os.path.join(base_path, fname),
                    "label":    label,
                })
    df = pd.DataFrame(rows)
    print(f"Audios encontrados : {len(df)}")
    print(f"  sin depresión (0): {(df.label == 0).sum()}")
    print(f"  con depresión (1): {(df.label == 1).sum()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES INTERNAS
# ═══════════════════════════════════════════════════════════════════════════

def _frame_len(sr: int) -> int:
    return int(FRAME_MS * sr / 1000)

def _hop_len(sr: int) -> int:
    return int(HOP_MS * sr / 1000)

def _safe_stats(arr: np.ndarray, prefix: str) -> dict:
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                f"{prefix}_p25": np.nan, f"{prefix}_p75": np.nan,
                f"{prefix}_iqr": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std":  float(np.std(arr)),
        f"{prefix}_p25":  float(np.percentile(arr, 25)),
        f"{prefix}_p75":  float(np.percentile(arr, 75)),
        f"{prefix}_iqr":  float(scipy_iqr(arr)),
    }

def _lpcc_frame(frame: np.ndarray, order: int = N_MFCC) -> np.ndarray:
    try:
        lpc_coeffs = librosa.lpc(frame, order=order)
        lpcc = np.zeros(order + 1)
        lpcc[0] = -np.log(np.abs(lpc_coeffs[0]) + 1e-10)
        for m in range(1, order + 1):
            lpcc[m] = -lpc_coeffs[m]
            for k in range(1, m):
                lpcc[m] -= (k / m) * lpcc[k] * lpc_coeffs[m - k]
        return lpcc[1:]
    except Exception:
        return np.full(order, np.nan)


def segment_audio(
    audio: np.ndarray,
    sr: int,
    segment_duration: float = SEG_DURATION,
    overlap: float = OVERLAP,
    min_seg_frac: float = MIN_SEG_FRAC,
) -> list[np.ndarray]:
    """Divide audio en segmentos con solapamiento configurable."""
    if segment_duration <= 0:
        raise ValueError("segment_duration debe ser > 0")
    if not 0 <= overlap < 1:
        raise ValueError("overlap debe estar en [0, 1)")
    if not 0 < min_seg_frac <= 1:
        raise ValueError("min_seg_frac debe estar en (0, 1]")

    n_samples = int(segment_duration * sr)
    if n_samples <= 0:
        raise ValueError("segment_duration produce un tamaño de segmento inválido")

    step = max(int(n_samples * (1 - overlap)), 1)
    min_length = int(n_samples * min_seg_frac)

    if len(audio) == 0:
        return []
    if len(audio) <= n_samples:
        return [np.pad(audio, (0, n_samples - len(audio)))]

    segments = []
    start = 0
    while start < len(audio):
        seg = audio[start:start + n_samples]
        if len(seg) < min_length:
            if not segments:
                segments.append(np.pad(seg, (0, n_samples - len(seg))))
            break
        if len(seg) < n_samples:
            seg = np.pad(seg, (0, n_samples - len(seg)))
        segments.append(seg)
        start += step

    return segments


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULOS DE EXTRACCIÓN  (lógica original intacta)
# ═══════════════════════════════════════════════════════════════════════════

def extract_mfcc(y: np.ndarray, sr: int) -> dict:
    """MFCCs 1-13 + Deltas + Delta-Delta (importantes en Gong et al. 2025)."""
    fl, hl = _frame_len(sr), _hop_len(sr)
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=fl, hop_length=hl, window='hann')
    delta    = librosa.feature.delta(mfcc)
    delta2   = librosa.feature.delta(mfcc, order=2)

    feats = {}
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"]   = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"]    = float(np.std(mfcc[i]))
        feats[f"mfcc{i+1}_p25"]    = float(np.percentile(mfcc[i], 25))
        feats[f"mfcc{i+1}_p75"]    = float(np.percentile(mfcc[i], 75))
        feats[f"dmfcc{i+1}_mean"]  = float(np.mean(delta[i]))
        feats[f"d2mfcc{i+1}_mean"] = float(np.mean(delta2[i]))
    return feats


def extract_lpcc(y: np.ndarray, sr: int) -> dict:
    """LPCCs 1-13 (Almaghrabi 2023, revisión en MDD)."""
    fl, hl = _frame_len(sr), _hop_len(sr)
    frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
    win    = np.hanning(fl)
    lpccs  = np.array([_lpcc_frame(frames[:, i] * win) for i in range(frames.shape[1])])

    feats = {}
    for i in range(N_MFCC):
        col = lpccs[:, i]
        feats[f"lpcc{i+1}_mean"] = float(np.nanmean(col))
        feats[f"lpcc{i+1}_std"]  = float(np.nanstd(col))
    return feats


def extract_pitch(y: np.ndarray, sr: int) -> dict:
    """
    F0: media, std, rango, percentiles, IQR y pendiente temporal.
    En depresión: F0↓, variabilidad↓, voz monótona (Regondi 2025).
    """
    fl, hl = _frame_len(sr), _hop_len(sr)
    f0 = librosa.yin(y, fmin=F0_FMIN, fmax=F0_FMAX,
                     sr=sr, frame_length=fl*2, hop_length=hl)
    voiced = f0[(f0 > F0_FMIN) & (f0 < F0_FMAX)]

    feats = _safe_stats(voiced, "f0")
    if len(voiced) > 1:
        feats["f0_range"]  = float(voiced.max() - voiced.min())
        feats["f0_median"] = float(np.median(voiced))
        x = np.arange(len(voiced))
        slope = np.polyfit(x, voiced, 1)[0]
        feats["f0_slope"]       = float(slope)
        rises = np.sum(np.diff(voiced) > 0)
        feats["f0_ratio_rises"] = float(rises / (len(voiced) - 1))
    else:
        feats.update({k: np.nan for k in
                      ["f0_range", "f0_median", "f0_slope", "f0_ratio_rises"]})

    total_frames  = len(f0)
    voiced_frames = np.sum((f0 > F0_FMIN) & (f0 < F0_FMAX))
    feats["voice_breaks_ratio"] = float(1 - voiced_frames / total_frames) if total_frames > 0 else np.nan
    return feats


def extract_spectral(y: np.ndarray, sr: int) -> dict:
    """
    Centroid, rolloff, bandwidth, flatness, spectral tilt.
    En depresión: centroid más bajo (timbre oscuro, Regondi 2025).
    """
    fl, hl = _frame_len(sr), _hop_len(sr)
    feats = {}
    feats.update(_safe_stats(
        librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=fl, hop_length=hl)[0], "spectral_centroid"))
    feats.update(_safe_stats(
        librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=fl, hop_length=hl,
                                          roll_percent=ROLLOFF_PCT)[0], "spectral_rolloff"))
    feats.update(_safe_stats(
        librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=fl, hop_length=hl)[0], "spectral_bandwidth"))
    feats.update(_safe_stats(
        librosa.feature.spectral_flatness(y=y, n_fft=fl, hop_length=hl)[0], "spectral_flatness"))

    S = np.abs(librosa.stft(y, n_fft=fl, hop_length=hl))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=fl)
    mean_spectrum = np.mean(S, axis=1)
    log_spectrum = 10 * np.log10(mean_spectrum + 1e-10)
    tilt = np.polyfit(freqs, log_spectrum, 1)[0] if len(freqs) > 1 else np.nan
    feats["spectral_tilt"] = float(tilt)

    low_mask  = (freqs >= 50)   & (freqs <= 1000)
    high_mask = (freqs >= 1000) & (freqs <= 5000)
    low_e  = np.sum(S[low_mask],  axis=0) + 1e-10
    high_e = np.sum(S[high_mask], axis=0) + 1e-10
    alpha_ratio = 10 * np.log10(high_e / low_e)
    feats.update(_safe_stats(alpha_ratio, "alpha_ratio"))
    return feats


def extract_energy(y: np.ndarray, sr: int) -> dict:
    """RMS / Intensidad. Depresión → voz más suave y monótona."""
    fl, hl = _frame_len(sr), _hop_len(sr)
    rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=hl)[0]
    return _safe_stats(rms, "rms")


def extract_teo(y: np.ndarray) -> dict:
    """
    Teager Energy Operator (TEO): captura no linealidades en producción.
    TEO[n] = x[n]^2 - x[n-1]*x[n+1]
    """
    if len(y) < 3:
        return {"teo_mean": np.nan, "teo_std": np.nan}
    teo = y[1:-1]**2 - y[:-2] * y[2:]
    return _safe_stats(teo, "teo")


def extract_cpp(y: np.ndarray, sr: int) -> dict:
    """
    Cepstral Peak Prominence (CPP): mide el pico cepstral relativo al ruido.
    Menor CPP → voz más tensa / breathiness → asociado a MDD.
    """
    fl, hl = _frame_len(sr), _hop_len(sr)
    frames  = librosa.util.frame(y, frame_length=fl, hop_length=hl)
    win     = np.hanning(fl)
    cpp_vals = []
    for i in range(frames.shape[1]):
        frame = frames[:, i] * win
        spec  = np.fft.rfft(frame, n=fl)
        log_spec = np.log(np.abs(spec) + 1e-10)
        cepstrum = np.fft.irfft(log_spec)
        q_min = max(int(sr / F0_FMAX), 1)
        q_max = min(int(sr / F0_FMIN), len(cepstrum) // 2)
        if q_max <= q_min:
            continue
        peak  = np.max(cepstrum[q_min:q_max])
        x     = np.arange(q_min, q_max)
        trend = np.polyval(np.polyfit(x, cepstrum[q_min:q_max], 1), x)
        cpp_vals.append(peak - np.mean(trend))
    return _safe_stats(np.array(cpp_vals), "cpp") if cpp_vals else {"cpp_mean": np.nan, "cpp_std": np.nan}


def extract_temporal(y: np.ndarray, sr: int) -> dict:
    """
    Duración de habla, silencios, # pausas, longitud media de pausa,
    tasa de articulación. En MDD: más pausas y más largas (Regondi 2025).
    """
    intervals = librosa.effects.split(y, top_db=VAD_TOP_DB)
    total_dur   = len(y) / sr
    speech_segs = [(e - s) / sr for s, e in intervals]
    speech_dur  = sum(speech_segs)
    n_pauses    = max(len(intervals) - 1, 0)
    pause_dur   = total_dur - speech_dur
    pause_ratio = pause_dur / total_dur if total_dur > 0 else np.nan

    pause_lengths = []
    for j in range(len(intervals) - 1):
        p_start = intervals[j][1] / sr
        p_end   = intervals[j+1][0] / sr
        pause_lengths.append(p_end - p_start)
    pause_lengths = np.array(pause_lengths) if pause_lengths else np.array([0.0])

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    speech_rate  = len(onset_frames) / speech_dur if speech_dur > 0 else np.nan

    return {
        "speech_duration_s":   float(speech_dur),
        "total_duration_s":    float(total_dur),
        "pause_total_s":       float(pause_dur),
        "pause_ratio":         float(pause_ratio),
        "n_pauses":            int(n_pauses),
        "pause_mean_length_s": float(np.mean(pause_lengths)),
        "pause_max_length_s":  float(np.max(pause_lengths)),
        "speech_rate_proxy":   float(speech_rate),
    }


def extract_praat_features(y: np.ndarray, sr: int) -> dict:
    """Jitter, Shimmer, HNR, Formantes F1–F3 y CPP-Praat via Parselmouth."""
    feats = {k: np.nan for k in [
        "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq3",
        "hnr_mean", "hnr_std",
        "f1_mean", "f1_std", "f2_mean", "f2_std", "f3_mean", "f3_std",
        "cpp_praat"
    ]}
    if not PARSELMOUTH_AVAILABLE:
        return feats
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        pp  = call(snd, "To PointProcess (periodic, cc)", F0_FMIN, F0_FMAX)

        feats["jitter_local"]  = float(call(pp, "Get jitter (local)",   0, 0, 0.0001, 0.02, 1.3))
        feats["jitter_rap"]    = float(call(pp, "Get jitter (rap)",      0, 0, 0.0001, 0.02, 1.3))
        feats["shimmer_local"] = float(call([snd, pp], "Get shimmer (local)",  0, 0, 0.0001, 0.02, 1.3, 1.6))
        feats["shimmer_apq3"]  = float(call([snd, pp], "Get shimmer (apq3)",   0, 0, 0.0001, 0.02, 1.3, 1.6))

        hnr_obj = call(snd, "To Harmonicity (cc)", 0.01, F0_FMIN, 0.1, 1.0)
        feats["hnr_mean"] = float(call(hnr_obj, "Get mean",              0, 0))
        feats["hnr_std"]  = float(call(hnr_obj, "Get standard deviation", 0, 0))

        formant_obj = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        duration = snd.get_total_duration()
        times    = np.arange(0.02, duration, 0.01)

        f1_vals, f2_vals, f3_vals = [], [], []
        for t in times:
            try:
                v1 = call(formant_obj, "Get value at time", 1, t, 'Hertz', 'Linear')
                v2 = call(formant_obj, "Get value at time", 2, t, 'Hertz', 'Linear')
                v3 = call(formant_obj, "Get value at time", 3, t, 'Hertz', 'Linear')
                if v1 and not np.isnan(v1): f1_vals.append(v1)
                if v2 and not np.isnan(v2): f2_vals.append(v2)
                if v3 and not np.isnan(v3): f3_vals.append(v3)
            except Exception:
                pass

        for fname, vals in [("f1", f1_vals), ("f2", f2_vals), ("f3", f3_vals)]:
            arr = np.array(vals)
            feats[f"{fname}_mean"] = float(np.mean(arr)) if len(arr) > 0 else np.nan
            feats[f"{fname}_std"]  = float(np.std(arr))  if len(arr) > 0 else np.nan

        try:
            cpp_obj = call(snd, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
            cpp_val = call(cpp_obj, "Get CPPS",
                           "yes", 0.02, 0.0, 60.0, 330.0, 0.05, "Parabolic", 0.001, 0.0,
                           "Exponential decay", "Robust")
            feats["cpp_praat"] = float(cpp_val)
        except Exception:
            pass

    except Exception as e:
        print(f"  [WARN] Error en extracción Praat: {e}")
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# PASO 2 — EXTRACCIÓN COMPLETA POR ARCHIVO
# ═══════════════════════════════════════════════════════════════════════════

def extract_depression_features_from_signal(y_proc: np.ndarray, sr: int) -> dict:
    """Pipeline completo de características para una señal de audio (depresión)."""
    features = {}
    features.update(extract_mfcc(y_proc, sr))
    features.update(extract_lpcc(y_proc, sr))
    features.update(extract_pitch(y_proc, sr))
    features.update(extract_spectral(y_proc, sr))
    features.update(extract_energy(y_proc, sr))
    features.update(extract_teo(y_proc))
    features.update(extract_cpp(y_proc, sr))
    features.update(extract_temporal(y_proc, sr))
    features.update(extract_praat_features(y_proc, sr))
    return features


def extract_depression_features(audio_path: str) -> dict:
    """Compatibilidad: extrae features de un WAV completo (sin segmentación)."""
    y, sr = librosa.load(audio_path, sr=SR_TARGET, mono=True)
    y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])
    y_trimmed, _ = librosa.effects.trim(y, top_db=VAD_TOP_DB)
    y_proc = y_trimmed if len(y_trimmed) > sr * 0.1 else y
    return extract_depression_features_from_signal(y_proc, sr)


# ═══════════════════════════════════════════════════════════════════════════
# PASO 3 — PROCESAR INVENTARIO COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def process_all(
    inventory: pd.DataFrame,
    segment_duration: float = SEG_DURATION,
    overlap: float = OVERLAP,
    min_seg_frac: float = MIN_SEG_FRAC,
) -> pd.DataFrame:
    rows = []
    print("\nExtrayendo features…")

    for _, audio_row in tqdm(inventory.iterrows(),
                              total=len(inventory), unit="audio"):
        try:
            y, sr = librosa.load(audio_row["path"], sr=SR_TARGET, mono=True)
            y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])
            y_trimmed, _ = librosa.effects.trim(y, top_db=VAD_TOP_DB)
            y_proc = y_trimmed if len(y_trimmed) > sr * 0.1 else y

            segments = segment_audio(
                y_proc,
                sr,
                segment_duration=segment_duration,
                overlap=overlap,
                min_seg_frac=min_seg_frac,
            )

            for seg_idx, seg in enumerate(segments):
                feats = extract_depression_features_from_signal(seg, sr)
                rows.append({
                    "audio_id": audio_row["audio_id"],
                    "seg_idx":  seg_idx,
                    "label":    audio_row["label"],
                    **feats,
                })
        except Exception as e:
            print(f"\n  [ERROR] {audio_row['audio_id']}: {e}")

    df = pd.DataFrame(rows)
    id_cols   = ["audio_id", "seg_idx", "label"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    return df[id_cols + feat_cols].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in ("audio_id", "seg_idx", "label")]
    print(f"\n{'═' * 55}")
    print("RESUMEN DEL DATASET FINAL — DEPRESIÓN")
    print(f"{'═' * 55}")
    print(f"Archivo            : {OUTPUT_CSV}")
    print(f"Filas totales      : {len(df)}")
    print(f"Audios únicos      : {df['audio_id'].nunique()}")
    segs_per_audio = len(df) / max(df['audio_id'].nunique(), 1)
    print(f"Segs por audio     : ~{segs_per_audio:.1f}")
    print(f"Features extraídas : {len(feat_cols)}")
    print(f"\n  Clase 0 (sin depresión): {(df.label == 0).sum()} segmentos")
    print(f"  Clase 1 (con depresión): {(df.label == 1).sum()} segmentos")
    nan_total = df[feat_cols].isnull().sum().sum()
    if nan_total:
        print(f"\n  ⚠️  NaN detectados: {nan_total}")
        print("     Imputa con mediana antes de entrenar.")
    else:
        print("\n  ✓ Sin valores NaN.")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS BÁSICOS
# ═══════════════════════════════════════════════════════════════════════════

def _run_tests():
    sr  = SR_TARGET
    dur = 2.0
    t   = np.linspace(0, dur, int(sr * dur), endpoint=False)

    tone = 0.5 * np.sin(2 * np.pi * 150 * t)
    f0_feats = extract_pitch(tone, sr)
    assert not np.isnan(f0_feats["f0_mean"]), "F0 no debería ser NaN"
    assert 110 < f0_feats["f0_mean"] < 200, f"F0 ≈150 Hz esperado, got {f0_feats['f0_mean']:.1f}"
    print("✓ Test 1 (pitch tono puro): OK")

    teo_feats = extract_teo(tone)
    assert not np.isnan(teo_feats["teo_mean"]), "TEO no debería ser NaN"
    print("✓ Test 2 (TEO): OK")

    noise = np.random.randn(int(sr * dur)).astype(np.float32) * 0.1
    mfcc_feats = extract_mfcc(noise, sr)
    assert f"mfcc{N_MFCC}_mean" in mfcc_feats
    print("✓ Test 3 (dimensiones MFCC): OK")

    temp = extract_temporal(tone, sr)
    assert 0 <= temp["pause_ratio"] <= 1
    assert temp["n_pauses"] >= 0
    print("✓ Test 4 (temporal / pausas): OK")

    cpp_feats = extract_cpp(tone, sr)
    assert "cpp_mean" in cpp_feats
    print("✓ Test 5 (CPP): OK")

    segs = segment_audio(tone, sr, segment_duration=1.0, overlap=0.5, min_seg_frac=0.5)
    assert len(segs) >= 1, "segment_audio debería producir al menos un segmento"
    print("✓ Test 6 (segmentación): OK")

    print("\n✓ Todos los tests pasaron correctamente.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de features para depresión")
    parser.add_argument("--test", action="store_true", help="Ejecuta tests rápidos")
    parser.add_argument("--segment-duration", type=float, default=SEG_DURATION,
                        help="Duración del segmento en segundos")
    parser.add_argument("--overlap", type=float, default=OVERLAP,
                        help="Solapamiento entre segmentos en [0, 1)")
    parser.add_argument("--min-seg-frac", type=float, default=MIN_SEG_FRAC,
                        help="Fracción mínima para conservar el último segmento")
    args = parser.parse_args()

    if args.test:
        _run_tests()
        raise SystemExit(0)

    print("=" * 55)
    print("PIPELINE — DETECCIÓN DE DEPRESIÓN EN AUDIO")
    print("=" * 55)
    print(f"Segmento: {args.segment_duration}s | Overlap: {int(args.overlap * 100)}%")

    inventory = collect_inventory()
    df        = process_all(
        inventory,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        min_seg_frac=args.min_seg_frac,
    )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅  Guardado: {OUTPUT_CSV}  ({len(df)} filas × {len(df.columns)} columnas)")
    print_summary(df)