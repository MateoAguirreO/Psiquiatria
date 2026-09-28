"""
WAV Pipeline Stage Exporter
============================
Runs the full preprocessing chain on a single participant file
and saves one WAV per stage so you can listen to each step.

Output files (in --output_dir)
-------------------------------
  stage0_raw.wav               — original full interview (both speakers)
  stage1_participant.wav       — only participant turns concatenated
  stage2_trimmed.wav           — silence trimmed
  stage3_normalised.wav        — peak normalised
  stage4_denoised.wav          — after Demucs  (skipped with --no_denoise)
  segments/
    segment_000.wav            — first 5s segment
    segment_001.wav
    ...

Usage
-----
python export_wav_stages.py \\
    /path/to/300_AUDIO.wav \\
    --transcript /path/to/300_TRANSCRIPT.csv \\
    --output_dir /path/to/output_folder

Skip Demucs (faster):
python export_wav_stages.py \\
    /path/to/300_AUDIO.wav \\
    --transcript /path/to/300_TRANSCRIPT.csv \\
    --output_dir /path/to/output_folder \\
    --no_denoise

Only save a few segments (e.g. first 3):
python export_wav_stages.py \\
    /path/to/300_AUDIO.wav \\
    --transcript /path/to/300_TRANSCRIPT.csv \\
    --output_dir /path/to/output_folder \\
    --max_segments 3
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_to_float32(signal: np.ndarray) -> np.ndarray:
    if signal.dtype == np.int16:
        return signal.astype(np.float32) / 32768.0
    if signal.dtype == np.int32:
        return signal.astype(np.float32) / 2147483648.0
    if signal.dtype == np.uint8:
        return (signal.astype(np.float32) - 128.0) / 128.0
    signal = signal.astype(np.float32)
    peak = np.abs(signal).max()
    if peak > 1.0:
        signal /= peak
    return signal

def to_int16(signal: np.ndarray) -> np.ndarray:
    return (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)

def ensure_mono(signal: np.ndarray) -> np.ndarray:
    return signal if signal.ndim == 1 else np.mean(signal, axis=1)

def peak_normalize(signal: np.ndarray) -> np.ndarray:
    peak = np.abs(signal).max()
    return signal / peak if peak > 0 else signal

def trim_silence(signal: np.ndarray, threshold_db: float = 30.0) -> np.ndarray:
    if signal.size == 0:
        return signal
    peak = np.abs(signal).max()
    if peak <= 1e-8:
        return signal
    threshold = peak * (10 ** (-threshold_db / 20.0))
    idx = np.where(np.abs(signal) >= threshold)[0]
    if idx.size == 0:
        return signal
    return signal[int(idx[0]): int(idx[-1]) + 1]

def load_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    data = ensure_mono(data)
    data = normalize_to_float32(data)
    return data, sr

def save_wav(path: Path, signal: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, to_int16(signal))

def print_stats(label: str, signal: np.ndarray, sr: int):
    duration = len(signal) / sr
    peak     = float(np.abs(signal).max()) if signal.size else 0.0
    rms      = float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0
    peak_db  = round(20 * np.log10(peak) if peak > 1e-9 else -np.inf, 1)
    rms_db   = round(20 * np.log10(rms)  if rms  > 1e-9 else -np.inf, 1)
    print(f"  {label:<35} {duration:>7.2f}s  "
          f"peak {peak_db:>6} dBFS  rms {rms_db:>6} dBFS  "
          f"samples {len(signal):,}")


# ─────────────────────────────────────────────────────────────────────────────
# Transcript
# ─────────────────────────────────────────────────────────────────────────────

def read_turns(transcript_path: str, participant_label: str = "Participant") -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas not installed: pip install pandas")
        return []
    try:
        df = pd.read_csv(transcript_path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        if "speaker" not in df.columns:
            df = pd.read_csv(transcript_path)
            df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"[ERROR] Could not read transcript: {e}")
        return []
    rows = df[df["speaker"].str.strip() == participant_label]
    turns = []
    for _, row in rows.iterrows():
        try:
            s, e = float(row["start_time"]), float(row["stop_time"])
            if e > s:
                turns.append({"start": s, "stop": e})
        except Exception:
            continue
    return turns

def extract_participant_audio(
    waveform: np.ndarray, sr: int,
    turns: list[dict], silence_sec: float = 0.1
) -> np.ndarray:
    silence = np.zeros(int(silence_sec * sr), dtype=np.float32)
    chunks  = []
    for t in turns:
        s = max(0, int(t["start"] * sr))
        e = min(len(waveform), int(t["stop"] * sr))
        chunk = waveform[s:e].astype(np.float32)
        if len(chunk) > 0:
            chunks.append(chunk)
            chunks.append(silence)
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Demucs
# ─────────────────────────────────────────────────────────────────────────────

def run_demucs(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Run Demucs using the current venv's Python only.
    Uses scipy to write/read WAV — never touches torchaudio.
    """
    demucs_python = str(Path(sys.executable))

    # Verify demucs is available in the current venv
    check = subprocess.run(
        [demucs_python, "-c", "import demucs"],
        capture_output=True,
    )
    if check.returncode != 0:
        print("[WARN] demucs not installed in current venv.")
        print("       Fix: pip install demucs")
        print("       Keeping normalised audio.")
        return waveform

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        tmp_wav = tmp_dir / "input.wav"

        # Write input with scipy only — no torchaudio
        wavfile.write(str(tmp_wav), sr, to_int16(waveform))

        print(f"  → Running Demucs (30–120 s) ...")

        # Completely isolated env — only current venv, no PYTHONPATH leaking
        import os
        env = os.environ.copy()
        env['TORCHAUDIO_BACKEND'] = 'soundfile'
        venv_bin = str(Path(demucs_python).parent)
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")
        # Remove any PYTHONPATH that could pull in the other venv
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONSTARTUP", None)

        result = subprocess.run(
            [demucs_python, "/tmp/run_demucs.py",
             "--two-stems=vocals",
             "-n", "htdemucs",
             "--out", str(tmp_dir),
             str(tmp_wav)],
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            print("[WARN] Demucs failed — keeping normalised audio.")
            last_lines = [l for l in result.stderr.strip().splitlines() if l.strip()][-3:]
            for line in last_lines:
                print(f"       {line}")
            return waveform

        vocals_path = tmp_dir / "htdemucs" / "input" / "vocals.wav"
        if not vocals_path.exists():
            print(f"[WARN] Demucs output not found at {vocals_path}")
            return waveform

        # Read output with scipy only — no torchaudio
        sr_v, vocals = wavfile.read(str(vocals_path))
        vocals = ensure_mono(vocals)
        vocals = normalize_to_float32(vocals)

        if sr_v != sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, sr_v)
            vocals = resample_poly(vocals, sr // g, sr_v // g).astype(np.float32)
            print(f"  → Resampled {sr_v} Hz → {sr} Hz")

        print(f"  → Demucs done. Final: {len(vocals)/sr:.1f}s at {sr} Hz")
        return vocals


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_audio(
    waveform: np.ndarray, sr: int,
    seg_dur: float = 5.0, overlap: float = 0.5, min_dur: float = 1.0
) -> list[dict]:
    seg_samples = int(seg_dur * sr)
    hop_samples = int(seg_samples * (1.0 - overlap))
    min_samples = int(min_dur * sr)
    segments, seg_id, start = [], 0, 0
    while start < len(waveform):
        end   = start + seg_samples
        chunk = waveform[start:end]
        if len(chunk) >= min_samples:
            if len(chunk) < seg_samples:
                chunk = np.pad(chunk, (0, seg_samples - len(chunk)))
            segments.append({
                "id":    seg_id,
                "start": start / sr,
                "end":   min(end, len(waveform)) / sr,
                "wave":  chunk,
            })
            seg_id += 1
        start += hop_samples
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Save one WAV per pipeline stage so you can listen to each step."
    )
    parser.add_argument("wav",
                        help="Path to the raw interview WAV (e.g. 300_AUDIO.wav)")
    parser.add_argument("--transcript",           metavar="PATH",
                        help="Path to the DAIC transcript CSV (_TRANSCRIPT.csv)")
    parser.add_argument("--output_dir",           default="pipeline_output",
                        help="Folder where stage WAVs will be saved (default: pipeline_output/)")
    parser.add_argument("--participant_label",    default="Participant")
    parser.add_argument("--no_denoise",           action="store_true",
                        help="Skip Demucs")
    parser.add_argument("--segment_duration",     type=float, default=5.0)
    parser.add_argument("--overlap_ratio",        type=float, default=0.5)
    parser.add_argument("--silence_threshold_db", type=float, default=30.0)
    parser.add_argument("--max_segments",         type=int,   default=None,
                        help="Only save the first N segments (default: all)")
    args = parser.parse_args()

    if not Path(args.wav).exists():
        print(f"[ERROR] File not found: {args.wav}")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  WAV Pipeline Stage Exporter")
    print(f"  Input:  {Path(args.wav).name}")
    print(f"  Output: {out.resolve()}")
    print(f"{'═'*60}\n")

    # ── Stage 0 — raw ────────────────────────────────────────────────────────
    print("[Stage 0] Loading raw audio ...")
    raw, sr = load_wav(args.wav)
    save_wav(out / "stage0_raw.wav", raw, sr)
    print_stats("stage0_raw.wav", raw, sr)

    # ── Stage 1 — participant extraction ─────────────────────────────────────
    if args.transcript:
        print(f"\n[Stage 1] Extracting participant turns ...")
        turns = read_turns(args.transcript, args.participant_label)
        print(f"  → {len(turns)} turns found  "
              f"({sum(t['stop']-t['start'] for t in turns):.1f}s of speech)")
        participant = extract_participant_audio(raw, sr, turns)
    else:
        print("\n[Stage 1] No transcript provided — using full audio.")
        participant = raw.copy()

    save_wav(out / "stage1_participant.wav", participant, sr)
    print_stats("stage1_participant.wav", participant, sr)

    # ── Stage 2 — silence trim ───────────────────────────────────────────────
    print("\n[Stage 2] Trimming silence ...")
    trimmed = trim_silence(participant, threshold_db=args.silence_threshold_db)
    save_wav(out / "stage2_trimmed.wav", trimmed, sr)
    print_stats("stage2_trimmed.wav", trimmed, sr)

    # ── Stage 3 — normalise ──────────────────────────────────────────────────
    print("\n[Stage 3] Peak normalising ...")
    normalised = peak_normalize(trimmed)
    save_wav(out / "stage3_normalised.wav", normalised, sr)
    print_stats("stage3_normalised.wav", normalised, sr)

    # ── Stage 4 — Demucs ─────────────────────────────────────────────────────
    if args.no_denoise:
        print("\n[Stage 4] Demucs skipped (--no_denoise).")
        denoised = normalised
    else:
        print("\n[Stage 4] Denoising with Demucs ...")
        denoised = run_demucs(normalised, sr)
        denoised = peak_normalize(denoised)
        save_wav(out / "stage4_denoised.wav", denoised, sr)
        print_stats("stage4_denoised.wav", denoised, sr)

    # ── Stage 5 — segments ───────────────────────────────────────────────────
    print(f"\n[Stage 5] Segmenting "
          f"({args.segment_duration}s, {int(args.overlap_ratio*100)}% overlap) ...")
    segments = segment_audio(
        denoised, sr,
        seg_dur=args.segment_duration,
        overlap=args.overlap_ratio,
    )

    to_save = segments if args.max_segments is None else segments[:args.max_segments]
    seg_dir = out / "segments"
    seg_dir.mkdir(exist_ok=True)

    for seg in to_save:
        fname = seg_dir / f"segment_{seg['id']:03d}.wav"
        save_wav(fname, seg["wave"], sr)

    print(f"  → {len(segments)} segments total, {len(to_save)} saved")
    for seg in to_save:
        print(f"     segment_{seg['id']:03d}.wav  "
              f"{seg['start']:.2f}s – {seg['end']:.2f}s")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Done. Files saved to: {out.resolve()}")
    print(f"{'═'*60}")
    print(f"\n  {out}/")
    print(f"  ├── stage0_raw.wav")
    print(f"  ├── stage1_participant.wav")
    print(f"  ├── stage2_trimmed.wav")
    print(f"  ├── stage3_normalised.wav")
    if not args.no_denoise:
        print(f"  ├── stage4_denoised.wav")
    print(f"  └── segments/")
    for seg in to_save:
        print(f"       segment_{seg['id']:03d}.wav")
    print()


if __name__ == "__main__":
    main()