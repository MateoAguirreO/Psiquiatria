"""
DAIC-WOZ SSL Embedding Pipeline — Merged Version
=================================================
Combines the best of two prior implementations:

  Base:              coworker's daic_embedding_pipeline.py
  Per-layer embeddings: Claude's daic_embedding_pipeline.py
  Denoising:         batch_copy_preprocess_audio.py  (Demucs, htdemucs vocals stem)
  Bug fix:           attention_mask=None for wav2vec batch (coworker finding)
  Audio extraction:  0.1 s silence between turns (coworker — better prosodic boundaries)
  Folder discovery:  flexible multi-pattern search (coworker)
  Label loading:     flexible column detection (Claude) + hardcoded split filenames (coworker)

Key design decisions
--------------------
* Demucs (--two-stems=vocals) is used for noise reduction, consistent with
  batch_copy_preprocess_audio.py.  noisereduce is NOT used.
* attention_mask is passed as None to wav2vec models to avoid the
  sample-level vs token-level dimension mismatch discovered by the coworker.
* Per-layer embeddings are extracted and saved (all transformer layers),
  controllable via --no_per_layer.
* 0.1 s silence is inserted between participant turns before concatenation.
* Silence trimming (threshold-based) is applied before Demucs.
* Audio helpers (ensure_mono, normalize_to_float32, to_int16) are taken
  from batch_copy_preprocess_audio.py — they handle int16 / int32 / uint8.

Dataset structure expected
--------------------------
    daic_root/
      300/          OR  300_P/   (both patterns are tried)
        300_AUDIO.wav   OR  300_P.wav   OR  300.wav
        300_TRANSCRIPT.csv   (columns: start_time, stop_time, speaker, value)

Split CSVs (place all in a single --labels_dir folder)
    train_split_Depression_AVEC2017.csv  →  Participant_ID, PHQ8_Binary
    dev_split_Depression_AVEC2017.csv    →  Participant_ID, PHQ8_Binary
    full_test_split.csv                  →  Participant_ID, PHQ_Binary

Output structure (identical to both prior versions)
    embeddings_daic/
      {condition}/
        {class_label}/
          {model_key}/
            {audio_id}/
              segment_{n}/
                embedding.json
      manifest.json

Usage
./.venv/bin/python daic_embedding_pipeline_merged.py --daic_root /home/ci2dt2-ai/Proyectos/Psiquiatria/DATA-DAIC --labels_dir /home/ci2dt2-ai/Proyectos/Psiquiatria/DATA-DAIC  --output_dir /home/ci2dt2-ai/Proyectos/Psiquiatria/embeddings_daic --condition depression --splits train dev test --models xlsr-300m xlsr-53 whisper-large-encoder wav2vec2-large-robust wavlm-large hubert-large --pooling attention --overlap_ratio 0.5xlsr-300m",
    "wav2vec2-large-robust",
Optional flags
--------------
  --no_denoise          Skip Demucs (faster, lower quality)
  --no_trim_silence     Skip silence trimming
  --no_per_layer        Skip per-layer embedding extraction
  --silence_threshold_db FLOAT   (default 30.0)
  --batch_size INT               (default 4)
  --no_fp16             Disable mixed precision
  --overwrite           Reprocess already-saved segments
  --participant_label   STR      (default "Participant")
  --list_models         Print available models and exit
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from scipy.io import wavfile
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    HubertModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WavLMModel,
    WhisperModel,
    WhisperProcessor,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DAICMergedPipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "xlsr-300m": {
        "hf_id":                "facebook/wav2vec2-xls-r-300m",
        "model_cls":            Wav2Vec2Model,
        "processor_id":         "facebook/wav2vec2-xls-r-300m",
        "n_transformer_layers": 24,
        "hidden_size":          1024,
        "multilingual":         True,
        "languages":            "128 languages (includes Spanish & English)",
        "priority":             1,
        "notes":                "Recommended #1 for Spanish.",
    },
    "xlsr-53": {
        "hf_id":                "facebook/wav2vec2-large-xlsr-53",
        "model_cls":            Wav2Vec2Model,
        "processor_id":         "facebook/wav2vec2-large-xlsr-53",
        "n_transformer_layers": 24,
        "hidden_size":          1024,
        "multilingual":         True,
        "languages":            "53 languages (includes Spanish & English)",
        "priority":             2,
        "notes":                "Recommended #2.",
    },
    "whisper-large-encoder": {
        "hf_id":                "openai/whisper-large-v3",
        "model_cls":            WhisperModel,
        "processor_id":         "openai/whisper-large-v3",
        "n_transformer_layers": 32,
        "hidden_size":          1280,
        "multilingual":         True,
        "languages":            "99 languages",
        "priority":             3,
        "notes":                "Recommended #3.",
    },
    "wav2vec2-large-robust": {
        "hf_id":                "facebook/wav2vec2-large-robust",
        "model_cls":            Wav2Vec2Model,
        "processor_id":         "facebook/wav2vec2-large-robust",
        "n_transformer_layers": 24,
        "hidden_size":          1024,
        "multilingual":         False,
        "languages":            "English (LibriSpeech + noisy)",
        "priority":             4,
        "notes":                "Robust English baseline.",
    },
    "wavlm-large": {
        "hf_id":                "microsoft/wavlm-large",
        "model_cls":            WavLMModel,
        "processor_id":         "microsoft/wavlm-large",
        "n_transformer_layers": 24,
        "hidden_size":          1024,
        "multilingual":         False,
        "languages":            "English (LibriSpeech 960h)",
        "priority":             5,
        "notes":                "SOTA on SUPERB benchmark.",
    },
    "hubert-large": {
        "hf_id":                "facebook/hubert-large-ls960-ft",
        "model_cls":            HubertModel,
        "processor_id":         "facebook/wav2vec2-large-xlsr-53",
        "n_transformer_layers": 24,
        "hidden_size":          1024,
        "multilingual":         False,
        "languages":            "English (LibriSpeech 960h)",
        "priority":             6,
        "notes":                "Strong English baseline.",
    },
}

DEFAULT_MODELS = list(MODEL_REGISTRY.keys())

# Split CSV filenames and their ID/label column names (coworker convention)
SPLIT_FILES = {
    "train": ("train_split_Depression_AVEC2017.csv", "Participant_ID", "PHQ8_Binary"),
    "dev":   ("dev_split_Depression_AVEC2017.csv",   "Participant_ID", "PHQ8_Binary"),
  "test":  ("full_test_split.csv",  "Participant_ID", "PHQ_Binary"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Paths
    daic_root:  str = ""
    labels_dir: str = ""
    output_dir: str = "embeddings_daic"
    condition:  str = "depression"

    # Splits to load labels from
    splits: list = field(default_factory=lambda: ["train", "dev", "test"])

    # Audio
    target_sr:            int   = 16_000
    participant_label:    str   = "Participant"
    turn_silence_sec:     float = 0.1   # silence inserted between participant turns

    # Preprocessing
    trim_silence:          bool  = True
    silence_threshold_db:  float = 30.0
    denoise:               bool  = True

    # Segmentation
    segment_duration:     float = 5.0
    overlap_ratio:        float = 0.5
    min_segment_duration: float = 1.0

    # Embedding
    pooling_strategy:   str  = "attention"
    extract_all_layers: bool = True
    batch_size:         int  = 4
    fp16:               bool = True

    # Output
    output_format: str  = "json"
    overwrite:     bool = False

    # Models
    models: list = field(default_factory=lambda: DEFAULT_MODELS)


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers  (from batch_copy_preprocess_audio.py — most robust versions)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_to_float32(signal: np.ndarray) -> np.ndarray:
    """Convert any integer PCM dtype to float32 in [-1, 1]."""
    if signal.dtype == np.int16:
        return signal.astype(np.float32) / 32768.0
    if signal.dtype == np.int32:
        return signal.astype(np.float32) / 2147483648.0
    if signal.dtype == np.uint8:
        return (signal.astype(np.float32) - 128.0) / 128.0
    signal = signal.astype(np.float32)
    peak = np.max(np.abs(signal)) if signal.size else 0.0
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
    peak = np.max(np.abs(signal))
    if peak <= 1e-8:
        return signal
    threshold = peak * (10 ** (-threshold_db / 20.0))
    idx = np.where(np.abs(signal) >= threshold)[0]
    if idx.size == 0:
        return signal
    return signal[int(idx[0]): int(idx[-1]) + 1]


def load_audio(path: str | Path, target_sr: int = 16_000) -> np.ndarray:
    """Load any audio file → mono float32 at target_sr. Uses scipy/librosa only."""
    try:
        sr, data = wavfile.read(str(path))
        data = ensure_mono(data)
        data = normalize_to_float32(data)
    except Exception:
        data, sr = librosa.load(str(path), sr=None, mono=True)
        data = data.astype(np.float32)

    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Demucs denoising  (from batch_copy_preprocess_audio.py)
# ─────────────────────────────────────────────────────────────────────────────

def _find_demucs_python() -> str | None:
    """Use only the current venv's Python — never a foreign venv."""
    python = str(Path(sys.executable))
    check = subprocess.run(
        [python, "-c", "import demucs"],
        capture_output=True,
    )
    return python if check.returncode == 0 else None


def _run_demucs(input_wav: Path, out_dir: Path) -> Path:
    """Run Demucs in an isolated subprocess; return path to vocals.wav.
    Uses a wrapper script that monkey-patches torchaudio.save → soundfile
    to avoid the torchcodec dependency inside torchaudio nightly builds.
    """
    demucs_python = _find_demucs_python()
    if demucs_python is None:
        raise RuntimeError(
            "demucs not installed in current venv. Run: pip install demucs"
        )

    # Write the wrapper script that patches torchaudio.save before calling demucs
    wrapper = out_dir / "_run_demucs_wrapper.py"
    wrapper.write_text(
        "import sys\n"
        "import soundfile as sf\n"
        "import torchaudio\n"
        "def _save_patched(path, src, sample_rate, **kwargs):\n"
        "    data = src.numpy()\n"
        "    if data.ndim == 2:\n"
        "        data = data.T\n"
        "    sf.write(str(path), data, sample_rate)\n"
        "torchaudio.save = _save_patched\n"
        "from demucs.__main__ import main\n"
        "main()\n"
    )

    import os
    env = os.environ.copy()
    env["PATH"] = str(Path(demucs_python).parent) + ":" + env.get("PATH", "")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)

    result = subprocess.run(
        [demucs_python, str(wrapper),
         "--two-stems=vocals",
         "-n", "htdemucs",
         "--out", str(out_dir),
         str(input_wav)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stderr
        )
    vocals_path = out_dir / "htdemucs" / input_wav.stem / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(f"Demucs did not produce vocals at: {vocals_path}")
    return vocals_path


def denoise_waveform(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Write waveform to a temp WAV, run Demucs vocal separation,
    return the denoised (vocals) signal resampled to sr.
    Falls back to the original signal on any error.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        tmp_wav = tmp_dir / "input.wav"
        wavfile.write(str(tmp_wav), sr, to_int16(waveform))

        try:
            vocals_path = _run_demucs(tmp_wav, tmp_dir)
            sr_v, vocals = wavfile.read(str(vocals_path))
            vocals = ensure_mono(vocals)
            vocals = normalize_to_float32(vocals)
            if sr_v != sr:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, sr_v)
                vocals = resample_poly(vocals, sr // g, sr_v // g).astype(np.float32)
                logger.debug(f"Demucs output resampled {sr_v} Hz → {sr} Hz")
            return vocals.astype(np.float32)
        except Exception as exc:
            logger.warning(f"Demucs failed — using original signal. Reason: {exc}")
            return waveform


# ─────────────────────────────────────────────────────────────────────────────
# Label loading
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(labels_dir: str, splits: list[str]) -> dict[int, int]:
    """
    Read AVEC2017 split CSVs from labels_dir.
    Returns {participant_id (int): binary_label (int)}.
    Uses the hardcoded filenames / column names from SPLIT_FILES.
    Falls back to auto-detecting columns if names differ slightly.
    """
    labels: dict[int, int] = {}
    labels_path = Path(labels_dir)

    for split in splits:
        if split not in SPLIT_FILES:
            logger.warning(f"Unknown split '{split}' — skipping.")
            continue

        fname, id_col, label_col = SPLIT_FILES[split]
        fpath = labels_path / fname

        if not fpath.exists():
            logger.warning(f"Label CSV not found: {fpath}")
            continue

        df = pd.read_csv(fpath)

        # Flexible column detection (fall back if exact names not found)
        if id_col not in df.columns:
            id_col = next(
                (c for c in df.columns if "participant" in c.lower()), None
            )
        if label_col not in df.columns:
            label_col = next(
                (c for c in df.columns if "binary" in c.lower() or "label" in c.lower()),
                None,
            )

        if id_col is None or label_col is None:
            logger.error(
                f"Could not find ID/label columns in {fname}. "
                f"Available: {df.columns.tolist()}"
            )
            continue

        for _, row in df.iterrows():
            labels[int(row[id_col])] = int(row[label_col])

        logger.info(f"Loaded {len(df)} labels from {fname} (split={split})")

    logger.info(f"Total labels loaded: {len(labels)}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery  (flexible, coworker style)
# ─────────────────────────────────────────────────────────────────────────────

def _find_audio_file(participant_dir: Path, pid: int) -> Optional[Path]:
    for candidate in [
        participant_dir / f"{pid}_AUDIO.wav",
        participant_dir / f"{pid}_P.wav",
        participant_dir / f"{pid}.wav",
    ]:
        if candidate.exists():
            return candidate
    wavs = [
        w for w in participant_dir.glob("*.wav")
        if "ellie" not in w.name.lower() and "interviewer" not in w.name.lower()
    ]
    return wavs[0] if wavs else None


def _find_transcript_file(participant_dir: Path, pid: int) -> Optional[Path]:
    for candidate in [
        participant_dir / f"{pid}_TRANSCRIPT.csv",
        participant_dir / f"{pid}_transcript.csv",
    ]:
        if candidate.exists():
            return candidate
    found = list(participant_dir.glob("*TRANSCRIPT*"))
    return found[0] if found else None


def build_entries(daic_root: str, labels: dict[int, int]) -> list[dict]:
    """
    Walk daic_root and build one entry dict per participant.
    Only participants present in labels are included.
    """
    root = Path(daic_root)
    entries = []

    for pid, lbl in sorted(labels.items()):
        # Try several folder naming conventions
        participant_dir = next(
            (
                d for d in [
                    root / str(pid),
                    root / f"{pid}_P",
                    root / f"{pid:03d}",
                ]
                if d.is_dir()
            ),
            None,
        )

        if participant_dir is None:
            logger.warning(f"[{pid}] Participant folder not found in {daic_root} — skipping.")
            continue

        audio_path = _find_audio_file(participant_dir, pid)
        if audio_path is None:
            logger.warning(f"[{pid}] No audio file found in {participant_dir} — skipping.")
            continue

        transcript_path = _find_transcript_file(participant_dir, pid)
        if transcript_path is None:
            logger.warning(f"[{pid}] No transcript found — full audio will be used (includes interviewer).")

        class_label = f"class_{lbl}"
        entries.append({
            "participant_id":  pid,
            "audio_id":        str(pid),
            "audio_path":      str(audio_path),
            "transcript_path": str(transcript_path) if transcript_path else None,
            "class_id":        str(lbl),
            "class_label":     class_label,
        })

    logger.info(f"Built {len(entries)} participant entries from {daic_root}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Transcript parsing
# ─────────────────────────────────────────────────────────────────────────────

def read_participant_turns(
    transcript_path: str,
    participant_label: str = "Participant",
) -> list[dict]:
    """
    Parse a DAIC-WOZ transcript CSV/TSV.
    Returns list of {start, stop} dicts for the participant's turns only.
    """
    try:
        df = pd.read_csv(transcript_path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        df = pd.read_csv(transcript_path)
        df.columns = [c.strip() for c in df.columns]

    required = {"start_time", "stop_time", "speaker"}
    if not required.issubset(df.columns):
        logger.error(
            f"Transcript missing required columns {required}. "
            f"Found: {df.columns.tolist()}"
        )
        return []

    rows = df[df["speaker"].str.strip() == participant_label]
    if rows.empty:
        logger.warning(f"No '{participant_label}' turns found in {transcript_path}")
        return []

    turns = []
    for _, row in rows.iterrows():
        try:
            start = float(row["start_time"])
            stop  = float(row["stop_time"])
            if stop > start:
                turns.append({"start": start, "stop": stop})
        except (ValueError, KeyError):
            continue

    return turns


# ─────────────────────────────────────────────────────────────────────────────
# Participant audio extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_participant_audio(
    waveform: np.ndarray,
    sr: int,
    turns: list[dict],
    silence_between_sec: float = 0.1,
) -> np.ndarray:
    """
    Concatenate participant speech turns.
    Inserts silence_between_sec of zeros between turns to preserve
    natural prosodic boundaries (coworker design choice).
    """
    silence = np.zeros(int(silence_between_sec * sr), dtype=np.float32)
    chunks = []

    for turn in turns:
        s = max(0, int(turn["start"] * sr))
        e = min(len(waveform), int(turn["stop"] * sr))
        chunk = waveform[s:e]
        if len(chunk) > 0:
            chunks.append(chunk.astype(np.float32))
            chunks.append(silence)

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_audio(
    waveform: np.ndarray,
    sr: int,
    segment_duration: float,
    overlap_ratio: float,
    min_duration: float,
) -> list[dict]:
    """5-second sliding window with 50% overlap (or configured values)."""
    seg_samples = int(segment_duration * sr)
    hop_samples = int(seg_samples * (1.0 - overlap_ratio))
    min_samples = int(min_duration * sr)
    n_total = len(waveform)

    segments = []
    seg_id = 0
    start = 0

    while start < n_total:
        end   = start + seg_samples
        chunk = waveform[start:end]
        if len(chunk) >= min_samples:
            if len(chunk) < seg_samples:
                chunk = np.pad(chunk, (0, seg_samples - len(chunk)))
            segments.append({
                "segment_id": seg_id,
                "start_time": round(start / sr, 4),
                "end_time":   round(min(end, n_total) / sr, 4),
                "waveform":   chunk,
            })
            seg_id += 1
        start += hop_samples

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Attention pooling
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = torch.nn.Linear(hidden_size, hidden_size)
        self.v = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.float()
        scores = torch.tanh(self.W(hidden_states))
        logits = self.v(scores).squeeze(-1)
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extractor
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingExtractor:
    def __init__(self, model_key: str, config: PipelineConfig, device: torch.device):
        self.model_key = model_key
        self.config    = config
        self.device    = device
        self.meta      = MODEL_REGISTRY[model_key]
        self.is_whisper = "whisper" in model_key.lower()
        self._load_model()
        self.attention_pool = (
            AttentionPooling(self.meta["hidden_size"]).to(device).eval()
        )

    def _load_model(self):
        hf_id   = self.meta["hf_id"]
        proc_id = self.meta["processor_id"]
        logger.info(f"Loading {self.model_key} ({hf_id}) ...")

        if self.is_whisper:
            self.processor = WhisperProcessor.from_pretrained(proc_id)
            self.model = WhisperModel.from_pretrained(hf_id, output_hidden_states=True)
        else:
            # Multi-level fallback for processor loading (Claude version robustness)
            for loader in [
                lambda: Wav2Vec2Processor.from_pretrained(proc_id),
                lambda: AutoProcessor.from_pretrained(proc_id),
                lambda: AutoFeatureExtractor.from_pretrained(proc_id),
                lambda: Wav2Vec2FeatureExtractor.from_pretrained(proc_id),
            ]:
                try:
                    self.processor = loader()
                    break
                except Exception:
                    continue
            self.model = self.meta["model_cls"].from_pretrained(
                hf_id, output_hidden_states=True
            )

        self.model.eval().to(self.device)
        if self.config.fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        logger.info(f"  ✓ {self.model_key} ready on {self.device}")

    @torch.no_grad()
    def extract_batch(
        self,
        waveforms: list[np.ndarray],
        segment_metas: list[dict],
    ) -> list[dict]:
        if self.is_whisper:
            return self._extract_whisper(waveforms, segment_metas)
        return self._extract_wav2vec(waveforms, segment_metas)

    def _extract_wav2vec(self, waveforms, segment_metas):
        inputs = self.processor(
            waveforms,
            sampling_rate=self.config.target_sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        if self.config.fp16 and self.device.type == "cuda":
            input_values = input_values.half()

        # NOTE: attention_mask is intentionally NOT passed to wav2vec models.
        # The processor returns a sample-level mask, but the model expects a
        # token-level mask — passing it causes a dimension mismatch error on
        # batches with variable-length inputs. Since our segments are all
        # fixed-length (padded in segment_audio), mean-pooling over all tokens
        # is correct and avoids the crash. (Bug identified by coworker.)
        outputs = self.model(
            input_values=input_values,
            attention_mask=None,
            output_hidden_states=True,
        )

        return self._pool_and_package(
            outputs.last_hidden_state,
            outputs.hidden_states[1:],
            attention_mask=None,
            segment_metas=segment_metas,
        )

    def _extract_whisper(self, waveforms, segment_metas):
        n_samples = getattr(
            getattr(self.processor, "feature_extractor", None),
            "n_samples",
            self.config.target_sr * 30,
        )
        inputs = self.processor(
            waveforms,
            sampling_rate=self.config.target_sr,
            return_tensors="pt",
            padding="max_length",
            max_length=n_samples,
            truncation=True,
        )
        input_features = inputs["input_features"].to(self.device)
        dtype = next(self.model.parameters()).dtype
        input_features = input_features.to(dtype=dtype)

        encoder_outputs = self.model.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )

        return self._pool_and_package(
            encoder_outputs.last_hidden_state,
            encoder_outputs.hidden_states[1:],
            attention_mask=None,
            segment_metas=segment_metas,
        )

    def _pool_and_package(
        self,
        last_hidden:    torch.Tensor,
        transformer_hs: tuple,
        attention_mask: Optional[torch.Tensor],
        segment_metas:  list[dict],
    ) -> list[dict]:
        strategy = self.config.pooling_strategy
        records  = []

        for i, meta in enumerate(segment_metas):
            h = last_hidden[i].unsqueeze(0).float()   # (1, T, D)

            # ── Main embedding ──────────────────────────────────────────────
            if strategy == "mean":
                emb = h.mean(dim=1).squeeze(0)

            elif strategy == "attention":
                emb = self.attention_pool(h, attention_mask).squeeze(0)

            elif strategy == "last":
                emb = h[0, -1, :]

            elif strategy == "weighted_sum":
                n_layers = len(transformer_hs)
                w = torch.softmax(torch.ones(n_layers, device=self.device), dim=0)
                stacked = torch.stack(
                    [transformer_hs[l][i].float() for l in range(n_layers)], dim=0
                )
                emb = (stacked * w.view(-1, 1, 1)).sum(dim=0).mean(dim=0)

            else:
                raise ValueError(f"Unknown pooling strategy: {strategy!r}")

            emb_list = emb.float().cpu().numpy().tolist()

            # ── Per-layer embeddings (Claude version feature) ───────────────
            per_layer = None
            if self.config.extract_all_layers and transformer_hs:
                per_layer = {}
                for layer_idx, layer_h in enumerate(transformer_hs):
                    lh = layer_h[i].float().unsqueeze(0)   # (1, T, D)
                    if strategy == "mean":
                        le = lh.mean(dim=1).squeeze(0)
                    elif strategy == "attention":
                        le = self.attention_pool(lh, attention_mask).squeeze(0)
                    elif strategy == "last":
                        le = lh[0, -1, :]
                    else:   # weighted_sum falls back to mean per layer
                        le = lh.mean(dim=1).squeeze(0)
                    per_layer[f"layer_{layer_idx + 1}"] = le.cpu().numpy().tolist()

            record = {
                "audio_id":      meta["audio_id"],
                "class_id":      meta.get("class_id"),
                "class_label":   meta.get("class_label"),
                "condition":     meta.get("condition"),
                "segment_id":    meta["segment_id"],
                "start_time":    meta["start_time"],
                "end_time":      meta["end_time"],
                "model":         self.model_key,
                "multilingual":  self.meta["multilingual"],
                "pooling":       strategy,
                "embedding_dim": len(emb_list),
                "embedding":     emb_list,
            }
            if per_layer is not None:
                record["per_layer_embeddings"] = per_layer

            records.append(record)

        return records


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_record(record: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def build_output_path(
    output_root: str,
    condition: str,
    audio_id: str,
    segment_id: int,
    model_key: str,
    class_label: Optional[str],
) -> Path:
    class_segment = class_label if class_label is not None else "class_unknown"
    return (
        Path(output_root)
        / condition
        / class_segment
        / model_key
        / audio_id
        / f"segment_{segment_id}"
        / "embedding.json"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DAICEmbeddingPipeline:
    """
    End-to-end pipeline:
      1. Load AVEC2017 labels from split CSVs
      2. Discover participant folders / audio / transcripts
      3. Per participant:
           a. Load full audio
           b. Extract participant turns from transcript
           c. Concatenate participant audio (with 0.1 s inter-turn silence)
           d. Trim leading/trailing silence
           e. Peak-normalise
           f. Demucs vocal separation (noise reduction)
           g. Peak-normalise again
           h. Sliding-window segmentation (5 s, 50 % overlap)
           i. SSL embedding extraction (all models)
           j. Save per segment × model × layer
      4. Write manifest.json
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = self._resolve_device()
        logger.info(f"Device: {self.device}")

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("No GPU found — running on CPU (slow for large models).")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Per-participant preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, entry: dict) -> np.ndarray:
        cfg = self.config
        pid = entry["participant_id"]

        # 1. Load full interview audio
        waveform = load_audio(entry["audio_path"], cfg.target_sr)

        # 2. Extract participant turns
        if entry["transcript_path"] is not None:
            turns = read_participant_turns(
                entry["transcript_path"],
                participant_label=cfg.participant_label,
            )
            if turns:
                logger.info(
                    f"[{pid}] {len(turns)} turns → "
                    f"{sum(t['stop']-t['start'] for t in turns):.1f}s of speech"
                )
                waveform = extract_participant_audio(
                    waveform, cfg.target_sr, turns,
                    silence_between_sec=cfg.turn_silence_sec,
                )
            else:
                logger.warning(f"[{pid}] No participant turns found — using full audio.")
        else:
            logger.warning(f"[{pid}] No transcript — using full audio (includes interviewer).")

        if waveform.size == 0:
            return waveform

        # 3. Silence trimming
        if cfg.trim_silence:
            waveform = trim_silence(waveform, cfg.silence_threshold_db)

        # 4. Peak normalise
        waveform = peak_normalize(waveform)

        # 5. Demucs denoising
        if cfg.denoise:
            waveform = denoise_waveform(waveform, cfg.target_sr)
            waveform = peak_normalize(waveform)  # re-normalise after denoising

        return waveform

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        cfg = self.config

        print(f"\n{'═'*60}")
        print(f"  DAIC-WOZ Merged Embedding Pipeline")
        print(f"  Root:      {cfg.daic_root}")
        print(f"  Labels:    {cfg.labels_dir}  (splits: {cfg.splits})")
        print(f"  Output:    {cfg.output_dir}")
        print(f"  Condition: {cfg.condition}")
        print(f"  Models:    {cfg.models}")
        print(f"  Denoise:   {'Demucs' if cfg.denoise else 'off'}")
        print(f"  Per-layer: {cfg.extract_all_layers}")
        print(f"{'═'*60}\n")

        labels  = load_labels(cfg.labels_dir, cfg.splits)
        entries = build_entries(cfg.daic_root, labels)

        if not entries:
            raise RuntimeError("No valid participants found. Check --daic_root and --labels_dir.")

        from collections import Counter
        counts = Counter(e["class_id"] for e in entries)
        for cls_id, n in sorted(counts.items()):
            print(f"  class_{cls_id}: {n} participants")
        print()

        # Load or initialise manifest
        manifest_path = Path(cfg.output_dir) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            logger.info(f"Existing manifest loaded: {len(manifest)} entries")
        else:
            manifest = []

        processed: set[tuple] = {
            (m["audio_id"], m["segment_id"], m["model"])
            for m in manifest
        }

        for model_key in cfg.models:
            if model_key not in MODEL_REGISTRY:
                logger.warning(f"Model '{model_key}' not in registry — skipping.")
                continue

            meta_m = MODEL_REGISTRY[model_key]
            logger.info(f"\n{'─'*60}")
            logger.info(f"Model: {model_key}  |  {meta_m['hf_id']}")
            logger.info(f"{'─'*60}")

            extractor = EmbeddingExtractor(model_key, cfg, self.device)

            for entry in tqdm(entries, desc=f"[{model_key}]"):
                pid = entry["participant_id"]
                try:
                    waveform = self.preprocess(entry)
                except Exception as exc:
                    logger.error(f"[{pid}] Preprocessing failed: {exc}", exc_info=True)
                    continue

                if waveform.size == 0:
                    logger.warning(f"[{pid}] Empty waveform after preprocessing — skipping.")
                    continue

                segments = segment_audio(
                    waveform, cfg.target_sr,
                    cfg.segment_duration, cfg.overlap_ratio, cfg.min_segment_duration,
                )
                if not segments:
                    logger.warning(f"[{pid}] No valid segments — skipping.")
                    continue

                logger.debug(f"[{pid}] {len(waveform)/cfg.target_sr:.1f}s → {len(segments)} segments")

                for batch_start in range(0, len(segments), cfg.batch_size):
                    batch = segments[batch_start: batch_start + cfg.batch_size]
                    waveforms_b = [s["waveform"] for s in batch]
                    metas_b = [
                        {
                            "audio_id":    entry["audio_id"],
                            "class_id":    entry["class_id"],
                            "class_label": entry["class_label"],
                            "condition":   cfg.condition,
                            "segment_id":  s["segment_id"],
                            "start_time":  s["start_time"],
                            "end_time":    s["end_time"],
                        }
                        for s in batch
                    ]

                    try:
                        records = extractor.extract_batch(waveforms_b, metas_b)
                    except Exception as exc:
                        logger.error(f"[{pid}] Batch embedding error: {exc}", exc_info=True)
                        continue

                    for record in records:
                        seg_id = record["segment_id"]
                        key    = (entry["audio_id"], seg_id, model_key)

                        if key in processed and not cfg.overwrite:
                            continue

                        out_path = build_output_path(
                            output_root=cfg.output_dir,
                            condition=cfg.condition,
                            audio_id=entry["audio_id"],
                            segment_id=seg_id,
                            model_key=model_key,
                            class_label=entry["class_label"],
                        )
                        try:
                            save_record(record, out_path)
                            manifest.append({
                                "condition":   cfg.condition,
                                "audio_id":    entry["audio_id"],
                                "class_id":    entry["class_id"],
                                "class_label": entry["class_label"],
                                "model":       model_key,
                                "segment_id":  seg_id,
                                "start_time":  record["start_time"],
                                "end_time":    record["end_time"],
                                "file":        str(out_path),
                            })
                            processed.add(key)
                        except Exception as exc:
                            logger.error(
                                f"[{entry['audio_id']}] seg_{seg_id} save error: {exc}",
                                exc_info=True,
                            )

            # Save manifest after every model (safe to interrupt between models)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Manifest saved: {len(manifest)} entries → {manifest_path}")

            del extractor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        logger.info(f"\n✅ Pipeline complete.")
        logger.info(f"   Manifest: {manifest_path}  ({len(manifest)} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DAIC-WOZ Merged SSL Embedding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--daic_root",   required=True,
                        help="Root folder containing participant subfolders (300/, 300_P/, …)")
    parser.add_argument("--labels_dir",  required=True,
                        help="Folder containing AVEC2017 split CSVs")
    parser.add_argument("--output_dir",  default="embeddings_daic")
    parser.add_argument("--condition",   default="depression")
    parser.add_argument("--splits",      nargs="+", default=["train", "dev", "test"],
                        choices=["train", "dev", "test"])
    parser.add_argument("--models",      nargs="+", default=DEFAULT_MODELS,
                        choices=DEFAULT_MODELS)
    parser.add_argument("--segment_duration",  type=float, default=5.0)
    parser.add_argument("--overlap_ratio",     type=float, default=0.5)
    parser.add_argument("--pooling",           default="attention",
                        choices=["mean", "attention", "weighted_sum", "last"])
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--silence_threshold_db", type=float, default=30.0)
    parser.add_argument("--participant_label", default="Participant")
    parser.add_argument("--no_denoise",        action="store_true",
                        help="Skip Demucs denoising")
    parser.add_argument("--no_trim_silence",   action="store_true",
                        help="Skip silence trimming")
    parser.add_argument("--no_per_layer",      action="store_true",
                        help="Skip per-layer embedding extraction")
    parser.add_argument("--no_fp16",           action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--overwrite",         action="store_true",
                        help="Reprocess already-saved segments")
    parser.add_argument("--list_models",       action="store_true",
                        help="Print available models and exit")

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:\n")
        for key, m in sorted(MODEL_REGISTRY.items(), key=lambda x: x[1]["priority"]):
            flag = "🌍" if m["multilingual"] else "🇬🇧"
            print(f"  [{m['priority']}] {flag}  {key:<30} {m['languages']}")
            print(f"        {m['notes']}")
            print()
        return

    config = PipelineConfig(
        daic_root=args.daic_root,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        condition=args.condition,
        splits=args.splits,
        models=args.models,
        segment_duration=args.segment_duration,
        overlap_ratio=args.overlap_ratio,
        pooling_strategy=args.pooling,
        batch_size=args.batch_size,
        silence_threshold_db=args.silence_threshold_db,
        participant_label=args.participant_label,
        trim_silence=not args.no_trim_silence,
        denoise=not args.no_denoise,
        extract_all_layers=not args.no_per_layer,
        fp16=not args.no_fp16,
        overwrite=args.overwrite,
    )

    pipeline = DAICEmbeddingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()