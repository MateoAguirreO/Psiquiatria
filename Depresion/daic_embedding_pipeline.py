"""
DAIC-WOZ Embedding Pipeline
============================
Adaptación de embedding_pipeline_5s.py para el dataset DAIC-WOZ.

Diferencias clave con el pipeline original:
  1. Usa los transcripts (_TRANSCRIPT.csv) para extraer SOLO los segmentos
     del Participant, eliminando a Ellie sin Demucs (más limpio).
  2. Aplica denoising (noisereduce) a cada segmento del participante antes
     de concatenar el audio limpio.
  3. Lee las etiquetas desde los CSV oficiales del AVEC2017.
  4. Produce el mismo formato de manifest.json y estructura de carpetas
     que embedding_pipeline_5s.py → compatible con dl_script1 y dl_script2.

Estructura de entrada esperada:
    daic_wav/
      300/
        300_AUDIO.wav          ← audio completo de la sesión
        300_TRANSCRIPT.csv     ← timestamps por turno (Ellie / Participant)
      301/
        ...

Estructura de salida (idéntica a embedding_pipeline_5s.py):
    embeddings_daic/
      depresion/
        class_0/
          xlsr-300m/
            300/
              segment_0/
                embedding.json
        class_1/
          ...
      manifest.json

Uso:
  python daic_embedding_pipeline.py \\
    --daic_dir   "/ruta/a/daic_wav/" \\
    --labels_dir "/ruta/a/csvs/" \\
    --output_dir "/ruta/a/embeddings_daic/" \\
    --models xlsr-300m wav2vec2-large-robust \\
    --splits train dev test

  # Para generar solo train+dev (más frecuente):
  python daic_embedding_pipeline.py \\
    --daic_dir   "/ruta/a/daic_wav/" \\
    --labels_dir "/ruta/a/csvs/" \\
    --output_dir "/ruta/a/embeddings_daic/" \\
    --models xlsr-300m wav2vec2-large-robust \\
    --splits train dev
"""

import argparse
import json
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from tqdm import tqdm
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    WavLMModel,
    WhisperModel,
    WhisperProcessor,
    HubertModel,
    AutoFeatureExtractor,
    AutoModel,
)

# noisereduce para denoising ligero (pip install noisereduce)
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DAIC_Pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Registro de modelos (igual que embedding_pipeline_5s.py)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "xlsr-300m": {
        "hf_id":              "facebook/wav2vec2-xls-r-300m",
        "model_cls":          Wav2Vec2Model,
        "processor_id":       "facebook/wav2vec2-xls-r-300m",
        "n_transformer_layers": 24,
        "hidden_size":        1024,
        "multilingual":       True,
        "languages":          "128 idiomas (incluye español e inglés)",
        "priority":           1,
    },
    "xlsr-53": {
        "hf_id":              "facebook/wav2vec2-large-xlsr-53",
        "model_cls":          Wav2Vec2Model,
        "processor_id":       "facebook/wav2vec2-large-xlsr-53",
        "n_transformer_layers": 24,
        "hidden_size":        1024,
        "multilingual":       True,
        "languages":          "53 idiomas (incluye español e inglés)",
        "priority":           2,
    },
    "whisper-large-encoder": {
        "hf_id":              "openai/whisper-large-v3",
        "model_cls":          WhisperModel,
        "processor_id":       "openai/whisper-large-v3",
        "n_transformer_layers": 32,
        "hidden_size":        1280,
        "multilingual":       True,
        "languages":          "99 idiomas",
        "priority":           3,
    },
    "wav2vec2-large-robust": {
        "hf_id":              "facebook/wav2vec2-large-robust",
        "model_cls":          Wav2Vec2Model,
        "processor_id":       "facebook/wav2vec2-large-robust",
        "n_transformer_layers": 24,
        "hidden_size":        1024,
        "multilingual":       False,
        "languages":          "Inglés (LibriSpeech + noisy)",
        "priority":           4,
    },
    "wavlm-large": {
        "hf_id":              "microsoft/wavlm-large",
        "model_cls":          WavLMModel,
        "processor_id":       "microsoft/wavlm-large",
        "n_transformer_layers": 24,
        "hidden_size":        1024,
        "multilingual":       False,
        "languages":          "Inglés",
        "priority":           5,
    },
    "hubert-large": {
        "hf_id":              "facebook/hubert-large-ls960-ft",
        "model_cls":          HubertModel,
        "processor_id":       "facebook/wav2vec2-large-xlsr-53",
        "n_transformer_layers": 24,
        "hidden_size":        1024,
        "multilingual":       False,
        "languages":          "Inglés",
        "priority":           6,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DAICConfig:
    # Rutas
    daic_dir:   str = ""         # carpeta raíz con subcarpetas por participante
    labels_dir: str = ""         # carpeta con los CSVs del AVEC2017
    output_dir: str = "embeddings_daic"
    condition:  str = "depresion"

    # Splits a procesar
    splits: list = field(default_factory=lambda: ["train", "dev", "test"])

    # Audio
    target_sr:            int   = 16_000
    segment_duration:     float = 5.0    # igual que embedding_pipeline_5s.py
    overlap_ratio:        float = 0.5    # 50% overlap → hop de 2.5s
    min_segment_duration: float = 1.0

    # Denoising
    denoise: bool = True
    # Cuántos segundos del inicio del audio usar como perfil de ruido
    # Los primeros segundos suelen ser silencio/ruido de fondo antes de que hable
    noise_profile_sec: float = 0.5

    # Modelos
    models: list = field(default_factory=lambda: ["xlsr-300m", "wav2vec2-large-robust"])

    # Extracción
    pooling_strategy: str = "attention"
    batch_size:       int = 4
    output_format:    str = "json"
    use_fp16:         bool = True

    # Nombre del hablante del participante en el transcript
    # (en algunos archivos puede variar — lo hacemos configurable)
    participant_speaker_label: str = "Participant"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Etiquetas — Lectura de CSVs AVEC2017
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev":   "dev_split_Depression_AVEC2017.csv",
    "test":  "full_test_split.csv",   # contiene etiquetas PHQ_Binary
}

# Columnas de ID y etiqueta difieren entre train/dev y test
SPLIT_COLS = {
    "train": ("Participant_ID", "PHQ8_Binary"),
    "dev":   ("Participant_ID", "PHQ8_Binary"),
    "test":  ("Participant_ID", "PHQ_Binary"),
}


def load_labels(labels_dir: str, splits: list[str]) -> dict[int, int]:
    """
    Lee los CSVs del AVEC2017 y devuelve {participant_id: binary_label}.
    Mezcla todos los splits pedidos.
    """
    labels = {}
    labels_dir = Path(labels_dir)

    for split in splits:
        fname = SPLIT_FILES.get(split)
        if fname is None:
            logger.warning(f"Split '{split}' no reconocido, omitiendo.")
            continue
        fpath = labels_dir / fname
        if not fpath.exists():
            logger.warning(f"CSV no encontrado: {fpath}")
            continue

        id_col, label_col = SPLIT_COLS[split]
        df = pd.read_csv(fpath)

        if id_col not in df.columns or label_col not in df.columns:
            logger.error(f"Columnas esperadas '{id_col}', '{label_col}' no están en {fname}")
            continue

        for _, row in df.iterrows():
            pid  = int(row[id_col])
            lbl  = int(row[label_col])
            labels[pid] = lbl
            logger.debug(f"  {split}: {pid} → clase {lbl}")

        logger.info(f"  Split '{split}': {len(df)} participantes cargados desde {fname}")

    logger.info(f"Total etiquetas cargadas: {len(labels)}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transcript — Extracción de segmentos del Participante
# ─────────────────────────────────────────────────────────────────────────────

def load_participant_segments_from_transcript(
    transcript_path: Path,
    participant_label: str = "Participant",
) -> list[dict]:
    """
    Lee el transcript TSV y devuelve solo los turnos del participante.
    Retorna lista de {start_time, stop_time} en segundos.
    """
    try:
        df = pd.read_csv(transcript_path, sep="\t")
    except Exception as e:
        logger.error(f"No se pudo leer transcript {transcript_path}: {e}")
        return []

    # Columnas esperadas: start_time, stop_time, speaker, value
    required = {"start_time", "stop_time", "speaker"}
    if not required.issubset(df.columns):
        logger.error(f"Transcript {transcript_path} no tiene columnas {required}")
        return []

    participant_rows = df[df["speaker"] == participant_label]
    if participant_rows.empty:
        logger.warning(f"No se encontraron turnos de '{participant_label}' en {transcript_path}")
        return []

    segments = []
    for _, row in participant_rows.iterrows():
        t_start = float(row["start_time"])
        t_stop  = float(row["stop_time"])
        if t_stop > t_start:
            segments.append({"start": t_start, "stop": t_stop})

    return segments


def extract_participant_audio(
    waveform: np.ndarray,
    sr: int,
    participant_segments: list[dict],
    min_duration: float = 0.1,
) -> np.ndarray:
    """
    Concatena los fragmentos de audio del participante en un único array.
    Inserta 0.1s de silencio entre turnos para preservar separación natural.
    """
    silence_samples = int(0.1 * sr)
    silence = np.zeros(silence_samples, dtype=np.float32)

    chunks = []
    for seg in participant_segments:
        s = int(seg["start"] * sr)
        e = int(seg["stop"]  * sr)
        # Clamp a los límites del array
        s = max(0, min(s, len(waveform)))
        e = max(s, min(e, len(waveform)))
        chunk = waveform[s:e]
        if len(chunk) >= int(min_duration * sr):
            chunks.append(chunk)
            chunks.append(silence)

    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Denoising
# ─────────────────────────────────────────────────────────────────────────────

def denoise_waveform(
    waveform: np.ndarray,
    sr: int,
    noise_profile_sec: float = 0.5,
) -> np.ndarray:
    """
    Denoising espectral con noisereduce.
    Usa los primeros 'noise_profile_sec' segundos como muestra de ruido.
    Si noisereduce no está instalado, devuelve el audio sin cambios.
    """
    if not NOISEREDUCE_AVAILABLE:
        logger.warning("noisereduce no instalado — omitiendo denoising. "
                       "Instalar con: pip install noisereduce")
        return waveform

    noise_samples = int(noise_profile_sec * sr)
    noise_clip = waveform[:noise_samples] if len(waveform) > noise_samples else waveform

    try:
        denoised = nr.reduce_noise(
            y=waveform,
            sr=sr,
            y_noise=noise_clip,
            stationary=False,   # Non-stationary: mejor para entrevistas clínicas
            prop_decrease=0.75, # Reducción moderada — no agresiva para no distorsionar voz
        )
        return denoised.astype(np.float32)
    except Exception as e:
        logger.warning(f"Denoising falló ({e}) — usando audio original.")
        return waveform


# ─────────────────────────────────────────────────────────────────────────────
# 4. Preprocesamiento general
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: Path, target_sr: int = 16_000) -> np.ndarray:
    """Carga audio, convierte a mono, resamplea."""
    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        waveform_np, sr = librosa.load(str(path), sr=None, mono=True)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze().numpy().astype(np.float32)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    peak = np.abs(waveform).max()
    return waveform / peak if peak > 0 else waveform


def segment_audio(
    waveform: np.ndarray,
    sr: int,
    segment_duration: float,
    overlap_ratio: float,
    min_duration: float,
) -> list[dict]:
    """
    Ventaneo deslizante con overlap — idéntico a embedding_pipeline_5s.py.
    5s + 50% overlap → hop de 2.5s.
    """
    seg_samples = int(segment_duration * sr)
    hop_samples = int(seg_samples * (1.0 - overlap_ratio))
    min_samples = int(min_duration * sr)
    n_total = len(waveform)

    segments = []
    seg_id = 0
    start = 0

    while start < n_total:
        end = start + seg_samples
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
# 5. Attention Pooling (igual que embedding_pipeline_5s.py)
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
# 6. Extractor de embeddings (igual que embedding_pipeline_5s.py)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingExtractor:
    def __init__(self, model_key: str, config: DAICConfig, device: torch.device):
        self.model_key = model_key
        self.config    = config
        self.device    = device
        self.meta      = MODEL_REGISTRY[model_key]
        self.is_whisper = "whisper" in model_key.lower()
        self._load_model()
        self.attention_pool = AttentionPooling(
            self.meta["hidden_size"]
        ).to(device).eval()

    def _load_model(self):
        hf_id   = self.meta["hf_id"]
        proc_id = self.meta["processor_id"]
        logger.info(f"  Cargando {hf_id} ...")

        if self.is_whisper:
            self.processor = WhisperProcessor.from_pretrained(proc_id)
            self.model     = WhisperModel.from_pretrained(hf_id).to(self.device).eval()
        else:
            self.processor = AutoFeatureExtractor.from_pretrained(proc_id)
            self.model     = self.meta["model_cls"].from_pretrained(
                hf_id, output_hidden_states=True
            ).to(self.device).eval()

        if self.config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        logger.info(f"  ✅ {self.model_key} listo en {self.device}")

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
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)

        if self.config.use_fp16 and self.device.type == "cuda":
            input_values = input_values.half()

        # Pasamos attention_mask solo si todos los segmentos son del mismo
        # tamaño (sin padding) — cuando hay padding el mask del processor
        # está en dimensión de samples, no de tokens, y causa el error
        # "size of tensor a (80000) must match tensor b (249)".
        # Sin attention_mask el modelo hace mean-pool sobre todos los tokens,
        # lo cual es correcto porque nuestros segmentos ya tienen tamaño fijo.
        with torch.no_grad():
            outputs = self.model(
                input_values,
                attention_mask=None,
                output_hidden_states=True,
            )

        last_hidden    = outputs.last_hidden_state
        transformer_hs = outputs.hidden_states[1:]

        return self._pool_and_package(
            last_hidden, transformer_hs, None, segment_metas
        )

    def _extract_whisper(self, waveforms, segment_metas):
        inputs = self.processor(
            waveforms,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs["input_features"].to(self.device)
        if self.config.use_fp16 and self.device.type == "cuda":
            input_features = input_features.half()

        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_features,
                output_hidden_states=True,
            )

        last_hidden    = encoder_outputs.last_hidden_state
        transformer_hs = encoder_outputs.hidden_states[1:]

        return self._pool_and_package(
            last_hidden, transformer_hs, None, segment_metas
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
            h = last_hidden[i].unsqueeze(0).float()
            mask_i = attention_mask[i].unsqueeze(0) if attention_mask is not None else None

            if strategy == "mean":
                if mask_i is not None:
                    mask_exp = mask_i.unsqueeze(-1).float()
                    emb = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-8)
                else:
                    emb = h.mean(dim=1)
                emb = emb.squeeze(0)

            elif strategy == "attention":
                emb = self.attention_pool(h, mask_i).squeeze(0)

            elif strategy == "last":
                if mask_i is not None:
                    lengths = mask_i.sum(dim=1).long()
                    emb = h[0, lengths[0] - 1, :]
                else:
                    emb = h[0, -1, :]

            else:  # weighted_sum
                n_layers = len(transformer_hs)
                weights  = torch.softmax(
                    torch.ones(n_layers, device=self.device), dim=0
                )
                stacked = torch.stack(
                    [transformer_hs[l][i].float() for l in range(n_layers)], dim=0
                )
                h_ws = (stacked * weights.view(-1, 1, 1)).sum(dim=0).unsqueeze(0)
                emb = h_ws.mean(dim=1).squeeze(0)

            records.append({
                **meta,
                "model":     self.model_key,
                "pooling":   strategy,
                "embedding": emb.cpu().float().tolist(),
            })

        return records


# ─────────────────────────────────────────────────────────────────────────────
# 7. Guardado — igual que embedding_pipeline_5s.py
# ─────────────────────────────────────────────────────────────────────────────

def save_json(record: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: v for k, v in record.items() if k != "waveform"}, f)


def build_output_path(
    output_root: str,
    condition: str,
    audio_id: str,
    segment_id: int,
    model_key: str,
    class_label: Optional[str],
    fmt: str,
) -> Path:
    class_segment = class_label if class_label is not None else "class_unknown"
    return Path(output_root) / condition / class_segment / model_key / audio_id / f"segment_{segment_id}" / f"embedding.{fmt}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

class DAICEmbeddingPipeline:

    def __init__(self, config: DAICConfig):
        self.config = config
        self.device = self._resolve_device()
        logger.info(f"Device: {self.device}")

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("GPU no disponible — usando CPU (lento para large models)")
        return torch.device("cpu")

    def _find_audio_file(self, participant_dir: Path, pid: int) -> Optional[Path]:
        """
        Busca el archivo de audio principal del participante.
        DAIC-WOZ nombra los archivos como: {pid}_AUDIO.wav o {pid}_P.wav
        """
        candidates = [
            participant_dir / f"{pid}_AUDIO.wav",
            participant_dir / f"{pid}_P.wav",
            participant_dir / f"{pid}.wav",
        ]
        # También buscar cualquier wav si los anteriores no existen
        for c in candidates:
            if c.exists():
                return c

        wavs = list(participant_dir.glob("*.wav"))
        # Excluir archivos de Ellie o del entrevistador si los hay separados
        wavs = [w for w in wavs if "ellie" not in w.name.lower()
                and "interviewer" not in w.name.lower()]
        if wavs:
            return wavs[0]
        return None

    def _find_transcript_file(self, participant_dir: Path, pid: int) -> Optional[Path]:
        """Busca el transcript CSV del participante."""
        candidates = [
            participant_dir / f"{pid}_TRANSCRIPT.csv",
            participant_dir / f"{pid}_transcript.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        csvs = list(participant_dir.glob("*TRANSCRIPT*"))
        return csvs[0] if csvs else None

    def process_participant(
        self,
        pid: int,
        class_id: str,
        class_label: str,
        extractor: EmbeddingExtractor,
        manifest: list,
        processed: set,
    ):
        """
        Procesa un participante completo:
          1. Carga audio
          2. Extrae turnos del participante desde el transcript
          3. Concatena audio solo del participante
          4. Denoise (opcional)
          5. Normaliza
          6. Segmenta en ventanas de 5s con 50% overlap
          7. Extrae embeddings y guarda
        """
        cfg = self.config
        daic_dir = Path(cfg.daic_dir)

        # Buscar carpeta del participante (puede ser "300", "300_P", etc.)
        participant_dirs = [
            daic_dir / str(pid),
            daic_dir / f"{pid}_P",
            daic_dir / f"{pid:03d}",
        ]
        participant_dir = next((d for d in participant_dirs if d.exists()), None)
        if participant_dir is None:
            logger.warning(f"  [{pid}] Carpeta no encontrada, omitiendo.")
            return

        # Buscar audio
        audio_path = self._find_audio_file(participant_dir, pid)
        if audio_path is None:
            logger.warning(f"  [{pid}] Audio no encontrado en {participant_dir}")
            return

        # Buscar transcript
        transcript_path = self._find_transcript_file(participant_dir, pid)
        if transcript_path is None:
            logger.warning(
                f"  [{pid}] Transcript no encontrado — "
                "se usará el audio completo (incluye a Ellie)"
            )

        # Cargar audio completo
        try:
            waveform = load_audio(audio_path, cfg.target_sr)
        except Exception as e:
            logger.error(f"  [{pid}] Error cargando audio: {e}")
            return

        # Extraer solo turnos del participante
        if transcript_path is not None:
            participant_turns = load_participant_segments_from_transcript(
                transcript_path,
                participant_label=cfg.participant_speaker_label,
            )
            if participant_turns:
                waveform = extract_participant_audio(
                    waveform, cfg.target_sr, participant_turns
                )
                logger.debug(
                    f"  [{pid}] Extraídos {len(participant_turns)} turnos del participante "
                    f"→ {len(waveform)/cfg.target_sr:.1f}s de audio limpio"
                )
            else:
                logger.warning(f"  [{pid}] No se encontraron turnos del participante en transcript")

        if len(waveform) == 0:
            logger.warning(f"  [{pid}] Audio vacío tras extracción de turnos, omitiendo.")
            return

        # Denoising
        if cfg.denoise:
            waveform = denoise_waveform(waveform, cfg.target_sr, cfg.noise_profile_sec)

        # Normalizar
        waveform = normalize_waveform(waveform)

        # Segmentar
        segments = segment_audio(
            waveform, cfg.target_sr,
            cfg.segment_duration, cfg.overlap_ratio, cfg.min_segment_duration,
        )

        if not segments:
            logger.warning(f"  [{pid}] Sin segmentos válidos tras segmentación.")
            return

        logger.debug(
            f"  [{pid}] clase={class_id} | "
            f"{len(waveform)/cfg.target_sr:.1f}s → {len(segments)} segmentos"
        )

        audio_id = str(pid)

        # Extraer embeddings por batch
        all_records = []
        for batch_start in range(0, len(segments), cfg.batch_size):
            batch = segments[batch_start: batch_start + cfg.batch_size]
            waveforms_batch = [s["waveform"] for s in batch]
            metas = [
                {
                    "audio_id":    audio_id,
                    "class_id":    class_id,
                    "class_label": class_label,
                    "condition":   cfg.condition,
                    "segment_id":  s["segment_id"],
                    "start_time":  s["start_time"],
                    "end_time":    s["end_time"],
                }
                for s in batch
            ]
            try:
                records = extractor.extract_batch(waveforms_batch, metas)
                all_records.extend(records)
            except Exception as e:
                logger.error(f"  [{pid}] Error en batch embedding: {e}")
                continue

        # Guardar registros
        model_key = extractor.model_key
        fmt = cfg.output_format

        for record in all_records:
            seg_id = record["segment_id"]
            key = (audio_id, seg_id, model_key)

            if key in processed:
                continue

            out_path = build_output_path(
                output_root=cfg.output_dir,
                condition=cfg.condition,
                audio_id=audio_id,
                segment_id=seg_id,
                model_key=model_key,
                class_label=class_label,
                fmt=fmt,
            )

            try:
                save_json(record, out_path)
                manifest.append({
                    "condition":   cfg.condition,
                    "audio_id":    audio_id,
                    "class_id":    class_id,
                    "class_label": class_label,
                    "model":       model_key,
                    "segment_id":  seg_id,
                    "start_time":  record["start_time"],
                    "end_time":    record["end_time"],
                    "file":        str(out_path),
                })
                processed.add(key)
            except Exception as e:
                logger.error(f"  [{pid}] seg_{seg_id}: Error al guardar: {e}")

    def run(self):
        cfg = self.config

        # Cargar etiquetas
        logger.info("Cargando etiquetas...")
        labels = load_labels(cfg.labels_dir, cfg.splits)
        if not labels:
            raise RuntimeError("No se cargaron etiquetas. Revisar rutas de CSV.")

        # Construir lista de participantes con clase
        entries = []
        for pid, lbl in sorted(labels.items()):
            class_id    = str(lbl)
            class_label = f"class_{lbl}"
            entries.append({
                "pid":         pid,
                "class_id":    class_id,
                "class_label": class_label,
            })

        n_class0 = sum(1 for e in entries if e["class_id"] == "0")
        n_class1 = sum(1 for e in entries if e["class_id"] == "1")
        logger.info(
            f"Participantes: {len(entries)} total | "
            f"clase0={n_class0} (no deprimido) | clase1={n_class1} (deprimido)"
        )

        # Manifest
        manifest_path = Path(cfg.output_dir) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            logger.info(f"Manifest previo cargado: {len(manifest)} entradas")
        else:
            manifest = []

        processed = {
            (m["audio_id"], m["segment_id"], m["model"])
            for m in manifest
        }

        # Loop por modelo
        for model_key in cfg.models:
            if model_key not in MODEL_REGISTRY:
                logger.warning(f"Modelo '{model_key}' no registrado, omitiendo.")
                continue

            meta_m = MODEL_REGISTRY[model_key]
            logger.info(f"\n{'═'*60}")
            logger.info(f"Modelo: {model_key} | multilingüe={meta_m['multilingual']}")
            logger.info(f"  Idiomas: {meta_m['languages']}")
            logger.info(f"{'═'*60}")

            extractor = EmbeddingExtractor(model_key, cfg, self.device)

            for entry in tqdm(entries, desc=f"[{model_key}]"):
                self.process_participant(
                    pid=entry["pid"],
                    class_id=entry["class_id"],
                    class_label=entry["class_label"],
                    extractor=extractor,
                    manifest=manifest,
                    processed=processed,
                )

            # Guardar manifest tras cada modelo
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"  Manifest guardado: {len(manifest)} entradas")

            del extractor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        logger.info(f"\n✅ Pipeline DAIC-WOZ completo.")
        logger.info(f"   Manifest: {manifest_path}")
        logger.info(f"   Total entradas: {len(manifest)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DAIC-WOZ Embedding Pipeline — compatible con embedding_pipeline_5s.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--daic_dir",   required=True,
                        help="Carpeta raíz de DAIC-WOZ (subcarpetas por participante: 300/, 301/, ...)")
    parser.add_argument("--labels_dir", required=True,
                        help="Carpeta con los CSVs del AVEC2017")
    parser.add_argument("--output_dir", default="embeddings_daic",
                        help="Directorio de salida de embeddings")
    parser.add_argument("--condition",  default="depresion",
                        help="Nombre de la condición (primer nivel de carpeta)")
    parser.add_argument("--models",     nargs="+",
                        default=["xlsr-300m", "wav2vec2-large-robust"],
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Modelos SSL a usar")
    parser.add_argument("--splits",     nargs="+",
                        default=["train", "dev", "test"],
                        choices=["train", "dev", "test"],
                        help="Splits a procesar")
    parser.add_argument("--segment_duration", type=float, default=5.0)
    parser.add_argument("--overlap_ratio",    type=float, default=0.5)
    parser.add_argument("--pooling",   default="attention",
                        choices=["mean", "attention", "weighted_sum", "last"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_denoise", action="store_true",
                        help="Desactivar denoising con noisereduce")
    parser.add_argument("--noise_profile_sec", type=float, default=0.5,
                        help="Segundos del inicio del audio a usar como perfil de ruido")
    parser.add_argument("--participant_label", default="Participant",
                        help="Etiqueta del hablante participante en el transcript")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Desactivar fp16 (usar si hay errores de precisión)")

    args = parser.parse_args()

    config = DAICConfig(
        daic_dir=args.daic_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        condition=args.condition,
        splits=args.splits,
        models=args.models,
        segment_duration=args.segment_duration,
        overlap_ratio=args.overlap_ratio,
        pooling_strategy=args.pooling,
        batch_size=args.batch_size,
        denoise=not args.no_denoise,
        noise_profile_sec=args.noise_profile_sec,
        participant_speaker_label=args.participant_label,
        use_fp16=not args.no_fp16,
    )

    logger.info("Configuración:")
    logger.info(f"  daic_dir:    {config.daic_dir}")
    logger.info(f"  labels_dir:  {config.labels_dir}")
    logger.info(f"  output_dir:  {config.output_dir}")
    logger.info(f"  splits:      {config.splits}")
    logger.info(f"  modelos:     {config.models}")
    logger.info(f"  segmentos:   {config.segment_duration}s con {config.overlap_ratio*100:.0f}% overlap")
    logger.info(f"  denoising:   {'sí' if config.denoise else 'no'}")
    logger.info(f"  pooling:     {config.pooling_strategy}")

    pipeline = DAICEmbeddingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()