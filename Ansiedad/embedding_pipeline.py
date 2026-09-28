"""
SSL Speech Embedding Pipeline for Depression/Anxiety Detection
==============================================================
Soporta estructura de dataset con clases:
    wav/
      0/   ← clase 0 (control / sin depresión)
          audio_001.wav
          audio_002.wav
      1/   ← clase 1 (depresión / ansiedad)
          audio_003.wav
          audio_004.wav
Se ejecuta asi:
python "embedding_pipeline.py" \
  --audio_dir "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Ansiedad/wav/" \
  --output_dir "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Ansiedad/embeddings/" \
  --segment_duration 5.0 \
  --overlap_ratio 0.5 \
  --pooling attention \
  --format json
"""

import os
import json
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torchaudio
import librosa
from tqdm import tqdm
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    HubertModel,
    WavLMModel,
    WhisperModel,
    WhisperProcessor,
    AutoProcessor,
    AutoFeatureExtractor,
    AutoModel,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EmbeddingPipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Registro de modelos — con priorización para español
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # ══════════════════════════════════════════════════════
    # GRUPO A — Multilingüe / Cross-lingual (RECOMENDADOS para español)
    # ══════════════════════════════════════════════════════

    "xlsr-300m": {
        # XLS-R: wav2vec2 entrenado en 128 idiomas, 436k horas
        # INCLUYE español de múltiples variedades (CommonVoice, MLS, BABEL)
        # Mejor opción para español colombiano según literatura cross-lingual
        "hf_id": "facebook/wav2vec2-xls-r-300m",
        "model_cls": Wav2Vec2Model,
        "processor_id": "facebook/wav2vec2-xls-r-300m",
        "n_transformer_layers": 24,
        "hidden_size": 1024,
        "multilingual": True,
        "languages": "128 idiomas (incluye español)",
        "priority": 1,
        "notes": (
            "RECOMENDADO #1 para español. XLS-R entrena con 436k h en 128 idiomas. "
            "Captura fonética y prosodia del español nativo. "
            "Maji et al. (Interspeech 2024) valida cross-lingual en depression detection."
        ),
    },

    "xlsr-53": {
        # XLSR-53: precursor de XLS-R, 53 idiomas
        # Fine-tuned en español por la comunidad HuggingFace
        "hf_id": "facebook/wav2vec2-large-xlsr-53",
        "model_cls": Wav2Vec2Model,
        "processor_id": "facebook/wav2vec2-large-xlsr-53",
        "n_transformer_layers": 24,
        "hidden_size": 1024,
        "multilingual": True,
        "languages": "53 idiomas (incluye español)",
        "priority": 2,
        "notes": (
            "RECOMENDADO #2. Base de todos los modelos multilingüe de español. "
            "Contrastive pre-training compartido cross-lingual = representaciones "
            "fonéticas más transferibles para depresión en español."
        ),
    },

    "whisper-large-encoder": {
        # Whisper large-v3: encoder multilingüe de OpenAI
        # Entrenado en 680k h de audio web en 99 idiomas incluido español
        # El encoder aprende representaciones acústico-prosódicas ricas
        # NO usar el decoder (solo queremos embeddings del encoder)
        "hf_id": "openai/whisper-large-v3",
        "model_cls": WhisperModel,
        "processor_id": "openai/whisper-large-v3",
        "n_transformer_layers": 32,
        "hidden_size": 1280,
        "multilingual": True,
        "languages": "99 idiomas (incluye español nativo)",
        "priority": 3,
        "notes": (
            "RECOMENDADO #3. Whisper encoder captura prosodia y ritmo del habla "
            "con calidad excepcional. Entrenado en 680k h incluyendo español diverso. "
            "Dimensión 1280. Usar SOLO encoder (sin decoder)."
        ),
    },

    # ══════════════════════════════════════════════════════
    # GRUPO B — Inglés puro (para ablation / baseline comparison)
    # ══════════════════════════════════════════════════════

    "wav2vec2-large-robust": {
        # Entrenado en audio ruidoso/telefónico (librispeech + libri-light)
        # Más robusto a condiciones de grabación clínica que el large estándar
        # INGLÉS PURO — usar como baseline comparativo
        "hf_id": "facebook/wav2vec2-large-robust",
        "model_cls": Wav2Vec2Model,
        "processor_id": "facebook/wav2vec2-large-robust",
        "n_transformer_layers": 24,
        "hidden_size": 1024,
        "multilingual": False,
        "languages": "Inglés (LibriSpeech + noisy)",
        "priority": 4,
        "notes": (
            "Baseline inglés. Huang et al. (2024) Sci.Rep. usa este tipo de modelo. "
            "Útil para comparar con modelos multilingüe en ablation study."
        ),
    },

    "wavlm-large": {
        # WavLM: SOTA en SUPERB benchmark (emotion, speaker, depression)
        # INGLÉS PURO — mejor modelo monolingüe para inglés
        "hf_id": "microsoft/wavlm-large",
        "model_cls": WavLMModel,
        "processor_id": "microsoft/wavlm-large",
        "n_transformer_layers": 24,
        "hidden_size": 1024,
        "multilingual": False,
        "languages": "Inglés (LibriSpeech 960h)",
        "priority": 5,
        "notes": (
            "SOTA en SUPERB. Maji et al. (Interspeech 2024): WavLM > HuBERT en "
            "depression detection. Para español, comparar vs xlsr-300m."
        ),
    },

    "hubert-large": {
        "hf_id": "facebook/hubert-large-ls960-ft",
        "model_cls": HubertModel,
        "processor_id": "facebook/wav2vec2-large-xlsr-53",  # compatible processor
        "n_transformer_layers": 24,
        "hidden_size": 1024,
        "multilingual": False,
        "languages": "Inglés (LibriSpeech 960h)",
        "priority": 6,
        "notes": "Fuerte baseline inglés. Buenas representaciones emocionales.",
    },
}

# Conjunto recomendado para español
DEFAULT_MODELS_SPANISH = []
# Suite completa incluyendo ablation
DEFAULT_MODELS_ALL = list(MODEL_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Configuración central
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Preprocesamiento
    target_sr: int = 16_000
    normalize_audio: bool = True

    # ── Segmentación
    # Literatura (Zhang et al. 2024): segmentos de 5-7s + 50% overlap
    # Para audios >3 min con 5s + 50% overlap → ~72 segmentos/audio
    # Esto multiplica x72 los datos de entrenamiento disponibles (~80 → ~5760 samples)
    segment_duration: float = 5.0
    overlap_ratio: float = 0.5
    min_segment_duration: float = 1.0

    # ── Extracción de embeddings
    # "attention" = Zhang et al. (2024) recomendado para depression
    # "weighted_sum" = SUPERB benchmark recomendado para paralinguistic tasks
    pooling_strategy: str = "attention"
    extract_all_layers: bool = True    # exportar hidden states por capa
    batch_size: int = 4
    fp16: bool = True

    # ── Salida
    output_dir: str = "embeddings"
    output_format: str = "json"        # "json" | "npy" | "parquet"

    # ── Filtros de nombre de archivo
    include_substrings: list[str] = field(default_factory=lambda: ["_recortado_denoised"])
    exclude_substrings: list[str] = field(default_factory=list)

    # ── Modelos
    models: list = field(default_factory=lambda: DEFAULT_MODELS_ALL)


# ─────────────────────────────────────────────────────────────────────────────
# Detección de estructura del dataset
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


def discover_dataset(
    audio_root: str,
    include_substrings: Optional[list[str]] = None,
    exclude_substrings: Optional[list[str]] = None,
) -> list[dict]:
    """
    Detecta automáticamente la estructura del dataset.

    Soporta:
      Modo A — Con clases explícitas (tu caso):
        wav/
          0/  audio_001.wav ...   ← class_id = "0"
          1/  audio_003.wav ...   ← class_id = "1"

      Modo B — Plano (sin clases):
        wav/
          audio_001.wav ...       ← class_id = None

    Retorna lista de dicts:
      { path, audio_id, class_id, class_label }
    """
    root = Path(audio_root)
    entries = []

    include_substrings = [s.lower() for s in (include_substrings or []) if s]
    exclude_substrings = [s.lower() for s in (exclude_substrings or []) if s]

    def _matches_filters(path: Path) -> bool:
        name = path.name.lower()
        if include_substrings and not any(s in name for s in include_substrings):
            return False
        if exclude_substrings and any(s in name for s in exclude_substrings):
            return False
        return True

    # Buscar subdirectorios que sean nombres de clase (0, 1, o strings)
    subdirs = [d for d in root.iterdir() if d.is_dir()]

    if subdirs:
        # Comprobar si los subdirectorios parecen clases
        class_dirs = [d for d in subdirs if d.name.isdigit() or
                      d.name.lower() in {"depresion", "control", "positive", "negative",
                                          "depression", "healthy", "ansiedad", "anxiety",
                                          "dep", "ctrl", "pos", "neg", "0", "1"}]
        if class_dirs:
            # Modo A — estructura con clases
            logger.info(f"Estructura detectada: dataset con {len(class_dirs)} clases")
            for class_dir in sorted(class_dirs):
                class_id = class_dir.name
                audio_files = sorted([
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in AUDIO_EXTENSIONS and _matches_filters(f)
                ])
                for f in audio_files:
                    entries.append({
                        "path": str(f),
                        "audio_id": f.stem,
                        "class_id": class_id,
                        "class_label": f"class_{class_id}",
                    })
                logger.info(f"  Clase '{class_id}': {len(audio_files)} archivos")
            return entries

    # Modo B — plano
    logger.info("Estructura detectada: dataset plano (sin clases)")
    audio_files = sorted([
        f for f in root.rglob("*")
        if f.suffix.lower() in AUDIO_EXTENSIONS and _matches_filters(f)
    ])
    for f in audio_files:
        entries.append({
            "path": str(f),
            "audio_id": f.stem,
            "class_id": None,
            "class_label": None,
        })
    logger.info(f"  Total archivos: {len(audio_files)}")
    return entries


def print_dataset_summary(entries: list[dict]):
    """Imprime resumen del dataset descubierto."""
    total = len(entries)
    print(f"\n{'─'*50}")
    print(f"  Dataset: {total} audios encontrados")
    if entries and entries[0]["class_id"] is not None:
        from collections import Counter
        counts = Counter(e["class_id"] for e in entries)
        for cls_id, count in sorted(counts.items()):
            pct = count / total * 100
            print(f"  Clase {cls_id}: {count} audios ({pct:.1f}%)")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocesamiento de audio
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(
    audio_path: str,
    target_sr: int = 16_000,
    normalize: bool = True,
) -> np.ndarray:
    """
    Carga audio, convierte a mono, resamplea a 16kHz, normaliza.
    Soporta .wav, .mp3, .flac, .ogg, .m4a
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        waveform_np, sr = librosa.load(str(path), sr=None, mono=True)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resampleo de alta calidad (ventana Kaiser)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    waveform_np = waveform.squeeze().numpy().astype(np.float32)

    # Normalización pico → previene saturación en transformers
    if normalize:
        peak = np.abs(waveform_np).max()
        if peak > 0:
            waveform_np = waveform_np / peak

    return waveform_np


def segment_audio(
    waveform: np.ndarray,
    sr: int,
    segment_duration: float,
    overlap_ratio: float,
    min_duration: float,
) -> list[dict]:
    """
    Ventaneo deslizante con overlap.

    Justificación (Zhang et al., 2024 + Du et al., 2023):
    - 5s + 50% overlap es el sweet spot para audios clínicos largos
    - Maximiza n° de segmentos (esencial con ~80 audios)
    - Preserva suficiente contexto prosódico para depression detection
    - Los modelos wav2vec2/XLS-R manejan hasta ~20s sin degradación
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
                "end_time": round(min(end, n_total) / sr, 4),
                "waveform": chunk,
            })
            seg_id += 1
        start += hop_samples

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Attention Pooling (Zhang et al., 2024 — estrategia recomendada)
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
        hidden_states = hidden_states.float()            # FIX: normaliza fp16→fp32
        scores = torch.tanh(self.W(hidden_states))
        logits = self.v(scores).squeeze(-1)
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Extractor por modelo
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingExtractor:
    """
    Carga un modelo SSL y extrae embeddings para batches de segmentos.
    Maneja correctamente Whisper (encoder-only) vs wav2vec2/HuBERT/WavLM.
    """

    def __init__(self, model_key: str, config: PipelineConfig, device: torch.device):
        self.model_key = model_key
        self.config = config
        self.device = device
        self.meta = MODEL_REGISTRY[model_key]
        self.is_whisper = "whisper" in model_key.lower()
        self._load_model()
        self.attention_pool = AttentionPooling(
            self.meta["hidden_size"]
        ).to(device).eval()

    def _load_model(self):
        hf_id = self.meta["hf_id"]
        proc_id = self.meta["processor_id"]
        logger.info(f"Cargando {self.model_key} ({hf_id}) ...")
        logger.info(f"  Multilingüe: {self.meta['multilingual']} | Idiomas: {self.meta['languages']}")

        if self.is_whisper:
            self.processor = WhisperProcessor.from_pretrained(proc_id)
            self.model = WhisperModel.from_pretrained(
                hf_id, output_hidden_states=True
            )
        else:
            ModelCls = self.meta["model_cls"]
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(proc_id)
            except Exception:
                try:
                    self.processor = AutoProcessor.from_pretrained(proc_id)
                except Exception:
                    logger.warning(
                        "Processor con tokenizer no disponible; usando feature extractor."
                    )
                    try:
                        self.processor = AutoFeatureExtractor.from_pretrained(proc_id)
                    except Exception:
                        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(proc_id)

            self.model = ModelCls.from_pretrained(
                hf_id, output_hidden_states=True
            )

        self.model.eval().to(self.device)

        if self.config.fp16 and self.device.type == "cuda":
            self.model = self.model.half()

    @torch.no_grad()
    def extract_batch(
        self,
        waveforms: list[np.ndarray],
        segment_meta: list[dict],
    ) -> list[dict]:

        if self.is_whisper:
            return self._extract_batch_whisper(waveforms, segment_meta)
        else:
            return self._extract_batch_wav2vec(waveforms, segment_meta)

    def _extract_batch_wav2vec(
        self,
        waveforms: list[np.ndarray],
        segment_meta: list[dict],
    ) -> list[dict]:
        """Extracción para familia wav2vec2 / HuBERT / WavLM / XLS-R."""
        inputs = self.processor(
            waveforms,
            sampling_rate=self.config.target_sr,
            return_tensors="pt",
            padding="max_length",
            max_length=3000,
            truncation=True,
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if self.config.fp16 and self.device.type == "cuda":
            input_values = input_values.half()

        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # hidden_states[0] = CNN encoder output; [1..L] = transformer layers
        transformer_hidden = outputs.hidden_states[1:]
        last_hidden = outputs.last_hidden_state

        return self._pool_and_package(
            last_hidden, transformer_hidden, attention_mask, segment_meta
        )

    def _extract_batch_whisper(
        self,
        waveforms: list[np.ndarray],
        segment_meta: list[dict],
    ) -> list[dict]:
        inputs = self.processor(
            waveforms,
            sampling_rate=self.config.target_sr,
            return_tensors="pt",
            padding="max_length",
            max_length=getattr(
                getattr(self.processor, "feature_extractor", None),
                "n_samples",
                self.config.target_sr * 30,
            ),
            truncation=True,
        )
        input_features = inputs["input_features"].to(self.device)

        # ── FIX: castear al dtype real del modelo (float16 o float32)
        #    en lugar de asumir que fp16 implica half()
        model_dtype = next(self.model.parameters()).dtype
        input_features = input_features.to(dtype=model_dtype)

        encoder_outputs = self.model.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )
        transformer_hidden = encoder_outputs.hidden_states[1:]
        last_hidden = encoder_outputs.last_hidden_state
        attention_mask = None

        return self._pool_and_package(
            last_hidden, transformer_hidden, attention_mask, segment_meta
        )

    def _pool_and_package(
        self,
        last_hidden: torch.Tensor,
        transformer_hidden: tuple,
        attention_mask: Optional[torch.Tensor],
        segment_meta: list[dict],
    ) -> list[dict]:
        results = []
        strategy = self.config.pooling_strategy

        for i, meta in enumerate(segment_meta):
            h_last = last_hidden[i]  # (T, H)
            h_last = h_last.float()  # ← AGREGAR ESTA LÍNEA — normaliza a fp32 para attention pooling

            mask_i = (
                attention_mask[i] if attention_mask is not None
                else torch.ones(h_last.shape[0], dtype=torch.long, device=self.device)
            )

            # FIX: CNN encoder hace downsampling, mask puede no coincidir con h_last
            if mask_i.shape[0] != h_last.shape[0]:
                mask_i = torch.ones(h_last.shape[0], dtype=torch.long, device=self.device)

            # ── Pooling principal ─────────────────────────────────────────
            if strategy == "mean":
                m = mask_i.float().unsqueeze(-1)
                emb = (h_last * m).sum(0) / m.sum(0).clamp(min=1)

            elif strategy == "last":
                last_idx = int(mask_i.sum().item()) - 1
                emb = h_last[last_idx]

            elif strategy == "attention":
                emb = self.attention_pool(
                    h_last.unsqueeze(0), mask_i.unsqueeze(0)
                ).squeeze(0)

            elif strategy == "weighted_sum":
                # Media sobre capas, luego attention pool → SUPERB recomendado
                per_layer = []
                for layer_h in transformer_hidden:
                    p = self.attention_pool(
                        layer_h[i].float().unsqueeze(0), mask_i.unsqueeze(0)
                    ).squeeze(0)
                    per_layer.append(p)
                emb = torch.stack(per_layer, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Pooling desconocido: {strategy}")

            emb_np = emb.float().cpu().numpy().tolist()

            # ── Per-layer embeddings (opcional) ──────────────────────────
            per_layer_embs = None
            if self.config.extract_all_layers:
                per_layer_embs = {}
                for layer_idx, layer_h in enumerate(transformer_hidden):
                    p = self.attention_pool(
                        layer_h[i].float().unsqueeze(0), mask_i.unsqueeze(0)
                    ).squeeze(0)
                    per_layer_embs[f"layer_{layer_idx + 1}"] = p.float().cpu().numpy().tolist()

            record = {
                "audio_id":    meta["audio_id"],
                "class_id":    meta.get("class_id"),      # "0" o "1"
                "class_label": meta.get("class_label"),   # "class_0" / "class_1"
                "segment_id":  meta["segment_id"],
                "start_time":  meta["start_time"],
                "end_time":    meta["end_time"],
                "model":       self.model_key,
                "multilingual": self.meta["multilingual"],
                "pooling":     strategy,
                "embedding_dim": len(emb_np),
                "embedding":   emb_np,
            }
            if per_layer_embs:
                record["per_layer_embeddings"] = per_layer_embs

            results.append(record)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Writers de salida
# ─────────────────────────────────────────────────────────────────────────────

def save_json(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def save_npy(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)
    meta = [{k: v for k, v in r.items() if k != "embedding"} for r in records]
    np.save(str(path.with_suffix(".npy")), embeddings)
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def save_parquet(records: list[dict], path: Path):
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas no instalado — usando JSON.")
        save_json(records, path.with_suffix(".json"))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in r.items() if k != "per_layer_embeddings"} for r in records]
    pd.DataFrame(rows).to_parquet(str(path.with_suffix(".parquet")), index=False)


SAVERS = {"json": save_json, "npy": save_npy, "parquet": save_parquet}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingPipeline:
    """
    Pipeline completo:
      Dataset (wav/0/ + wav/1/) → segmentación → SSL embedding → salida estructurada

    Estructura de salida:
      embeddings/
        xlsr-300m/
          class_0/
            audio_001.json
            audio_002.json
          class_1/
            audio_003.json
        xlsr-53/
          class_0/ ...
          class_1/ ...
        manifest.json   ← índice global de todos los embeddings con class_id
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
        logger.warning("GPU no disponible — usando CPU (puede ser lento para modelos large)")
        return torch.device("cpu")

    def process_audio(
        self,
        entry: dict,
        extractor: EmbeddingExtractor,
    ) -> list[dict]:
        """Procesa un audio: carga → segmenta → extrae embeddings."""
        cfg = self.config
        waveform = load_and_preprocess(entry["path"], cfg.target_sr, cfg.normalize_audio)
        duration = len(waveform) / cfg.target_sr

        segments = segment_audio(
            waveform, cfg.target_sr,
            cfg.segment_duration, cfg.overlap_ratio, cfg.min_segment_duration
        )

        logger.debug(
            f"  {entry['audio_id']} (clase={entry['class_id']}): "
            f"{duration:.1f}s → {len(segments)} segmentos"
        )

        if not segments:
            logger.warning(f"  {entry['audio_id']}: sin segmentos válidos, omitiendo.")
            return []

        all_records = []
        for batch_start in range(0, len(segments), cfg.batch_size):
            batch_segs = segments[batch_start: batch_start + cfg.batch_size]
            waveforms = [s["waveform"] for s in batch_segs]
            metas = [
                {
                    "audio_id":    entry["audio_id"],
                    "class_id":    entry["class_id"],
                    "class_label": entry["class_label"],
                    "segment_id":  s["segment_id"],
                    "start_time":  s["start_time"],
                    "end_time":    s["end_time"],
                }
                for s in batch_segs
            ]
            records = extractor.extract_batch(waveforms, metas)
            all_records.extend(records)

        return all_records

    def run(self, audio_root: str):
        """
        Ejecuta el pipeline completo sobre el directorio de audios.
        Detecta automáticamente la estructura wav/0/ y wav/1/.
        """
        print(f"Audio root: {audio_root}")
        entries = discover_dataset(
            audio_root,
            include_substrings=self.config.include_substrings,
            exclude_substrings=self.config.exclude_substrings,
        )
        if not entries:
            raise RuntimeError(f"No se encontraron audios en {audio_root}")
        print_dataset_summary(entries)

        saver = SAVERS[self.config.output_format]
        manifest_path = Path(self.config.output_dir) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            logger.info(f"Manifest previo cargado: {len(manifest)} entradas")
        else:
            manifest = []

        for model_key in self.config.models:
            if model_key not in MODEL_REGISTRY:
                logger.warning(f"Modelo '{model_key}' no registrado — omitiendo.")
                continue

            meta = MODEL_REGISTRY[model_key]
            logger.info(f"\n{'═'*60}")
            logger.info(f"Modelo: {model_key}")
            logger.info(f"  HF ID:       {meta['hf_id']}")
            logger.info(f"  Multilingüe: {meta['multilingual']}")
            logger.info(f"  Idiomas:     {meta['languages']}")
            logger.info(f"  Prioridad:   #{meta['priority']}")
            logger.info(f"{'═'*60}")

            extractor = EmbeddingExtractor(model_key, self.config, self.device)

            for entry in tqdm(entries, desc=f"[{model_key}]"):
                # Organizar por modelo / clase / audio
                if entry["class_id"] is not None:
                    out_dir = Path(self.config.output_dir) / model_key / f"class_{entry['class_id']}"
                else:
                    out_dir = Path(self.config.output_dir) / model_key
                out_dir.mkdir(parents=True, exist_ok=True)

                ext = self.config.output_format
                out_path = out_dir / f"{entry['audio_id']}.{ext}"

                if out_path.exists():
                    logger.debug(f"  {entry['audio_id']}: ya existe, omitiendo.")
                    continue

                try:
                    records = self.process_audio(entry, extractor)
                    if records:
                        saver(records, out_path)
                        manifest.append({
                            "audio_id":    entry["audio_id"],
                            "class_id":    entry["class_id"],
                            "class_label": entry["class_label"],
                            "model":       model_key,
                            "n_segments":  len(records),
                            "file":        str(out_path),
                        })
                except Exception as e:
                    logger.error(f"  {entry['audio_id']}: ERROR — {e}", exc_info=True)

            del extractor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Guardar manifest global
        manifest_path = Path(self.config.output_dir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"\n✅ Pipeline completo. Manifest: {manifest_path}")
        logger.info(f"   Total embeddings generados: {len(manifest)}")


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: cargar manifest para ML downstream
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest_as_dataframe(embeddings_root: str, model_key: str):
    """
    Carga todos los embeddings de un modelo en una matriz numpy lista para sklearn.

    Retorna:
      X      : (n_audios, embed_dim)  — media de segmentos por audio
      y      : (n_audios,)            — etiquetas numéricas (0 o 1)
      ids    : list de audio_ids
    """
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [m for m in manifest if m["model"] == model_key]
    if not entries:
        raise ValueError(f"No hay entradas para modelo '{model_key}' en el manifest.")

    X_rows, y_rows, id_rows = [], [], []

    for entry in entries:
        with open(entry["file"]) as f:
            records = json.load(f)
        embs = np.array([r["embedding"] for r in records], dtype=np.float32)
        X_rows.append(embs.mean(axis=0))  # mean pooling a nivel audio
        y_rows.append(int(entry["class_id"]) if entry["class_id"] is not None else -1)
        id_rows.append(entry["audio_id"])

    return np.stack(X_rows), np.array(y_rows), id_rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "SSL Embedding Pipeline — Depresión/Ansiedad (Español)\n"
            "Soporta estructura wav/0/ y wav/1/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--audio_dir", required=True,
        help="Directorio raíz del dataset (e.g., 'wav/' con subcarpetas 0/ y 1/)"
    )
    parser.add_argument("--output_dir", default="embeddings")
    parser.add_argument(
        "--models", nargs="+",
        default=DEFAULT_MODELS_ALL,
        choices=DEFAULT_MODELS_ALL,
        help=(
            "Modelos a usar. Recomendados para español: xlsr-300m xlsr-53 whisper-large-encoder\n"
            "Baseline inglés: wav2vec2-large-robust wavlm-large hubert-large"
        ),
    )
    parser.add_argument("--segment_duration", type=float, default=5.0,
                        help="Duración de cada segmento en segundos (default: 5.0)")
    parser.add_argument("--overlap_ratio", type=float, default=0.5,
                        help="Fracción de overlap entre segmentos (default: 0.5 = 50%%)")
    parser.add_argument("--pooling", default="attention",
                        choices=["mean", "attention", "weighted_sum", "last"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--format", default="json", choices=["json", "npy", "parquet"])
    parser.add_argument(
        "--include_substring",
        action="append",
        default=["_recortado_denoised"],
        help="Solo incluir archivos cuyo nombre contenga este texto (se puede repetir)",
    )
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_per_layer", action="store_true")
    parser.add_argument("--list_models", action="store_true",
                        help="Listar modelos disponibles con descripción y salir")

    args = parser.parse_args()

    if args.list_models:
        print("\nModelos disponibles:\n")
        for key, m in sorted(MODEL_REGISTRY.items(), key=lambda x: x[1]["priority"]):
            flag = "🌍" if m["multilingual"] else "🇺🇸"
            print(f"  [{m['priority']}] {flag} {key:<30} {m['languages']}")
            print(f"       {m['notes'][:90]}...")
            print()
        return

    config = PipelineConfig(
        output_dir=args.output_dir,
        models=args.models,
        segment_duration=args.segment_duration,
        overlap_ratio=args.overlap_ratio,
        pooling_strategy=args.pooling,
        batch_size=args.batch_size,
        output_format=args.format,
        fp16=not args.no_fp16,
        extract_all_layers=not args.no_per_layer,
        include_substrings=args.include_substring
    )

    pipeline = EmbeddingPipeline(config)
    pipeline.run(args.audio_dir)


if __name__ == "__main__":
    main()
