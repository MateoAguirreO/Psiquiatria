"""
SSL Speech Embedding Pipeline — Estructura Jerárquica por Segmento
==================================================================
Estructura de salida:
    embeddings_v2/
      ansiedad/          ← nombre de la carpeta raíz de audios (condición clínica)
        class_0/
          xlsr-300m/
            audio_001/
              segment_0/
                embedding.json
            wavlm-large/
              audio_001/
                segment_0/
                  embedding.json
        class_1/
          ...
      ansiedad/
        ...
      manifest.json       ← índice global

Dataset esperado (igual que antes):
    wav/
      0/   ← clase 0 (control)
          audio_001.wav
      1/   ← clase 1 (caso clínico)
          audio_003.wav

Uso:
python embedding_pipeline_5s.py \
  --audio_dir "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Ansiedad/wav" \
  --output_dir "/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Ansiedad/embeddings_v2/" \
  --condition "ansiedad" \
  --models xlsr-300m xlsr-53 whisper-large-encoder wav2vec2-large-robust wavlm-large hubert-large \
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
logger = logging.getLogger("EmbeddingPipelineV2")


# ─────────────────────────────────────────────────────────────────────────────
# Registro de modelos — TODOS habilitados
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # ══════════════════════════════════════════════════════
    # GRUPO A — Multilingüe / Cross-lingual (RECOMENDADOS para español)
    # ══════════════════════════════════════════════════════

    "xlsr-300m": {
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
            "Captura fonética y prosodia del español nativo."
        ),
    },

    "xlsr-53": {
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
            "Contrastive pre-training compartido cross-lingual."
        ),
    },

    "whisper-large-encoder": {
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
            "con calidad excepcional. Usar SOLO encoder (sin decoder)."
        ),
    },

    # ══════════════════════════════════════════════════════
    # GRUPO B — Inglés puro (ablation / baseline)
    # ══════════════════════════════════════════════════════

    "wav2vec2-large-robust": {
        "hf_id": "facebook/wav2vec2-large-robust",
        "model_cls": Wav2Vec2Model,
        "processor_id": "facebook/wav2vec2-large-robust",
        "n_transformer_layers": 24,
        "hidden_size": 1024,
        "multilingual": False,
        "languages": "Inglés (LibriSpeech + noisy)",
        "priority": 4,
        "notes": "Baseline inglés robusto a audio ruidoso/telefónico.",
    },

    "wavlm-large": {
        "hf_id": "microsoft/wavlm-large",
        "model_cls": WavLMModel,
        "processor_id": "microsoft/wavlm-large",
        "n_transformer_layers": 24,
        "hidden_size": 1024,
        "multilingual": False,
        "languages": "Inglés (LibriSpeech 960h)",
        "priority": 5,
        "notes": "SOTA en SUPERB. Maji et al. (Interspeech 2024): WavLM > HuBERT en depression.",
    },

    "hubert-large": {
        "hf_id": "facebook/hubert-large-ls960-ft",
        "model_cls": HubertModel,
        "processor_id": "facebook/wav2vec2-large-xlsr-53",
        "n_transformer_layers": 24,
        "hidden_size": 1024,
        "multilingual": False,
        "languages": "Inglés (LibriSpeech 960h)",
        "priority": 6,
        "notes": "Fuerte baseline inglés. Buenas representaciones emocionales.",
    },
}

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
    segment_duration: float = 5.0
    overlap_ratio: float = 0.5
    min_segment_duration: float = 1.0

    # ── Extracción
    pooling_strategy: str = "attention"
    extract_all_layers: bool = True
    batch_size: int = 4
    fp16: bool = True

    # ── Salida
    output_dir: str = "embeddings_v2"
    output_format: str = "json"

    # ── Condición clínica (nombre de la carpeta de primer nivel)
    # Ejemplos: "depresion", "ansiedad", "tdah", etc.
    condition: str = "depresion"

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

    Modo A — Con clases explícitas:
        wav/
          0/  audio_001.wav ...   ← class_id = "0"
          1/  audio_003.wav ...   ← class_id = "1"

    Modo B — Plano:
        wav/
          audio_001.wav ...       ← class_id = None
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

    subdirs = [d for d in root.iterdir() if d.is_dir()]

    if subdirs:
        class_dirs = [d for d in subdirs if d.name.isdigit() or
                      d.name.lower() in {"depresion", "control", "positive", "negative",
                                          "depression", "healthy", "ansiedad", "anxiety",
                                          "dep", "ctrl", "pos", "neg", "0", "1"}]
        if class_dirs:
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
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

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

    waveform_np = waveform.squeeze().numpy().astype(np.float32)

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
    Con 5s + 50% overlap avanza 2.5s por ventana.
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
# Attention Pooling (Zhang et al., 2024)
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
# Extractor por modelo
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingExtractor:
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
            self.model = WhisperModel.from_pretrained(hf_id, output_hidden_states=True)
        else:
            ModelCls = self.meta["model_cls"]
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(proc_id)
            except Exception:
                try:
                    self.processor = AutoProcessor.from_pretrained(proc_id)
                except Exception:
                    logger.warning("Processor con tokenizer no disponible; usando feature extractor.")
                    try:
                        self.processor = AutoFeatureExtractor.from_pretrained(proc_id)
                    except Exception:
                        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(proc_id)

            self.model = ModelCls.from_pretrained(hf_id, output_hidden_states=True)

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

    def _extract_batch_wav2vec(self, waveforms, segment_meta):
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

        transformer_hidden = outputs.hidden_states[1:]
        last_hidden = outputs.last_hidden_state

        return self._pool_and_package(last_hidden, transformer_hidden, attention_mask, segment_meta)

    def _extract_batch_whisper(self, waveforms, segment_meta):
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

        return self._pool_and_package(last_hidden, transformer_hidden, attention_mask, segment_meta)

    def _pool_and_package(self, last_hidden, transformer_hidden, attention_mask, segment_meta):
        results = []
        strategy = self.config.pooling_strategy

        for i, meta in enumerate(segment_meta):
            h_last = last_hidden[i].float()

            mask_i = (
                attention_mask[i] if attention_mask is not None
                else torch.ones(h_last.shape[0], dtype=torch.long, device=self.device)
            )

            if mask_i.shape[0] != h_last.shape[0]:
                mask_i = torch.ones(h_last.shape[0], dtype=torch.long, device=self.device)

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

            per_layer_embs = None
            if self.config.extract_all_layers:
                per_layer_embs = {}
                for layer_idx, layer_h in enumerate(transformer_hidden):
                    p = self.attention_pool(
                        layer_h[i].float().unsqueeze(0), mask_i.unsqueeze(0)
                    ).squeeze(0)
                    per_layer_embs[f"layer_{layer_idx + 1}"] = p.float().cpu().numpy().tolist()

            record = {
                "audio_id":       meta["audio_id"],
                "class_id":       meta.get("class_id"),
                "class_label":    meta.get("class_label"),
                "condition":      meta.get("condition"),
                "segment_id":     meta["segment_id"],
                "start_time":     meta["start_time"],
                "end_time":       meta["end_time"],
                "model":          self.model_key,
                "multilingual":   self.meta["multilingual"],
                "pooling":        strategy,
                "embedding_dim":  len(emb_np),
                "embedding":      emb_np,
            }
            if per_layer_embs:
                record["per_layer_embeddings"] = per_layer_embs

            results.append(record)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Writers de salida
# ─────────────────────────────────────────────────────────────────────────────

def save_json(record: dict, path: Path):
    """Guarda un único segmento como JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def save_npy(record: dict, path: Path):
    """Guarda embedding numpy + metadata JSON para un segmento."""
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = np.array(record["embedding"], dtype=np.float32)
    np.save(str(path.with_suffix(".npy")), emb)
    meta = {k: v for k, v in record.items() if k not in ("embedding", "per_layer_embeddings")}
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def save_parquet(record: dict, path: Path):
    """Para segmentos individuales usa JSON; parquet tiene sentido al agregar."""
    save_json(record, path.with_suffix(".json"))


SAVERS = {"json": save_json, "npy": save_npy, "parquet": save_parquet}


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de ruta jerárquica
# ─────────────────────────────────────────────────────────────────────────────

def build_output_path(
    output_root: str,
    condition: str,
    audio_id: str,
    segment_id: int,
    model_key: str,
    class_label: Optional[str],
    fmt: str,
) -> Path:
    """
    Construye la ruta:
        output_root/
          {condition}/
            {class_label}/
              {model_key}/
                {audio_id}/
                  segment_{segment_id}/
                    embedding.{fmt}

    Si class_label es None (dataset plano), usa class_unknown.
    """
    class_segment = class_label if class_label is not None else "class_unknown"
    parts = [
        output_root,
        condition,
        class_segment,
        model_key,
        audio_id,
        f"segment_{segment_id}",
    ]

    ext = "json" if fmt == "parquet" else fmt  # parquet → JSON a nivel segmento
    return Path(*parts) / f"embedding.{ext}"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingPipeline:
    """
    Pipeline con estructura de salida jerárquica por segmento:
        embeddings_v2/
          {condition}/
            {class_label}/
              {model}/
                {audio_id}/
                  segment_{n}/
                    embedding.json
          manifest.json
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

    def process_audio(self, entry: dict, extractor: EmbeddingExtractor) -> list[dict]:
        """Carga → segmenta → extrae embeddings por segmento."""
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
                    "condition":   cfg.condition,
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
        Guarda cada segmento × modelo en su propia carpeta jerárquica.
        """
        print(f"Audio root:  {audio_root}")
        print(f"Condición:   {self.config.condition}")
        print(f"Output root: {self.config.output_dir}")

        entries = discover_dataset(
            audio_root,
            include_substrings=self.config.include_substrings,
            exclude_substrings=self.config.exclude_substrings,
        )
        if not entries:
            raise RuntimeError(f"No se encontraron audios en {audio_root}")
        print_dataset_summary(entries)

        fmt = self.config.output_format
        saver = SAVERS[fmt]

        # Cargar manifest previo si existe
        manifest_path = Path(self.config.output_dir) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            logger.info(f"Manifest previo cargado: {len(manifest)} entradas")
        else:
            manifest = []

        # Índice rápido para saltar archivos ya procesados
        processed = {
            (m["audio_id"], m["segment_id"], m["model"])
            for m in manifest
        }

        for model_key in self.config.models:
            if model_key not in MODEL_REGISTRY:
                logger.warning(f"Modelo '{model_key}' no registrado — omitiendo.")
                continue

            meta_m = MODEL_REGISTRY[model_key]
            logger.info(f"\n{'═'*60}")
            logger.info(f"Modelo: {model_key}")
            logger.info(f"  HF ID:       {meta_m['hf_id']}")
            logger.info(f"  Multilingüe: {meta_m['multilingual']}")
            logger.info(f"  Idiomas:     {meta_m['languages']}")
            logger.info(f"  Prioridad:   #{meta_m['priority']}")
            logger.info(f"{'═'*60}")

            extractor = EmbeddingExtractor(model_key, self.config, self.device)

            for entry in tqdm(entries, desc=f"[{model_key}]"):
                try:
                    records = self.process_audio(entry, extractor)
                except Exception as e:
                    logger.error(f"  {entry['audio_id']}: ERROR al procesar — {e}", exc_info=True)
                    continue

                for record in records:
                    seg_id = record["segment_id"]
                    key = (record["audio_id"], seg_id, model_key)

                    if key in processed:
                        logger.debug(f"  {record['audio_id']} seg_{seg_id} [{model_key}]: ya existe, omitiendo.")
                        continue

                    out_path = build_output_path(
                        output_root=self.config.output_dir,
                        condition=self.config.condition,
                        audio_id=record["audio_id"],
                        segment_id=seg_id,
                        model_key=model_key,
                        class_label=record.get("class_label"),
                        fmt=fmt,
                    )

                    try:
                        saver(record, out_path)
                        manifest.append({
                            "condition":   self.config.condition,
                            "audio_id":    record["audio_id"],
                            "class_id":    record["class_id"],
                            "class_label": record["class_label"],
                            "model":       model_key,
                            "segment_id":  seg_id,
                            "start_time":  record["start_time"],
                            "end_time":    record["end_time"],
                            "file":        str(out_path),
                        })
                        processed.add(key)
                    except Exception as e:
                        logger.error(f"  {record['audio_id']} seg_{seg_id}: ERROR al guardar — {e}", exc_info=True)

            # Guardar manifest parcial después de cada modelo
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"  Manifest guardado: {len(manifest)} entradas acumuladas")

            del extractor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        logger.info(f"\n✅ Pipeline completo.")
        logger.info(f"   Manifest final: {manifest_path}")
        logger.info(f"   Total entradas en manifest: {len(manifest)}")


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: cargar manifest para ML downstream
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings_for_classification(
    embeddings_root: str,
    model_key: str,
    condition: Optional[str] = None,
    aggregation: str = "mean",
) -> tuple:
    """
    Carga embeddings del manifest y los agrega a nivel de audio completo.

    Args:
        embeddings_root : carpeta raíz (donde está manifest.json)
        model_key       : ej. "xlsr-300m"
        condition       : filtrar por condición clínica (None = todas)
        aggregation     : "mean" | "max" | "first"

    Retorna:
        X   : (n_audios, embed_dim)
        y   : (n_audios,)  — etiqueta numérica (0 o 1)
        ids : lista de audio_ids
    """
    manifest_path = Path(embeddings_root) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [
        m for m in manifest
        if m["model"] == model_key
        and (condition is None or m.get("condition") == condition)
    ]
    if not entries:
        raise ValueError(
            f"No hay entradas para modelo='{model_key}' condición='{condition}'."
        )

    # Agrupar por audio_id
    from collections import defaultdict
    audio_segments: dict = defaultdict(list)
    audio_meta: dict = {}
    for e in entries:
        aid = e["audio_id"]
        with open(e["file"]) as f:
            record = json.load(f)
        audio_segments[aid].append(np.array(record["embedding"], dtype=np.float32))
        audio_meta[aid] = e  # última entrada guarda class_id

    X_rows, y_rows, id_rows = [], [], []
    for aid, segs in audio_segments.items():
        stacked = np.stack(segs)
        if aggregation == "mean":
            agg = stacked.mean(axis=0)
        elif aggregation == "max":
            agg = stacked.max(axis=0)
        elif aggregation == "first":
            agg = stacked[0]
        else:
            agg = stacked.mean(axis=0)

        X_rows.append(agg)
        cid = audio_meta[aid].get("class_id")
        y_rows.append(int(cid) if cid is not None else -1)
        id_rows.append(aid)

    return np.stack(X_rows), np.array(y_rows), id_rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "SSL Embedding Pipeline V2 — Estructura Jerárquica por Segmento\n"
            "Salida: output_dir/{condition}/{audio_id}/segment_{n}/{model}/{class}/embedding.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--audio_dir", required=True,
        help="Directorio raíz del dataset (con subcarpetas 0/ y 1/)"
    )
    parser.add_argument(
        "--output_dir", default="embeddings_v2",
        help="Directorio raíz de salida"
    )
    parser.add_argument(
        "--condition", default="depresion",
        help=(
            "Nombre de la condición clínica — primer nivel de la carpeta de salida. "
            "Ej: 'depresion', 'ansiedad', 'tdah'"
        )
    )
    parser.add_argument(
        "--models", nargs="+",
        default=DEFAULT_MODELS_ALL,
        choices=DEFAULT_MODELS_ALL,
        help="Modelos a ejecutar. Por defecto: TODOS."
    )
    parser.add_argument(
        "--segment_duration", type=float, default=5.0,
        help="Duración de cada segmento en segundos (default: 5.0)"
    )
    parser.add_argument(
        "--overlap_ratio", type=float, default=0.5,
        help="Fracción de overlap entre segmentos (default: 0.5 = 50%%)"
    )
    parser.add_argument(
        "--pooling", default="attention",
        choices=["mean", "attention", "weighted_sum", "last"]
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--format", default="json",
        choices=["json", "npy", "parquet"]
    )
    parser.add_argument(
        "--include_substring",
        action="append",
        default=["_recortado_denoised"],
        help="Solo incluir archivos cuyo nombre contenga este texto (repetible)"
    )
    parser.add_argument(
        "--exclude_substring",
        action="append",
        default=[],
        help="Excluir archivos cuyo nombre contenga este texto (repetible)"
    )
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_per_layer", action="store_true")
    parser.add_argument(
        "--list_models", action="store_true",
        help="Listar modelos disponibles y salir"
    )

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
        condition=args.condition,
        models=args.models,
        segment_duration=args.segment_duration,
        overlap_ratio=args.overlap_ratio,
        pooling_strategy=args.pooling,
        batch_size=args.batch_size,
        output_format=args.format,
        fp16=not args.no_fp16,
        extract_all_layers=not args.no_per_layer,
        include_substrings=args.include_substring,
        exclude_substrings=args.exclude_substring,
    )

    pipeline = EmbeddingPipeline(config)
    pipeline.run(args.audio_dir)


if __name__ == "__main__":
    main()