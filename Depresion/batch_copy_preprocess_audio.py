import argparse
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile


SOURCE_EDITED_DIR = Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/DatosPreprocesados/wav_audios_edit")
DEST_DATASETS = [
    Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Ansiedad/wav"),
    Path("/home/ci2dt2-ai/Proyectos/Psiquiatria/Clasificación Final/Depresión/wav"),
]
CLASS_FOLDERS = ["0", "1"]


@dataclass
class Stats:
    matched_ids: int = 0
    copied_trimmed: int = 0
    generated_denoised: int = 0
    missing_source: int = 0
    errors: int = 0
    missing_ids: list = field(default_factory=list)


def normalize_to_float32(signal: np.ndarray) -> np.ndarray:
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
    return signal[int(idx[0]) : int(idx[-1]) + 1]


def extract_id(filename: str) -> str | None:
    m = re.match(r"^(\d{3})", filename)
    return m.group(1) if m else None


def build_source_map(source_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in source_dir.glob("*_editado.wav"):
        m = re.match(r"^(\d{3})_editado\.wav$", path.name)
        if m:
            mapping[m.group(1)] = path
    return mapping


def collect_ids_in_class(class_dir: Path) -> set[str]:
    ids = set()
    for p in class_dir.glob("*.wav"):
        if p.name.endswith("_recortado.wav") or p.name.endswith("_recortado_denoised.wav"):
            continue
        audio_id = extract_id(p.name)
        if audio_id:
            ids.add(audio_id)
    return ids


def run_demucs(input_wav: Path, out_dir: Path) -> Path:
    result = subprocess.run(
        [
            "python", "-m", "demucs",
            "--two-stems=vocals",
            "--out", str(out_dir),
            str(input_wav),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)

    stem_name = input_wav.stem
    vocals_path = out_dir / "htdemucs" / stem_name / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(f"Demucs no generó vocals en: {vocals_path}")
    return vocals_path


def process_id(
    source_path: Path,
    out_trimmed: Path,
    out_denoised: Path,
    overwrite: bool,
    threshold_db: float,
) -> tuple[bool, bool]:
    wrote_trimmed = False
    wrote_denoised = False

    need_trimmed = overwrite or not out_trimmed.exists()
    need_denoised = overwrite or not out_denoised.exists()

    if not need_trimmed and not need_denoised:
        return False, False

    # --- Paso 1: recorte de silencios ---
    if need_trimmed:
        sr, signal = wavfile.read(source_path)
        signal = ensure_mono(signal)
        signal = normalize_to_float32(signal)
        signal = trim_silence(signal, threshold_db=threshold_db)
        wavfile.write(out_trimmed, sr, to_int16(signal))
        wrote_trimmed = True

    # --- Paso 2: separación vocal con Demucs ---
    if need_denoised:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            vocals_path = run_demucs(out_trimmed, tmp_dir)

            sr_v, vocals = wavfile.read(vocals_path)
            vocals = ensure_mono(vocals)
            vocals = normalize_to_float32(vocals)
            wavfile.write(out_denoised, sr_v, to_int16(vocals))
            wrote_denoised = True

    return wrote_trimmed, wrote_denoised


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline: recorte de silencios + separación vocal (Demucs)"
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Sobrescribe archivos ya existentes")
    parser.add_argument("--threshold-db", type=float, default=30.0,
                        help="Umbral en dB para recorte de silencios")
    args = parser.parse_args()

    source_map = build_source_map(SOURCE_EDITED_DIR)
    if not source_map:
        raise FileNotFoundError(
            f"No se encontraron archivos *_editado.wav en: {SOURCE_EDITED_DIR}"
        )

    print(f"Source map: {len(source_map)} archivos _editado.wav encontrados")

    total = Stats()

    for dataset_root in DEST_DATASETS:
        print(f"\n=== Dataset: {dataset_root} ===")
        for cls in CLASS_FOLDERS:
            class_dir = dataset_root / cls
            if not class_dir.is_dir():
                print(f"[WARN] Carpeta no encontrada: {class_dir}")
                continue

            ids = sorted(collect_ids_in_class(class_dir))
            print(f"Clase {cls}: {len(ids)} IDs base detectados")

            stats = Stats()

            for audio_id in ids:
                src = source_map.get(audio_id)
                if src is None:
                    stats.missing_source += 1
                    stats.missing_ids.append(audio_id)
                    continue

                stats.matched_ids += 1
                out_trimmed = class_dir / f"{audio_id}_recortado.wav"
                out_denoised = class_dir / f"{audio_id}_recortado_denoised.wav"

                try:
                    wrote_trimmed, wrote_denoised = process_id(
                        source_path=src,
                        out_trimmed=out_trimmed,
                        out_denoised=out_denoised,
                        overwrite=args.overwrite,
                        threshold_db=args.threshold_db,
                    )
                    if wrote_trimmed:
                        stats.copied_trimmed += 1
                    if wrote_denoised:
                        stats.generated_denoised += 1
                except subprocess.CalledProcessError as exc:
                    stats.errors += 1
                    print(f"  [ERROR Demucs] {dataset_root.name}/{cls}/{audio_id}: {exc.stderr[-300:]}")
                except Exception as exc:
                    stats.errors += 1
                    print(f"  [ERROR] {dataset_root.name}/{cls}/{audio_id}: {exc}")

            print(
                f"  matched={stats.matched_ids} | recortado_nuevo={stats.copied_trimmed} "
                f"| denoised_nuevo={stats.generated_denoised} | missing_source={stats.missing_source} "
                f"| errors={stats.errors}"
            )

            if stats.missing_ids:
                print(f"  [MISSING IDs] {', '.join(stats.missing_ids)}")

            total.matched_ids += stats.matched_ids
            total.copied_trimmed += stats.copied_trimmed
            total.generated_denoised += stats.generated_denoised
            total.missing_source += stats.missing_source
            total.errors += stats.errors
            total.missing_ids.extend(stats.missing_ids)

    print("\n=== RESUMEN GLOBAL ===")
    print(
        f"matched={total.matched_ids} | recortado_nuevo={total.copied_trimmed} "
        f"| denoised_nuevo={total.generated_denoised} | missing_source={total.missing_source} "
        f"| errors={total.errors}"
    )
    if total.missing_ids:
        unique_missing = sorted(set(total.missing_ids))
        print(f"[MISSING IDs GLOBAL] {', '.join(unique_missing)}")


if __name__ == "__main__":
    main()