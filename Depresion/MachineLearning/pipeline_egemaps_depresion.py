"""
=============================================================
PIPELINE DE EXTRACCIÓN eGeMAPS (openSMILE)
Trastorno objetivo: DEPRESIÓN
=============================================================
Extrae el set funcional eGeMAPSv02 (88 features, vía la librería `opensmile`)
por segmento, reutilizando EXACTAMENTE el mismo inventario de audios y la
misma segmentación (5s, 50% overlap, preénfasis + trim VAD) que
pipeline_feat_depresion.py — importando sus funciones/constantes en vez de
reimplementarlas — para garantizar correspondencia 1:1 de (audio_id, seg_idx)
con features_depresion_5.csv.

Salida:
  features_depresion_egemaps.csv

Uso:
  python pipeline_egemaps_depresion.py            # extracción completa
  python pipeline_egemaps_depresion.py --limit 2  # smoke test (2 audios)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import opensmile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import kd_common  # noqa: E402

OUTPUT_CSV = Path(__file__).parent / "features_depresion_egemaps.csv"


def extract_egemaps_dataset(limit: int | None = None) -> pd.DataFrame:
    cfg = kd_common.get_config("depresion")
    mod = kd_common.load_pipeline_module(cfg.pipeline_feat_path)
    inventory = mod.collect_inventory()
    if limit is not None:
        inventory = inventory.head(limit)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    rows = []
    for _, audio_row in tqdm(inventory.iterrows(), total=len(inventory), unit="audio"):
        segments = kd_common.segments_for_audio(mod, audio_row["path"])
        for seg_idx, seg in enumerate(segments):
            feats = smile.process_signal(seg, mod.SR_TARGET).reset_index(drop=True).iloc[0].to_dict()
            rows.append({
                "audio_id": audio_row["audio_id"],
                "seg_idx": seg_idx,
                "label": audio_row["label"],
                **feats,
            })

    df = pd.DataFrame(rows)
    id_cols = ["audio_id", "seg_idx", "label"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    return df[id_cols + feat_cols].reset_index(drop=True)


def verify_correspondence(df: pd.DataFrame, cfg) -> None:
    ref = pd.read_csv(cfg.librosa_features_csv, usecols=["audio_id", "seg_idx"])
    ref_keys = set(map(tuple, ref.to_numpy()))
    new_keys = set(map(tuple, df[["audio_id", "seg_idx"]].to_numpy()))
    missing = ref_keys - new_keys
    extra = new_keys - ref_keys
    if missing or extra:
        print(f"[WARN] Desalineado con el CSV librosa: {len(missing)} faltantes, {len(extra)} extra")
    else:
        print(f"✓ Correspondencia exacta con {cfg.librosa_features_csv.name}: {len(ref_keys)} muestras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracción eGeMAPS para depresión")
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo N audios (smoke test)")
    args = parser.parse_args()

    print("=" * 55)
    print("PIPELINE — eGeMAPS (openSMILE) — DEPRESIÓN")
    print("=" * 55)

    df = extract_egemaps_dataset(limit=args.limit)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅  Guardado: {OUTPUT_CSV}  ({len(df)} filas × {len(df.columns)} columnas)")

    if args.limit is None:
        cfg = kd_common.get_config("depresion")
        verify_correspondence(df, cfg)
