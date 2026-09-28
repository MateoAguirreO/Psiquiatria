"""
Embedding Analysis & Visualization Utilities
=============================================
Tools for inspecting, validating, and exploring the extracted embeddings.
Useful for sanity checks before downstream classification/clustering.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("EmbeddingAnalysis")


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_embeddings(json_path: str) -> list[dict]:
    """Load all segment embeddings for a single audio file."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_model_embeddings(model_dir: str) -> dict[str, list[dict]]:
    """
    Load all JSON files in a model directory.
    Returns: { audio_id: [segment_records] }
    """
    model_dir = Path(model_dir)
    result = {}
    for p in sorted(model_dir.glob("*.json")):
        records = load_audio_embeddings(str(p))
        if records:
            audio_id = records[0]["audio_id"]
            result[audio_id] = records
    return result


def embeddings_to_matrix(
    records: list[dict],
    pooling: str = "mean",
) -> np.ndarray:
    """
    Collapse segment-level embeddings to a single audio-level vector.

    pooling options:
      "mean"  — average over segments (recommended for classification)
      "max"   — element-wise max (captures salient features)
      "concat"— concatenate stats (mean + std); doubles dimensionality
    """
    matrix = np.array([r["embedding"] for r in records], dtype=np.float32)

    if pooling == "mean":
        return matrix.mean(axis=0)
    elif pooling == "max":
        return matrix.max(axis=0)
    elif pooling == "concat":
        return np.concatenate([matrix.mean(axis=0), matrix.std(axis=0)])
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


def build_feature_matrix(
    model_dir: str,
    audio_pooling: str = "mean",
) -> tuple[np.ndarray, list[str]]:
    """
    Build X matrix (n_audios × embedding_dim) ready for sklearn/AutoML.

    Returns:
        X: (n_audios, dim) float32 array
        audio_ids: list of audio identifiers (same order as X rows)
    """
    all_records = load_model_embeddings(model_dir)
    audio_ids = sorted(all_records.keys())
    X = np.stack([
        embeddings_to_matrix(all_records[aid], audio_pooling)
        for aid in audio_ids
    ], axis=0)
    return X, audio_ids


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_stats(embeddings_root: str):
    """Print overview of extracted embeddings."""
    root = Path(embeddings_root)
    print(f"\n{'='*60}")
    print(f"Embeddings root: {root.resolve()}")
    print(f"{'='*60}")

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue

        json_files = list(model_dir.glob("*.json"))
        if not json_files:
            continue

        # Sample stats from first file
        sample = load_audio_embeddings(str(json_files[0]))
        if not sample:
            continue

        n_segs = len(sample)
        emb_dim = len(sample[0]["embedding"])
        duration_covered = sample[-1]["end_time"] - sample[0]["start_time"]

        print(f"\nModel: {model_dir.name}")
        print(f"  Audio files processed : {len(json_files)}")
        print(f"  Embedding dimension   : {emb_dim}")
        print(f"  Segments (1st audio)  : {n_segs}")
        print(f"  Time covered (1st)    : {duration_covered:.1f}s")
        print(f"  Pooling strategy      : {sample[0].get('pooling', 'unknown')}")

        total_segments = sum(
            len(load_audio_embeddings(str(f))) for f in json_files
        )
        print(f"  Total segments        : {total_segments}")
        size_mb = sum(f.stat().st_size for f in json_files) / 1e6
        print(f"  Total JSON size       : {size_mb:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# Quick dimensionality reduction preview (no matplotlib required at import)
# ─────────────────────────────────────────────────────────────────────────────

def reduce_and_cluster(
    X: np.ndarray,
    audio_ids: list[str],
    labels: Optional[list] = None,
    n_components: int = 2,
    method: str = "pca",
    save_path: Optional[str] = None,
):
    """
    Dimensionality reduction + optional scatter plot.

    method: "pca" | "umap" | "tsne"
    labels: optional list of class labels per audio for color coding
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping visualization.")
        return

    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
        title = f"PCA — {n_components}D ({reducer.explained_variance_ratio_[:2].sum()*100:.1f}% variance)"

    elif method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(30, len(X) - 1)
        reducer = TSNE(n_components=n_components, perplexity=perp, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
        title = "t-SNE"

    elif method == "umap":
        try:
            import umap
        except ImportError:
            logger.error("Install umap-learn: pip install umap-learn")
            return
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
        title = "UMAP"

    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter_kwargs = dict(s=80, alpha=0.8, edgecolors="white", linewidths=0.5)

    if labels is not None:
        unique = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab10", len(unique))
        for i, lbl in enumerate(unique):
            mask = [j for j, l in enumerate(labels) if l == lbl]
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=str(lbl),
                       color=cmap(i), **scatter_kwargs)
        ax.legend(title="Label", framealpha=0.9)
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], color="#3b82f6", **scatter_kwargs)

    # Annotate with audio IDs
    for i, aid in enumerate(audio_ids):
        ax.annotate(aid, (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=6, alpha=0.6, ha="center", va="bottom")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity matrix
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute NxN cosine similarity matrix for audio-level embeddings."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / (norms + 1e-8)
    return X_norm @ X_norm.T


# ─────────────────────────────────────────────────────────────────────────────
# Segment-level temporal embedding export (for sequence models: LSTM, Transformer)
# ─────────────────────────────────────────────────────────────────────────────

def export_temporal_sequences(
    model_dir: str,
    output_path: str,
    max_segments: Optional[int] = None,
    pad_to_length: Optional[int] = None,
):
    """
    Export embeddings as temporal sequences (n_audios, n_segments, embed_dim).
    Useful for LSTM/Transformer downstream classifiers.

    Saves: .npz with arrays 'X', 'lengths', 'audio_ids'

    max_segments: truncate sequences to this length
    pad_to_length: zero-pad all sequences to this length (for batch training)
    """
    all_records = load_model_embeddings(model_dir)
    audio_ids = sorted(all_records.keys())

    sequences = []
    lengths = []

    for aid in audio_ids:
        segs = sorted(all_records[aid], key=lambda r: r["segment_id"])
        embs = np.array([r["embedding"] for r in segs], dtype=np.float32)

        if max_segments:
            embs = embs[:max_segments]

        lengths.append(len(embs))
        sequences.append(embs)

    if pad_to_length is None:
        pad_to_length = max(lengths)

    embed_dim = sequences[0].shape[1]
    X = np.zeros((len(sequences), pad_to_length, embed_dim), dtype=np.float32)

    for i, seq in enumerate(sequences):
        n = min(len(seq), pad_to_length)
        X[i, :n] = seq[:n]

    np.savez_compressed(
        output_path,
        X=X,
        lengths=np.array(lengths),
        audio_ids=np.array(audio_ids),
    )
    logger.info(f"Temporal sequences saved: {output_path}")
    logger.info(f"  Shape: {X.shape}  (audios × max_segments × embed_dim)")
    return X, lengths, audio_ids


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analysis_utils.py <embeddings_root>")
        sys.exit(1)

    print_dataset_stats(sys.argv[1])
