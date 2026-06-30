"""
Stage 3: Semantic Retrieval

Uses sentence-transformers/all-MiniLM-L6-v2 for lightweight CPU-friendly embeddings.
Computes cosine similarity between job description and each candidate.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Model will be loaded lazily
_model = None
_model_name = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer model: %s", _model_name)
        _model = SentenceTransformer(_model_name)
        logger.info("Model loaded successfully.")
    return _model


def encode_texts(
    texts: list[str],
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts into embeddings using the sentence-transformer model.

    Args:
        texts: List of text strings to encode.
        batch_size: Batch size for encoding.
        show_progress: Whether to show a progress bar.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    logger.info("Encoding %d texts in batches of %d...", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
    )

    logger.info(
        "Encoding complete. Shape: %s", embeddings.shape
    )
    return embeddings


def encode_single(text: str) -> np.ndarray:
    """Encode a single text into an embedding vector."""
    model = _get_model()
    embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding[0]  # Shape: (embedding_dim,)


def compute_similarities(
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between JD embedding and all candidate embeddings.

    Since embeddings are L2-normalized, cosine similarity = dot product.

    Args:
        jd_embedding: Shape (embedding_dim,)
        candidate_embeddings: Shape (n_candidates, embedding_dim)

    Returns:
        numpy array of shape (n_candidates,) with similarity scores in [0, 1].
    """
    # Dot product with normalized vectors = cosine similarity
    similarities = candidate_embeddings @ jd_embedding

    # Clamp to [0, 1] (some very dissimilar texts can have negative cosine)
    similarities = np.clip(similarities, 0.0, 1.0)

    logger.info(
        "Similarity stats: min=%.4f, max=%.4f, mean=%.4f, median=%.4f",
        similarities.min(),
        similarities.max(),
        similarities.mean(),
        np.median(similarities),
    )

    return similarities
