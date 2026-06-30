"""
Stage 7: Score Adjustment & Re-Ranking

Computes weighted composite base score from sub-scores,
applies honeypot-adjusted scoring, and selects top 100 candidates.

Formula:
    base_score = weighted sum of sub-scores
    adjusted_score = base_score * consistency_score * (1 - honeypot_score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score weights for composite base score (must sum to 1.0)
# ---------------------------------------------------------------------------

SCORE_WEIGHTS: dict[str, float] = {
    "semantic_score": 0.15,
    "retrieval_score": 0.22,
    "ranking_score": 0.18,
    "production_score": 0.13,
    "behavioral_score": 0.05,
    "experience_score": 0.08,
    "education_score": 0.07,
    "career_score": 0.12,
}


@dataclass
class RankedCandidate:
    """A candidate with all computed scores and final ranking."""
    candidate_id: str
    rank: int
    adjusted_score: float
    base_score: float
    consistency_score: float
    honeypot_score: float
    sub_scores: dict[str, float]
    # Original candidate index (for retrieving raw data later)
    original_index: int


def compute_base_score(sub_scores: dict[str, float]) -> float:
    """Compute the weighted composite base score.

    All sub-scores should be on 0-100 scale.
    Output is also on 0-100 scale.

    Args:
        sub_scores: Dict of score name to value (0-100).

    Returns:
        Weighted composite score (0-100).
    """
    score = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        score += weight * sub_scores.get(key, 0.0)
    return score


def compute_adjusted_score(
    base_score: float,
    consistency_score: float,
    honeypot_score: float,
) -> float:
    """Apply honeypot-adjusted scoring.

    adjusted_score = base_score * consistency_score * (1 - honeypot_score)

    Args:
        base_score: Composite base score (0-100).
        consistency_score: Consistency score (0.0-1.0).
        honeypot_score: Honeypot score (0.0-1.0).

    Returns:
        Adjusted score (0-100).
    """
    return base_score * consistency_score * (1.0 - honeypot_score)


def rank_candidates(
    candidate_ids: list[str],
    all_sub_scores: list[dict[str, float]],
    consistency_scores: list[float],
    honeypot_scores: list[float],
    top_k: int = 100,
) -> list[RankedCandidate]:
    """Rank all candidates and return the top-k.

    Args:
        candidate_ids: List of candidate IDs.
        all_sub_scores: List of sub-score dicts (one per candidate).
        consistency_scores: List of consistency scores (0.0-1.0).
        honeypot_scores: List of honeypot scores (0.0-1.0).
        top_k: Number of top candidates to return.

    Returns:
        List of RankedCandidate objects, sorted by adjusted_score descending.
    """
    n = len(candidate_ids)
    logger.info("Ranking %d candidates, selecting top %d...", n, top_k)

    # Compute all adjusted scores using vectorized numpy operations
    base_scores = np.zeros(n)
    adjusted_scores = np.zeros(n)

    for i in range(n):
        base_scores[i] = compute_base_score(all_sub_scores[i])
        adjusted_scores[i] = compute_adjusted_score(
            base_scores[i],
            consistency_scores[i],
            honeypot_scores[i],
        )

    # Get top-k indices
    top_indices = np.argsort(adjusted_scores)[::-1][:top_k]

    # Build ranked candidates
    ranked: list[RankedCandidate] = []
    for rank, idx in enumerate(top_indices, 1):
        ranked.append(RankedCandidate(
            candidate_id=candidate_ids[idx],
            rank=rank,
            adjusted_score=float(adjusted_scores[idx]),
            base_score=float(base_scores[idx]),
            consistency_score=float(consistency_scores[idx]),
            honeypot_score=float(honeypot_scores[idx]),
            sub_scores=all_sub_scores[idx],
            original_index=int(idx),
        ))

    logger.info(
        "Top candidate: %s (score=%.2f), Bottom of top-%d: %s (score=%.2f)",
        ranked[0].candidate_id, ranked[0].adjusted_score,
        top_k,
        ranked[-1].candidate_id, ranked[-1].adjusted_score,
    )

    # Log score distribution
    top_scores = [r.adjusted_score for r in ranked]
    logger.info(
        "Top-%d score distribution: min=%.2f, max=%.2f, mean=%.2f, median=%.2f",
        top_k,
        min(top_scores), max(top_scores),
        np.mean(top_scores), np.median(top_scores),
    )

    return ranked
