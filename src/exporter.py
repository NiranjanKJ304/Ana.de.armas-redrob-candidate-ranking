"""
Stage 10: CSV Export

Writes the final submission.csv with exactly 100 ranked candidates.
Format: candidate_id,rank,score,reasoning
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .ranker import RankedCandidate

logger = logging.getLogger(__name__)


def normalize_scores(ranked: list[RankedCandidate]) -> list[float]:
    """Normalize adjusted_scores to [0.0, 1.0] range for the submission.

    Uses min-max normalization across the top-100:
    - Rank 1 gets score closest to 1.0
    - Rank 100 gets score closest to 0.01

    Args:
        ranked: List of RankedCandidate objects.

    Returns:
        List of normalized scores in [0.01, 1.0].
    """
    if not ranked:
        return []

    scores = [r.adjusted_score for r in ranked]
    max_score = max(scores)
    min_score = min(scores)

    if max_score == min_score:
        # All scores identical — distribute evenly
        return [1.0 - (i / len(ranked)) for i in range(len(ranked))]

    # Linear normalization to [0.01, 1.0]
    normalized = []
    for r in ranked:
        # Map to [0.01, 1.0] preserving relative differences
        norm = 0.01 + 0.99 * (r.adjusted_score - min_score) / (max_score - min_score)
        normalized.append(round(norm, 2))

    return normalized


def export_csv(
    ranked: list[RankedCandidate],
    reasoning_list: list[str],
    output_path: str | Path,
) -> Path:
    """Export ranked candidates to submission.csv.

    Args:
        ranked: List of top-100 RankedCandidate objects.
        reasoning_list: List of reasoning strings (same order as ranked).
        output_path: Path to write the CSV file.

    Returns:
        Path to the written CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize scores
    normalized_scores = normalize_scores(ranked)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        # Header
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        # Data rows
        for i, (r, reasoning) in enumerate(zip(ranked, reasoning_list)):
            writer.writerow([
                r.candidate_id,
                r.rank,
                normalized_scores[i],
                reasoning,
            ])

    logger.info(
        "Exported %d candidates to %s", len(ranked), output_path
    )

    # Validation
    _validate_submission(output_path, len(ranked))

    return output_path


def _validate_submission(path: Path, expected_rows: int) -> None:
    """Validate the submission CSV format.

    Args:
        path: Path to the CSV file.
        expected_rows: Expected number of data rows.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        assert header == ["candidate_id", "rank", "score", "reasoning"], \
            f"Invalid header: {header}"

        rows = list(reader)
        assert len(rows) == expected_rows, \
            f"Expected {expected_rows} rows, got {len(rows)}"

        # Check ranks are 1 to expected_rows
        ranks = [int(row[1]) for row in rows]
        assert ranks == list(range(1, expected_rows + 1)), \
            "Ranks must be sequential from 1"

        # Check scores are in [0, 1]
        scores = [float(row[2]) for row in rows]
        assert all(0.0 <= s <= 1.0 for s in scores), \
            "All scores must be in [0.0, 1.0]"

        # Check scores are non-increasing
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 0.001, \
                f"Scores must be non-increasing: rank {i} has score {scores[i]} > rank {i-1} score {scores[i-1]}"

    logger.info("Submission validation passed: %s", path)
