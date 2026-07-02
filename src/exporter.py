"""
Stage 10: CSV Export

Writes the final submission.csv with exactly 100 ranked candidates.
Format: candidate_id,rank,score,reasoning
"""

from __future__ import annotations

import csv
import logging
import io
import sys
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
    """Export ranked candidates to submission.csv after tie-breaking and validation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize scores
    normalized_scores = normalize_scores(ranked)

    # Combine data for sorting
    combined = list(zip(ranked, reasoning_list, normalized_scores))
    
    # Primary key: score descending, Secondary key: candidate_id ascending
    combined.sort(key=lambda x: (-x[2], x[0].candidate_id))
    
    # Regenerate ranks sequentially (1...100) after sorting
    for i, item in enumerate(combined, 1):
        item[0].rank = i

    # Validation in memory BEFORE writing the final file
    _validate_submission(combined)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r, reasoning, norm_score in combined:
            writer.writerow([r.candidate_id, r.rank, norm_score, reasoning])

    logger.info("Exported %d candidates to %s", len(combined), output_path)

    return output_path


def _validate_submission(combined: list[tuple[RankedCandidate, str, float]]) -> None:
    """Validate the submission CSV format in memory before writing."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n===========================================")
    print("Validating submission.csv")
    print("===========================================")
    
    try:
        # Build virtual CSV to strictly test formatting
        mem_file = io.StringIO()
        writer = csv.writer(mem_file, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r, reasoning, norm_score in combined:
            writer.writerow([r.candidate_id, r.rank, norm_score, reasoning])
            
        # Parse it back
        mem_file.seek(0)
        reader = csv.reader(mem_file)
        
        # 1. Check Header
        header = next(reader)
        if header != ["candidate_id", "rank", "score", "reasoning"]:
            raise ValueError(f"Invalid header: {header}")
        print("✓ Header")
        
        rows = list(reader)
        
        # 2. Check Candidate IDs (uniqueness and count)
        if len(rows) != len(combined):
            raise ValueError(f"Expected {len(combined)} rows, got {len(rows)}")
            
        c_ids = [row[0] for row in rows]
        if len(set(c_ids)) != len(c_ids):
            raise ValueError("Duplicate candidate IDs found")
        print("✓ Candidate IDs")
            
        # 3. Check Ranks
        ranks = [int(row[1]) for row in rows]
        expected_ranks = list(range(1, len(combined) + 1))
        if ranks != expected_ranks:
            raise ValueError("Ranks must be exactly 1 to 100 sequentially")
        print("✓ Ranks")
        
        # 4. Check Score Ordering
        scores = [float(row[2]) for row in rows]
        for i in range(1, len(scores)):
            if scores[i] > scores[i - 1]:
                raise ValueError(f"Scores must be non-increasing: rank {i+1} has score {scores[i]} > rank {i} score {scores[i-1]}")
        print("✓ Score Ordering")
        
        # 5. Check Tie-breaking
        for i in range(1, len(rows)):
            if scores[i] == scores[i - 1]:
                if c_ids[i] < c_ids[i - 1]:
                    raise ValueError(f"Tie-breaking invalid: rank {i+1} ({c_ids[i]}) should be before rank {i} ({c_ids[i-1]}) for score {scores[i]}")
        print("✓ Tie-breaking")
        
        # 6. CSV Format
        print("✓ CSV Format\n")
        print("Submission validation passed.")
        
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")
        sys.exit(1)
