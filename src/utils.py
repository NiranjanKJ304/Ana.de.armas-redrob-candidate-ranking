"""
Shared Utilities — Redrob Candidate Ranking System

Provides:
- Text normalization (computed once, reused across stages)
- Keyword matching helpers
- Degree ranking helpers
- Professional stage logging
- Memory tracking
- Dynamic pre-filter K computation
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import CandidateRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalized text container — compute once, reuse everywhere
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NormalizedText:
    """Pre-normalized text fields for a single candidate.

    Computed once during parsing and reused by feature extraction,
    honeypot detection, and reasoning generation to avoid redundant
    string allocations and repeated ``.lower()`` calls.
    """
    career_text: str       # " ".join(entry.description for entry in career_history).lower()
    full_text: str         # headline + summary + career_text + skill_text, lowered
    skill_text: str        # " ".join(skill.name for skill in skills).lower()
    skill_names: tuple[str, ...]  # tuple of each skill.name.lower()
    headline_lower: str    # candidate.headline.lower()
    summary_lower: str     # candidate.summary.lower()
    career_titles: str     # " ".join(entry.title for entry in career_history).lower()


def normalize_text(candidate: "CandidateRecord") -> NormalizedText:
    """Normalize all candidate text fields once for reuse across stages.

    Args:
        candidate: Parsed candidate record.

    Returns:
        NormalizedText with all lowered text fields pre-computed.
    """
    career_text = " ".join(
        entry.description for entry in candidate.career_history
    ).lower()

    headline_lower = candidate.headline.lower()
    summary_lower = candidate.summary.lower()

    skill_names = tuple(s.name.lower() for s in candidate.skills)
    skill_text = " ".join(skill_names)

    full_text = f"{headline_lower} {summary_lower} {career_text} {skill_text}"

    career_titles = " ".join(
        entry.title for entry in candidate.career_history
    ).lower()

    return NormalizedText(
        career_text=career_text,
        full_text=full_text,
        skill_text=skill_text,
        skill_names=skill_names,
        headline_lower=headline_lower,
        summary_lower=summary_lower,
        career_titles=career_titles,
    )


# ---------------------------------------------------------------------------
# Keyword matching helpers
# ---------------------------------------------------------------------------

def check_keywords_in_text(text: str, keywords: frozenset[str]) -> int:
    """Count how many keywords appear in the text.

    Args:
        text: Pre-lowered text to search in.
        keywords: Set of lowered keyword strings.

    Returns:
        Number of keywords found in the text.
    """
    count = 0
    for kw in keywords:
        if kw in text:
            count += 1
    return count


def check_keywords_in_skills(
    skill_names: tuple[str, ...],
    keywords: frozenset[str],
) -> int:
    """Count how many skill names match any keyword.

    Args:
        skill_names: Tuple of pre-lowered skill name strings.
        keywords: Set of lowered keyword strings.

    Returns:
        Number of skills matching at least one keyword.
    """
    count = 0
    for name in skill_names:
        if name in keywords or any(kw in name for kw in keywords):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Degree / tier ranking helpers
# ---------------------------------------------------------------------------

def get_degree_rank(degree: str, degree_ranks: dict[str, int]) -> int:
    """Map a degree string to a numeric rank (higher = better).

    Args:
        degree: Raw degree string from education entry.
        degree_ranks: Mapping of degree substrings to rank values.

    Returns:
        Numeric rank for the degree; 0 if unrecognized.
    """
    degree_lower = degree.lower().strip()
    for key, rank in degree_ranks.items():
        if key in degree_lower:
            return rank
    return 0


def get_tier_rank(tier: str, tier_ranks: dict[str, int]) -> int:
    """Map a tier string to a numeric rank (higher = better).

    Args:
        tier: Tier string (e.g. "tier_1").
        tier_ranks: Mapping of tier strings to rank values.

    Returns:
        Numeric rank for the tier; 1 if unrecognized.
    """
    return tier_ranks.get(tier.lower().strip(), 1)


# ---------------------------------------------------------------------------
# Dynamic pre-filter K
# ---------------------------------------------------------------------------

def dynamic_prefilter_k(n_candidates: int) -> int:
    """Compute the number of candidates to keep for semantic embedding.

    Formula: 5% of dataset, floored at 500, capped at 2000.

    Args:
        n_candidates: Total number of candidates in the dataset.

    Returns:
        Number of candidates to pre-filter for embedding.
    """
    k = int(n_candidates * 0.05)
    k = max(k, 500)
    k = min(k, 2000)
    return min(k, n_candidates)


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def get_memory_mb() -> float:
    """Get current process RSS memory in megabytes.

    Falls back to 0.0 if psutil is not available.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def release_memory(*objects_to_del: str) -> None:
    """Force garbage collection to release unused memory.

    Call after deleting large objects at stage boundaries.
    """
    gc.collect()


# ---------------------------------------------------------------------------
# Professional stage logging
# ---------------------------------------------------------------------------

class StageLogger:
    """Context manager for professional stage-by-stage terminal logging.

    Usage::

        with StageLogger(1, "Parsing Candidates") as sl:
            # ... do work ...
            sl.set_metric("Candidates Processed", 100000)
    """

    _SEPARATOR = "=" * 50

    def __init__(self, stage_num: int, stage_name: str) -> None:
        self.stage_num = stage_num
        self.stage_name = stage_name
        self.metrics: list[tuple[str, str]] = []
        self._start_time: float = 0.0
        self._start_mem: float = 0.0
        self._logger = logging.getLogger("pipeline")

    def __enter__(self) -> "StageLogger":
        self._start_time = time.time()
        self._start_mem = get_memory_mb()
        print(f"\n{self._SEPARATOR}")
        print(f"Stage {self.stage_num:<3}: {self.stage_name}")
        print(self._SEPARATOR)
        return self

    def set_metric(self, name: str, value: object) -> None:
        """Record a metric to be displayed at stage completion."""
        self.metrics.append((name, str(value)))

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        elapsed = time.time() - self._start_time
        current_mem = get_memory_mb()

        for name, value in self.metrics:
            print(f"{name:<30}: {value}")

        print(f"{'Time Taken':<30}: {elapsed:.1f} sec")
        if current_mem > 0:
            print(f"{'Memory Used':<30}: {current_mem:.0f} MB")
        print(self._SEPARATOR)

        self._logger.info(
            "Stage %d [%s] completed in %.1fs",
            self.stage_num, self.stage_name, elapsed,
        )


class PipelineSummary:
    """Collect and display pipeline-wide summary statistics."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.start_mem: float = get_memory_mb()
        self.peak_mem: float = self.start_mem
        self.total_candidates: int = 0

    def update_peak_memory(self) -> None:
        """Update peak memory tracking."""
        current = get_memory_mb()
        if current > self.peak_mem:
            self.peak_mem = current

    def display(self) -> None:
        """Print the final pipeline summary."""
        total_time = time.time() - self.start_time
        self.update_peak_memory()

        sep = "=" * 50
        print(f"\n{sep}")
        print("Pipeline Summary")
        print(sep)
        print(f"{'Total Candidates':<30}: {self.total_candidates}")
        print(f"{'Total Runtime':<30}: {total_time:.1f} sec ({total_time / 60:.1f} min)")
        if self.peak_mem > 0:
            print(f"{'Peak Memory Usage':<30}: {self.peak_mem:.0f} MB")
        cpu_time = time.process_time()
        print(f"{'CPU Time':<30}: {cpu_time:.1f} sec")
        if total_time > 0:
            print(f"{'Average Candidate/sec':<30}: {self.total_candidates / total_time:.0f}")
        print(sep)
