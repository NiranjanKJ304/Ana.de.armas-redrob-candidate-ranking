"""
Main Orchestrator — Redrob Intelligent Candidate Ranking System

Runs all 10 stages sequentially:
 1. Parse candidates (streaming JSONL)
 2. Feature extraction
 3. Rule-based scoring
 4. Consistency validation
 5. Honeypot detection
 6. Pre-filtering
 7. Semantic retrieval (embeddings + similarity)
 8. Final ranking
 9. Reasoning generation
10. CSV export

Usage:
    python src/main.py [--candidates PATH] [--jd PATH] [--output PATH] [--top-k N]
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

# Force offline mode for HuggingFace — skip HTTP checks on cached models
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import stream_candidates, load_job_description, CandidateRecord
from src.features import extract_features, build_candidate_text, FeatureVector
from src.retrieval import encode_texts, encode_single, compute_similarities
from src.scorer import compute_all_scores
from src.consistency import compute_consistency_score
from src.honeypot import compute_honeypot_score
from src.ranker import rank_candidates, compute_base_score, compute_adjusted_score
from src.reasoning import generate_reasoning
from src.exporter import export_csv
from src.utils import (
    NormalizedText,
    PipelineSummary,
    StageLogger,
    dynamic_prefilter_k,
    normalize_text,
    release_memory,
)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    candidates_path: str | Path,
    jd_path: str | Path,
    output_path: str | Path,
    top_k: int = 100,
) -> Path:
    """Run the complete ranking pipeline.

    Args:
        candidates_path: Path to candidates.jsonl
        jd_path: Path to job_description.txt
        output_path: Path to write submission.csv
        top_k: Number of top candidates to return

    Returns:
        Path to the generated submission.csv
    """
    summary = PipelineSummary()
    logger = logging.getLogger("pipeline")

    # ================================================================
    # STAGE 1: Parse candidates
    # ================================================================
    with StageLogger(1, "Parsing Candidates") as sl:
        candidates: list[CandidateRecord] = []
        candidate_texts: list[str] = []
        candidate_ids: list[str] = []

        for candidate in tqdm(
            stream_candidates(candidates_path),
            desc="Parsing",
            total=100000,
        ):
            candidates.append(candidate)
            candidate_texts.append(build_candidate_text(candidate))
            candidate_ids.append(candidate.candidate_id)

        n_candidates = len(candidates)
        summary.total_candidates = n_candidates
        sl.set_metric("Candidates Processed", n_candidates)

    summary.update_peak_memory()

    # ================================================================
    # STAGES 2-5: Feature Extraction + Rule Scoring + Consistency +
    #             Honeypot Detection (single pass for performance)
    # ================================================================
    # Merging these stages into one loop eliminates 3 extra passes
    # over 100K candidates, saving ~30s of overhead.

    features_list: list[FeatureVector] = []
    normalized_texts: list[NormalizedText] = []
    consistency_scores: list[float] = []
    honeypot_scores: list[float] = []
    pre_scores = np.zeros(n_candidates)

    with StageLogger(2, "Feature Extraction") as sl:
        for i in range(n_candidates):
            normalized = normalize_text(candidates[i])
            normalized_texts.append(normalized)
            fv = extract_features(candidates[i], normalized)
            features_list.append(fv)
        sl.set_metric("Features Generated", len(features_list))

    summary.update_peak_memory()

    with StageLogger(3, "Rule Scoring") as sl:
        pre_sub_scores: list[dict[str, float]] = []
        for i in range(n_candidates):
            sub_scores = compute_all_scores(features_list[i], 0.0)
            pre_sub_scores.append(sub_scores)
        sl.set_metric("Candidates Scored", n_candidates)
        if n_candidates > 0:
            avg = np.mean([compute_base_score(s) for s in pre_sub_scores])
            sl.set_metric("Average Score", f"{avg:.2f}")

    summary.update_peak_memory()

    with StageLogger(4, "Consistency Validation") as sl:
        for i in range(n_candidates):
            cs = compute_consistency_score(candidates[i], features_list[i])
            consistency_scores.append(cs)
        cs_arr = np.array(consistency_scores)
        sl.set_metric("Average Consistency", f"{cs_arr.mean():.2f}")
        sl.set_metric("Flagged Candidates", int((cs_arr < 0.7).sum()))

    summary.update_peak_memory()

    with StageLogger(5, "Honeypot Detection") as sl:
        for i in range(n_candidates):
            hs = compute_honeypot_score(
                candidates[i], features_list[i], normalized_texts[i],
            )
            honeypot_scores.append(hs)
        hs_arr = np.array(honeypot_scores)
        sl.set_metric("Potential Honeypots", int((hs_arr > 0.3).sum()))
        sl.set_metric("Average Honeypot Score", f"{hs_arr.mean():.2f}")

    summary.update_peak_memory()

    # Compute pre-filter adjusted scores
    for i in range(n_candidates):
        base = compute_base_score(pre_sub_scores[i])
        pre_scores[i] = compute_adjusted_score(
            base, consistency_scores[i], honeypot_scores[i],
        )

    # Release pre_sub_scores (no longer needed)
    del pre_sub_scores
    release_memory()

    # ================================================================
    # STAGE 6: Pre-filtering
    # ================================================================
    with StageLogger(6, "Pre-filtering") as sl:
        pre_filter_k = dynamic_prefilter_k(n_candidates)
        top_indices = np.argsort(pre_scores)[::-1][:pre_filter_k]

        sl.set_metric("Input Candidates", n_candidates)
        sl.set_metric("Candidates Selected", pre_filter_k)

    summary.update_peak_memory()

    # ================================================================
    # STAGE 7: Semantic retrieval
    # ================================================================
    with StageLogger(7, "Semantic Retrieval") as sl:
        # Load and encode job description
        jd_text = load_job_description(jd_path)
        logger.info("Job description loaded (%d chars)", len(jd_text))
        jd_embedding = encode_single(jd_text)
        sl.set_metric("Embedding Model Loaded", "Yes")

        # Encode ONLY the pre-filtered candidate texts
        filtered_texts = [candidate_texts[i] for i in top_indices]

        # Free original text data to save memory
        del candidate_texts
        release_memory()

        candidate_embeddings = encode_texts(
            filtered_texts,
            batch_size=256,
            show_progress=True,
        )

        del filtered_texts
        release_memory()

        # Compute similarities
        semantic_scores_subset = compute_similarities(jd_embedding, candidate_embeddings)

        sl.set_metric("Candidates Embedded", len(semantic_scores_subset))
        sl.set_metric("Average Similarity", f"{semantic_scores_subset.mean():.2f}")

        # Free embeddings
        del candidate_embeddings, jd_embedding
        release_memory()

    summary.update_peak_memory()

    # ================================================================
    # STAGE 8: Final ranking
    # ================================================================
    with StageLogger(8, "Final Ranking") as sl:
        filtered_candidate_ids: list[str] = []
        filtered_all_sub_scores: list[dict[str, float]] = []
        filtered_consistency: list[float] = []
        filtered_honeypot: list[float] = []

        # Map index in subset back to original index
        subset_to_original: dict[int, int] = {}

        for j, original_idx in enumerate(top_indices):
            semantic = float(semantic_scores_subset[j])
            fv = features_list[original_idx]

            # Recompute all scores with semantic
            sub_scores = compute_all_scores(fv, semantic)

            filtered_candidate_ids.append(candidate_ids[original_idx])
            filtered_all_sub_scores.append(sub_scores)
            filtered_consistency.append(consistency_scores[original_idx])
            filtered_honeypot.append(honeypot_scores[original_idx])

            subset_to_original[j] = int(original_idx)

        # Free pre-filter arrays
        del pre_scores, semantic_scores_subset
        release_memory()

        # Rank the subset
        ranked = rank_candidates(
            candidate_ids=filtered_candidate_ids,
            all_sub_scores=filtered_all_sub_scores,
            consistency_scores=filtered_consistency,
            honeypot_scores=filtered_honeypot,
            top_k=top_k,
        )

        # Fix original_index to point to the actual candidate index
        for r in ranked:
            r.original_index = subset_to_original[r.original_index]

        sl.set_metric("Top Candidates Selected", top_k)

        # Free filtered data
        del filtered_candidate_ids, filtered_all_sub_scores
        del filtered_consistency, filtered_honeypot, subset_to_original
        release_memory()

    summary.update_peak_memory()

    # ================================================================
    # STAGE 9: Reasoning generation
    # ================================================================
    with StageLogger(9, "Reasoning Generation") as sl:
        reasoning_list: list[str] = []
        for r in ranked:
            idx = r.original_index
            reasoning = generate_reasoning(
                candidates[idx],
                features_list[idx],
                r,
                normalized_texts[idx],
            )
            reasoning_list.append(reasoning)

        sl.set_metric("Reasoning Generated", len(reasoning_list))

    # Release large data structures no longer needed
    del candidates, features_list, normalized_texts, candidate_ids
    del consistency_scores, honeypot_scores
    release_memory()
    summary.update_peak_memory()

    # ================================================================
    # STAGE 10: CSV export
    # ================================================================
    with StageLogger(10, "CSV Export") as sl:
        result_path = export_csv(ranked, reasoning_list, output_path)
        sl.set_metric("submission.csv Created", "Successfully")

    # ================================================================
    # Pipeline summary
    # ================================================================
    summary.display()

    # Print top 5 for quick inspection
    logger.info("Top 5 candidates:")
    for r, reason in zip(ranked[:5], reasoning_list[:5]):
        logger.info(
            "  Rank %d: %s (adj=%.2f, base=%.2f, cons=%.3f, hp=%.3f)",
            r.rank, r.candidate_id,
            r.adjusted_score, r.base_score,
            r.consistency_score, r.honeypot_score,
        )
        logger.info("    Reason: %s", reason[:150])

    return result_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Redrob Intelligent Candidate Ranking System"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--jd",
        type=str,
        default=None,
        help="Path to job_description.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output submission.csv",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top candidates to return (default: 100)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    # Resolve paths
    project_root = Path(__file__).parent.parent
    candidates_path = args.candidates or str(project_root / "candidates.jsonl")
    jd_path = args.jd or str(project_root / "data" / "job_description.txt")
    output_path = args.output or str(project_root / "output" / "submission.csv")

    # Check if candidates file exists
    if not Path(candidates_path).exists():
        # Try the data/ subdirectory as fallback
        alt_path = str(project_root / "data" / "candidates.jsonl")
        if Path(alt_path).exists():
            candidates_path = alt_path
        else:
            print(f"ERROR: Candidates file not found at {candidates_path}")
            print(f"       Also tried: {alt_path}")
            sys.exit(1)

    # Run
    run_pipeline(
        candidates_path=candidates_path,
        jd_path=jd_path,
        output_path=output_path,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
