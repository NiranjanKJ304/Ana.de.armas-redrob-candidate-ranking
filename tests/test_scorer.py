"""Tests for Stage 4: Rule-Based Scoring."""

import pytest

from src.parser import parse_candidate
from src.features import extract_features
from src.scorer import (
    compute_retrieval_score,
    compute_ranking_score,
    compute_production_score,
    compute_behavioral_score,
    compute_experience_score,
    compute_education_score,
    compute_career_score,
    compute_all_scores,
)
from src.utils import normalize_text


# Reuse candidate fixtures from test_features
from tests.test_features import PERFECT_ML_CANDIDATE, NON_TECH_CANDIDATE


def _parse_and_extract(data: dict):
    """Helper: parse candidate, normalize, extract features."""
    record = parse_candidate(data)
    normalized = normalize_text(record)
    fv = extract_features(record, normalized)
    return record, fv


class TestRetrievalScore:
    """Test retrieval score computation."""

    def test_perfect_candidate_high_score(self):
        """Perfect ML candidate should score high on retrieval."""
        _, fv = _parse_and_extract(PERFECT_ML_CANDIDATE)
        score = compute_retrieval_score(fv)
        assert score >= 40, f"Expected >= 40, got {score}"

    def test_non_tech_candidate_low_score(self):
        """Non-tech candidate should score low on retrieval."""
        _, fv = _parse_and_extract(NON_TECH_CANDIDATE)
        score = compute_retrieval_score(fv)
        assert score <= 10, f"Expected <= 10, got {score}"


class TestExperienceScore:
    """Test experience fit scoring."""

    def test_ideal_experience(self):
        """7 years should score near 100 (ideal for 5-9yr JD)."""
        _, fv = _parse_and_extract(PERFECT_ML_CANDIDATE)
        score = compute_experience_score(fv)
        assert score >= 90, f"Expected >= 90, got {score}"

    def test_low_experience(self):
        """1 year should score low."""
        candidate = {**NON_TECH_CANDIDATE}
        candidate["profile"] = {**candidate["profile"], "years_of_experience": 1.0}
        _, fv = _parse_and_extract(candidate)
        score = compute_experience_score(fv)
        assert score <= 30, f"Expected <= 30, got {score}"


class TestBehavioralScore:
    """Test behavioral score computation."""

    def test_high_engagement_candidate(self):
        """Highly engaged candidate should score well."""
        _, fv = _parse_and_extract(PERFECT_ML_CANDIDATE)
        score = compute_behavioral_score(fv)
        assert score >= 50, f"Expected >= 50, got {score}"

    def test_low_engagement_candidate(self):
        """Low engagement candidate should score lower."""
        _, fv = _parse_and_extract(NON_TECH_CANDIDATE)
        score = compute_behavioral_score(fv)
        assert score <= 40, f"Expected <= 40, got {score}"


class TestAllScores:
    """Test compute_all_scores integration."""

    def test_returns_all_score_types(self):
        """Should return all expected score keys."""
        _, fv = _parse_and_extract(PERFECT_ML_CANDIDATE)
        scores = compute_all_scores(fv, 0.85)

        expected_keys = {
            "semantic_score", "retrieval_score", "ranking_score",
            "production_score", "behavioral_score", "experience_score",
            "education_score", "career_score",
        }
        assert set(scores.keys()) == expected_keys

    def test_scores_in_range(self):
        """All scores should be in [0, 100]."""
        _, fv = _parse_and_extract(PERFECT_ML_CANDIDATE)
        scores = compute_all_scores(fv, 0.85)

        for key, value in scores.items():
            assert 0 <= value <= 100, f"{key}={value} out of range"

    def test_semantic_score_scaling(self):
        """Semantic score should be cosine similarity * 100."""
        _, fv = _parse_and_extract(PERFECT_ML_CANDIDATE)
        scores = compute_all_scores(fv, 0.92)
        assert scores["semantic_score"] == 92.0
