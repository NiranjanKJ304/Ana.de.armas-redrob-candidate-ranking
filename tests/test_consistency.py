"""Tests for Stage 5: Consistency Validation."""

import pytest

from src.parser import parse_candidate
from src.features import extract_features
from src.consistency import compute_consistency_score
from src.utils import normalize_text


# ---------------------------------------------------------------------------
# Test candidates with various anomalies
# ---------------------------------------------------------------------------

CONSISTENT_CANDIDATE = {
    "candidate_id": "CAND_CONSISTENT",
    "profile": {
        "anonymized_name": "Consistent User",
        "headline": "ML Engineer | 7 years",
        "summary": "Senior ML engineer.",
        "location": "Pune",
        "country": "India",
        "years_of_experience": 7.0,
        "current_title": "ML Engineer",
        "current_company": "TechCo",
        "current_company_size": "201-500",
        "current_industry": "Software",
    },
    "career_history": [
        {
            "company": "TechCo", "title": "ML Engineer",
            "start_date": "2021-01-01", "end_date": None,
            "duration_months": 42, "is_current": True,
            "industry": "Software", "company_size": "201-500",
            "description": "Built ML systems for search ranking.",
        },
        {
            "company": "StartupX", "title": "Data Scientist",
            "start_date": "2018-01-01", "end_date": "2020-12-31",
            "duration_months": 36, "is_current": False,
            "industry": "Technology", "company_size": "51-200",
            "description": "Developed recommendation models.",
        },
    ],
    "education": [
        {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
         "start_year": 2013, "end_year": 2017, "grade": "8.5", "tier": "tier_1"},
    ],
    "skills": [
        {"name": "Python", "proficiency": "advanced", "endorsements": 50, "duration_months": 60},
        {"name": "PyTorch", "proficiency": "advanced", "endorsements": 30, "duration_months": 48},
    ],
    "certifications": [],
    "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 85.0, "signup_date": "2024-01-01",
        "last_active_date": "2025-06-01", "open_to_work_flag": True,
        "profile_views_received_30d": 10, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.75, "avg_response_time_hours": 24.0,
        "skill_assessment_scores": {}, "connection_count": 500,
        "endorsements_received": 80, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 25.0, "max": 35.0},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 70, "search_appearance_30d": 40,
        "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.90,
        "offer_acceptance_rate": 0.80, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    },
}

SALARY_INVERTED = {
    **CONSISTENT_CANDIDATE,
    "candidate_id": "CAND_SALARY_INV",
    "redrob_signals": {
        **CONSISTENT_CANDIDATE["redrob_signals"],
        "expected_salary_range_inr_lpa": {"min": 50.0, "max": 20.0},
    },
}


def _parse_and_extract(data: dict):
    """Helper: parse candidate, normalize, extract features."""
    record = parse_candidate(data)
    normalized = normalize_text(record)
    fv = extract_features(record, normalized)
    return record, fv


class TestConsistencyScore:
    """Test consistency score computation."""

    def test_consistent_candidate_high_score(self):
        """A consistent candidate should score close to 1.0."""
        record, fv = _parse_and_extract(CONSISTENT_CANDIDATE)
        score = compute_consistency_score(record, fv)
        assert score >= 0.8, f"Expected >= 0.8, got {score}"

    def test_salary_inversion_penalty(self):
        """Salary min > max should reduce score."""
        record, fv = _parse_and_extract(SALARY_INVERTED)
        score = compute_consistency_score(record, fv)
        assert score < 1.0, f"Expected < 1.0, got {score}"

    def test_score_in_range(self):
        """Score should always be in [0, 1]."""
        record, fv = _parse_and_extract(CONSISTENT_CANDIDATE)
        score = compute_consistency_score(record, fv)
        assert 0.0 <= score <= 1.0

    def test_education_timeline_anomaly(self):
        """PhD before Bachelor's should be penalized."""
        candidate = {
            **CONSISTENT_CANDIDATE,
            "candidate_id": "CAND_EDU_ANOMALY",
            "education": [
                {"institution": "Uni A", "degree": "Ph.D", "field_of_study": "CS",
                 "start_year": 2005, "end_year": 2009, "grade": "9.0", "tier": "tier_1"},
                {"institution": "Uni B", "degree": "B.Tech", "field_of_study": "CS",
                 "start_year": 2015, "end_year": 2019, "grade": "8.0", "tier": "tier_2"},
            ],
        }
        record, fv = _parse_and_extract(candidate)
        score = compute_consistency_score(record, fv)
        assert score < 0.9, f"Expected < 0.9 for PhD before Bachelors, got {score}"
