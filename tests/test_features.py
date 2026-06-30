"""Tests for Stage 2: Feature Extraction."""

import pytest

from src.parser import parse_candidate
from src.features import extract_features, build_candidate_text
from src.utils import normalize_text


# ---------------------------------------------------------------------------
# Candidate archetypes for testing
# ---------------------------------------------------------------------------

PERFECT_ML_CANDIDATE = {
    "candidate_id": "CAND_PERFECT",
    "profile": {
        "anonymized_name": "Perfect ML",
        "headline": "Senior ML Engineer | Search & Retrieval | FAISS | PyTorch",
        "summary": "7 years building production search and ranking systems with embeddings.",
        "location": "Pune, Maharashtra",
        "country": "India",
        "years_of_experience": 7.0,
        "current_title": "Senior Machine Learning Engineer",
        "current_company": "ProductCo",
        "current_company_size": "201-500",
        "current_industry": "Software",
    },
    "career_history": [
        {
            "company": "ProductCo",
            "title": "Senior ML Engineer",
            "start_date": "2021-01-01",
            "end_date": None,
            "duration_months": 42,
            "is_current": True,
            "industry": "Software",
            "company_size": "201-500",
            "description": "Built ranking and retrieval systems using FAISS, embeddings, and learning-to-rank. Deployed models to production with MLflow.",
        },
    ],
    "education": [{"institution": "IIT Delhi", "degree": "B.Tech", "field_of_study": "Computer Science", "start_year": 2013, "end_year": 2017, "grade": "9.0", "tier": "tier_1"}],
    "skills": [
        {"name": "Python", "proficiency": "advanced", "endorsements": 50, "duration_months": 80},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
        {"name": "PyTorch", "proficiency": "advanced", "endorsements": 40, "duration_months": 60},
        {"name": "Elasticsearch", "proficiency": "intermediate", "endorsements": 20, "duration_months": 24},
        {"name": "XGBoost", "proficiency": "advanced", "endorsements": 25, "duration_months": 48},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 90.0, "signup_date": "2024-01-01",
        "last_active_date": "2025-06-01", "open_to_work_flag": True,
        "profile_views_received_30d": 20, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.85, "avg_response_time_hours": 12.0,
        "skill_assessment_scores": {}, "connection_count": 800,
        "endorsements_received": 165, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 45.0},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 80, "search_appearance_30d": 60,
        "saved_by_recruiters_30d": 10, "interview_completion_rate": 0.95,
        "offer_acceptance_rate": 0.80, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    },
}

NON_TECH_CANDIDATE = {
    "candidate_id": "CAND_NONTECH",
    "profile": {
        "anonymized_name": "Non Tech",
        "headline": "Accountant | 10+ years experience",
        "summary": "Experienced accountant in manufacturing.",
        "location": "Mumbai",
        "country": "India",
        "years_of_experience": 10.0,
        "current_title": "Accountant",
        "current_company": "MfgCorp",
        "current_company_size": "1001-5000",
        "current_industry": "Manufacturing",
    },
    "career_history": [
        {
            "company": "MfgCorp", "title": "Accountant",
            "start_date": "2020-01-01", "end_date": None,
            "duration_months": 60, "is_current": True,
            "industry": "Manufacturing", "company_size": "1001-5000",
            "description": "Month-end close, financial reporting, tax filings.",
        },
    ],
    "education": [{"institution": "Local College", "degree": "B.Com", "field_of_study": "Commerce", "start_year": 2010, "end_year": 2013, "grade": "70%", "tier": "tier_4"}],
    "skills": [
        {"name": "Excel", "proficiency": "advanced", "endorsements": 10, "duration_months": 100},
        {"name": "Accounting", "proficiency": "advanced", "endorsements": 5, "duration_months": 120},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 50.0, "signup_date": "2024-06-01",
        "last_active_date": "2025-01-01", "open_to_work_flag": False,
        "profile_views_received_30d": 2, "applications_submitted_30d": 1,
        "recruiter_response_rate": 0.30, "avg_response_time_hours": 72.0,
        "skill_assessment_scores": {}, "connection_count": 100,
        "endorsements_received": 15, "notice_period_days": 90,
        "expected_salary_range_inr_lpa": {"min": 8.0, "max": 12.0},
        "preferred_work_mode": "office", "willing_to_relocate": False,
        "github_activity_score": -1, "search_appearance_30d": 5,
        "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.50,
        "offer_acceptance_rate": -1, "verified_email": True,
        "verified_phone": False, "linkedin_connected": False,
    },
}


class TestFeatureExtraction:
    """Test feature extraction for different candidate archetypes."""

    def test_perfect_candidate_flags(self):
        """Test that a perfect ML candidate has relevant flags set."""
        record = parse_candidate(PERFECT_ML_CANDIDATE)
        normalized = normalize_text(record)
        fv = extract_features(record, normalized)

        assert fv.has_retrieval_experience is True
        assert fv.has_ranking_experience is True
        assert fv.has_production_ml is True
        assert fv.has_python is True
        assert fv.is_relevant_title is True
        assert fv.is_non_tech_title is False

    def test_non_tech_candidate_flags(self):
        """Test that a non-tech candidate has no ML flags."""
        record = parse_candidate(NON_TECH_CANDIDATE)
        normalized = normalize_text(record)
        fv = extract_features(record, normalized)

        assert fv.has_retrieval_experience is False
        assert fv.has_ranking_experience is False
        assert fv.has_production_ml is False
        assert fv.is_non_tech_title is True
        assert fv.is_relevant_title is False

    def test_skill_counts(self):
        """Test skill count extraction."""
        record = parse_candidate(PERFECT_ML_CANDIDATE)
        normalized = normalize_text(record)
        fv = extract_features(record, normalized)

        assert fv.total_skill_count == 5
        assert fv.retrieval_skill_count >= 1  # FAISS, Elasticsearch
        assert fv.relevant_skill_count > 0

    def test_behavioral_features(self):
        """Test behavioral feature extraction."""
        record = parse_candidate(PERFECT_ML_CANDIDATE)
        normalized = normalize_text(record)
        fv = extract_features(record, normalized)

        assert fv.open_to_work is True
        assert fv.recruiter_response_rate == 0.85
        assert fv.github_activity_score == 80
        assert fv.notice_period_days == 30

    def test_education_features(self):
        """Test education feature extraction."""
        record = parse_candidate(PERFECT_ML_CANDIDATE)
        normalized = normalize_text(record)
        fv = extract_features(record, normalized)

        assert fv.best_tier == "tier_1"
        assert fv.has_ml_ai_specialization is True  # CS counts

    def test_career_stability(self):
        """Test career stability metrics."""
        record = parse_candidate(PERFECT_ML_CANDIDATE)
        normalized = normalize_text(record)
        fv = extract_features(record, normalized)

        assert fv.num_companies == 1
        assert fv.is_title_chaser is False
        assert fv.is_consulting_only is False


class TestBuildCandidateText:
    """Test candidate text builder."""

    def test_text_includes_key_sections(self):
        """Test that built text includes headline, summary, and skills."""
        record = parse_candidate(PERFECT_ML_CANDIDATE)
        text = build_candidate_text(record)

        assert "ML Engineer" in text
        assert "FAISS" in text
        assert "PyTorch" in text

    def test_text_truncation(self):
        """Test that text is truncated to reasonable length."""
        record = parse_candidate(PERFECT_ML_CANDIDATE)
        text = build_candidate_text(record)
        assert len(text) <= 2000
