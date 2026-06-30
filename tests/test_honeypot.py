"""Tests for Stage 6: Honeypot Detection Engine."""

import pytest

from src.parser import parse_candidate
from src.features import extract_features
from src.honeypot import compute_honeypot_score
from src.utils import normalize_text


# ---------------------------------------------------------------------------
# Genuine candidate — should score low honeypot
# ---------------------------------------------------------------------------

GENUINE_ML_CANDIDATE = {
    "candidate_id": "CAND_GENUINE",
    "profile": {
        "anonymized_name": "Genuine ML",
        "headline": "Senior ML Engineer | Search & Retrieval",
        "summary": "7 years building production ML systems for search and ranking.",
        "location": "Pune",
        "country": "India",
        "years_of_experience": 7.0,
        "current_title": "Senior Machine Learning Engineer",
        "current_company": "ProductCo",
        "current_company_size": "201-500",
        "current_industry": "Software",
    },
    "career_history": [
        {
            "company": "ProductCo", "title": "Senior ML Engineer",
            "start_date": "2021-01-01", "end_date": None,
            "duration_months": 42, "is_current": True,
            "industry": "Software", "company_size": "201-500",
            "description": "Built retrieval system using FAISS and sentence-transformers. "
                           "Deployed ranking models with MLflow. Designed evaluation framework "
                           "with NDCG and MRR metrics. Production ML pipeline serving 10M queries/day.",
        },
        {
            "company": "SearchCo", "title": "ML Engineer",
            "start_date": "2018-01-01", "end_date": "2020-12-31",
            "duration_months": 36, "is_current": False,
            "industry": "Technology", "company_size": "51-200",
            "description": "Developed learning-to-rank models using XGBoost. "
                           "Built vector search infrastructure with Elasticsearch. "
                           "Implemented A/B testing framework for search quality.",
        },
    ],
    "education": [
        {"institution": "IIT Delhi", "degree": "B.Tech", "field_of_study": "Computer Science",
         "start_year": 2013, "end_year": 2017, "grade": "9.0", "tier": "tier_1"},
    ],
    "skills": [
        {"name": "Python", "proficiency": "advanced", "endorsements": 50, "duration_months": 80},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
        {"name": "PyTorch", "proficiency": "advanced", "endorsements": 40, "duration_months": 60},
        {"name": "Elasticsearch", "proficiency": "intermediate", "endorsements": 20, "duration_months": 24},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 90.0, "signup_date": "2024-01-01",
        "last_active_date": "2025-06-01", "open_to_work_flag": True,
        "profile_views_received_30d": 20, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.85, "avg_response_time_hours": 12.0,
        "skill_assessment_scores": {}, "connection_count": 800,
        "endorsements_received": 140, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 45.0},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 80, "search_appearance_30d": 60,
        "saved_by_recruiters_30d": 10, "interview_completion_rate": 0.95,
        "offer_acceptance_rate": 0.80, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    },
}


# ---------------------------------------------------------------------------
# Honeypot candidate — non-tech with AI buzzwords
# ---------------------------------------------------------------------------

HONEYPOT_BUZZWORD = {
    "candidate_id": "CAND_HONEYPOT_BUZZ",
    "profile": {
        "anonymized_name": "Honeypot Buzz",
        "headline": "Customer Support | AI enthusiast",
        "summary": "Curious about ChatGPT, LangChain, RAG, and Prompt Engineering. "
                   "Experimented with GenAI tools.",
        "location": "Mumbai",
        "country": "India",
        "years_of_experience": 3.0,
        "current_title": "Customer Support",
        "current_company": "TCS",
        "current_company_size": "10001+",
        "current_industry": "IT Services",
    },
    "career_history": [
        {
            "company": "TCS", "title": "Customer Support",
            "start_date": "2022-01-01", "end_date": None,
            "duration_months": 30, "is_current": True,
            "industry": "IT Services", "company_size": "10001+",
            "description": "Customer support team lead. Managed tier-1 and tier-2 tickets. "
                           "Built knowledge base. No ML or AI involvement.",
        },
    ],
    "education": [
        {"institution": "Local College", "degree": "B.Com", "field_of_study": "Commerce",
         "start_year": 2016, "end_year": 2019, "grade": "65%", "tier": "tier_4"},
    ],
    "skills": [
        {"name": "ChatGPT", "proficiency": "advanced", "endorsements": 5, "duration_months": 6},
        {"name": "LangChain", "proficiency": "intermediate", "endorsements": 2, "duration_months": 4},
        {"name": "Prompt Engineering", "proficiency": "advanced", "endorsements": 3, "duration_months": 8},
        {"name": "RAG", "proficiency": "intermediate", "endorsements": 1, "duration_months": 3},
        {"name": "GenAI", "proficiency": "beginner", "endorsements": 0, "duration_months": 2},
        {"name": "Excel", "proficiency": "intermediate", "endorsements": 10, "duration_months": 30},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 40.0, "signup_date": "2024-06-01",
        "last_active_date": "2025-01-01", "open_to_work_flag": True,
        "profile_views_received_30d": 2, "applications_submitted_30d": 8,
        "recruiter_response_rate": 0.20, "avg_response_time_hours": 96.0,
        "skill_assessment_scores": {}, "connection_count": 50,
        "endorsements_received": 21, "notice_period_days": 90,
        "expected_salary_range_inr_lpa": {"min": 8.0, "max": 12.0},
        "preferred_work_mode": "office", "willing_to_relocate": False,
        "github_activity_score": -1, "search_appearance_30d": 5,
        "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.40,
        "offer_acceptance_rate": -1, "verified_email": True,
        "verified_phone": False, "linkedin_connected": False,
    },
}


# ---------------------------------------------------------------------------
# Honeypot candidate — title/career description mismatch
# ---------------------------------------------------------------------------

HONEYPOT_TITLE_MISMATCH = {
    "candidate_id": "CAND_HONEYPOT_TITLE",
    "profile": {
        "anonymized_name": "Honeypot Title",
        "headline": "Accountant | Helping teams scale",
        "summary": "Professional accountant with AI curiosity.",
        "location": "Delhi",
        "country": "India",
        "years_of_experience": 8.0,
        "current_title": "Accountant",
        "current_company": "MfgCorp",
        "current_company_size": "1001-5000",
        "current_industry": "Manufacturing",
    },
    "career_history": [
        {
            "company": "MfgCorp", "title": "Accountant",
            "start_date": "2020-01-01", "end_date": None,
            "duration_months": 48, "is_current": True,
            "industry": "Manufacturing", "company_size": "1001-5000",
            "description": "Senior accounting role. Month-end close, financial reporting, "
                           "statutory compliance, tax filings.",
        },
        {
            "company": "Wipro", "title": "HR Manager",
            "start_date": "2017-01-01", "end_date": "2019-12-31",
            "duration_months": 36, "is_current": False,
            "industry": "IT Services", "company_size": "10001+",
            "description": "HR operations. Recruitment, onboarding, performance reviews.",
        },
    ],
    "education": [
        {"institution": "Local College", "degree": "M.Sc", "field_of_study": "Commerce",
         "start_year": 2010, "end_year": 2014, "grade": "70%", "tier": "tier_4"},
    ],
    "skills": [
        {"name": "Image Classification", "proficiency": "advanced", "endorsements": 50, "duration_months": 38},
        {"name": "Deep Learning", "proficiency": "advanced", "endorsements": 30, "duration_months": 35},
        {"name": "PyTorch", "proficiency": "intermediate", "endorsements": 15, "duration_months": 20},
        {"name": "SQL", "proficiency": "beginner", "endorsements": 5, "duration_months": 12},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 60.0, "signup_date": "2024-01-01",
        "last_active_date": "2025-06-01", "open_to_work_flag": True,
        "profile_views_received_30d": 5, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.35, "avg_response_time_hours": 72.0,
        "skill_assessment_scores": {}, "connection_count": 200,
        "endorsements_received": 100, "notice_period_days": 60,
        "expected_salary_range_inr_lpa": {"min": 15.0, "max": 22.0},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": -1, "search_appearance_30d": 20,
        "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.60,
        "offer_acceptance_rate": -1, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    },
}


def _parse_and_extract(data: dict):
    """Helper: parse candidate, normalize, extract features."""
    record = parse_candidate(data)
    normalized = normalize_text(record)
    fv = extract_features(record, normalized)
    return record, fv, normalized


class TestHoneypotDetection:
    """Test honeypot detection engine."""

    def test_genuine_candidate_low_score(self):
        """Genuine ML candidate should have low honeypot score (< 0.15)."""
        record, fv, normalized = _parse_and_extract(GENUINE_ML_CANDIDATE)
        score = compute_honeypot_score(record, fv, normalized)
        assert score < 0.15, f"Genuine candidate scored too high: {score}"

    def test_buzzword_honeypot_high_score(self):
        """Buzzword honeypot should have high honeypot score (> 0.25)."""
        record, fv, normalized = _parse_and_extract(HONEYPOT_BUZZWORD)
        score = compute_honeypot_score(record, fv, normalized)
        assert score > 0.25, f"Buzzword honeypot scored too low: {score}"

    def test_title_mismatch_honeypot(self):
        """Title/career mismatch honeypot should have elevated score."""
        record, fv, normalized = _parse_and_extract(HONEYPOT_TITLE_MISMATCH)
        score = compute_honeypot_score(record, fv, normalized)
        assert score > 0.15, f"Title mismatch honeypot scored too low: {score}"

    def test_score_in_range(self):
        """Honeypot score should always be in [0, 1]."""
        for candidate_data in [GENUINE_ML_CANDIDATE, HONEYPOT_BUZZWORD, HONEYPOT_TITLE_MISMATCH]:
            record, fv, normalized = _parse_and_extract(candidate_data)
            score = compute_honeypot_score(record, fv, normalized)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_empty_candidate_no_crash(self):
        """Empty candidate should not crash."""
        minimal = {
            "candidate_id": "CAND_EMPTY",
            "profile": {"years_of_experience": 0},
            "career_history": [],
            "education": [],
            "skills": [],
            "certifications": [],
            "languages": [],
            "redrob_signals": {},
        }
        record, fv, normalized = _parse_and_extract(minimal)
        score = compute_honeypot_score(record, fv, normalized)
        assert 0.0 <= score <= 1.0

    def test_genuine_vs_honeypot_separation(self):
        """Genuine candidates should score significantly lower than honeypots."""
        genuine_record, genuine_fv, genuine_norm = _parse_and_extract(GENUINE_ML_CANDIDATE)
        genuine_score = compute_honeypot_score(genuine_record, genuine_fv, genuine_norm)

        honeypot_record, honeypot_fv, honeypot_norm = _parse_and_extract(HONEYPOT_BUZZWORD)
        honeypot_score = compute_honeypot_score(honeypot_record, honeypot_fv, honeypot_norm)

        assert honeypot_score > genuine_score + 0.15, \
            f"Insufficient separation: genuine={genuine_score}, honeypot={honeypot_score}"
