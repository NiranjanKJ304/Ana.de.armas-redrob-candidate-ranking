"""Tests for Stage 1: Candidate Parser."""

import json
import tempfile
from pathlib import Path

import pytest

from src.parser import (
    CandidateRecord,
    CareerEntry,
    EducationEntry,
    SkillEntry,
    parse_candidate,
    stream_candidates,
    load_job_description,
)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_CANDIDATE = {
    "candidate_id": "CAND_TEST_001",
    "profile": {
        "anonymized_name": "Test User",
        "headline": "ML Engineer | Search & Retrieval",
        "summary": "Senior ML engineer with 7 years of experience.",
        "location": "Pune, Maharashtra",
        "country": "India",
        "years_of_experience": 7.0,
        "current_title": "ML Engineer",
        "current_company": "TechCorp",
        "current_company_size": "201-500",
        "current_industry": "Software",
    },
    "career_history": [
        {
            "company": "TechCorp",
            "title": "ML Engineer",
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": 30,
            "is_current": True,
            "industry": "Software",
            "company_size": "201-500",
            "description": "Built retrieval systems using FAISS and sentence-transformers.",
        },
        {
            "company": "DataCo",
            "title": "Data Scientist",
            "start_date": "2019-01-01",
            "end_date": "2021-12-31",
            "duration_months": 36,
            "is_current": False,
            "industry": "Technology",
            "company_size": "51-200",
            "description": "Developed ML models for search ranking.",
        },
    ],
    "education": [
        {
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2012,
            "end_year": 2016,
            "grade": "8.5 CGPA",
            "tier": "tier_1",
        },
    ],
    "skills": [
        {"name": "Python", "proficiency": "advanced", "endorsements": 50, "duration_months": 60},
        {"name": "FAISS", "proficiency": "intermediate", "endorsements": 20, "duration_months": 24},
        {"name": "PyTorch", "proficiency": "advanced", "endorsements": 30, "duration_months": 48},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 85.0,
        "signup_date": "2024-01-01",
        "last_active_date": "2025-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 15,
        "applications_submitted_30d": 5,
        "recruiter_response_rate": 0.75,
        "avg_response_time_hours": 24.0,
        "skill_assessment_scores": {},
        "connection_count": 500,
        "endorsements_received": 100,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 25.0, "max": 35.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 75,
        "search_appearance_30d": 50,
        "saved_by_recruiters_30d": 8,
        "interview_completion_rate": 0.90,
        "offer_acceptance_rate": 0.80,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}


class TestParseCandidate:
    """Test parse_candidate function."""

    def test_basic_parsing(self):
        """Test parsing a valid candidate."""
        record = parse_candidate(SAMPLE_CANDIDATE)
        assert record.candidate_id == "CAND_TEST_001"
        assert record.anonymized_name == "Test User"
        assert record.years_of_experience == 7.0
        assert record.current_title == "ML Engineer"

    def test_career_history(self):
        """Test career history parsing."""
        record = parse_candidate(SAMPLE_CANDIDATE)
        assert len(record.career_history) == 2
        assert record.career_history[0].company == "TechCorp"
        assert record.career_history[0].is_current is True
        assert record.career_history[1].duration_months == 36

    def test_education(self):
        """Test education parsing."""
        record = parse_candidate(SAMPLE_CANDIDATE)
        assert len(record.education) == 1
        assert record.education[0].degree == "B.Tech"
        assert record.education[0].tier == "tier_1"

    def test_skills(self):
        """Test skill parsing."""
        record = parse_candidate(SAMPLE_CANDIDATE)
        assert len(record.skills) == 3
        assert record.skills[0].name == "Python"
        assert record.skills[0].proficiency == "advanced"

    def test_signals(self):
        """Test redrob signals parsing."""
        record = parse_candidate(SAMPLE_CANDIDATE)
        assert record.signals.recruiter_response_rate == 0.75
        assert record.signals.github_activity_score == 75
        assert record.signals.open_to_work_flag is True
        assert record.signals.expected_salary_range is not None
        assert record.signals.expected_salary_range.min_lpa == 25.0

    def test_missing_fields(self):
        """Test parsing with missing optional fields."""
        minimal = {
            "candidate_id": "CAND_MINIMAL",
            "profile": {},
            "redrob_signals": {},
        }
        record = parse_candidate(minimal)
        assert record.candidate_id == "CAND_MINIMAL"
        assert record.years_of_experience == 0.0
        assert len(record.career_history) == 0
        assert len(record.skills) == 0

    def test_frozen_dataclass(self):
        """Test that CandidateRecord is immutable."""
        record = parse_candidate(SAMPLE_CANDIDATE)
        with pytest.raises(AttributeError):
            record.candidate_id = "NEW_ID"


class TestStreamCandidates:
    """Test stream_candidates function."""

    def test_stream_valid_jsonl(self, tmp_path):
        """Test streaming from a valid JSONL file."""
        jsonl_path = tmp_path / "test.jsonl"
        lines = [json.dumps(SAMPLE_CANDIDATE) + "\n"] * 3
        jsonl_path.write_text("".join(lines), encoding="utf-8")

        results = list(stream_candidates(str(jsonl_path)))
        assert len(results) == 3
        assert all(r.candidate_id == "CAND_TEST_001" for r in results)

    def test_stream_handles_malformed_lines(self, tmp_path):
        """Test that malformed lines are skipped gracefully."""
        jsonl_path = tmp_path / "test.jsonl"
        lines = [
            json.dumps(SAMPLE_CANDIDATE) + "\n",
            "INVALID JSON LINE\n",
            json.dumps(SAMPLE_CANDIDATE) + "\n",
        ]
        jsonl_path.write_text("".join(lines), encoding="utf-8")

        results = list(stream_candidates(str(jsonl_path)))
        assert len(results) == 2

    def test_stream_handles_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        jsonl_path = tmp_path / "test.jsonl"
        lines = [
            json.dumps(SAMPLE_CANDIDATE) + "\n",
            "\n",
            "  \n",
            json.dumps(SAMPLE_CANDIDATE) + "\n",
        ]
        jsonl_path.write_text("".join(lines), encoding="utf-8")

        results = list(stream_candidates(str(jsonl_path)))
        assert len(results) == 2

    def test_stream_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            list(stream_candidates("/nonexistent/path.jsonl"))


class TestLoadJobDescription:
    """Test load_job_description function."""

    def test_load_valid_file(self, tmp_path):
        """Test loading a valid JD file."""
        jd_path = tmp_path / "jd.txt"
        jd_path.write_text("Senior AI Engineer", encoding="utf-8")
        result = load_job_description(str(jd_path))
        assert result == "Senior AI Engineer"

    def test_load_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_job_description("/nonexistent/jd.txt")
