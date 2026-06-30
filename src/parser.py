"""
Stage 1: Candidate Parsing — Streaming JSONL Reader

Streams candidates.jsonl one record at a time using orjson for fast parsing.
Never loads all 100k records into memory simultaneously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import orjson

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CareerEntry:
    """A single career history entry."""
    company: str
    title: str
    start_date: Optional[str]
    end_date: Optional[str]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass(frozen=True)
class EducationEntry:
    """A single education entry."""
    institution: str
    degree: str
    field_of_study: str
    start_year: Optional[int]
    end_year: Optional[int]
    grade: str
    tier: str


@dataclass(frozen=True)
class SkillEntry:
    """A single skill entry."""
    name: str
    proficiency: str
    endorsements: int
    duration_months: int


@dataclass(frozen=True)
class SalaryRange:
    """Expected salary range in INR LPA."""
    min_lpa: float
    max_lpa: float


@dataclass(frozen=True)
class RedrobSignals:
    """Platform behavioral signals."""
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, Any]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range: Optional[SalaryRange]
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass(frozen=True)
class CandidateRecord:
    """Fully parsed and typed candidate record."""
    candidate_id: str
    # Profile fields
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    # Nested structures
    career_history: tuple[CareerEntry, ...]
    education: tuple[EducationEntry, ...]
    skills: tuple[SkillEntry, ...]
    certifications: tuple[str, ...]
    languages: tuple[dict[str, str], ...]
    signals: RedrobSignals


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _safe_str(val: Any, default: str = "") -> str:
    """Safely convert to string."""
    if val is None:
        return default
    return str(val).strip()


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        if val is None or val == -1:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely convert to int."""
    try:
        if val is None or val == -1:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_bool(val: Any, default: bool = False) -> bool:
    """Safely convert to bool."""
    if val is None:
        return default
    return bool(val)


def _parse_career_entry(raw: dict[str, Any]) -> CareerEntry:
    """Parse a single career history entry."""
    return CareerEntry(
        company=_safe_str(raw.get("company")),
        title=_safe_str(raw.get("title")),
        start_date=raw.get("start_date"),
        end_date=raw.get("end_date"),
        duration_months=_safe_int(raw.get("duration_months")),
        is_current=_safe_bool(raw.get("is_current")),
        industry=_safe_str(raw.get("industry")),
        company_size=_safe_str(raw.get("company_size")),
        description=_safe_str(raw.get("description")),
    )


def _parse_education_entry(raw: dict[str, Any]) -> EducationEntry:
    """Parse a single education entry."""
    return EducationEntry(
        institution=_safe_str(raw.get("institution")),
        degree=_safe_str(raw.get("degree")),
        field_of_study=_safe_str(raw.get("field_of_study")),
        start_year=raw.get("start_year"),
        end_year=raw.get("end_year"),
        grade=_safe_str(raw.get("grade")),
        tier=_safe_str(raw.get("tier"), "tier_4"),
    )


def _parse_skill_entry(raw: dict[str, Any]) -> SkillEntry:
    """Parse a single skill entry."""
    return SkillEntry(
        name=_safe_str(raw.get("name")),
        proficiency=_safe_str(raw.get("proficiency"), "beginner"),
        endorsements=_safe_int(raw.get("endorsements")),
        duration_months=_safe_int(raw.get("duration_months")),
    )


def _parse_signals(raw: dict[str, Any]) -> RedrobSignals:
    """Parse the redrob_signals block."""
    salary_raw = raw.get("expected_salary_range_inr_lpa", {})
    salary = None
    if salary_raw and isinstance(salary_raw, dict):
        min_val = _safe_float(salary_raw.get("min"), -1.0)
        max_val = _safe_float(salary_raw.get("max"), -1.0)
        if min_val >= 0 or max_val >= 0:
            salary = SalaryRange(min_lpa=min_val, max_lpa=max_val)

    return RedrobSignals(
        profile_completeness_score=_safe_float(raw.get("profile_completeness_score")),
        signup_date=_safe_str(raw.get("signup_date")),
        last_active_date=_safe_str(raw.get("last_active_date")),
        open_to_work_flag=_safe_bool(raw.get("open_to_work_flag")),
        profile_views_received_30d=_safe_int(raw.get("profile_views_received_30d")),
        applications_submitted_30d=_safe_int(raw.get("applications_submitted_30d")),
        recruiter_response_rate=_safe_float(raw.get("recruiter_response_rate")),
        avg_response_time_hours=_safe_float(raw.get("avg_response_time_hours")),
        skill_assessment_scores=raw.get("skill_assessment_scores", {}),
        connection_count=_safe_int(raw.get("connection_count")),
        endorsements_received=_safe_int(raw.get("endorsements_received")),
        notice_period_days=_safe_int(raw.get("notice_period_days")),
        expected_salary_range=salary,
        preferred_work_mode=_safe_str(raw.get("preferred_work_mode")),
        willing_to_relocate=_safe_bool(raw.get("willing_to_relocate")),
        github_activity_score=_safe_float(raw.get("github_activity_score"), -1.0),
        search_appearance_30d=_safe_int(raw.get("search_appearance_30d")),
        saved_by_recruiters_30d=_safe_int(raw.get("saved_by_recruiters_30d")),
        interview_completion_rate=_safe_float(raw.get("interview_completion_rate")),
        offer_acceptance_rate=_safe_float(raw.get("offer_acceptance_rate"), -1.0),
        verified_email=_safe_bool(raw.get("verified_email")),
        verified_phone=_safe_bool(raw.get("verified_phone")),
        linkedin_connected=_safe_bool(raw.get("linkedin_connected")),
    )


def parse_candidate(raw: dict[str, Any]) -> CandidateRecord:
    """Parse a raw JSON dict into a typed CandidateRecord."""
    profile = raw.get("profile", {})
    signals_raw = raw.get("redrob_signals", {})

    career_entries = tuple(
        _parse_career_entry(c) for c in raw.get("career_history", [])
    )
    education_entries = tuple(
        _parse_education_entry(e) for e in raw.get("education", [])
    )
    skill_entries = tuple(
        _parse_skill_entry(s) for s in raw.get("skills", [])
    )
    cert_entries = tuple(
        _safe_str(c) if isinstance(c, str) else _safe_str(c.get("name", ""))
        for c in raw.get("certifications", [])
    )
    lang_entries = tuple(raw.get("languages", []))

    return CandidateRecord(
        candidate_id=_safe_str(raw.get("candidate_id")),
        anonymized_name=_safe_str(profile.get("anonymized_name")),
        headline=_safe_str(profile.get("headline")),
        summary=_safe_str(profile.get("summary")),
        location=_safe_str(profile.get("location")),
        country=_safe_str(profile.get("country")),
        years_of_experience=_safe_float(profile.get("years_of_experience")),
        current_title=_safe_str(profile.get("current_title")),
        current_company=_safe_str(profile.get("current_company")),
        current_company_size=_safe_str(profile.get("current_company_size")),
        current_industry=_safe_str(profile.get("current_industry")),
        career_history=career_entries,
        education=education_entries,
        skills=skill_entries,
        certifications=cert_entries,
        languages=lang_entries,
        signals=_parse_signals(signals_raw),
    )


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------

def stream_candidates(path: str | Path) -> Iterator[CandidateRecord]:
    """
    Stream candidates from a JSONL file, yielding one CandidateRecord at a time.

    Handles malformed lines gracefully with logging.
    Memory-efficient: never loads the full file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    parsed = 0
    errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = orjson.loads(line)
                candidate = parse_candidate(raw)
                parsed += 1
                yield candidate
            except Exception as e:
                errors += 1
                if errors <= 10:
                    logger.warning(
                        "Failed to parse line %d: %s", line_num, str(e)[:200]
                    )

    logger.info(
        "Streaming complete: %d parsed, %d errors out of %d lines",
        parsed, errors, parsed + errors,
    )


def load_job_description(path: str | Path) -> str:
    """Load job description text from file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Job description file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
