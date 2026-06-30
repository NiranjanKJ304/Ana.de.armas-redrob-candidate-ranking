"""
Stage 5: Consistency Validation

Detects data anomalies in candidate profiles and returns a consistency_score (0.0-1.0).
Higher score = more consistent profile. Penalties are subtracted from a base of 1.0.
"""

from __future__ import annotations

import logging

from .keywords import DEGREE_RANKS
from .parser import CandidateRecord
from .features import FeatureVector
from .utils import get_degree_rank

logger = logging.getLogger(__name__)


def _check_education_timeline(candidate: CandidateRecord) -> float:
    """Check for impossible education timelines.

    Returns penalty (0.0 = no issue, up to 0.2 = severe issue).
    """
    if len(candidate.education) < 2:
        return 0.0

    penalty = 0.0

    # Sort education by end_year
    sorted_edu = sorted(
        [e for e in candidate.education if e.end_year is not None],
        key=lambda e: e.end_year,
    )

    # Check PhD before Bachelor's
    for i, edu_a in enumerate(sorted_edu):
        for edu_b in sorted_edu[i + 1:]:
            rank_a = get_degree_rank(edu_a.degree, DEGREE_RANKS)
            rank_b = get_degree_rank(edu_b.degree, DEGREE_RANKS)
            # Higher degree ending before lower degree starts
            if rank_a > rank_b and edu_a.end_year and edu_b.start_year:
                if edu_a.end_year < edu_b.start_year:
                    penalty += 0.4

    # Check implausible graduation dates
    for edu in sorted_edu:
        if edu.end_year and edu.end_year < 1990:
            penalty += 0.05
        if edu.start_year and edu.end_year:
            duration = edu.end_year - edu.start_year
            if duration < 0:
                penalty += 0.1
            elif duration > 10:
                penalty += 0.05

    return min(penalty, 0.2)


def _check_experience_duration(candidate: CandidateRecord) -> float:
    """Check if years_of_experience conflicts with career_history total.

    Returns penalty (0.0-0.15).
    """
    if not candidate.career_history:
        return 0.0

    total_career_months = sum(
        entry.duration_months for entry in candidate.career_history
    )
    claimed_months = candidate.years_of_experience * 12

    if claimed_months <= 0:
        return 0.05

    # Allow some overlap but flag large discrepancies
    ratio = total_career_months / claimed_months if claimed_months > 0 else 0

    if ratio > 1.5:
        return 0.15
    elif ratio < 0.3 and candidate.years_of_experience > 3:
        return 0.10

    return 0.0


def _check_skill_duration(candidate: CandidateRecord) -> float:
    """Check for skill duration inconsistencies.

    Returns penalty (0.0-0.15).
    """
    if not candidate.skills:
        return 0.0

    career_months = candidate.years_of_experience * 12
    penalty = 0.0
    flags = 0

    for skill in candidate.skills:
        # Skill duration exceeds total career
        if career_months > 0 and skill.duration_months > career_months * 1.2:
            flags += 1

        # Advanced proficiency with very short duration
        if skill.proficiency == "advanced" and skill.duration_months < 6:
            flags += 1

    total_skills = len(candidate.skills)
    if total_skills > 0:
        flag_ratio = flags / total_skills
        if flag_ratio > 0.5:
            penalty = 0.30
        elif flag_ratio > 0.25:
            penalty = 0.20
        elif flag_ratio > 0:
            penalty = 0.10

    return penalty


def _check_salary(candidate: CandidateRecord) -> float:
    """Check for salary inconsistencies.

    Returns penalty (0.0-0.10).
    """
    salary = candidate.signals.expected_salary_range
    if salary is None:
        return 0.0

    penalty = 0.0

    # min > max
    if salary.min_lpa > 0 and salary.max_lpa > 0:
        if salary.min_lpa > salary.max_lpa:
            penalty += 0.20

    # Unrealistic values
    if salary.max_lpa > 200 and candidate.years_of_experience < 10:
        penalty += 0.05
    if salary.min_lpa < 1 and candidate.years_of_experience > 5:
        penalty += 0.03

    return min(penalty, 0.10)


def _check_title_mismatch(candidate: CandidateRecord) -> float:
    """Check if headline title conflicts with current_title.

    Returns penalty (0.0-0.10).
    """
    headline = candidate.headline.lower().strip()
    title = candidate.current_title.lower().strip()

    if not headline or not title:
        return 0.0

    # Extract the first part of headline (before |)
    headline_title = headline.split("|")[0].strip()

    # Check if title appears in headline at all
    if title in headline:
        return 0.0

    # Check if headline title and current title are in completely different domains
    tech_keywords = {"engineer", "developer", "scientist", "ml", "ai", "data", "software"}
    non_tech_keywords = {"accountant", "hr", "marketing", "sales", "support", "manager"}

    headline_tech = any(kw in headline_title for kw in tech_keywords)
    headline_non_tech = any(kw in headline_title for kw in non_tech_keywords)
    title_tech = any(kw in title for kw in tech_keywords)
    title_non_tech = any(kw in title for kw in non_tech_keywords)

    # Major domain mismatch
    if (headline_tech and title_non_tech) or (headline_non_tech and title_tech):
        return 0.10

    # Minor mismatch
    if headline_title and title and headline_title != title:
        return 0.03

    return 0.0


def _check_education_career_overlap(candidate: CandidateRecord) -> float:
    """Check for impossible education-career overlaps.

    Returns penalty (0.0-0.10).
    """
    if not candidate.education or not candidate.career_history:
        return 0.0

    penalty = 0.0

    for edu in candidate.education:
        if not edu.start_year or not edu.end_year:
            continue
        for career in candidate.career_history:
            if not career.start_date:
                continue
            try:
                career_start_year = int(career.start_date[:4])
            except (ValueError, IndexError):
                continue

            if (edu.start_year <= career_start_year <= edu.end_year
                    and career.duration_months > 12):
                penalty += 0.05

    return min(penalty, 0.10)


def _check_keyword_stuffing_light(candidate: CandidateRecord) -> float:
    """Light keyword stuffing check: >15 advanced skills with <3yr experience.

    Returns penalty (0.0-0.15).
    """
    advanced_count = sum(
        1 for s in candidate.skills if s.proficiency == "advanced"
    )

    if advanced_count > 15 and candidate.years_of_experience < 3:
        return 0.15
    elif advanced_count > 10 and candidate.years_of_experience < 2:
        return 0.10

    return 0.0


def compute_consistency_score(
    candidate: CandidateRecord,
    features: FeatureVector,
) -> float:
    """Compute overall consistency score for a candidate.

    Args:
        candidate: Parsed candidate record.
        features: Extracted feature vector.

    Returns:
        Score in [0.0, 1.0] where 1.0 = fully consistent.
    """
    base = 1.0

    penalties = {
        "education_timeline": _check_education_timeline(candidate),
        "experience_duration": _check_experience_duration(candidate),
        "skill_duration": _check_skill_duration(candidate),
        "salary": _check_salary(candidate),
        "title_mismatch": _check_title_mismatch(candidate),
        "education_career_overlap": _check_education_career_overlap(candidate),
        "keyword_stuffing": _check_keyword_stuffing_light(candidate),
    }

    total_penalty = sum(penalties.values())
    score = max(0.0, base - total_penalty)

    if total_penalty > 0.1:
        logger.debug(
            "Consistency penalties for %s: %s (total=%.3f, score=%.3f)",
            candidate.candidate_id, penalties, total_penalty, score,
        )

    return score
