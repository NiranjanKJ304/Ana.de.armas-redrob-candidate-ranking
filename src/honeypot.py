"""
Stage 6: Honeypot Detection Engine

Detects ~80 intentionally misleading or inconsistent candidates in the dataset.

Returns honeypot_score (0.0-1.0) where:
- 0.0 = genuine candidate
- 1.0 = strong honeypot candidate

Runs 7 independent detectors, each producing a sub-score (0.0-1.0).
The final honeypot_score is a weighted combination.

Scoring formula:
    adjusted_score = base_score * consistency_score * (1 - honeypot_score)

Never immediately discards candidates — applies gradual penalties.

Accepts pre-normalized text from ``utils.normalize_text()`` to avoid
rebuilding career_text in each detector.
"""

from __future__ import annotations

import logging

from .keywords import (
    AI_ML_KEYWORDS,
    BUZZWORD_KEYWORDS,
    EVIDENCE_KEYWORDS,
    FOUNDATIONAL_SKILLS_KEYWORDS,
    ML_TITLE_TERMS,
    NON_TECH_DESC_TERMS,
    NON_TECH_TERMS,
    NON_TECH_TITLES,
    PRODUCTION_ML_KEYWORDS,
    RETRIEVAL_KEYWORDS,
    SEARCH_KEYWORDS,
    TECH_DESC_TERMS,
    TECH_TERMS,
)
from .features import FeatureVector
from .parser import CandidateRecord
from .utils import NormalizedText, check_keywords_in_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Honeypot detector weights
# ---------------------------------------------------------------------------

DETECTOR_WEIGHTS: dict[str, float] = {
    "career_inconsistency": 0.20,
    "timeline_inconsistency": 0.15,
    "skill_inconsistency": 0.15,
    "salary_inconsistency": 0.05,
    "title_inconsistency": 0.15,
    "buzzword_inflation": 0.20,
    "keyword_stuffing": 0.10,
}


# ---------------------------------------------------------------------------
# Detector 1: Career Inconsistency
# ---------------------------------------------------------------------------

def _detect_career_inconsistency(
    candidate: CandidateRecord,
    features: FeatureVector,
    career_text: str,
    career_titles: str,
) -> float:
    """Detect mismatch between claimed skills and actual work history.

    - AI/ML skills without AI/ML career descriptions
    - Search/retrieval skills without search career evidence
    - Tech skills with all non-tech career titles
    """
    if not candidate.skills:
        return 0.0

    mismatches = 0
    total_checks = 0

    # Check 1: AI/ML skills without AI/ML career evidence
    if features.ai_ml_skill_count > 0:
        total_checks += 1
        ai_career_evidence = check_keywords_in_text(career_text, AI_ML_KEYWORDS)
        if ai_career_evidence == 0:
            mismatches += 1

    # Check 2: Retrieval/search skills without career evidence
    if features.retrieval_skill_count > 0:
        total_checks += 1
        retrieval_career = check_keywords_in_text(career_text, RETRIEVAL_KEYWORDS)
        search_career = check_keywords_in_text(career_text, SEARCH_KEYWORDS)
        if retrieval_career + search_career == 0:
            mismatches += 1

    # Check 3: Tech skills but all non-tech career titles
    if features.ai_ml_skill_count + features.retrieval_skill_count > 2:
        total_checks += 1
        all_non_tech = all(
            any(nt in entry.title.lower() for nt in NON_TECH_TITLES)
            for entry in candidate.career_history
        ) if candidate.career_history else False
        if all_non_tech:
            mismatches += 1

    # Check 4: Production skills without production career evidence
    if features.production_skill_count > 0:
        total_checks += 1
        prod_career = check_keywords_in_text(career_text, PRODUCTION_ML_KEYWORDS)
        if prod_career == 0:
            mismatches += 1

    # Check 5: Fake AI Transition
    if features.is_relevant_title and candidate.career_history:
        total_checks += 1
        has_any_ml_history = False
        for entry in candidate.career_history:
            if check_keywords_in_text(entry.description.lower(), AI_ML_KEYWORDS) > 0:
                has_any_ml_history = True
                break
        if not has_any_ml_history:
            mismatches += 1

    if total_checks == 0:
        return 0.0

    return min(mismatches / total_checks, 1.0)


# ---------------------------------------------------------------------------
# Detector 2: Timeline Inconsistency
# ---------------------------------------------------------------------------

def _detect_timeline_inconsistency(candidate: CandidateRecord) -> float:
    """Detect impossible chronologies in education and career history."""
    anomalies: float = 0
    checks = 0

    # Check 1: Education ordering
    if len(candidate.education) >= 2:
        degrees_with_years = [
            (e.degree, e.start_year, e.end_year)
            for e in candidate.education
            if e.start_year and e.end_year
        ]

        degree_ranks: dict[str, int] = {
            "ph.d": 3, "phd": 3, "doctorate": 3,
            "m.tech": 2, "mtech": 2, "m.s": 2, "ms": 2, "m.sc": 2,
            "m.e.": 2, "m.e": 2, "mba": 2,
            "b.tech": 1, "btech": 1, "b.e.": 1, "b.e": 1,
            "b.s.": 1, "b.sc": 1, "bsc": 1,
        }

        def _rank(d: str) -> int:
            d_lower = d.lower().strip()
            for key, r in degree_ranks.items():
                if key in d_lower:
                    return r
            return 0

        for i, (deg_a, start_a, end_a) in enumerate(degrees_with_years):
            for deg_b, start_b, end_b in degrees_with_years[i + 1:]:
                checks += 1
                rank_a = _rank(deg_a)
                rank_b = _rank(deg_b)
                # Higher degree completed before lower degree started
                if rank_a > rank_b and end_a < start_b:
                    anomalies += 1
                elif rank_b > rank_a and end_b < start_a:
                    anomalies += 1

    # Check 2: Education dates sanity
    for edu in candidate.education:
        checks += 1
        if edu.start_year and edu.end_year:
            if edu.end_year < edu.start_year:
                anomalies += 1
            elif edu.end_year - edu.start_year > 10:
                anomalies += 0.5

    # Check 3: Experience duration conflicts
    if candidate.career_history:
        total_career_months = sum(
            e.duration_months for e in candidate.career_history
        )
        claimed_months = candidate.years_of_experience * 12
        checks += 1

        if claimed_months > 0 and total_career_months > claimed_months * 1.5:
            anomalies += 1
        elif claimed_months > 0 and total_career_months < claimed_months * 0.3:
            anomalies += 0.5

    # Check 4: Career before education completion
    if candidate.education and candidate.career_history:
        earliest_edu_end = min(
            (e.end_year for e in candidate.education if e.end_year),
            default=None,
        )
        if earliest_edu_end:
            for career in candidate.career_history:
                if career.start_date:
                    try:
                        career_start_year = int(career.start_date[:4])
                        checks += 1
                        if career_start_year < earliest_edu_end - 2:
                            anomalies += 0.5
                    except (ValueError, IndexError):
                        pass

    if checks == 0:
        return 0.0

    return min(anomalies / checks, 1.0)


# ---------------------------------------------------------------------------
# Detector 3: Skill Duration Inconsistency
# ---------------------------------------------------------------------------

def _detect_skill_inconsistency(candidate: CandidateRecord) -> float:
    """Detect inflated skill claims based on duration mismatches."""
    if not candidate.skills:
        return 0.0

    career_months = candidate.years_of_experience * 12
    flagged: float = 0
    total = len(candidate.skills)

    for skill in candidate.skills:
        # Skill duration exceeds total career
        if career_months > 0 and skill.duration_months > career_months * 1.2:
            flagged += 1
            continue

        # Advanced proficiency with very short experience
        if skill.proficiency == "advanced" and skill.duration_months < 12:
            flagged += 1
            continue

    # Many skills with very low average duration
    if total > 10:
        avg_duration = sum(s.duration_months for s in candidate.skills) / total
        if avg_duration < 6:
            flagged += total * 0.2

    if total == 0:
        return 0.0

    return min(flagged / total, 1.0)


# ---------------------------------------------------------------------------
# Detector 4: Salary Inconsistency
# ---------------------------------------------------------------------------

def _detect_salary_inconsistency(candidate: CandidateRecord) -> float:
    """Detect salary red flags: inverted range, unrealistic values."""
    salary = candidate.signals.expected_salary_range
    if salary is None:
        return 0.0

    flags = 0
    checks = 0

    # Inverted range
    if salary.min_lpa > 0 and salary.max_lpa > 0:
        checks += 1
        if salary.min_lpa > salary.max_lpa:
            flags += 1

    # Unrealistic values for experience level
    yoe = candidate.years_of_experience
    if salary.max_lpa > 0 and yoe > 0:
        checks += 1
        if yoe < 3 and salary.max_lpa > 50:
            flags += 1
        elif yoe < 5 and salary.max_lpa > 100:
            flags += 1

    if checks == 0:
        return 0.0

    return min(flags / checks, 1.0)


# ---------------------------------------------------------------------------
# Detector 5: Title Inconsistency
# ---------------------------------------------------------------------------

def _detect_title_inconsistency(
    candidate: CandidateRecord,
    career_text: str,
    headline_lower: str,
) -> float:
    """Detect mismatches between headline, current_title, and career descriptions."""
    mismatches = 0
    checks = 0

    current_title = candidate.current_title.lower()

    # Check 1: Headline vs current_title domain mismatch
    if headline_lower and current_title:
        checks += 1
        headline_is_tech = any(t in headline_lower for t in TECH_TERMS)
        headline_is_non_tech = any(t in headline_lower for t in NON_TECH_TERMS)
        title_is_tech = any(t in current_title for t in TECH_TERMS)
        title_is_non_tech = any(t in current_title for t in NON_TECH_TERMS)

        if (headline_is_tech and title_is_non_tech) or \
           (headline_is_non_tech and title_is_tech):
            mismatches += 1

    # Check 2: Title vs career descriptions domain
    if candidate.career_history:
        current_is_tech = any(
            t in current_title
            for t in {"engineer", "developer", "scientist", "ml", "ai"}
        )
        career_has_tech = check_keywords_in_text(career_text, AI_ML_KEYWORDS) > 0

        checks += 1
        if current_is_tech and not career_has_tech:
            mismatches += 1

    # Check 3: Career descriptions contradict titles
    if candidate.career_history:
        for entry in candidate.career_history:
            title_lower = entry.title.lower()
            desc_lower = entry.description.lower()
            checks += 1

            title_is_ml = any(t in title_lower for t in ML_TITLE_TERMS)
            desc_is_non_tech = any(
                t in desc_lower for t in NON_TECH_DESC_TERMS
            ) and not any(
                t in desc_lower for t in TECH_DESC_TERMS
            )

            if title_is_ml and desc_is_non_tech:
                mismatches += 1

    if checks == 0:
        return 0.0

    return min(mismatches / checks, 1.0)


# ---------------------------------------------------------------------------
# Detector 6: Buzzword Inflation
# ---------------------------------------------------------------------------

def _detect_buzzword_inflation(
    features: FeatureVector,
    career_text: str,
    full_text: str,
) -> float:
    """Detect profiles with AI buzzwords but no substantive career evidence."""
    # Count buzzwords in profile + skills (not career)
    buzzword_count = check_keywords_in_text(full_text, BUZZWORD_KEYWORDS)

    # Count evidence in career descriptions
    evidence_count = check_keywords_in_text(career_text, EVIDENCE_KEYWORDS)

    if buzzword_count < 2:
        return 0.0

    # Buzzword-to-evidence ratio
    ratio = buzzword_count / max(evidence_count, 1)

    if ratio > 3.0:
        score = ratio / (ratio + 1)
    elif ratio > 1.5 and evidence_count < 3:
        score = 0.3
    else:
        score = 0.0

    # Extra penalty: many buzzwords, non-tech title, no career ML evidence
    if buzzword_count >= 3 and features.is_non_tech_title and evidence_count == 0:
        score = max(score, 0.8)

    # Extra penalty: no foundational skills
    foundational_count = check_keywords_in_text(full_text, FOUNDATIONAL_SKILLS_KEYWORDS)
    if buzzword_count >= 2 and foundational_count == 0:
        score = max(score, 0.7)

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Detector 7: Keyword Stuffing
# ---------------------------------------------------------------------------

def _detect_keyword_stuffing(
    features: FeatureVector,
    career_text: str,
) -> float:
    """Detect excessive AI keyword accumulation in skills with low career support."""
    # AI keywords in skills
    ai_in_skills = features.ai_ml_skill_count + features.retrieval_skill_count

    # AI keywords backed by career descriptions
    ai_in_career = (
        check_keywords_in_text(career_text, AI_ML_KEYWORDS)
        + check_keywords_in_text(career_text, RETRIEVAL_KEYWORDS)
    )

    if ai_in_skills <= 2:
        return 0.0

    # Stuffing ratio
    ratio = ai_in_skills / max(ai_in_career, 1)

    # Penalize ratios > 2x
    if ratio > 2:
        score = max(0, (ratio - 2) / 8)
    else:
        score = 0.0

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Honeypot Score Aggregation
# ---------------------------------------------------------------------------

def compute_honeypot_score(
    candidate: CandidateRecord,
    features: FeatureVector,
    normalized: NormalizedText,
) -> float:
    """Compute the overall honeypot score for a candidate.

    Args:
        candidate: Parsed candidate record.
        features: Extracted feature vector.
        normalized: Pre-normalized text fields.

    Returns:
        Score in [0.0, 1.0] where 0.0 = genuine, 1.0 = strong honeypot.
    """
    career_text = normalized.career_text
    career_titles = normalized.career_titles
    headline_lower = normalized.headline_lower
    full_text = normalized.full_text

    scores: dict[str, float] = {
        "career_inconsistency": _detect_career_inconsistency(
            candidate, features, career_text, career_titles,
        ),
        "timeline_inconsistency": _detect_timeline_inconsistency(candidate),
        "skill_inconsistency": _detect_skill_inconsistency(candidate),
        "salary_inconsistency": _detect_salary_inconsistency(candidate),
        "title_inconsistency": _detect_title_inconsistency(
            candidate, career_text, headline_lower,
        ),
        "buzzword_inflation": _detect_buzzword_inflation(
            features, career_text, full_text,
        ),
        "keyword_stuffing": _detect_keyword_stuffing(features, career_text),
    }

    # Weighted combination
    honeypot_score = sum(
        DETECTOR_WEIGHTS[key] * value
        for key, value in scores.items()
    )

    # Clamp to [0, 1]
    honeypot_score = max(0.0, min(1.0, honeypot_score))

    if honeypot_score > 0.3:
        logger.debug(
            "High honeypot score for %s: %.3f (details: %s)",
            candidate.candidate_id, honeypot_score,
            {k: f"{v:.2f}" for k, v in scores.items() if v > 0},
        )

    return honeypot_score
