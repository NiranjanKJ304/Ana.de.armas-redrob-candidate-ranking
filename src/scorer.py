"""
Stage 4: Rule-Based Evidence Scoring

Computes independent sub-scores (all normalized 0-100) for each candidate:
- retrieval_score: retrieval/search infrastructure skills
- ranking_score: ranking systems, evaluation frameworks
- production_score: production ML deployment experience
- behavioral_score: platform engagement signals
- experience_score: experience fit for 5-9yr sweet spot
- education_score: degree level + tier + specialization
- career_score: title relevance + company type + industry
"""

from __future__ import annotations

import logging
import math

from .features import FeatureVector

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Individual score functions
# ---------------------------------------------------------------------------

def compute_retrieval_score(fv: FeatureVector) -> float:
    """
    Score retrieval/search infrastructure experience (0-100).

    Based on: retrieval skills, vector DB experience, embeddings,
    search experience, and career evidence.
    """
    score = 0.0

    # Retrieval skill count (each worth up to 5 points, max 15)
    score += min(fv.retrieval_skill_count * 5, 15)

    # Boolean flags (5 points each, max 20)
    if fv.has_retrieval_experience:
        score += 5
    if fv.has_vector_databases:
        score += 5
    if fv.has_embeddings:
        score += 5
    if fv.has_search_experience:
        score += 5

    # Career description evidence (up to 65)
    career_evidence = fv.career_mentions_retrieval + fv.career_mentions_search
    score += min(career_evidence * 15, 65)

    return _clamp(score)


def compute_ranking_score(fv: FeatureVector) -> float:
    """
    Score ranking/evaluation experience (0-100).

    Based on: LTR, evaluation frameworks, ranking mentions, A/B testing.
    """
    score = 0.0

    # Ranking skill count (each worth 5 points, max 15)
    score += min(fv.ranking_skill_count * 5, 15)

    # Boolean flags
    if fv.has_ranking_experience:
        score += 10
    if fv.has_learning_to_rank:
        score += 10
    if fv.has_evaluation_frameworks:
        score += 5
    if fv.has_ab_testing:
        score += 5

    # Career evidence (up to 55)
    score += min(fv.career_mentions_ranking * 15, 55)

    return _clamp(score)


def compute_production_score(fv: FeatureVector) -> float:
    """
    Score production ML experience (0-100).

    Based on: MLOps tools, deployment, model serving, career evidence.
    """
    score = 0.0

    # Production skill count (5 each, max 15)
    score += min(fv.production_skill_count * 5, 15)

    # Boolean flags
    if fv.has_production_ml:
        score += 10
    if fv.has_python:
        score += 5

    # LLM fine-tuning is a nice-to-have
    if fv.has_llm_finetuning:
        score += 5

    # Career evidence (up to 65)
    score += min(fv.career_mentions_production * 15, 65)

    return _clamp(score)


def compute_behavioral_score(fv: FeatureVector) -> float:
    """
    Score platform behavioral signals (0-100).

    Composite of engagement metrics: recruiter response rate, GitHub activity,
    profile completeness, open_to_work, saved_by_recruiters, etc.
    """
    score = 0.0

    # Recruiter response rate (up to 25)
    score += fv.recruiter_response_rate * 25

    # GitHub activity (up to 15, -1 means not available)
    if fv.github_activity_score > 0:
        score += min(fv.github_activity_score / 100 * 15, 15)

    # Profile completeness (up to 10)
    score += min(fv.profile_completeness_score / 100 * 10, 10)

    # Open to work (5 points)
    if fv.open_to_work:
        score += 5

    # Saved by recruiters (up to 10)
    score += min(fv.saved_by_recruiters_30d * 2, 10)

    # Interview completion rate (up to 10)
    score += fv.interview_completion_rate * 10

    # Notice period bonus (shorter is better, up to 10)
    if 0 < fv.notice_period_days <= 30:
        score += 10
    elif 30 < fv.notice_period_days <= 60:
        score += 7
    elif 60 < fv.notice_period_days <= 90:
        score += 4
    elif fv.notice_period_days > 90:
        score += 1

    # Profile views (up to 5)
    score += min(fv.profile_views_30d * 0.5, 5)

    # Verification bonuses
    if fv.verified_email:
        score += 2
    if fv.linkedin_connected:
        score += 3

    return _clamp(score)


def compute_experience_score(fv: FeatureVector) -> float:
    """
    Score experience fit for 5-9yr sweet spot (0-100).

    Uses a bell curve centered around 7 years (JD ideal: 6-8).
    """
    yoe = fv.years_of_experience
    ideal_center = 7.0
    ideal_sigma = 2.5

    # Gaussian bell curve: peak at ideal_center, drops off with sigma
    fit = math.exp(-0.5 * ((yoe - ideal_center) / ideal_sigma) ** 2)
    score = fit * 100

    # Hard penalty for <3yr or >15yr (outside reasonable range)
    if yoe < 3.0:
        score *= 0.3
    elif yoe > 15.0:
        score *= 0.5

    return _clamp(score)


def compute_education_score(fv: FeatureVector) -> float:
    """
    Score education quality (0-100).

    Based on: degree level, institution tier, ML/AI specialization.
    """
    score = 0.0

    # Degree level (up to 30)
    degree_lower = fv.highest_degree.lower().strip() if fv.highest_degree else ""
    if "ph.d" in degree_lower or "phd" in degree_lower:
        score += 30
    elif "m.tech" in degree_lower or "mtech" in degree_lower:
        score += 25
    elif any(d in degree_lower for d in ("m.s", "ms", "m.sc", "msc", "m.e", "me")):
        score += 20
    elif any(d in degree_lower for d in ("b.tech", "btech", "b.e", "be")):
        score += 15
    elif any(d in degree_lower for d in ("b.s", "bs", "b.sc", "bsc")):
        score += 12
    else:
        score += 5

    # Tier (up to 30)
    tier_map = {"tier_1": 30, "tier_2": 22, "tier_3": 12, "tier_4": 5}
    score += tier_map.get(fv.best_tier, 5)

    # ML/AI specialization (40 points)
    if fv.has_ml_ai_specialization:
        score += 40

    return _clamp(score)


def compute_career_score(fv: FeatureVector) -> float:
    """
    Score career alignment (0-100).

    Based on: title relevance, company type, career stability, industry.
    """
    score = 0.0

    # Title relevance (up to 35)
    if fv.is_relevant_title:
        score += 35
    elif not fv.is_non_tech_title:
        # Some tech title but not directly relevant
        score += 10

    # Company type penalty
    if fv.is_consulting_only:
        score += 0  # No points for consulting-only
    elif not fv.is_consulting_only and fv.num_companies > 0:
        score += 20  # Has at least some non-consulting experience

    # Career stability (up to 15)
    if fv.is_title_chaser:
        score += 0
    elif fv.avg_tenure_months >= 24:
        score += 15
    elif fv.avg_tenure_months >= 18:
        score += 10
    else:
        score += 5

    # NLP/IR experience (up to 15)
    if fv.has_nlp_ir:
        score += 15

    # Recommendation systems (up to 10)
    if fv.has_recommendation_systems:
        score += 10

    # Penalty for CV/speech/robotics only (no NLP/IR)
    if fv.has_cv_speech_robotics and not fv.has_nlp_ir:
        score *= 0.3

    return _clamp(score)


def compute_all_scores(fv: FeatureVector, semantic_score: float) -> dict[str, float]:
    """
    Compute all sub-scores for a candidate.

    Args:
        fv: Extracted feature vector.
        semantic_score: Cosine similarity score from Stage 3 (0.0-1.0).

    Returns:
        Dictionary with all sub-scores (0-100 scale).
    """
    return {
        "semantic_score": _clamp(semantic_score * 100),
        "retrieval_score": compute_retrieval_score(fv),
        "ranking_score": compute_ranking_score(fv),
        "production_score": compute_production_score(fv),
        "behavioral_score": compute_behavioral_score(fv),
        "experience_score": compute_experience_score(fv),
        "education_score": compute_education_score(fv),
        "career_score": compute_career_score(fv),
    }
