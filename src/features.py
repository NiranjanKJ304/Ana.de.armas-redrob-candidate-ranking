"""
Stage 2: Feature Extraction

Extracts all feature dimensions from a CandidateRecord:
- Profile features (experience, title, industry)
- Career/domain features (search, ranking, retrieval, production ML, etc.)
- Behavioral features (from redrob_signals)
- Education features (degree, tier, specialization)
- Career stability (tenure, company count)

Accepts pre-normalized text from ``utils.normalize_text()`` to avoid
redundant string allocations and repeated ``.lower()`` calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .keywords import (
    AI_ML_KEYWORDS,
    BUZZWORD_KEYWORDS,
    CONSULTING_COMPANIES,
    CV_SPEECH_ROBOTICS_KEYWORDS,
    DEGREE_RANKS,
    EMBEDDING_KEYWORDS,
    LLM_FINETUNING_KEYWORDS,
    ML_AI_FIELDS,
    NLP_IR_KEYWORDS,
    NON_TECH_TITLES,
    PRODUCTION_ML_KEYWORDS,
    PYTHON_KEYWORDS,
    RANKING_KEYWORDS,
    RECOMMENDATION_KEYWORDS,
    RELEVANT_TITLES,
    RETRIEVAL_KEYWORDS,
    SEARCH_KEYWORDS,
    TIER_RANKS,
)
from .parser import CandidateRecord
from .utils import (
    NormalizedText,
    check_keywords_in_skills,
    check_keywords_in_text,
    get_degree_rank,
    get_tier_rank,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """All extracted features for a single candidate."""
    candidate_id: str

    # Profile features
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_industry: str = ""
    location: str = ""
    country: str = ""

    # Career/domain boolean flags
    has_search_experience: bool = False
    has_ranking_experience: bool = False
    has_retrieval_experience: bool = False
    has_recommendation_systems: bool = False
    has_production_ml: bool = False
    has_embeddings: bool = False
    has_vector_databases: bool = False
    has_evaluation_frameworks: bool = False
    has_ab_testing: bool = False
    has_learning_to_rank: bool = False
    has_llm_finetuning: bool = False
    has_nlp_ir: bool = False
    has_python: bool = False
    has_cv_speech_robotics: bool = False

    # Skill counts
    relevant_skill_count: int = 0
    total_skill_count: int = 0
    total_skill_endorsements: int = 0
    retrieval_skill_count: int = 0
    ranking_skill_count: int = 0
    production_skill_count: int = 0
    ai_ml_skill_count: int = 0
    buzzword_skill_count: int = 0

    # Behavioral features
    open_to_work: bool = False
    recruiter_response_rate: float = 0.0
    github_activity_score: float = -1.0
    profile_views_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    notice_period_days: int = 0
    profile_completeness_score: float = 0.0
    avg_response_time_hours: float = 0.0
    willing_to_relocate: bool = False
    search_appearance_30d: int = 0
    connection_count: int = 0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False

    # Education features
    highest_degree: str = ""
    best_tier: str = "tier_4"
    has_ml_ai_specialization: bool = False
    education_count: int = 0

    # Career stability
    avg_tenure_months: float = 0.0
    num_companies: int = 0
    is_title_chaser: bool = False
    total_career_months: int = 0
    is_consulting_only: bool = False
    is_non_tech_title: bool = False
    is_relevant_title: bool = False

    # Career text signals (from descriptions)
    career_mentions_search: int = 0
    career_mentions_ranking: int = 0
    career_mentions_retrieval: int = 0
    career_mentions_production: int = 0
    career_mentions_ml: int = 0
    career_mentions_embeddings: int = 0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    candidate: CandidateRecord,
    normalized: NormalizedText,
) -> FeatureVector:
    """Extract all features from a CandidateRecord using pre-normalized text.

    Args:
        candidate: Parsed candidate record.
        normalized: Pre-normalized text (from ``normalize_text``).

    Returns:
        FeatureVector with all extracted features.
    """
    fv = FeatureVector(candidate_id=candidate.candidate_id)
    full_text = normalized.full_text

    # ---- Profile ----
    fv.years_of_experience = candidate.years_of_experience
    fv.current_title = candidate.current_title
    fv.current_company = candidate.current_company
    fv.current_industry = candidate.current_industry
    fv.location = candidate.location
    fv.country = candidate.country

    # ---- Career/domain flags (using pre-lowered full_text) ----
    fv.has_search_experience = check_keywords_in_text(full_text, SEARCH_KEYWORDS) > 0
    fv.has_ranking_experience = check_keywords_in_text(full_text, RANKING_KEYWORDS) > 0
    fv.has_retrieval_experience = check_keywords_in_text(full_text, RETRIEVAL_KEYWORDS) > 0
    fv.has_recommendation_systems = check_keywords_in_text(full_text, RECOMMENDATION_KEYWORDS) > 0
    fv.has_production_ml = check_keywords_in_text(full_text, PRODUCTION_ML_KEYWORDS) > 0
    fv.has_embeddings = any(
        kw in full_text for kw in ("embedding", "embeddings", "sentence-transformers")
    )
    fv.has_vector_databases = any(
        kw in full_text
        for kw in ("faiss", "pinecone", "qdrant", "milvus", "weaviate", "opensearch")
    )
    fv.has_evaluation_frameworks = any(
        kw in full_text
        for kw in ("ndcg", "mrr", "map", "evaluation framework", "a/b testing")
    )
    fv.has_ab_testing = "a/b test" in full_text or "ab test" in full_text
    fv.has_learning_to_rank = any(
        kw in full_text
        for kw in ("learning to rank", "learning-to-rank", "ltr", "lambdamart")
    )
    fv.has_llm_finetuning = check_keywords_in_text(full_text, LLM_FINETUNING_KEYWORDS) > 0
    fv.has_nlp_ir = check_keywords_in_text(full_text, NLP_IR_KEYWORDS) > 0
    fv.has_python = check_keywords_in_text(full_text, PYTHON_KEYWORDS) > 0
    fv.has_cv_speech_robotics = check_keywords_in_text(full_text, CV_SPEECH_ROBOTICS_KEYWORDS) > 0

    # ---- Skill counts (using pre-lowered skill names) ----
    skill_names = normalized.skill_names
    fv.total_skill_count = len(candidate.skills)
    fv.total_skill_endorsements = sum(s.endorsements for s in candidate.skills)
    fv.retrieval_skill_count = check_keywords_in_skills(skill_names, RETRIEVAL_KEYWORDS)
    fv.ranking_skill_count = check_keywords_in_skills(skill_names, RANKING_KEYWORDS)
    fv.production_skill_count = check_keywords_in_skills(skill_names, PRODUCTION_ML_KEYWORDS)
    fv.ai_ml_skill_count = check_keywords_in_skills(skill_names, AI_ML_KEYWORDS)
    fv.buzzword_skill_count = check_keywords_in_skills(skill_names, BUZZWORD_KEYWORDS)
    fv.relevant_skill_count = (
        fv.retrieval_skill_count + fv.ranking_skill_count
        + fv.production_skill_count + fv.ai_ml_skill_count
    )

    # ---- Behavioral features ----
    sig = candidate.signals
    fv.open_to_work = sig.open_to_work_flag
    fv.recruiter_response_rate = sig.recruiter_response_rate
    fv.github_activity_score = sig.github_activity_score
    fv.profile_views_30d = sig.profile_views_received_30d
    fv.saved_by_recruiters_30d = sig.saved_by_recruiters_30d
    fv.interview_completion_rate = sig.interview_completion_rate
    fv.notice_period_days = sig.notice_period_days
    fv.profile_completeness_score = sig.profile_completeness_score
    fv.avg_response_time_hours = sig.avg_response_time_hours
    fv.willing_to_relocate = sig.willing_to_relocate
    fv.search_appearance_30d = sig.search_appearance_30d
    fv.connection_count = sig.connection_count
    fv.verified_email = sig.verified_email
    fv.verified_phone = sig.verified_phone
    fv.linkedin_connected = sig.linkedin_connected

    # ---- Education ----
    best_degree_rank = 0
    best_tier_rank = 0
    fv.education_count = len(candidate.education)

    for edu in candidate.education:
        d_rank = get_degree_rank(edu.degree, DEGREE_RANKS)
        if d_rank > best_degree_rank:
            best_degree_rank = d_rank
            fv.highest_degree = edu.degree

        t_rank = get_tier_rank(edu.tier, TIER_RANKS)
        if t_rank > best_tier_rank:
            best_tier_rank = t_rank
            fv.best_tier = edu.tier

        # Check ML/AI specialization
        if any(kw in edu.field_of_study.lower() for kw in ML_AI_FIELDS):
            fv.has_ml_ai_specialization = True

    # ---- Career stability ----
    fv.num_companies = len(candidate.career_history)
    fv.total_career_months = sum(
        entry.duration_months for entry in candidate.career_history
    )

    if fv.num_companies > 0:
        fv.avg_tenure_months = fv.total_career_months / fv.num_companies
    else:
        fv.avg_tenure_months = 0.0

    fv.is_title_chaser = (
        fv.avg_tenure_months < 18 and fv.num_companies >= 4
    )

    # Check consulting-only
    if fv.num_companies > 0:
        consulting_count = sum(
            1 for entry in candidate.career_history
            if entry.company.lower().strip() in CONSULTING_COMPANIES
        )
        fv.is_consulting_only = consulting_count == fv.num_companies

    # Check non-tech vs relevant title
    title_lower = candidate.current_title.lower().strip()
    fv.is_non_tech_title = title_lower in NON_TECH_TITLES or any(
        t in title_lower for t in NON_TECH_TITLES
    )
    fv.is_relevant_title = title_lower in RELEVANT_TITLES or any(
        t in title_lower for t in RELEVANT_TITLES
    )

    # ---- Career text signals (using pre-lowered career_text) ----
    career_text = normalized.career_text
    fv.career_mentions_search = check_keywords_in_text(career_text, SEARCH_KEYWORDS)
    fv.career_mentions_ranking = check_keywords_in_text(career_text, RANKING_KEYWORDS)
    fv.career_mentions_retrieval = check_keywords_in_text(career_text, RETRIEVAL_KEYWORDS)
    fv.career_mentions_production = check_keywords_in_text(career_text, PRODUCTION_ML_KEYWORDS)
    fv.career_mentions_ml = check_keywords_in_text(career_text, AI_ML_KEYWORDS)
    fv.career_mentions_embeddings = check_keywords_in_text(career_text, EMBEDDING_KEYWORDS)

    return fv


def build_candidate_text(candidate: CandidateRecord) -> str:
    """Build composite text for semantic embedding.

    Combines headline, summary, career descriptions, and skill names.
    Truncated to ~2000 chars (model handles tokenization).

    Args:
        candidate: Parsed candidate record.

    Returns:
        Composite text string for embedding.
    """
    parts: list[str] = []

    if candidate.headline:
        parts.append(candidate.headline)

    if candidate.summary:
        parts.append(candidate.summary)

    for entry in candidate.career_history:
        if entry.description:
            parts.append(entry.description)

    skill_names = ", ".join(s.name for s in candidate.skills if s.name)
    if skill_names:
        parts.append(f"Skills: {skill_names}")

    text = " ".join(parts)
    # Truncate to ~2000 chars (model handles tokenization)
    return text[:2000]
