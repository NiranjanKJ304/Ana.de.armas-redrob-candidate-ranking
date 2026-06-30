"""
Stage 8: Reasoning Generation

Generates factual, evidence-based reasoning strings for ranked candidates.
Uses actual candidate data — no hallucinations, no generic statements.

Accepts pre-normalized text to avoid re-scanning career descriptions.
"""

from __future__ import annotations

import logging

from .keywords import (
    CAREER_HIGHLIGHT_KEYWORDS,
    HIGHLIGHT_SKILLS,
)
from .parser import CandidateRecord
from .features import FeatureVector
from .ranker import RankedCandidate
from .utils import NormalizedText

logger = logging.getLogger(__name__)


def _get_matching_skills(candidate: CandidateRecord) -> dict[str, list[str]]:
    """Get candidate skills grouped by JD-relevant categories.

    Args:
        candidate: Parsed candidate record.

    Returns:
        Dict mapping category name to list of matching skill names.
    """
    result: dict[str, list[str]] = {cat: [] for cat in HIGHLIGHT_SKILLS}

    for skill in candidate.skills:
        skill_lower = skill.name.lower()
        for category, keywords in HIGHLIGHT_SKILLS.items():
            if skill_lower in keywords or any(kw in skill_lower for kw in keywords):
                result[category].append(skill.name)
                break  # Only categorize each skill once

    return result


def _extract_career_highlights(career_text: str) -> list[str]:
    """Extract notable achievements/highlights from pre-lowered career text.

    Args:
        career_text: Pre-lowered career description text.

    Returns:
        Sorted list of up to 5 matching highlight keywords.
    """
    highlights = set()
    for keyword in CAREER_HIGHLIGHT_KEYWORDS:
        if keyword in career_text:
            highlights.add(keyword)
    return sorted(highlights)[:5]


def generate_reasoning(
    candidate: CandidateRecord,
    features: FeatureVector,
    ranked: RankedCandidate,
    normalized: NormalizedText,
) -> str:
    """Generate a factual, evidence-based reasoning string for a ranked candidate.

    Uses actual candidate data only:
    - Title + years of experience
    - Top matching skills
    - Career highlights
    - Behavioral signals
    - Location relevance

    Args:
        candidate: Parsed candidate record.
        features: Extracted feature vector.
        ranked: Ranking result for this candidate.
        normalized: Pre-normalized text fields.

    Returns:
        Reasoning string (max 400 chars, CSV-friendly).
    """
    parts: list[str] = []

    # 1. Title + experience + location
    title = features.current_title or "Professional"
    yoe = features.years_of_experience
    location_str = f" based in {features.location}" if features.location else ""
    parts.append(f"{title} with {yoe:.1f} years of experience{location_str}.")

    # 2. Production/relevant skills
    matching_skills = _get_matching_skills(candidate)
    all_relevant: list[str] = []
    for category in ["retrieval", "ranking", "production", "llm", "core_ml"]:
        all_relevant.extend(matching_skills[category])

    if all_relevant:
        top_skills = all_relevant[:4]
        skills_str = ", ".join(top_skills)
        parts.append(f"Demonstrated technical expertise in {skills_str}.")

    # 3. Career highlights (using pre-normalized career_text)
    highlights = _extract_career_highlights(normalized.career_text)
    if highlights:
        highlights_str = ", ".join(highlights[:3])
        parts.append(f"Career evidence includes specific work on {highlights_str}.")

    # 4. Behavioral signals
    behavioral_parts: list[str] = []
    rr = features.recruiter_response_rate
    if rr > 0:
        pct = int(rr * 100)
        if pct >= 60:
            behavioral_parts.append(f"{pct}% recruiter response rate")

    github = features.github_activity_score
    if github > 0:
        behavioral_parts.append(f"active GitHub presence (score {int(github)})")

    if behavioral_parts:
        parts.append(f"Platform signals show {' and '.join(behavioral_parts)}.")

    # Assemble
    reasoning = " ".join(parts)

    # Ensure it's not too long (CSV-friendly)
    if len(reasoning) > 400:
        reasoning = reasoning[:397] + "..."

    return reasoning
