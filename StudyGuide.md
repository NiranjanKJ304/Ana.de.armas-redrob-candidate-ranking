# 📚 Study Guide — Redrob Intelligent Candidate Ranking System

A comprehensive walkthrough of the architecture, pipeline stages, scoring logic, and module responsibilities.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Execution Flow](#execution-flow)
- [Module-by-Module Breakdown](#module-by-module-breakdown)
- [Pipeline Stage Details](#pipeline-stage-details)
- [Feature Engineering Workflow](#feature-engineering-workflow)
- [Scoring Workflow](#scoring-workflow)
- [Semantic Retrieval Workflow](#semantic-retrieval-workflow)
- [Ranking Workflow](#ranking-workflow)
- [Honeypot Detection Workflow](#honeypot-detection-workflow)
- [File Responsibilities](#file-responsibilities)

---

## Architecture Overview

The system processes ~100K candidate profiles against a single job description using a **10-stage pipeline** that runs entirely offline on CPU. No APIs, no LLMs, no cloud services.

```
┌──────────────────────────────────────────────────────────────┐
│                      run.py (Entry Point)                    │
│  Directory setup → Model check → Model patch → Pipeline     │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                    src/main.py (Orchestrator)                 │
│                                                              │
│  Stage 1: parser.py        → CandidateRecord (frozen DC)     │
│  Stage 2: features.py      → FeatureVector (60+ features)    │
│  Stage 3: scorer.py        → 8 sub-scores (0-100 each)       │
│  Stage 4: consistency.py   → consistency_score (0.0-1.0)     │
│  Stage 5: honeypot.py      → honeypot_score (0.0-1.0)       │
│  Stage 6: ranker.py        → pre-filter top 2,000            │
│  Stage 7: retrieval.py     → semantic similarity (0.0-1.0)   │
│  Stage 8: ranker.py        → final top 100 ranking           │
│  Stage 9: reasoning.py     → evidence-based text per rank    │
│  Stage 10: exporter.py     → output/submission.csv           │
│                                                              │
│  Shared: keywords.py, utils.py                               │
└──────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Single-pass normalization** — `normalize_text()` runs once per candidate, producing a `NormalizedText` object reused by stages 2, 5, and 9.
2. **Centralized keywords** — All keyword sets live in `keywords.py`; no duplications.
3. **Aggressive memory management** — `del` + `gc.collect()` at every stage boundary.
4. **Dynamic pre-filtering** — Only top 2,000 candidates get the expensive embedding step.
5. **Deterministic output** — No randomness; identical input always produces identical output.

---

## Execution Flow

```
python run.py
    │
    ├── 1. setup_directories()     → Create models/, logs/, output/, cache/
    ├── 2. validate_inputs()       → Check candidates.jsonl + job_description.txt
    ├── 3. check_model()           → Check models/all-MiniLM-L6-v2/model.safetensors
    │       └── If missing: download_model() → extract_model() → verify_model()
    ├── 4. Patch src.retrieval._model_name → "models/all-MiniLM-L6-v2"
    ├── 5. Set HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
    └── 6. run_pipeline()          → Execute 10 stages via src/main.py
```

### Data Flow Through Pipeline

```
candidates.jsonl (100K lines)
    │
    ▼ Stage 1: stream_candidates()
CandidateRecord[] (100K frozen dataclasses)
    │
    ├──▼ Stage 2: normalize_text() → extract_features()
    │   NormalizedText[] + FeatureVector[] (100K each)
    │
    ├──▼ Stage 3: compute_all_scores(fv, semantic=0.0)
    │   sub_scores[] (100K dicts, 8 keys each)
    │
    ├──▼ Stage 4: compute_consistency_score()
    │   consistency_scores[] (100K floats)
    │
    ├──▼ Stage 5: compute_honeypot_score()
    │   honeypot_scores[] (100K floats)
    │
    ├──▼ Stage 6: pre-filter via adjusted_score
    │   top_indices (2,000 indices)
    │
    ├──▼ Stage 7: encode_texts() → compute_similarities()
    │   semantic_scores_subset (2,000 floats)
    │
    ├──▼ Stage 8: compute_all_scores(fv, semantic) → rank_candidates()
    │   RankedCandidate[] (top 100)
    │
    ├──▼ Stage 9: generate_reasoning()
    │   reasoning_list (100 strings)
    │
    └──▼ Stage 10: export_csv()
        output/submission.csv
```

---

## Module-by-Module Breakdown

### `run.py` — Entry Point

The **only command** a user runs. Handles:
- Directory creation
- Input validation with user-friendly error messages
- Model existence check (auto-downloads if missing via streaming HTTP + tqdm)
- Runtime monkey-patching of `src.retrieval._model_name` to load from local `models/` directory
- Offline mode enforcement (`HF_HUB_OFFLINE=1`)
- Console + file logging setup
- Pipeline execution delegation to `src/main.py`

### `load_model.py` — Model Downloader

Standalone script and importable module. Exposes `ensure_model() -> bool`:
1. Check `models/all-MiniLM-L6-v2/model.safetensors`
2. If missing: stream-download ZIP → extract to `models/` → verify → delete ZIP
3. Returns `True` if model is ready

### `src/main.py` — Pipeline Orchestrator

Contains `run_pipeline()` which executes all 10 stages sequentially. Each stage:
- Wrapped in a `StageLogger` context manager for professional terminal output
- Followed by `release_memory()` for garbage collection
- Reports metrics (candidates processed, scores, timings)

Also provides a CLI via `main()` with `argparse` for direct invocation.

### `src/parser.py` — JSONL Parser

Streams `candidates.jsonl` line-by-line using `orjson` for fast parsing. Each line is converted to a `CandidateRecord` (frozen dataclass). Nested structures:
- `CareerEntry` — company, title, duration, description
- `EducationEntry` — institution, degree, field, tier
- `SkillEntry` — name, proficiency, endorsements
- `RedrobSignals` — 23 behavioral platform signals
- `SalaryRange` — min/max in INR LPA

### `src/features.py` — Feature Extraction

Extracts 60+ features into a `FeatureVector` dataclass:
- **Domain flags**: has_retrieval, has_ranking, has_production, has_search, etc.
- **Skill counts**: retrieval_skill_count, ranking_skill_count, etc.
- **Career signals**: career_text_retrieval, career_text_ranking
- **Behavioral**: profile_completeness, recruiter_response_rate, github_activity
- **Education**: highest_degree_rank, highest_tier_rank, has_ml_specialization
- **Career stability**: avg_tenure_months, total_companies, career_progression

### `src/keywords.py` — Centralized Keywords

Single source of truth for 25+ keyword sets. All `frozenset[str]` for O(1) lookups:
- Domain: `RETRIEVAL_KEYWORDS`, `RANKING_KEYWORDS`, `PRODUCTION_ML_KEYWORDS`, etc.
- Education: `DEGREE_RANKS`, `TIER_RANKS`, `ML_AI_FIELDS`
- Honeypot: `BUZZWORD_KEYWORDS`, `NON_TECH_TITLES`
- Reasoning: `HIGHLIGHT_SKILLS`, `CAREER_HIGHLIGHT_KEYWORDS`

### `src/utils.py` — Shared Utilities

- **`NormalizedText`** — frozen dataclass with pre-lowered text fields
- **`normalize_text()`** — builds NormalizedText from CandidateRecord (called once per candidate)
- **`check_keywords_in_text()`** / **`check_keywords_in_skills()`** — keyword matching
- **`get_degree_rank()`** / **`get_tier_rank()`** — education ranking helpers
- **`StageLogger`** — context manager for boxed terminal output
- **`PipelineSummary`** — collects and displays pipeline-wide statistics
- **`dynamic_prefilter_k()`** — computes pre-filter K: `min(max(5% of N, 500), 2000)`

### `src/scorer.py` — Rule-Based Scoring

Computes 8 sub-scores (each 0-100):
1. **semantic_score** — cosine similarity × 100
2. **retrieval_score** — keyword and skill matching for IR domain
3. **ranking_score** — LTR/search ranking expertise
4. **production_score** — MLOps/deployment experience
5. **behavioral_score** — platform engagement signals
6. **experience_score** — years of experience fit (ideal: 5-9 years)
7. **education_score** — degree level + institution tier + ML specialization
8. **career_score** — stability, progression, industry relevance

### `src/consistency.py` — Consistency Validation

Returns a score from 0.0 (heavily anomalous) to 1.0 (fully consistent). Checks:
1. Experience vs career history duration mismatch
2. Salary range inversion (min > max)
3. Degree timeline anomalies (PhD before Bachelor's)
4. Skill endorsement vs experience mismatch
5. Title inflation (senior title with < 3 years experience)
6. Job hopping detection (many short stints)
7. Profile completeness cross-validation

### `src/honeypot.py` — Honeypot Detection

Returns a score from 0.0 (genuine) to 1.0 (likely fake). Detectors:
1. Buzzword density analysis
2. Title-skill mismatch
3. Career description keyword stuffing
4. Engagement signal anomalies
5. Education credential inflation
6. Skill endorsement inflation
7. Generic/template profile detection

### `src/retrieval.py` — Semantic Retrieval

Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings):
1. Lazy-load model via `_get_model()`
2. Encode job description → 384-dim vector
3. Encode 2,000 pre-filtered candidate texts → (2000, 384) matrix
4. Cosine similarity = dot product (embeddings are L2-normalized)
5. Clip to [0, 1]

### `src/ranker.py` — Score Weighting & Ranking

Weighted composite scoring:
```
base_score = Σ (weight_i × sub_score_i)

Weights:
  semantic: 0.15, retrieval: 0.22, ranking: 0.18, production: 0.13
  behavioral: 0.05, experience: 0.08, education: 0.07, career: 0.12

adjusted_score = base_score × consistency_score × (1 - honeypot_score)
```

### `src/reasoning.py` — Reasoning Generation

Generates factual, CSV-friendly reasoning strings (≤400 chars) using:
- Title + years of experience + location
- Matching JD-relevant skills (categorized)
- Career highlights from description keywords
- Platform behavioral signals (recruiter response rate, GitHub)

### `src/exporter.py` — CSV Export

Writes `output/submission.csv` with:
- Min-max score normalization to [0.01, 1.0]
- Full validation: header check, sequential ranks, score ranges, monotonicity

---

## Feature Engineering Workflow

```
CandidateRecord + NormalizedText
    │
    ├── Profile features
    │   ├── years_of_experience
    │   ├── current_title, location, industry
    │   └── has_relevant_title (matches RELEVANT_TITLES keywords)
    │
    ├── Domain keyword matching (against full_text)
    │   ├── has_retrieval, retrieval_skill_count
    │   ├── has_ranking, ranking_skill_count
    │   ├── has_production, production_skill_count
    │   ├── has_search, has_recommendation
    │   ├── has_nlp, has_python, has_llm
    │   └── has_cv_speech_robotics, has_embedding
    │
    ├── Career text signals
    │   ├── career_text_retrieval (keywords in career descriptions)
    │   ├── career_text_ranking
    │   └── career_text_production
    │
    ├── Behavioral features (from RedrobSignals)
    │   ├── profile_completeness, open_to_work
    │   ├── recruiter_response_rate, avg_response_time
    │   ├── github_activity_score, connection_count
    │   └── verified_email, verified_phone, linkedin_connected
    │
    ├── Education features
    │   ├── highest_degree_rank (PhD=5, Masters=4, ...)
    │   ├── highest_tier_rank (tier_1=4, tier_2=3, ...)
    │   └── has_ml_specialization
    │
    └── Career stability features
        ├── avg_tenure_months
        ├── total_companies
        └── career_progression_score
```

---

## Scoring Workflow

```
FeatureVector
    │
    ├── compute_retrieval_score()    → 0-100 (keyword + skill matching)
    ├── compute_ranking_score()      → 0-100 (LTR/search expertise)
    ├── compute_production_score()   → 0-100 (MLOps experience)
    ├── compute_behavioral_score()   → 0-100 (platform engagement)
    ├── compute_experience_score()   → 0-100 (years-of-experience fit)
    ├── compute_education_score()    → 0-100 (degree + tier + specialization)
    ├── compute_career_score()       → 0-100 (stability + progression)
    └── semantic_score               → cosine_similarity × 100
    │
    ▼
    compute_base_score() = weighted sum (weights sum to 1.0)
    │
    ▼
    compute_adjusted_score() = base × consistency × (1 - honeypot)
```

---

## Semantic Retrieval Workflow

```
Job Description Text
    │
    ▼ encode_single()
    JD Embedding (384-dim, L2-normalized)
    │
    │   Top 2,000 Candidate Texts (from pre-filter)
    │       │
    │       ▼ encode_texts(batch_size=256)
    │       Candidate Embeddings (2000 × 384, L2-normalized)
    │
    ▼
    cosine_similarity = dot_product(JD, Candidates)  ← O(2000 × 384)
    │
    ▼
    clip to [0.0, 1.0]
    │
    ▼
    semantic_scores (2000 floats)
```

---

## Ranking Workflow

### Pre-Filter (Stage 6)
```
100K candidates
    │
    ▼ compute_all_scores(fv, semantic=0.0)  ← no semantic yet
    │ compute_adjusted_score()
    │
    ▼ np.argsort()[::-1][:2000]
    │
    Top 2,000 by rule-based adjusted score
```

### Final Ranking (Stage 8)
```
2,000 pre-filtered candidates + semantic_scores
    │
    ▼ compute_all_scores(fv, semantic)  ← WITH semantic now
    │ compute_adjusted_score()
    │
    ▼ rank_candidates(top_k=100)
    │
    Top 100 RankedCandidate objects
```

---

## Honeypot Detection Workflow

```
CandidateRecord + FeatureVector + NormalizedText
    │
    ├── Buzzword density    → ratio of buzzwords in full_text
    ├── Title mismatch      → senior title but no matching skills
    ├── Keyword stuffing    → repeated keywords in career descriptions
    ├── Engagement anomaly  → very high engagement but no substance
    ├── Education inflation → advanced degree claims vs career evidence
    ├── Endorsement anomaly → very high endorsements with low experience
    └── Template detection  → generic/copy-paste profile patterns
    │
    ▼
    honeypot_score = weighted combination of 7 detectors
    Clamped to [0.0, 1.0]
    │
    ▼
    adjusted_score = base_score × consistency × (1 - honeypot_score)
    (High honeypot → score reduced toward 0)
```

---

## File Responsibilities

| File | Lines | Responsibility |
|---|---|---|
| `run.py` | ~220 | Entry point, setup, model management, pipeline execution |
| `load_model.py` | ~120 | Standalone model download with `ensure_model()` |
| `src/main.py` | ~410 | 10-stage pipeline orchestrator with `StageLogger` |
| `src/parser.py` | ~320 | Streaming JSONL parser, 7 frozen dataclasses |
| `src/features.py` | ~320 | 60+ feature extraction into `FeatureVector` |
| `src/keywords.py` | ~250 | 25+ centralized keyword sets |
| `src/scorer.py` | ~200 | 8 rule-based sub-score functions |
| `src/consistency.py` | ~200 | 7 consistency anomaly detectors |
| `src/honeypot.py` | ~490 | 7 honeypot fraud detectors |
| `src/retrieval.py` | ~110 | SentenceTransformer embedding + similarity |
| `src/ranker.py` | ~160 | Weighted scoring + top-K selection |
| `src/reasoning.py` | ~135 | Evidence-based reasoning generation |
| `src/exporter.py` | ~135 | CSV export + validation |
| `src/utils.py` | ~290 | Normalization, logging, memory, pre-filter |
