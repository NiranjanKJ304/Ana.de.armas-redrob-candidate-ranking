# 🤖 Redrob Intelligent Candidate Ranking System

An offline, CPU-only pipeline that ranks **100,000+ candidates** against a job description, detects honeypot/fake profiles, and outputs the **top 100** best-fit candidates with evidence-based reasoning.

> **🚀 Quick Start (Running the Project):**
> 1. **Install requirements:**
>    ```bash
>    pip install -r requirements.txt
>    ```
> 2. **Download the model (Optional - pre-downloads the model to reduce pipeline execution time):**
>    ```bash
>    python load_model.py
>    ```
>
> - [ensure (#candidates.jsonl) file inside a project folder]
> 3. **Run the ranking pipeline:**
>    ```bash
>    python run.py
>    ```

### 🎬 Demo Videos

- [📹 Video 1 — Project Demo](https://drive.google.com/file/d/1b62BRxjQkIul1n3hUmM8h7DOc3grmGTz/view?usp=drive_link)
- [📹 Video 2 — Project Walkthrough](https://drive.google.com/file/d/1Ir3bCvoOgojRf-gcW0eXLLzeaZ2rBg0u/view?usp=drive_link)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Run the Pipeline](#run-the-pipeline)
- [Input Files](#input-files)
- [Output Format](#output-format)
- [Pipeline Stages](#pipeline-stages)
- [Technologies Used](#technologies-used)
- [Run Tests](#run-tests)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

| Feature | Detail |
|---|---|
| **Input** | `candidates.jsonl` (~100K profiles, ~487MB) + `data/job_description.txt` |
| **Output** | `output/submission.csv` (100 ranked candidates with scores & reasoning) |
| **Runtime** | ~4–5 minutes on CPU |
| **Model** | `all-MiniLM-L6-v2` (auto-downloaded via `load_model.py`) |
| **API Calls** | None — fully offline after setup |
| **GPU Required** | No — runs entirely on CPU |
| **RAM** | ≤16 GB |

---

## Features

- **Streaming JSONL Parser** — processes 100K candidates without loading all into memory at once
- **60+ Feature Extraction** — skills, career history, education, behavioral signals
- **7 Rule-Based Sub-Scores** — retrieval, ranking, production, behavioral, experience, education, career
- **7 Consistency Validators** — timeline anomalies, salary inversions, skill-experience mismatches
- **7 Honeypot Detectors** — buzzword stuffing, title inflation, fake profile patterns
- **Semantic Retrieval** — sentence-transformers embeddings + cosine similarity against job description
- **Dynamic Pre-Filtering** — top 2,000 candidates selected for expensive embedding (5% of dataset, capped)
- **Evidence-Based Reasoning** — factual explanations using actual candidate data (no LLM hallucinations)
- **Professional Stage Logging** — boxed terminal output with metrics per stage
- **Aggressive Memory Management** — `gc.collect()` at stage boundaries to stay within 16 GB

---

## System Architecture

```
candidates.jsonl ──► parser.py ──► features.py ──► scorer.py ──┐
                                                                ├──► ranker.py (pre-score)
                                      consistency.py ──────────┤
                                      honeypot.py ─────────────┘
                                                                │
                                                         Top 2,000 candidates
                                                                │
job_description.txt ──► retrieval.py ──► cosine similarity ────┤
                                                                │
                                             scorer.py (rescore with semantic) ──► ranker.py
                                                                │
                                                             Top 100
                                                                │
                                             reasoning.py ──► exporter.py ──► submission.csv
```

### End-to-End Workflow

```
python run.py
    │
    ├── Check directories (models/, logs/, output/, cache/)
    ├── Check model (models/all-MiniLM-L6-v2/model.safetensors)
    │     └── If missing: download ZIP → extract → verify → delete ZIP
    ├── Patch model path for offline loading
    └── Execute 10-stage pipeline
          ├── Stage 1:  Parse 100K candidates (streaming JSONL)
          ├── Stage 2:  Extract 60+ features per candidate
          ├── Stage 3:  Rule-based scoring (7 sub-scores)
          ├── Stage 4:  Consistency validation (7 anomaly checks)
          ├── Stage 5:  Honeypot detection (7 fraud detectors)
          ├── Stage 6:  Pre-filter top 2,000 by adjusted score
          ├── Stage 7:  Semantic embeddings + cosine similarity
          ├── Stage 8:  Final ranking (rescore with semantic, select top 100)
          ├── Stage 9:  Reasoning generation
          └── Stage 10: CSV export + validation
```

---

## Folder Structure

```
hack2skill/
├── run.py                     # 🎯 Main entry point — runs the full pipeline
├── load_model.py              # Model downloader (standalone or imported by run.py)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git exclusions
│
├── data/
│   └── job_description.txt    # Input: Job description (plain text)
├── candidates.jsonl           # Input: ~100K candidate profiles (~487MB)
│
├── src/
│   ├── __init__.py            # Package marker
│   ├── main.py                # Pipeline orchestrator (10 stages)
│   ├── parser.py              # Stage 1: Streaming JSONL parser + dataclasses
│   ├── features.py            # Stage 2: Feature extraction (~60 features)
│   ├── scorer.py              # Stage 3: Rule-based sub-score computation
│   ├── consistency.py         # Stage 4: Data consistency validation
│   ├── honeypot.py            # Stage 5: Fake profile detection
│   ├── retrieval.py           # Stage 7: Semantic embedding + similarity
│   ├── ranker.py              # Stage 6/8: Score weighting + final ranking
│   ├── reasoning.py           # Stage 9: Evidence-based reasoning generation
│   ├── exporter.py            # Stage 10: CSV export + validation
│   ├── keywords.py            # Centralized keyword dictionaries (25+ sets)
│   └── utils.py               # Shared utilities (normalization, logging, memory)
│
├── models/                    # Downloaded embedding model (auto-created)
│   └── all-MiniLM-L6-v2/     # SentenceTransformer model files
├── output/                    # Pipeline output (auto-created)
│   └── submission.csv         # Top 100 ranked candidates
├── logs/                      # Execution logs (auto-created)
│   └── pipeline.log           # Full pipeline log
├── cache/                     # Reserved for future caching (auto-created)
│
├── tests/                     # Unit tests (40 tests)
│   ├── test_parser.py         # 11 tests — parsing, streaming, error handling
│   ├── test_features.py       # 8 tests — feature flags, counts, text builder
│   ├── test_scorer.py         # 7 tests — score ranges, all-scores dict, scaling
│   ├── test_consistency.py    # 4 tests — anomaly detection, penalty ranges
│   └── test_honeypot.py       # 6 tests — genuine vs honeypot separation
│
├── README.md                  # This file
├── StudyGuide.md              # Detailed architecture & code walkthrough
├── audit_report.md            # Production audit & quality assessment
└── implementation_plan.md     # Implementation plan & status
```

---

## Prerequisites

- **Python 3.10+** (required for `X | Y` type hint syntax)
- **pip** (Python package manager)
- **~2 GB disk space** (for dependencies + ML model)
- **≤16 GB RAM** (for processing 100K candidates)
- **Internet** (one-time only, for model download during setup)

---

## Installation & Setup

### Step 1: Clone or navigate to the project

```bash
cd path/to/hack2skill
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the embedding model

```bash
python load_model.py
```

This downloads `all-MiniLM-L6-v2` from HuggingFace, extracts it to `models/`, and verifies the installation. If the model already exists, it skips automatically.

### Step 4: Verify input files exist

```
hack2skill/
├── candidates.jsonl           ← Required (~487MB)
└── data/
    └── job_description.txt    ← Required (~2KB)
```

---

## Run the Pipeline

```bash
python run.py
```

This is the **only command needed** after setup. It will:
1. Check directories and create any missing ones
2. Verify the model exists (or auto-download it)
3. Validate input files
4. Execute all 10 pipeline stages
5. Output `output/submission.csv`

### Running with custom file paths:

By default, the script expects `candidates.jsonl` in the project root or in the `data/` directory. However, you can reference the file from **any directory on your computer** using command-line arguments:

```bash
python run.py --candidates C:\path\to\your\candidates.jsonl --jd data\job_description.txt --output output\submission.csv --top-k 100
```

#### CLI Options:
* `--candidates`: Path to the `candidates.jsonl` file (can be anywhere on your computer).
* `--jd`: Path to the job description file (defaults to `data/job_description.txt`).
* `--output`: Path to save the final CSV output (defaults to `output/submission.csv`).
* `--top-k`: Number of top candidates to output in the CSV (default: `100`).

### Console Output Example

```
Starting Candidate Ranking System...

Checking directories...                ✓
Checking embedding model...            ✓

Loading embedding model...             ✓

Starting pipeline execution...

==================================================
Stage 1  : Parsing Candidates
==================================================
Candidates Processed          : 100000
Time Taken                    : 17.9 sec
==================================================

...

==================================================
Pipeline Summary
==================================================
Total Candidates              : 100000
Total Runtime                 : 286.5 sec (4.8 min)
==================================================

Pipeline completed successfully.
Processed Candidates: 100000
Time Taken:           286.5 seconds
Output File:          output\submission.csv
```

---

## Input Files

### `candidates.jsonl`

One JSON object per line, each representing a candidate profile with:
- **Profile**: name, headline, summary, location, experience, title
- **Career History**: companies, titles, durations, descriptions
- **Education**: institutions, degrees, fields, tiers
- **Skills**: names, proficiency levels, endorsements
- **Redrob Signals**: platform engagement, recruiter response rate, GitHub activity

### `data/job_description.txt`

Plain text job description used for semantic similarity matching.

---

## Output Format

The pipeline generates `output/submission.csv`:

```csv
"candidate_id","rank","score","reasoning"
"CAND_0079387","1","1.0","AI Engineer with 6.9 years of experience..."
"CAND_0098846","2","0.97","AI Engineer with 7.6 years..."
...
"CAND_0054123","100","0.01","ML Engineer with 5.2 years..."
```

| Column | Type | Range | Description |
|---|---|---|---|
| `candidate_id` | string | — | Unique candidate identifier |
| `rank` | integer | 1–100 | 1 = best fit |
| `score` | float | 0.01–1.0 | Normalized relevance score |
| `reasoning` | string | ≤400 chars | Factual, evidence-based justification |

---

## Pipeline Stages

| Stage | Module | Description | Time |
|---|---|---|---|
| 1 | `parser.py` | Stream JSONL, convert to typed dataclasses | ~18s |
| 2 | `features.py` | Extract 60+ features (skills, career, education, behavioral) | ~70s |
| 3 | `scorer.py` | Compute 7 rule-based sub-scores (without semantic) | ~1s |
| 4 | `consistency.py` | Run 7 anomaly detectors (timelines, salary, skills) | ~1s |
| 5 | `honeypot.py` | Run 7 fraud detectors (buzzwords, title mismatch, stuffing) | ~29s |
| 6 | `ranker.py` | Pre-filter top 2,000 by adjusted score | <1s |
| 7 | `retrieval.py` | Encode JD + 2,000 candidates → cosine similarity | ~140s |
| 8 | `ranker.py` | Rescore with semantic, select top 100 | ~3s |
| 9 | `reasoning.py` | Generate evidence-based explanations | <1s |
| 10 | `exporter.py` | Normalize scores, write CSV, validate | <1s |
| **Total** | | | **~4.5 min** |

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| sentence-transformers | Semantic text embeddings |
| PyTorch | Neural network backend (CPU) |
| NumPy | Array operations, cosine similarity |
| orjson | Fast JSONL parsing |
| tqdm | Progress bars |
| requests | Model download (setup only) |
| pytest | Unit testing framework |

---

## Run Tests

```bash
python -m pytest tests/ -v
```

### Test Coverage (40 tests)

| Module | Tests |
|---|---|
| `parser.py` | 11 tests — parsing, streaming, error handling |
| `features.py` | 8 tests — feature flags, counts, text builder |
| `scorer.py` | 7 tests — score ranges, all-scores dict, scaling |
| `consistency.py` | 4 tests — anomaly detection, penalty ranges |
| `honeypot.py` | 6 tests — genuine vs honeypot separation |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `UnicodeEncodeError` on Windows | `run.py` auto-sets UTF-8 encoding. If using `src/main.py` directly, set `PYTHONIOENCODING=utf-8` |
| Model download fails | Check internet connection, then retry `python load_model.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: candidates.jsonl` | Ensure `candidates.jsonl` is in the project root or `data/` directory |
| Out of memory | The pipeline uses ≤16 GB RAM. Close other applications if needed |
| Runtime > 5 minutes | Expected on slower CPUs. The dynamic pre-filter caps embedding at 2,000 candidates |

---

## Future Improvements

- **GPU acceleration** — Use CUDA for embedding to reduce Stage 7 from ~140s to ~10s
- **Model caching** — Cache embeddings to disk for repeated runs with the same candidates
- **Configurable weights** — Expose scoring weights as a YAML/JSON config file
- **Batch processing** — Process candidates in chunks for datasets larger than 100K
- **Additional models** — Support larger embedding models (e.g., BGE-base, E5-large) for improved accuracy
- **API wrapper** — Optional REST API for integration with HR platforms

---

## License

This project was built for the Redrob Hack2Skill Challenge.
