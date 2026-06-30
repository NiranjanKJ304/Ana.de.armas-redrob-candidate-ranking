"""
Centralized Keyword Dictionaries

Single source of truth for all keyword sets used across the pipeline:
- Feature extraction
- Honeypot detection
- Reasoning generation
- Consistency validation

Import from here instead of duplicating keyword lists in each module.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Domain keyword sets (used for feature flags + honeypot + reasoning)
# ---------------------------------------------------------------------------

RETRIEVAL_KEYWORDS: frozenset[str] = frozenset({
    "faiss", "pinecone", "qdrant", "milvus", "weaviate", "opensearch",
    "elasticsearch", "vector search", "vector database", "vector db",
    "embeddings", "embedding", "sentence-transformers", "sentence transformers",
    "bge", "e5", "retrieval", "information retrieval", "semantic search",
    "bm25", "hybrid search", "dense retrieval", "sparse retrieval",
    "approximate nearest neighbor", "ann", "hnsw", "ivf",
})

RANKING_KEYWORDS: frozenset[str] = frozenset({
    "learning to rank", "learning-to-rank", "ltr", "xgboost", "lambdamart",
    "ndcg", "mrr", "map", "mean average precision", "mean reciprocal rank",
    "a/b testing", "ab testing", "a/b test", "evaluation framework",
    "ranking", "re-ranking", "reranking", "ranker", "candidate ranking",
    "search ranking", "relevance", "precision", "recall", "f1",
    "dcg", "discounted cumulative gain",
})

PRODUCTION_ML_KEYWORDS: frozenset[str] = frozenset({
    "mlops", "mlflow", "kubeflow", "bentoml", "seldon", "triton",
    "model serving", "model deployment", "production ml", "production machine learning",
    "docker", "kubernetes", "k8s", "ci/cd", "feature store", "feature engineering",
    "model monitoring", "data pipeline", "airflow", "weights & biases", "wandb",
    "sagemaker", "vertex ai", "ml pipeline", "model registry",
})

SEARCH_KEYWORDS: frozenset[str] = frozenset({
    "search engineer", "search", "search systems", "search infrastructure",
    "search relevance", "query understanding", "query processing",
    "search quality", "solr", "lucene", "inverted index",
})

RECOMMENDATION_KEYWORDS: frozenset[str] = frozenset({
    "recommendation system", "recommendation systems", "recommender",
    "collaborative filtering", "content-based filtering", "matrix factorization",
    "recommendation engine", "personalization",
})

NLP_IR_KEYWORDS: frozenset[str] = frozenset({
    "nlp", "natural language processing", "nlu", "text classification",
    "named entity recognition", "ner", "sentiment analysis", "tokenization",
    "transformers", "bert", "gpt", "llm", "large language model",
    "information retrieval", "ir", "text mining", "text analysis",
    "hugging face", "huggingface", "spacy", "nltk",
})

LLM_FINETUNING_KEYWORDS: frozenset[str] = frozenset({
    "lora", "qlora", "peft", "fine-tuning", "finetuning", "fine tuning",
    "fine-tuning llms", "model fine-tuning", "adapter", "instruction tuning",
    "rlhf", "dpo", "supervised fine-tuning", "sft",
})

AI_ML_KEYWORDS: frozenset[str] = frozenset({
    "machine learning", "deep learning", "neural network", "pytorch",
    "tensorflow", "scikit-learn", "sklearn", "keras", "ai", "artificial intelligence",
    "ml", "data science", "statistical modeling", "regression", "classification",
    "clustering", "reinforcement learning", "computer vision", "nlp",
    "natural language processing", "generative ai", "genai",
})

PYTHON_KEYWORDS: frozenset[str] = frozenset({
    "python", "flask", "fastapi", "django", "pandas", "numpy",
    "scipy", "pyspark",
})

BUZZWORD_KEYWORDS: frozenset[str] = frozenset({
    "prompt engineering", "chatgpt", "langchain", "rag",
    "genai", "generative ai", "gpt", "copilot", "openai",
    "ai-powered", "ai powered", "ai-driven",
})

CV_SPEECH_ROBOTICS_KEYWORDS: frozenset[str] = frozenset({
    "computer vision", "image classification", "object detection", "yolo",
    "image segmentation", "opencv", "cnn", "convolutional neural network",
    "speech recognition", "speech synthesis", "asr", "tts", "text to speech",
    "robotics", "ros", "robot operating system", "autonomous", "lidar",
    "slam", "perception", "point cloud",
})

FOUNDATIONAL_SKILLS_KEYWORDS: frozenset[str] = frozenset({
    "pytorch", "tensorflow", "scikit-learn", "numpy", "pandas",
})

EMBEDDING_KEYWORDS: frozenset[str] = frozenset({
    "embedding", "embeddings", "sentence-transformers", "vector",
})

# ---------------------------------------------------------------------------
# Company / title classification sets
# ---------------------------------------------------------------------------

CONSULTING_COMPANIES: frozenset[str] = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "ltimindtree",
    "l&t infotech", "hexaware", "cyient", "zensar",
})

NON_TECH_TITLES: frozenset[str] = frozenset({
    "accountant", "hr manager", "hr", "human resources", "customer support",
    "marketing manager", "marketing", "sales", "sales manager",
    "business analyst", "project manager", "operations manager",
    "administrative", "receptionist", "executive assistant",
    "content writer", "copywriter", "graphic designer",
})

RELEVANT_TITLES: frozenset[str] = frozenset({
    "machine learning engineer", "ml engineer", "ai engineer",
    "senior ai engineer", "senior ml engineer", "senior machine learning engineer",
    "data scientist", "senior data scientist", "nlp engineer", "senior nlp engineer",
    "search engineer", "applied ml engineer", "recommendation systems engineer",
    "staff machine learning engineer", "lead ai engineer", "principal engineer",
    "research scientist", "applied scientist",
})

# ---------------------------------------------------------------------------
# Education sets
# ---------------------------------------------------------------------------

ML_AI_FIELDS: frozenset[str] = frozenset({
    "computer science", "cs", "artificial intelligence", "ai",
    "machine learning", "ml", "data science", "information technology",
    "it", "software engineering", "computational linguistics",
    "statistics", "mathematics", "applied mathematics",
})

DEGREE_RANKS: dict[str, int] = {
    "ph.d": 6, "phd": 6, "ph.d.": 6, "doctorate": 6,
    "m.tech": 5, "mtech": 5, "m.tech.": 5,
    "m.s.": 4, "m.s": 4, "ms": 4, "m.sc": 4, "m.sc.": 4, "msc": 4,
    "m.e.": 4, "me": 4, "m.e": 4, "mba": 4,
    "b.tech": 3, "btech": 3, "b.tech.": 3,
    "b.e.": 3, "be": 3, "b.e": 3, "b.s.": 3, "bs": 3, "b.sc": 3,
    "b.sc.": 3, "bsc": 3, "bca": 2, "b.c.a": 2,
    "diploma": 1,
}

TIER_RANKS: dict[str, int] = {
    "tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1,
}

# ---------------------------------------------------------------------------
# Honeypot-specific evidence keywords
# ---------------------------------------------------------------------------

EVIDENCE_KEYWORDS: frozenset[str] = frozenset({
    "retrieval system", "retrieval systems", "ranking system", "ranking systems",
    "production ml", "production model", "search system", "search systems",
    "embedding", "embeddings", "vector database", "vector db",
    "model training", "model deployment", "model serving",
    "evaluation framework", "evaluation metric", "ndcg", "mrr",
    "fine-tuning", "fine-tuned", "finetuning", "finetuned",
    "pipeline", "ml pipeline", "data pipeline",
    "feature engineering", "feature store",
    "mlops", "mlflow", "kubeflow",
    "a/b test", "ab test", "experiment",
    "faiss", "pinecone", "qdrant", "milvus", "weaviate",
    "elasticsearch", "opensearch",
    "recommendation", "collaborative filtering",
    "transformer", "bert", "attention mechanism",
    "batch processing", "real-time inference",
    "scikit-learn", "pytorch", "tensorflow",
})

# ---------------------------------------------------------------------------
# Reasoning-specific keyword groups
# ---------------------------------------------------------------------------

HIGHLIGHT_SKILLS: dict[str, frozenset[str]] = {
    "retrieval": frozenset({
        "faiss", "pinecone", "qdrant", "milvus", "weaviate", "opensearch",
        "elasticsearch", "vector search", "embeddings", "bm25",
        "information retrieval", "semantic search", "dense retrieval",
    }),
    "ranking": frozenset({
        "learning to rank", "learning-to-rank", "xgboost", "ndcg",
        "mrr", "a/b testing", "evaluation framework", "ranking",
        "reranking",
    }),
    "production": frozenset({
        "mlops", "mlflow", "kubeflow", "bentoml", "docker", "kubernetes",
        "model serving", "model deployment", "feature engineering",
        "weights & biases", "sagemaker",
    }),
    "llm": frozenset({
        "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
        "hugging face transformers", "llms", "rag",
    }),
    "core_ml": frozenset({
        "pytorch", "tensorflow", "scikit-learn", "deep learning",
        "machine learning", "nlp", "python", "statistical modeling",
        "recommendation systems",
    }),
}

CAREER_HIGHLIGHT_KEYWORDS: tuple[str, ...] = (
    "semantic search", "vector search", "ranking system", "ranking pipeline",
    "retrieval system", "retrieval pipeline", "recommendation system",
    "embedding", "fine-tuned", "fine-tuning", "evaluation framework",
    "eval framework", "search system", "search infrastructure",
    "production ml", "model deployment", "deployed", "production",
    "learning to rank", "a/b test", "ndcg", "recall", "precision",
    "feature engineering", "ml pipeline", "data pipeline",
)

# ---------------------------------------------------------------------------
# Title domain classification (used in consistency + honeypot)
# ---------------------------------------------------------------------------

TECH_TERMS: frozenset[str] = frozenset({
    "engineer", "developer", "scientist", "ml", "ai", "data",
    "nlp", "search", "software", "architect",
})

NON_TECH_TERMS: frozenset[str] = frozenset({
    "accountant", "hr", "marketing", "sales", "support",
    "manager", "executive", "coordinator", "analyst",
})

ML_TITLE_TERMS: frozenset[str] = frozenset({
    "ml", "ai", "machine learning", "data scientist",
})

NON_TECH_DESC_TERMS: frozenset[str] = frozenset({
    "accounting", "audit", "tax", "hr", "recruitment",
    "customer support", "ticket", "helpdesk",
})

TECH_DESC_TERMS: frozenset[str] = frozenset({
    "model", "algorithm", "neural", "embedding", "pipeline",
})
