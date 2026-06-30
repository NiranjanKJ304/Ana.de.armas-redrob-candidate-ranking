# 🤖 SentenceTransformer Embedding Model Documentation

> **Quick Start:** To run the entire candidate ranking system (which automatically downloads the model if missing), execute:
> ```bash
> python run.py
> ```

This directory contains the local SentenceTransformer embedding model used for calculating the semantic similarity score between candidates and the job description.

---

## 📥 Model Source Details

* **Model ID:** `sentence-transformers/all-MiniLM-L6-v2`
* **Download Source (ZIP):** [niranjankj/hackathon-assets](https://huggingface.co/datasets/niranjankj/hackathon-assets/resolve/main/all-MiniLM-L6-v2.zip)
* **Local Extraction Path:** `models/all-MiniLM-L6-v2/`
* **Local Marker File:** `models/all-MiniLM-L6-v2/model.safetensors`

---

## 🛠️ How to Download the Model

There are two ways to download this model:

### Option 1: Automatic Download (Recommended)
Simply run the pipeline using:
```bash
python run.py
```
If the model directory is missing or incomplete, the script will automatically download the ZIP, extract it here, verify the integrity of the files, and delete the temporary ZIP.

### Option 2: Standalone Download
You can run the dedicated downloader script:
```bash
python load_model.py
```

---

## 🧠 Loading & Offline Enforcement

During runtime:
1. `run.py` dynamically patches the `SentenceTransformer` loader to use this local directory (`models/all-MiniLM-L6-v2`) instead of retrieving it online.
2. The environment variables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set programmatically to ensure that **no network requests are ever made** after setup, keeping the pipeline 100% offline and deterministic.
