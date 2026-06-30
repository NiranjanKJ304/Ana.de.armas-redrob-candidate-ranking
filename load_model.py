"""
Model Loader for the Redrob Intelligent Candidate Ranking System.

Automatically downloads and extracts the SentenceTransformer model if missing.
Exposes `ensure_model()` for reuse in the main orchestration script.
"""

import os
import sys
import zipfile
import logging
from pathlib import Path

# Setup a local logger for this module
logger = logging.getLogger("load_model")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def ensure_model() -> bool:
    """
    Check if the local embedding model exists. If not, download and extract it.
    Returns True if the model is ready, False if an error occurred.
    """
    model_dir = Path("models/all-MiniLM-L6-v2")
    model_marker = model_dir / "model.safetensors"
    models_root = Path("models")

    # 1. Check if model already exists
    if model_marker.exists():
        logger.info("Model already exists at %s. Skipping download.", model_marker)
        return True

    # 2. Create models directory if needed
    models_root.mkdir(parents=True, exist_ok=True)

    # 3. Download the ZIP
    import requests
    try:
        from tqdm import tqdm
    except ImportError:
        logger.error("tqdm is required. Please install it (pip install tqdm).")
        return False

    url = "https://huggingface.co/datasets/niranjankj/hackathon-assets/resolve/main/all-MiniLM-L6-v2.zip"
    zip_path = Path("models/all-MiniLM-L6-v2.zip")
    
    logger.info("Model not found. Downloading from %s", url)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading embedding model",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
                
        logger.info("Download complete. Extracting ZIP...")
    except Exception as e:
        logger.error("Failed to download the model: %s", e)
        if zip_path.exists():
            zip_path.unlink()
        return False

    # 4. Extract the ZIP
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(models_root)
        logger.info("Extraction complete.")
    except Exception as e:
        logger.error("Failed to extract the model ZIP: %s", e)
        if zip_path.exists():
            zip_path.unlink()
        return False

    # 5. Verify the extraction
    if not model_marker.exists():
        logger.error("Verification failed: %s not found after extraction.", model_marker)
        return False
        
    logger.info("Model verified successfully.")

    # 6. Delete the ZIP
    if zip_path.exists():
        try:
            zip_path.unlink()
            logger.info("Deleted temporary ZIP file.")
        except Exception as e:
            logger.warning("Could not delete temporary ZIP file: %s", e)

    return True

if __name__ == "__main__":
    # Force UTF-8 encoding for standard output/error on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        
    print("\nStarting model load process...\n")
    success = ensure_model()
    
    if success:
        print("\nModel is ready for use.")
        sys.exit(0)
    else:
        print("\nModel setup failed.")
        sys.exit(1)
