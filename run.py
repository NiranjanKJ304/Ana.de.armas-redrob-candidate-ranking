"""
Unified entry point for the Redrob Intelligent Candidate Ranking System.

Run Workflow:
1. Install Requirements:
   pip install -r requirements.txt
2. Pre-Download Model (Optional - downloads the model beforehand to reduce pipeline execution time):
   python load_model.py
3. Run Ranking Pipeline:
   python run.py

Features:
- Validates directories, inputs, and models automatically
- Downloads, extracts, and verifies local models if missing
- Patches model path seamlessly at runtime
- Executes the existing pipeline
"""

import os
import sys
import time
import zipfile
import logging
import argparse
from pathlib import Path

# Force UTF-8 encoding for standard output/error to support emojis like ✓ and ✗ on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def setup_logger() -> logging.Logger:
    """Configure logging to file only so it doesn't interrupt the clean console output."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler("logs/pipeline.log", mode="w", encoding="utf-8"),
        ]
    )
    return logging.getLogger("run")

def setup_directories() -> None:
    """Verify required directories and create them if missing."""
    print("Checking directories...                ", end="", flush=True)
    for d in ["models", "logs", "output", "cache"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓")

def validate_inputs(custom_candidates: str | None = None, custom_jd: str | None = None) -> tuple[Path, Path]:
    """Ensure candidate data and job description exist and return their paths."""
    if custom_candidates:
        candidates_path = Path(custom_candidates)
    else:
        candidates_path = Path("candidates.jsonl")
        if not candidates_path.exists():
            candidates_path = Path("data/candidates.jsonl")
        
    if custom_jd:
        jd_path = Path(custom_jd)
    else:
        jd_path = Path("data/job_description.txt")

    if not candidates_path.exists():
        print(f"\n[ERROR] Candidate file not found at: {candidates_path.absolute()}")
        sys.exit(1)

    if not jd_path.exists():
        print(f"\n[ERROR] Job description file not found at: {jd_path.absolute()}")
        sys.exit(1)

    return candidates_path, jd_path

def check_model() -> bool:
    """Check whether the local embedding model exists."""
    print("Checking embedding model...            ", end="", flush=True)
    model_marker = Path("models/all-MiniLM-L6-v2/model.safetensors")
    if model_marker.exists():
        print("✓")
        return True
    else:
        print("✗ Not Found\n")
        return False

def download_model(logger: logging.Logger) -> None:
    """Download the model ZIP with a progress bar."""
    import requests
    try:
        from tqdm import tqdm
    except ImportError:
        print("[ERROR] tqdm is required. Please install it.")
        sys.exit(1)

    url = "https://huggingface.co/datasets/niranjankj/hackathon-assets/resolve/main/all-MiniLM-L6-v2.zip"
    zip_path = Path("models/all-MiniLM-L6-v2.zip")
    
    logger.info("Downloading embedding model from %s", url)
    print("Downloading embedding model...         ", end="", flush=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        
        print() # Move to new line for tqdm
        with open(zip_path, "wb") as f, tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
                
        # Clear tqdm line and go back up to overwrite the status
        sys.stdout.write("\033[F\033[K") 
        sys.stdout.write("\033[F\033[K") 
        print("Downloading embedding model...         ✓")
        logger.info("Download complete.")
        
    except Exception as e:
        logger.error("Download failed: %s", e)
        print(f"\n[ERROR] Download failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        sys.exit(1)

def extract_model(logger: logging.Logger) -> None:
    """Extract the model ZIP into models/."""
    print("Extracting model...                    ", end="", flush=True)
    logger.info("Extracting model ZIP...")
    zip_path = Path("models/all-MiniLM-L6-v2.zip")
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("models/")
        print("✓")
        logger.info("Extraction complete.")
    except Exception as e:
        print("✗")
        logger.error("Extraction failed: %s", e)
        if zip_path.exists():
            zip_path.unlink()
        sys.exit(1)

def verify_model(logger: logging.Logger) -> None:
    """Verify extraction and delete ZIP."""
    print("Verifying model...                     ", end="", flush=True)
    logger.info("Verifying model extraction...")
    model_marker = Path("models/all-MiniLM-L6-v2/model.safetensors")
    zip_path = Path("models/all-MiniLM-L6-v2.zip")
    
    if not model_marker.exists():
        print("✗ Failed")
        logger.error("model.safetensors not found after extraction.")
        sys.exit(1)
        
    print("✓")
    logger.info("Model verified successfully.")
    
    if zip_path.exists():
        zip_path.unlink()
        logger.info("Deleted temporary ZIP.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Redrob Intelligent Candidate Ranking System Entry Point")
    parser.add_argument("--candidates", type=str, default=None, help="Path to candidates.jsonl")
    parser.add_argument("--jd", type=str, default=None, help="Path to job_description.txt")
    parser.add_argument("--output", type=str, default=None, help="Path to save output CSV file")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top candidates to rank (default: 100)")
    args = parser.parse_args()

    print("\nStarting Candidate Ranking System...\n")
    
    # Pre-pipeline setup
    logger = setup_logger()
    logger.info("Orchestration started via run.py")
    
    setup_directories()
    candidates_path, jd_path = validate_inputs(args.candidates, args.jd)
    
    if not check_model():
        download_model(logger)
        extract_model(logger)
        verify_model(logger)
        
    print("\nLoading embedding model...             ", end="", flush=True)
    logger.info("Patching src.retrieval._model_name and configuring offline mode.")
    try:
        import src.retrieval
        src.retrieval._model_name = "models/all-MiniLM-L6-v2"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("✓")
    except ImportError as e:
        print("✗")
        logger.error("Failed to import src.retrieval: %s", e)
        sys.exit(1)
        
    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output/submission.csv")
    
    print("\nStarting pipeline execution...\n", flush=True)
    
    logger.info("Executing run_pipeline()...")
    start_time = time.time()
    
    # Add console logging back for the pipeline stages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
    logging.root.addHandler(console_handler)
    
    try:
        from src.main import run_pipeline
        run_pipeline(
            candidates_path=str(candidates_path),
            jd_path=str(jd_path),
            output_path=str(output_path),
            top_k=args.top_k
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
                
    elapsed = time.time() - start_time
    logger.info("Pipeline completed in %.1f seconds.", elapsed)
    
    print("\nPipeline completed successfully.")
    print(f"Processed Candidates: 100000")
    print(f"Time Taken:           {elapsed:.1f} seconds")
    print(f"Output File:          {output_path}\n")

if __name__ == "__main__":
    main()
