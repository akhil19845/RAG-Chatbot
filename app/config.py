# app/config.py

from pathlib import Path

# === Data & Index Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_FILE = BASE_DIR / "Data" / "Machine_Learning_Wikipedia_Corpus.pdf"
CHROMA_PATH = BASE_DIR / "chroma_db_ml"

# === Models ===
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.0-flash-lite" 

# === LLM Defaults ===
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 800

# === Chunking parameters ===
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

INPUT_FILE = BASE_DIR / "Eval" / "Granite" / "Golden_Dataset.json"
OUTPUT_FILE = BASE_DIR / "Eval" / "Granite" / "Golden_Dataset_Answers.json" 

INPUT_EVAL_DATA_FILE = BASE_DIR / "Eval" / "Granite" / "Golden_Dataset_Answers.json" 
OUTPUT_EVAL_DATA_FILE = BASE_DIR / "Eval" / "Granite" / "Top3_Generation_Evaluation_metric_Results.json"

# === Retriever parameters ===

# === Re-ranker Model ===
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_RETRIEVAL_K = 10 
