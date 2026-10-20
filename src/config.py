"""Central configuration: paths, model choices, retrieval settings and seeds.

Values that differ per machine or contain secrets (API keys, model names) come
from the .env file; see .env.example. Everything else is set here, so the
application code contains no hard-coded tuning values.
"""

import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# --- Embeddings and vector store -----------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "wine_pairing")

# all-MiniLM-L6-v2 reads at most 256 tokens (~1000 characters) per chunk, so
# chunks stay below that; the overlap keeps sentences that cross a boundary.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_BATCH_SIZE = 64

# --- Retrieval -----------------------------------------------------------------
N_WINE_RESULTS = 4  # chunks about the wine itself (grape, region, style)
N_THEORY_RESULTS = 2  # chunks with general pairing principles
N_RECIPE_RESULTS = 4  # recipe chunks the answer can suggest

# Cosine distance above which the best wine chunk counts as a loose match.
# In testing, covered questions scored 0.25-0.39 and an off-topic one 0.55.
MAX_WINE_DISTANCE = float(os.getenv("MAX_WINE_DISTANCE", "0.5"))

# Words that never identify a specific wine when checking coverage
# ("Chianti Classico" -> only "chianti" counts).
GENERIC_WINE_WORDS = {
    "wine", "red", "white", "rose", "sparkling", "dry", "sweet", "the", "from", "and",
    "bottle", "classico", "riserva", "reserva", "grand", "cru", "vintage",
}

# --- LLM -----------------------------------------------------------------------
LLM_TEMPERATURE = 0.0  # deterministic, factual answers
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Ollama's default context is small and silently truncates long prompts; the
# answer prompt with 10 chunks needs about 3k tokens.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.1-8b-instant",
    "google": "gemini-2.0-flash",
}

# --- Reproducibility -------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """
    Seed every random number generator the project touches.

    Retrieval itself is deterministic, but this keeps embedding and any
    library-level sampling reproducible. LLM output is made repeatable with
    temperature 0 and, where the provider supports it, the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch comes with sentence-transformers, but stay safe
        pass
