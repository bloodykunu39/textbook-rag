"""Central settings for the textbook RAG pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BOOK_PDF = DATA_DIR / "raw" / "book.pdf"
QUERIES_JSON = DATA_DIR / "raw" / "queries.json"
GOLD_JSON = DATA_DIR / "gold_sections.json"
INDEX_DIR = ROOT / "index"          # Chroma DB + cached chunk table
OUTPUT_DIR = ROOT / "outputs"

# Printed page number = PDF page index (1-based) - PAGE_OFFSET (front matter).
PAGE_OFFSET = 12

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Models
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# The 6 GB GPU holds the LLM (Ollama) + reranker; one query embedding is fast on CPU.
# Index building still embeds on the GPU.
QUERY_EMBED_DEVICE = "cpu"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "llama3.2:3b"           # served by Ollama (4-bit GGUF, ~2 GB VRAM)
# bge v1.5 recommends this prefix for short queries (not for passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Retrieval
CANDIDATES = 80                     # dense candidates handed to the reranker
TOP_K = 8                           # chunks passed to the LLM
COLLECTION = "psychology2e"
