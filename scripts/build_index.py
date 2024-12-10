"""Parse the PDF, chunk it with section metadata, embed, and store in Chroma."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import ingest  # noqa: E402

if __name__ == "__main__":
    ingest.build_index()
