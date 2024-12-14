"""Dense retrieval from Chroma followed by cross-encoder reranking."""
from dataclasses import dataclass
from functools import lru_cache

import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Hit:
    id: str
    text: str
    page: int
    section: str
    section_title: str
    score: float


def _fp16(device: str) -> dict:
    return {"dtype": torch.float16} if device == "cuda" else {}


@lru_cache(maxsize=2)
def embedder(device: str = config.QUERY_EMBED_DEVICE) -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL, device=device, model_kwargs=_fp16(device))


@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    return CrossEncoder(config.RERANK_MODEL, device=DEVICE, max_length=512,
                        model_kwargs=_fp16(DEVICE))


@lru_cache(maxsize=1)
def collection():
    client = chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"))
    return client.get_collection(config.COLLECTION)


def embed_passages(texts: list[str]):
    return embedder(DEVICE).encode(texts, batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=True, convert_to_numpy=True)


def dense_search(text: str, k: int, is_query: bool) -> list[Hit]:
    """Search with a raw question (is_query=True, gets the BGE prefix) or a passage (HyDE)."""
    if is_query:
        text = config.BGE_QUERY_PREFIX + text
    vec = embedder().encode([text], normalize_embeddings=True, convert_to_numpy=True)
    res = collection().query(query_embeddings=vec.tolist(), n_results=k)
    return [
        Hit(id=i, text=doc, page=m["page"], section=m["section"],
            section_title=m["section_title"], score=1.0 - dist)
        for i, doc, m, dist in zip(res["ids"][0], res["documents"][0],
                                   res["metadatas"][0], res["distances"][0])
    ]


def rerank(question: str, hits: list[Hit], k: int) -> list[Hit]:
    """Score (question, passage) pairs with the cross-encoder and keep the top k."""
    if not hits:
        return []
    scores = reranker().predict([(question, h.text) for h in hits], batch_size=8,
                                activation_fn=torch.nn.Sigmoid())
    for h, s in zip(hits, scores):
        h.score = float(s)
    return sorted(hits, key=lambda h: h.score, reverse=True)[:k]
