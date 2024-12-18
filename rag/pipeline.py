"""End-to-end question answering: HyDE -> dense retrieval -> rerank -> grounded answer."""
from dataclasses import dataclass, field

from . import config, generate, retrieve


@dataclass
class RagResult:
    question: str
    answer: str
    hyde: str
    hits: list = field(default_factory=list)

    @property
    def references(self) -> dict:
        sections = []
        for h in self.hits:
            ref = f"{h.section} {h.section_title}"
            if ref not in sections:
                sections.append(ref)
        return {"sections": sections, "pages": sorted({h.page for h in self.hits})}


def retrieve_for(question: str, hyde: str | None, top_k: int = config.TOP_K,
                 candidates: int = config.CANDIDATES):
    """Dense search with the HyDE passage (or the raw question if hyde is None), then
    rerank against the *original question* rather than the hypothetical passage."""
    if hyde:
        hits = retrieve.dense_search(hyde, candidates, is_query=False)
    else:
        hits = retrieve.dense_search(question, candidates, is_query=True)
    return retrieve.rerank(question, hits, top_k)


def ask(question: str, use_hyde: bool = True, top_k: int = config.TOP_K) -> RagResult:
    hyde = generate.hypothetical_passage(question) if use_hyde else ""
    hits = retrieve_for(question, hyde or None, top_k=top_k)
    text = generate.answer(question, [h.text for h in hits])
    return RagResult(question=question, answer=text, hyde=hyde, hits=hits)
