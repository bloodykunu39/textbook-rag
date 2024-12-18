"""LLM calls via a local Ollama server (chat API, so the model's own chat template is used)."""
import ollama

from . import config

HYDE_SYSTEM = (
    "You write a short passage, in the style of an introductory psychology textbook, "
    "that would answer the user's question. Write 3-5 factual sentences. "
    "No preamble, no follow-up questions."
)

ANSWER_SYSTEM = (
    "You are a teaching assistant answering questions about the textbook Psychology 2e.\n"
    "Rules:\n"
    "1. Use only facts from the numbered context passages.\n"
    "2. Answer in one or two paragraphs (4-8 sentences) that directly answer the question.\n"
    "3. Only if the question itself asks for parts, stages, types or a comparison, name "
    "each one the context gives and explain it in a sentence.\n"
    "4. If the context does not answer the question, say so instead of guessing.\n"
    "5. No introductions, headings or sign-offs."
)


def _chat(system: str, user: str, temperature: float, num_predict: int) -> str:
    resp = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={"temperature": temperature, "num_predict": num_predict, "num_ctx": 4096},
    )
    return resp["message"]["content"].strip()


def hypothetical_passage(question: str) -> str:
    """HyDE: draft a textbook-style passage to use as the dense-retrieval query."""
    return _chat(HYDE_SYSTEM, question, temperature=0.0, num_predict=160)


def answer(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return _chat(ANSWER_SYSTEM, user, temperature=0.2, num_predict=450)
