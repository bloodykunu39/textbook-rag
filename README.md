# Textbook RAG: question answering over *Psychology 2e*

A local retrieval-augmented generation pipeline that answers questions about the OpenStax
*Psychology 2e* textbook (753 pages) and cites the chapter sections it used. It runs fully
offline on a 6 GB consumer GPU (RTX 3050).

Questions: the 50-question set from the CASML Generative AI Hackathon task (book +
`queries.json`, available as the Kaggle dataset `vansh63/casml-dataset`). The textbook is
© OpenStax, licensed CC BY 4.0, and is not redistributed here.

## Pipeline

```
PDF ──► outline-aware parsing ──► 2,691 chunks (800 chars, 100 overlap, section metadata)
                                         │  bge-large-en-v1.5 embeddings
                                         ▼
question ──► HyDE passage (Llama 3.2 3B) ──► Chroma top-80 ──► bge-reranker-v2-m3 vs question ──► top-8
                                                                                                  │
                                                         Llama 3.2 3B (Ollama, 4-bit) ◄───────────┘
                                                                     │
                                                         answer + section/page references
```

| Component | Choice |
|---|---|
| Parsing | PyMuPDF; PDF outline gives section boundaries; running headers/footers removed |
| Embeddings | `BAAI/bge-large-en-v1.5` (query instruction prefix for raw questions) |
| Vector store | ChromaDB (cosine) |
| Query expansion | HyDE: the LLM drafts a textbook-style passage used as the search query |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder, fp16 |
| Generator | `llama3.2:3b` via Ollama (Q4, ~2 GB VRAM) |

## Design decisions

1. **Rerank against the question, not the HyDE passage.** HyDE helps the dense search, but
   the cross-encoder should judge relevance to what the user actually asked. Reranking
   against the HyDE text is a common shortcut; the evaluation below compares both.
2. **Section-aware chunks.** Sections are cut at their headings using the PDF outline, so
   no chunk spans two sections, and every chunk carries its section number at index time.
   That makes citations a metadata lookup instead of a page-range guess.
3. **Cleaner text.** Running headers ("1.2 • History of Psychology 9") and footers are
   dropped. Unicode is normalised rather than deleted.
4. **Fits 6 GB VRAM.** A 4-bit LLM, an fp16 reranker, and a CPU query embedder share the
   GPU. A full-precision 3B model alone would need about 6.5 GB.

## Retrieval evaluation

`data/gold_sections.json` hand-labels the relevant section(s) for each of the 50 questions.
A retrieved chunk is relevant if it comes from a gold section. k = 8.

| Strategy | Hit@8 | Precision@8 | MRR |
|---|---|---|---|
| Dense (question) | 0.980 | 0.522 | **0.820** |
| Dense (HyDE) | 0.960 | **0.557** | 0.798 |
| Dense (question) + rerank | **1.000** | 0.517 | 0.806 |
| Dense (HyDE) + rerank vs HyDE passage | 0.980 | 0.495 | 0.792 |
| Dense (HyDE) + rerank vs question *(used)* | **1.000** | 0.522 | 0.808 |

**How to read this:**
- Reranking against the question beats reranking against the HyDE passage on every metric.
- All strategies are close. These are short definition-style questions over one
  well-structured book, so plain dense retrieval is already strong.
- With 50 queries, one query changes Hit@8 by 0.02. Treat differences under about 0.03
  as noise.
- The gold labels were made by one person from the table of contents and keyword search.

## Generation

- All 50 questions are answered in **~4.7 minutes** on an RTX 3050 (~5.7 s per question
  including HyDE), output to `outputs/submission.csv`.
- Answers are 38–174 words (median 95).

**Known limitation:** the 3B model sometimes omits or garbles details that are present in
the retrieved context. For example, its answer on sleep stages left out REM sleep and
misattributed delta waves to stage 2. Answer quality is not yet measured automatically.
Next steps: an LLM-as-judge faithfulness check, or a larger model on a bigger GPU.

## Run it

```bash
conda create -n textbook_rag python=3.11 -y
conda activate textbook_rag
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Ollama (no root needed): https://ollama.com/download
ollama serve &            # in another terminal
ollama pull llama3.2:3b

# put book.pdf and queries.json in data/raw/
python scripts/build_index.py          # ~45 s on GPU
python scripts/evaluate_retrieval.py   # ~3 min (HyDE passages cached in outputs/)
python scripts/answer_queries.py       # ~5 min for 50 questions

uvicorn api:app --port 8000
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "What is classical conditioning?"}'
```

## Layout

```
rag/config.py        paths, model names, chunk + retrieval settings
rag/ingest.py        PDF outline -> sections -> cleaned chunks -> Chroma
rag/retrieve.py      dense search + cross-encoder rerank
rag/generate.py      HyDE + answer prompts (Ollama chat API)
rag/pipeline.py      ask(question) -> answer + references
scripts/             build_index, evaluate_retrieval, answer_queries
api.py               FastAPI service
data/gold_sections.json   relevance labels for evaluation
```
