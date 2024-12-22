"""Compare retrieval strategies against hand-labelled gold sections.

A retrieved chunk is relevant if its section is one of the gold sections for that query.
Metrics: Hit@k (any relevant chunk in top k), Precision@k (share of top-k chunks that are
relevant) and MRR (1 / rank of the first relevant chunk).

HyDE passages are generated once and cached in outputs/hyde_cache.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config, generate, retrieve  # noqa: E402

K = config.TOP_K


def load_hyde_cache(queries) -> dict:
    path = config.OUTPUT_DIR / "hyde_cache.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    for q in queries:
        if q["query_id"] not in cache:
            print(f"HyDE {q['query_id']:>2}: {q['question']}")
            cache[q["query_id"]] = generate.hypothetical_passage(q["question"])
            path.write_text(json.dumps(cache, indent=2))
    return cache


def strategies(question: str, hyde: str) -> dict:
    """Each strategy returns a ranked list of hits (top K)."""
    q_dense = retrieve.dense_search(question, config.CANDIDATES, is_query=True)
    h_dense = retrieve.dense_search(hyde, config.CANDIDATES, is_query=False)
    return {
        "dense (question)": q_dense[:K],
        "dense (HyDE)": h_dense[:K],
        "dense (question) + rerank": retrieve.rerank(question, q_dense, K),
        "dense (HyDE) + rerank vs HyDE passage": retrieve.rerank(hyde, h_dense, K),
        "dense (HyDE) + rerank vs question": retrieve.rerank(question, h_dense, K),
    }


def score(hits, gold: set[str]) -> tuple[float, float, float]:
    rel = [h.section in gold for h in hits]
    first = next((i for i, r in enumerate(rel, 1) if r), None)
    return float(any(rel)), sum(rel) / len(rel), (1.0 / first if first else 0.0)


def main():
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    queries = json.loads(config.QUERIES_JSON.read_text())
    gold = json.loads(config.GOLD_JSON.read_text())
    hyde = load_hyde_cache(queries)

    totals: dict[str, list[float]] = {}
    per_query = []
    for q in queries:
        g = set(gold[q["query_id"]])
        row = {"query_id": q["query_id"], "question": q["question"]}
        for name, hits in strategies(q["question"], hyde[q["query_id"]]).items():
            s = score(hits, g)
            acc = totals.setdefault(name, [0.0, 0.0, 0.0])
            for i in range(3):
                acc[i] += s[i]
            row[name] = {"hit": s[0], "precision": round(s[1], 3), "mrr": round(s[2], 3),
                         "sections": [h.section for h in hits]}
        per_query.append(row)

    n = len(queries)
    lines = [f"Retrieval evaluation on {n} queries (k={K})", "",
             f"| Strategy | Hit@{K} | Precision@{K} | MRR |", "|---|---|---|---|"]
    for name, (hit, prec, mrr) in totals.items():
        lines.append(f"| {name} | {hit / n:.3f} | {prec / n:.3f} | {mrr / n:.3f} |")
    report = "\n".join(lines)
    print("\n" + report)
    (config.OUTPUT_DIR / "retrieval_eval.md").write_text(report + "\n")
    (config.OUTPUT_DIR / "retrieval_eval_per_query.json").write_text(
        json.dumps(per_query, indent=2))


if __name__ == "__main__":
    main()
