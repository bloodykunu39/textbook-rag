"""Answer every question in queries.json and write outputs/submission.csv."""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config, pipeline  # noqa: E402


def main():
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    queries = json.loads(config.QUERIES_JSON.read_text())
    out_path = config.OUTPUT_DIR / "submission.csv"
    timings = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "context", "answer", "references"])
        for q in queries:
            start = time.perf_counter()
            res = pipeline.ask(q["question"])
            timings.append(time.perf_counter() - start)
            context = "\n\n".join(h.text for h in res.hits)
            writer.writerow([q["query_id"], context, res.answer, json.dumps(res.references)])
            f.flush()
            print(f"[{q['query_id']:>2}] {timings[-1]:5.1f}s  {q['question']}")
            print(f"     -> {res.references['sections'][:3]}")
    print(f"\nWrote {out_path}")
    print(f"Mean {sum(timings) / len(timings):.1f}s per question, total {sum(timings) / 60:.1f} min")


if __name__ == "__main__":
    main()
