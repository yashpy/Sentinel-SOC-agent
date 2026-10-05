"""Retrieval evaluation: Hit@1, Recall@5, MRR@10 and latency for each retrieval mode.

Usage: python -m sentinel.rag.evaluate
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sentinel import config
from sentinel.rag.retriever import MODES, get_kb


def _relevant(hit: dict, gold: list[str]) -> bool:
    return any(t.split(".")[0] in gold for t in hit["techniques"])


def evaluate(queries: list[dict], modes=MODES) -> dict:
    kb = get_kb()
    known = {d.doc_id.split(".")[0] for d in kb.docs if d.kind == "technique"}
    missing = sorted({g for q in queries for g in q["gold"]} - known)
    out = {"n_queries": len(queries), "corpus_docs": len(kb.docs), "unknown_gold_ids": missing}
    for mode in modes:
        ranks, lat = [], []
        for q in queries:
            t = time.perf_counter()
            hits = kb.search(q["q"], k=10, mode=mode, kind="technique")
            lat.append(time.perf_counter() - t)
            ranks.append(next((i + 1 for i, h in enumerate(hits) if _relevant(h, q["gold"])), None))
        r = np.array([x or 0 for x in ranks])
        out[mode] = dict(hit_at_1=round(float(np.mean(r == 1)), 4),
                         recall_at_5=round(float(np.mean((r >= 1) & (r <= 5))), 4),
                         mrr_at_10=round(float(np.mean([1 / x if x else 0 for x in ranks])), 4),
                         p50_latency_ms=round(float(np.median(lat)) * 1000, 1))
    return out


def main() -> None:
    queries = json.loads((Path(__file__).parent / "eval_queries.json").read_text())
    res = evaluate(queries)
    (config.RESULTS_DIR / "rag_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
