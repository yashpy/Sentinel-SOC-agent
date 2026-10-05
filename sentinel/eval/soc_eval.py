"""End-to-end SOC agent evaluation on the held-out alert queue, with ablations.

Configurations
  naive_llm        : one LLM call over raw audit-log rows (no tools, no RAG, no verifier)
  agent_tools      : LangGraph agent with investigators, no RAG, no verifier
  agent_full       : + hybrid RAG (ATT&CK + runbooks) + verifier/self-correction
  agent_full_fast  : agent_full but every node on the small model (cost/quality routing study)

Results are cached per (config, alert) in results/soc_runs.jsonl, so runs are resumable.
Usage: python -m sentinel.eval.soc_eval --configs agent_full --limit 20 --workers 4
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from sentinel import config
from sentinel.agents import llm as L
from sentinel.agents.alerts import alert_queue, ground_truth, to_alert
from sentinel.agents.soc_graph import ATTACK_TYPES, build_graph, normalize_hypothesis
from sentinel.agents.tools import get_store
from sentinel.data.scenarios import ATTACKS, technique_match

RUNS_PATH = config.RESULTS_DIR / "soc_runs.jsonl"
SPLIT = "test"
CONFIGS = {
    "naive_llm": None,
    "agent_tools": dict(use_rag=False, use_verifier=False),
    "agent_full": dict(use_rag=True, use_verifier=True, use_triage_llm=False),
    "agent_full_fast": dict(use_rag=True, use_verifier=True, analyst_model=config.LLM_FAST),
}
_lock = threading.Lock()


def run_naive(alert: dict) -> dict:
    raw, n = get_store().raw_log(alert["user_id"], alert["day"], max_rows=60)
    system = ("You are a SOC analyst. Given raw SaaS audit-log rows for one user-day, decide if it is an attack. "
              'Reply JSON: {"verdict": "malicious"|"benign", "attack_type": one of ' + str(ATTACK_TYPES) +
              ', "technique_id": "MITRE ATT&CK ID or null", "confidence": 0-1, "action": one of '
              + str(config.ACTIONS) + ', "rationale": "short"}')
    user = f"Detector signals: {alert['signals']}\nRaw events ({n} rows, CSV):\n{raw}"
    data, tr = L.call_json(config.LLM_STRONG, system, user, node="naive", max_tokens=300)
    h = normalize_hypothesis(data)
    tier = config.ACTION_TIERS.get(h["action"], "read_only")
    status = "executed" if tier != "destructive" else "pending_approval"  # naive baseline: no verifier gate
    return dict(hypothesis=h, decision=dict(action=h["action"], tier=tier, status=status), trace=[tr])


def run_one(cfg_name: str, row, graph) -> dict:
    alert = to_alert(row)
    t0 = time.perf_counter()
    try:
        if cfg_name == "naive_llm":
            out = run_naive(alert)
        else:
            out = graph.invoke({"alert": alert, "settings": {**CONFIGS[cfg_name], "approval": "batch"}})
        err = None
    except Exception as e:  # keep the harness alive; failures are scored as wrong
        out, err = {}, repr(e)[:300]
    trace = out.get("trace", [])
    return dict(config=cfg_name, alert_id=alert["alert_id"], truth=ground_truth(row), error=err,
                ml_alert=alert["ml_alert"], source=alert["source"],
                hypothesis=out.get("hypothesis"), first_hypothesis=out.get("first_hypothesis"),
                decision=out.get("decision"), attempts=out.get("attempts", 1), plan=out.get("plan"),
                retrieved=[r["doc_id"] for r in out.get("retrieved", [])],
                critiques=[p for t in trace if t.get("node") == "verifier" for p in t.get("problems", [])],
                latency_s=round(time.perf_counter() - t0, 2),
                llm_calls=sum(1 for t in trace if t.get("input_tokens") is not None),
                input_tokens=sum(t.get("input_tokens", 0) or 0 for t in trace),
                output_tokens=sum(t.get("output_tokens", 0) or 0 for t in trace),
                json_ok=all(t.get("parsed", True) for t in trace))


def runs_path():
    return RUNS_PATH if SPLIT == "test" else config.RESULTS_DIR / f"soc_runs_{SPLIT}.jsonl"


def load_runs() -> list[dict]:
    """Latest record per (config, alert); errored attempts are superseded by successful retries."""
    path = runs_path()
    if not path.exists():
        return []
    latest: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["config"], r["alert_id"])] = r
    return list(latest.values())


def run(configs: list[str], limit: int | None, workers: int, max_runs: int | None = None) -> None:
    q = alert_queue("union", split=SPLIT)
    if limit:
        q = q.sample(n=min(limit, len(q)), random_state=0) if limit < len(q) else q
    done = {(r["config"], r["alert_id"]) for r in load_runs() if not r.get("error")}
    from sentinel.rag.retriever import get_kb

    get_kb(), get_store()  # warm shared singletons before threads start
    graph = build_graph()
    jobs = [(c, row) for c in configs for _, row in q.iterrows()
            if (c, to_alert(row)["alert_id"]) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} cached)")
    jobs = jobs[:max_runs] if max_runs else jobs
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, c, row, graph) for c, row in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            with _lock, runs_path().open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {time.time() - t0:.0f}s elapsed", flush=True)


def score(rec: dict) -> dict:
    t, h, d = rec["truth"], rec.get("hypothesis") or {}, rec.get("decision") or {}
    mal = t["label"] == 1
    pred_mal = h.get("verdict") == "malicious"
    action = d.get("action", "none")
    containment = config.ACTION_TIERS.get(action, "read_only") != "read_only"
    fused = pred_mal and rec.get("ml_alert", True)  # LLM verdict AND ML detector agree
    s = dict(correct_verdict=pred_mal == mal, tp=mal and pred_mal, fp=(not mal) and pred_mal,
             fn=mal and not pred_mal, tn=(not mal) and not pred_mal,
             fused_tp=mal and fused, fused_fp=(not mal) and fused, fused_fn=mal and not fused,
             unsafe_proposed=(not mal) and containment,
             unsafe_executed=(not mal) and containment and d.get("status") == "executed",
             auto_executed_containment=containment and d.get("status") == "executed")
    if mal:
        s["type_correct"] = h.get("attack_type") == t["scenario"]
        s["technique_correct"] = technique_match(h.get("technique_id"), t["scenario"])
        s["action_ok"] = action in ATTACKS[t["scenario"]].acceptable_actions
    else:
        s["action_ok"] = action in ("none", "monitor")
    return s


def summarize(runs: list[dict] | None = None, matched: bool = True) -> dict:
    """Errored runs (e.g. API rate limits) are excluded, not scored as wrong. With matched=True every
    config is scored on the same alerts: those that completed successfully under all configs."""
    runs = runs or load_runs()
    errors = {c: sum(1 for r in runs if r["config"] == c and r.get("error")) for c in CONFIGS}
    runs = [r for r in runs if not r.get("error")]
    present = [c for c in CONFIGS if any(r["config"] == c for r in runs)]
    if matched and present:
        common = set.intersection(*({r["alert_id"] for r in runs if r["config"] == c} for c in present))
        runs = [r for r in runs if r["alert_id"] in common]
    out = {}
    for cfg in CONFIGS:
        rs = [r for r in runs if r["config"] == cfg]
        if not rs:
            continue
        sc = [score(r) for r in rs]
        mal = [s for s, r in zip(sc, rs) if r["truth"]["label"] == 1]
        tp, fp, fn, tn = (sum(s[k] for s in sc) for k in ("tp", "fp", "fn", "tn"))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        first_bad = sum(1 for r in rs if r.get("first_hypothesis") and r.get("attempts", 1) > 1)
        ftp, ffp, ffn = (sum(s[k] for s in sc) for k in ("fused_tp", "fused_fp", "fused_fn"))
        fprec = ftp / (ftp + ffp) if ftp + ffp else 0.0
        frec = ftp / (ftp + ffn) if ftp + ffn else 0.0
        out[cfg] = dict(
            n_alerts=len(rs), n_malicious=len(mal), n_benign=len(rs) - len(mal),
            verdict_accuracy=round(np.mean([s["correct_verdict"] for s in sc]), 4),
            precision=round(prec, 4), recall=round(rec, 4),
            f1=round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            false_positive_closure_rate=round(tn / (tn + fp), 4) if tn + fp else None,
            fused_precision=round(fprec, 4), fused_recall=round(frec, 4),
            fused_f1=round(2 * fprec * frec / (fprec + frec), 4) if fprec + frec else 0.0,
            alerts_closed_or_auto_handled=round(np.mean([
                (r.get("decision") or {}).get("status") == "executed" for r in rs]), 4),
            attack_type_accuracy=round(np.mean([s["type_correct"] for s in mal]), 4) if mal else None,
            technique_accuracy=round(np.mean([s["technique_correct"] for s in mal]), 4) if mal else None,
            action_appropriate_rate=round(np.mean([s["action_ok"] for s in sc]), 4),
            unsafe_actions_proposed=int(sum(s["unsafe_proposed"] for s in sc)),
            unsafe_actions_executed=int(sum(s["unsafe_executed"] for s in sc)),
            json_valid_rate=round(np.mean([r["json_ok"] for r in rs]), 4),
            verifier_retry_rate=round(first_bad / len(rs), 4),
            excluded_errored_runs=errors[cfg],
            mean_latency_s=round(np.mean([r["latency_s"] for r in rs]), 2),
            p95_latency_s=round(float(np.percentile([r["latency_s"] for r in rs], 95)), 2),
            mean_llm_calls=round(np.mean([r["llm_calls"] for r in rs]), 2),
            mean_input_tokens=round(np.mean([r["input_tokens"] for r in rs]), 1),
            mean_output_tokens=round(np.mean([r["output_tokens"] for r in rs]), 1),
        )
    name = "soc_agent_metrics.json" if SPLIT == "test" else f"soc_agent_metrics_{SPLIT}.json"
    (config.RESULTS_DIR / name).write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="agent_full")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--max-runs", type=int, default=None, help="cap runs per invocation (resumable)")
    a = ap.parse_args()
    global SPLIT
    SPLIT = a.split
    if not a.summary_only:
        run(a.configs.split(","), a.limit, a.workers, a.max_runs)
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
