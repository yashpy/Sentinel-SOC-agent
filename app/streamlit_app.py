"""Sentinel SOC: portfolio app.

Live: ML detector scoring + SHAP explanations, hybrid RAG search over MITRE ATT&CK.
Recorded: LangGraph agent investigations and evaluation metrics (no API key needed to view).
Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from sentinel import config  # noqa: E402
from sentinel.ml.features import FEATURE_COLUMNS  # noqa: E402

st.set_page_config(page_title="Sentinel SOC Agent", page_icon="🛡️", layout="wide")


def load_json(name: str) -> dict:
    p = config.RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def scored() -> pd.DataFrame:
    return pd.read_parquet(config.GEN_DIR / "scored_test.parquet")


@st.cache_data
def agent_runs() -> list[dict]:
    p = config.RESULTS_DIR / "soc_runs.jsonl"
    if not p.exists():
        return []
    latest = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["config"], r["alert_id"])] = r
    return list(latest.values())


@st.cache_resource
def kb():
    from sentinel.rag.retriever import get_kb

    return get_kb()


det, rag, agent = load_json("detector_metrics.json"), load_json("rag_metrics.json"), load_json("soc_agent_metrics.json")

st.title("🛡️ Sentinel: Autonomous SOC Agent")
st.caption("ML threat detection → LangGraph multi-agent investigation → RAG over MITRE ATT&CK → "
           "verified, human-gated response. All numbers below are measured on held-out data.")

tab_overview, tab_alerts, tab_agent, tab_rag, tab_arch = st.tabs(
    ["📊 Results", "🚨 Alert queue (live ML)", "🤖 Agent investigations", "🔎 ATT&CK search (live RAG)", "🏗️ Architecture"])

# ------------------------------------------------------------------ results
with tab_overview:
    if det:
        h = det["hybrid"]
        c = st.columns(4)
        c[0].metric("Detector precision", f"{h['precision']:.0%}", f"{h['precision'] - det['rules']['precision']:+.0%} vs rules")
        c[1].metric("Detector recall", f"{h['recall']:.0%}")
        c[2].metric("False positives", h["false_positives"], f"{h['false_positives'] - det['rules']['false_positives']} vs rules",
                    delta_color="inverse")
        c[3].metric("XGBoost PR-AUC", f"{det['xgboost']['pr_auc']:.3f}")
        st.subheader("Detection: rules vs anomaly detection vs supervised ML (held-out future days)")
        rows = [dict(model=k, **{m: det[k][m] for m in ("precision", "recall", "f1", "false_positives", "alerts")})
                for k in ("rules", "isolation_forest", "xgboost", "hybrid")]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Split: train days {det['split']['train_days']}, test days {det['split']['test_days']} "
                   f"({det['split']['test_user_days']:,} user-days, {det['split']['test_malicious']} attacks, "
                   f"{det['split']['test_benign_lookalikes']} benign look-alikes). Threshold frozen from out-of-fold training scores.")
    if agent:
        st.subheader("Agent: LangGraph agent vs single-shot LLM on raw logs")
        keys = ["verdict_accuracy", "precision", "recall", "f1", "false_positive_closure_rate", "attack_type_accuracy",
                "technique_accuracy", "action_appropriate_rate", "unsafe_actions_proposed", "unsafe_actions_executed",
                "mean_input_tokens", "mean_latency_s", "n_alerts"]
        st.dataframe(pd.DataFrame({k: {m: v.get(m) for m in keys} for k, v in agent.items()}), width="stretch")
    if rag:
        st.subheader("Retrieval quality (40 analyst queries → ATT&CK technique)")
        rr = [dict(mode=m, **rag[m]) for m in ("bm25", "dense", "hybrid", "hybrid_rerank") if m in rag]
        st.dataframe(pd.DataFrame(rr), hide_index=True, width="stretch")

# ------------------------------------------------------------------ alerts
with tab_alerts:
    df = scored()
    alerts = df[(df.alert == 1) | (df.rule_alert == 1)].sort_values("score", ascending=False)
    st.write(f"**{len(alerts)}** alerts in the test period ({int((df.alert == 1).sum())} from the ML detector, "
             f"{int((df.rule_alert == 1).sum())} from SIEM rules).")
    show_truth = st.toggle("Reveal ground truth (for evaluation only)", value=False)
    cols = ["user_id", "day", "score", "alert", "rule_alert"] + (["scenario", "label"] if show_truth else [])
    st.dataframe(alerts[cols], hide_index=True, width="stretch", height=260)
    pick = st.selectbox("Explain an alert", alerts.apply(lambda r: f"{r.user_id} day {int(r.day)}", axis=1))
    if pick:
        u, d = pick.split(" day ")
        r = alerts[(alerts.user_id == u) & (alerts.day == int(d))].iloc[0]
        contrib = pd.DataFrame({"feature": FEATURE_COLUMNS, "value": [r[c] for c in FEATURE_COLUMNS],
                                "shap": [r[f"shap_{c}"] for c in FEATURE_COLUMNS]})
        contrib = contrib.reindex(contrib.shap.abs().sort_values(ascending=False).index).head(10)
        st.plotly_chart(px.bar(contrib[::-1], x="shap", y="feature", orientation="h", hover_data=["value"],
                               title=f"Why the model scored {u} at {r.score:.3f} (TreeSHAP contributions)"),
                        width="stretch")

# ------------------------------------------------------------------ agent
with tab_agent:
    runs = [r for r in agent_runs() if r["config"] == "agent_full" and r.get("hypothesis")]
    if not runs:
        st.info("No recorded agent runs yet. Run `python -m sentinel.eval.soc_eval`.")
    else:
        st.write(f"{len(runs)} recorded investigations by the LangGraph agent (gpt-oss-120b on Groq).")
        table = pd.DataFrame([dict(alert=r["alert_id"], truth=r["truth"]["scenario"], verdict=r["hypothesis"]["verdict"],
                                   attack_type=r["hypothesis"]["attack_type"], technique=r["hypothesis"]["technique_id"],
                                   action=(r["decision"] or {}).get("action"), status=(r["decision"] or {}).get("status"),
                                   attempts=r["attempts"], latency_s=r["latency_s"]) for r in runs])
        st.dataframe(table, hide_index=True, width="stretch", height=300)
        sel = st.selectbox("Open investigation", table.alert)
        r = next(x for x in runs if x["alert_id"] == sel)
        h = r["hypothesis"]
        c = st.columns(4)
        c[0].metric("Verdict", h["verdict"])
        c[1].metric("ATT&CK", h["technique_id"] or "n/a")
        c[2].metric("Action", (r["decision"] or {}).get("action"))
        c[3].metric("Status", (r["decision"] or {}).get("status"))
        st.markdown(f"**Rationale:** {h['rationale']}")
        st.markdown(f"**Cited evidence:** {', '.join(h['evidence_ids'])}  \n**Retrieved:** {', '.join(r['retrieved'])}")
        if r.get("critiques"):
            st.warning("Verifier rejected the first answer: " + "; ".join(r["critiques"]))
        st.caption(f"Ground truth: {r['truth']['scenario']}")

# ------------------------------------------------------------------ rag
with tab_rag:
    q = st.text_input("Describe suspicious activity", "user phone flooded with push notifications until they hit approve")
    mode = st.radio("Retriever", ["hybrid", "dense", "bm25"], horizontal=True)
    if q:
        with st.spinner("Searching 706 documents..."):
            hits = kb().search(q, k=5, mode=mode)
        for h_ in hits:
            tid = h_["techniques"][0] if h_["techniques"] else ""
            url = f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/" if h_["kind"] == "technique" else ""
            st.markdown(f"**{h_['title']}** · _{h_['kind']}_ {f'[↗]({url})' if url else ''}")
            st.caption(h_["text"][:400] + "…")

# ------------------------------------------------------------------ architecture
with tab_arch:
    st.code("""
 audit logs ──► feature store (user-day vs own baseline, no leakage)
              ──► XGBoost + critical rules ──► alert (+ TreeSHAP signals)
                                                    │
 LangGraph ┌───────────────────────────────────────▼──────────────────────┐
           │ triage ─► Send() fan-out ─► investigators ×4 (read-only tools)│
           │        ─► collect ─► hybrid RAG (BM25 + bge-small, RRF)       │
           │        ─► analyst LLM ─► verifier ──reject──► analyst (retry) │
           │                              └─pass─► tiered action gate      │
           │                                       (destructive ⇒ interrupt│
           │                                        for human approval)    │
           │        ─► incident report      SQLite checkpointing            │
           └───────────────────────────────────────────────────────────────┘
""", language="text")
