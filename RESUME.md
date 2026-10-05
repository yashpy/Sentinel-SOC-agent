# Resume entry

**Sentinel: Autonomous SOC Agent** | Python, LangGraph, XGBoost, RAG, Groq, Streamlit
[github.com/yashpy/Sentinel-SOC-agent](https://github.com/yashpy/Sentinel-SOC-agent) · [Live demo](<STREAMLIT_URL>)

- Built an autonomous security-operations agent in **LangGraph** (parallel `Send()` investigators, verifier with a self-correction loop, `interrupt()` human approval, SQLite checkpointing). On held-out alerts it raised verdict accuracy from **60% to 78%** and ATT&CK technique accuracy from **24% to 60%** compared with a single-shot LLM over raw logs.
- Designed tiered response guardrails (read-only / reversible / destructive). They cut unsafe containment actions executed on benign users from **11 to 0**, and the agent auto-closed **60%** of false-positive alerts compared with 13% for the baseline.
- Trained an **XGBoost** threat detector on 34 per-user behavioral features with a time-based split, plus TreeSHAP explanations. It reached **98% precision at 96% recall** (PR-AUC 0.99), with **96% fewer false positives** and 35% fewer alerts than a SIEM-rules baseline.
- Built **hybrid RAG** over 697 MITRE ATT&CK techniques and SOC runbooks (BM25 + bge-small embeddings, reciprocal rank fusion). Benchmarked 4 retrievers on 40 queries: **78% Hit@1, 100% Recall@5**. The cross-encoder reranker made results worse, so it was left out.
- Wrote a reproducible simulator (202k audit events, 8 attack types, 7 benign look-alikes) and an eval harness with ablations and per-alert cost/latency tracking. Deployed a Streamlit dashboard for live detection and retrieval.

**Skills:** LLMs, RAG (hybrid retrieval, embeddings, reranking, retrieval eval), LangGraph/LangChain, agentic workflows, prompt and context engineering, LLM evaluation, XGBoost, scikit-learn, SHAP, pandas, Streamlit, Git.

## Interview notes (be ready to explain)
- **Why the agent beats one-shot:** tools turn ~100 raw log rows into ~15 facts compared against each user's own baseline. The verifier rejects ungrounded ATT&CK IDs and benign verdicts paired with containment.
- **Honest limitations:** simulated data, n=40 agent sample (free-tier rate limits), and LLM-alone precision of 0.79. The system is reliable because the LLM and ML detector have to agree before auto-containment.
- **No leakage:** features only use each user's past days. The threshold is chosen on out-of-fold training scores, and prompts were tuned on a separate dev split.
