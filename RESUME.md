# Resume entry

**Sentinel: Autonomous SOC Agent** | Python, LangGraph, XGBoost, RAG, Groq, Streamlit
[github.com/yashpy/Sentinel-SOC-agent](https://github.com/yashpy/Sentinel-SOC-agent) · [Live demo](<STREAMLIT_URL>)

- Built an autonomous security-operations agent in **LangGraph** (parallel `Send()` investigators, verifier with a self-correction loop, `interrupt()` human approval, SQLite checkpointing). It raised verdict accuracy from **68% to 88%** over a single-shot LLM on raw logs while using **23% fewer tokens**.
- Ran ablations on held-out alerts showing that tool-based context engineering drove the accuracy gain, while hybrid RAG improved MITRE ATT&CK technique mapping from **32% to 60%**.
- Designed tiered, human-gated response guardrails. Unsafe containment actions executed on benign users dropped from **11 to 0**, and agent+detector agreement reached **100% precision at 94% recall** across 80 alerts.
- Trained an **XGBoost** threat detector on 34 per-user behavioral features with a time-based split and TreeSHAP explanations. It reached **98% precision at 96% recall** (PR-AUC 0.99), with **96% fewer false positives** than a SIEM-rules baseline.
- Built **hybrid RAG** over 697 ATT&CK techniques and runbooks (BM25 + bge-small embeddings, reciprocal rank fusion). Benchmarked 4 retrievers: **78% Hit@1, 100% Recall@5**. Dropped a cross-encoder reranker that made results worse.
- Wrote a reproducible simulator (202k audit events, 8 attack types, 7 benign look-alikes) and a resumable eval harness tracking accuracy, cost and latency. Deployed a Streamlit dashboard.

**Skills:** LLMs, RAG (hybrid retrieval, embeddings, reranking, retrieval eval), LangGraph/LangChain, agentic workflows, context engineering, LLM evaluation and ablations, XGBoost, scikit-learn, SHAP, pandas, Streamlit, Git.

## Interview notes (be ready to explain)
- **Ablation finding:** structured tool facts beat raw logs (+20 points accuracy, fewer tokens). RAG + verifier improved ATT&CK mapping but cut recall (0.88 vs 1.00), because strict grounding pushes borderline attacks to "benign". Next step: let the tools-only path decide the verdict and use RAG only to label the technique.
- **Why it's safe:** auto-containment needs both the ML detector and a confident, verified LLM verdict. Destructive actions always wait for a human.
- **Honest limitations:** simulated data, and the matched comparison is n=40 because of free-tier limits (200K tokens/day).
- **No leakage:** features only use each user's past. The threshold is chosen on out-of-fold training scores, and prompts were tuned on a separate dev split.
