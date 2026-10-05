# 🛡️ Sentinel: Autonomous SOC Agent

An autonomous security-operations agent for SaaS audit logs, similar in spirit to Salesforce Agentforce in Security Center. An ML detector raises alerts. A **LangGraph** multi-agent workflow investigates each alert with read-only tools and grounds its reasoning in **MITRE ATT&CK through hybrid RAG**. A deterministic verifier checks the answer, and response actions go through a tiered, human-approval gate.

All numbers below are **measured on held-out data**. The scripts to reproduce them are listed at the end.

## Architecture
```
 audit logs ─► feature store (each user-day vs that user's own past, no leakage)
            ─► XGBoost + critical rules ─► alert (+ TreeSHAP signals)
                                                  │
 LangGraph ┌─────────────────────────────────────▼───────────────────────┐
           │ triage ─► Send() fan-out ─► 4 investigators (read-only tools)│
           │        ─► collect ─► hybrid RAG (BM25 + bge-small, RRF)      │
           │        ─► analyst LLM ─► verifier ──reject──► analyst (retry)│
           │                             └─pass─► tiered action gate      │
           │                     (destructive ⇒ interrupt() for a human)  │
           │        ─► incident report             SQLite checkpointing   │
           └──────────────────────────────────────────────────────────────┘
```

## Results

**1. ML threat detection.** Data: 202k simulated audit events, 400 users, 289 labeled incidents (8 attack types plus 7 benign look-alikes). Time-based split, tested on days 30–43: 3,846 user-days, 51 attacks, 50 benign look-alikes. The alert threshold was frozen using out-of-fold training scores.

| Model | Precision | Recall | False positives | Alerts |
|---|---|---|---|---|
| SIEM rules baseline | 0.64 | 0.96 | 28 | 77 |
| IsolationForest | 0.19 | 0.94 | 200 | 248 |
| **XGBoost + critical rules** | **0.98** | **0.96** | **1** | **50** |

That's 96% fewer false positives and 35% fewer alerts than the rules baseline at the same recall. XGBoost PR-AUC is 0.991.

**2. Retrieval (RAG).** Corpus: 697 active ATT&CK techniques + 9 SOC runbooks. Tested on 40 analyst-style queries.

| Mode | Hit@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| BM25 | 0.55 | 0.85 | 0.66 |
| Dense (bge-small) | 0.78 | 1.00 | 0.86 |
| Hybrid (RRF) | 0.75 | 0.93 | 0.84 |
| Hybrid + cross-encoder rerank | 0.65 | 0.93 | 0.77 |

The reranker made results worse on this domain, so it's off by default.

**3. Agent vs single-shot LLM on raw logs.** 40-alert random sample of the held-out alert queue (25 attacks, 15 benign). Both use the same model, gpt-oss-120b on Groq.

| Metric | Raw-log LLM | **LangGraph agent** |
|---|---|---|
| Verdict accuracy | 0.60 | **0.78** |
| Precision | 0.63 | **0.79** |
| False positives correctly closed | 13% | **60%** |
| Attack-type accuracy | 0.36 | **0.60** |
| ATT&CK technique accuracy | 0.24 | **0.60** |
| Unsafe actions *executed* on benign users | 11 | **0** |
| Precision when the agent and ML detector agree | 0.95 | **1.00** (recall 0.88) |
| Mean latency / input tokens | 25 s / 1.6k | 28 s / 2.0k |

**Limitations:** the agent sample is small (n=40), and the free-tier rate limits caused a few failed runs (1 agent, 3 baseline). The data is simulated.

## Run it
```bash
pip install -r requirements.txt
python -m sentinel.data.generator && python -m sentinel.ml.features && python -m sentinel.ml.detector
python -m sentinel.rag.evaluate
GROQ_API_KEY=... SENTINEL_LLM_STRONG=groq:openai/gpt-oss-120b \
  python -m sentinel.eval.soc_eval --configs agent_full,naive_llm --limit 40
streamlit run app/streamlit_app.py
```
The agent code works with any model provider: `ollama:<model>`, `groq:<model>`, or any spec that LangChain's `init_chat_model` accepts.
