"""Autonomous SOC agent built with LangGraph.

    alert ─► triage (fast LLM) ─► fan-out (Send) ─► investigators ×N (read-only tools, parallel)
          ─► collect ─► retrieve (hybrid RAG: ATT&CK + runbooks) ─► analyst (strong LLM)
          ─► verifier ──fail──► analyst (bounded retries with critique)
                      └─pass──► action_gate (tiered; destructive => interrupt for human approval)
          ─► report

State is checkpointed (SQLite) so an investigation paused for approval can be resumed later.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from sentinel import config
from sentinel.agents import llm as L
from sentinel.agents.tools import INVESTIGATORS, TOOLS, get_store, user_profile
from sentinel.data.scenarios import ATTACKS

MAX_ANALYST_ATTEMPTS = 2  # one self-correction round
AUTO_CONTAIN_MIN_CONFIDENCE = 0.8
ATTACK_TYPES = list(ATTACKS) + ["benign"]
ATTACK_GLOSSARY = "\n".join(f"- {k}: {v.description}" for k, v in ATTACKS.items())


class SOCState(TypedDict, total=False):
    alert: dict
    settings: dict
    profile: dict
    plan: list[str]
    triage_reason: str
    findings: Annotated[list[dict], operator.add]
    evidence: list[str]
    retrieved: list[dict]
    hypothesis: dict
    first_hypothesis: dict
    critique: str
    attempts: int
    verified: bool
    decision: dict
    report: str
    trace: Annotated[list[dict], operator.add]


DEFAULT_SETTINGS = dict(triage_model=config.LLM_FAST, analyst_model=config.LLM_STRONG, use_rag=True,
                        use_verifier=True, use_triage_llm=True, approval="interactive", rag_mode="hybrid")


def _s(state: SOCState) -> dict:
    return {**DEFAULT_SETTINGS, **(state.get("settings") or {})}


# ---------------------------------------------------------------- nodes
def triage(state: SOCState) -> dict:
    a, s = state["alert"], _s(state)
    profile = user_profile(get_store(), a["user_id"], a["day"])
    if not s["use_triage_llm"]:
        return dict(profile=profile, plan=list(INVESTIGATORS), triage_reason="static: run all")
    system = ("You are a SOC triage lead. Choose which investigations to run for an alert. "
              "Investigators: auth (logins, failures, MFA), data_access (exports, record views, API, tokens), "
              "privilege (permission grants, auth-policy changes), session (session reuse, impossible travel, "
              "datacenter IPs). Reply JSON: {\"investigators\": [..], \"reason\": \"...\"}. "
              "Pick every investigator that could be relevant; when unsure include it.")
    user = f"User profile: {profile}\nDetector signals (feature=value, contribution): {a.get('signals', [])}"
    data, tr = L.call_json(s["triage_model"], system, user, node="triage", max_tokens=120)
    plan = [i for i in data.get("investigators", []) if i in INVESTIGATORS] or list(INVESTIGATORS)
    if "auth" not in plan:  # cheap guardrail: identity context is always needed
        plan.insert(0, "auth")
    return dict(profile=profile, plan=plan, triage_reason=str(data.get("reason", ""))[:300], trace=[tr])


def fan_out(state: SOCState) -> list[Send]:
    return [Send("investigate", {"alert": state["alert"], "investigator": inv}) for inv in state["plan"]]


def investigate(payload: dict) -> dict:
    a, inv = payload["alert"], payload["investigator"]
    facts = TOOLS[inv](get_store(), a["user_id"], a["day"])
    return dict(findings=[dict(investigator=inv, facts=facts)],
                trace=[dict(node=f"investigate:{inv}", model="tool", n_facts=len(facts))])


def collect(state: SOCState) -> dict:
    order = {k: i for i, k in enumerate(INVESTIGATORS)}
    ev, n = [], 1
    for f in sorted(state["findings"], key=lambda f: order[f["investigator"]]):
        for fact in f["facts"]:
            ev.append(f"E{n} [{f['investigator']}] {fact}")
            n += 1
    return dict(evidence=ev, attempts=0)


def retrieve(state: SOCState) -> dict:
    s = _s(state)
    if not s["use_rag"]:
        return dict(retrieved=[])
    from sentinel.rag.retriever import get_kb

    kb = get_kb()
    query = retrieval_query(state["alert"], state["evidence"])
    runbooks = kb.search(query, k=3, mode=s["rag_mode"], kind="runbook")
    techniques = kb.search(query, k=4, mode=s["rag_mode"], kind="technique")
    # Runbook-guided expansion: techniques named by the top runbooks become first-class candidates.
    seen = {t["doc_id"] for t in techniques}
    linked_ids = dict.fromkeys(t for rb in runbooks for t in rb["techniques"] if t in kb.by_id and t not in seen)
    linked_hits = [dict(doc_id=d.doc_id, kind=d.kind, title=d.title, techniques=d.techniques, text=d.text)
                   for d in (kb.by_id[t] for t in list(linked_ids)[:4])]
    hits = runbooks + linked_hits + techniques
    return dict(retrieved=hits, trace=[dict(node="retrieve", model=f"rag:{s['rag_mode']}", query=query[:300],
                                            hits=[h["doc_id"] for h in hits])])


_ANOMALY_MARKERS = ("new IP", "new device", "new country", "NEW", "DENIED", "HIGH-RISK", "THEMSELVES", "WEAKENS",
                    "reused", "Impossible", "never seen", "SUCCESSFUL login from the same IP", "token")


def is_anomalous(fact: str) -> bool:
    import re

    if any(m in fact for m in _ANOMALY_MARKERS):
        return True
    return bool(re.search(r"ratio (\d{2,}|[3-9])\.\dx|\((\d{2,}|[3-9])\.\dx\)", fact))


def retrieval_query(alert: dict, evidence: list[str]) -> str:
    """Semantic query: anomalous facts with IPs/numbers stripped (they only add lexical noise)."""
    import re

    facts = [e.split("] ", 1)[1] for e in evidence if any(m in e for m in _ANOMALY_MARKERS)]
    facts += [e.split("] ", 1)[1] for e in evidence if re.search(r"ratio (\d{2,}|[5-9])\.\dx|\((\d{2,}|[5-9])\.\dx\)", e)]
    signals = [s.split("=")[0] for s in alert.get("signals", [])]
    text = " ".join(signals) + ". " + " ".join(dict.fromkeys(facts))
    text = re.sub(r"\b\d+(\.\d+){1,3}\b|\b[\w-]*\d[\w-]*\b|[(),;:=+]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:1000]


ACTION_HELP = ("none (benign, close), monitor (benign or low confidence), revoke_sessions, reset_password, "
               "enforce_mfa, remove_permission, freeze_user")


def analyst(state: SOCState) -> dict:
    s = _s(state)
    ctx = ""
    for h in state.get("retrieved", []):
        body = h["text"][:480] if h["kind"] == "runbook" else h["title"] + ": " + h["text"].split(". ", 2)[-1][:160]
        ctx += f"- [{h['doc_id']}] {body}\n"
    system = (
        "You are a senior SOC analyst investigating a SaaS audit-log alert. Roughly 40% of alerts in this queue "
        "are false positives (travel on a known device, VPN, new laptop, forgotten password from a known IP, "
        "quarter-end reports from the corporate network, admin onboarding of OTHER users, CI tokens in the home "
        "country). One unusual fact alone is not an attack; attacks combine unfamiliar access with abuse. "
        "Decide from the EVIDENCE only. Compare against the user's own baseline and the runbooks' benign "
        "look-alikes. 'known IP/device/country' facts are benign context. First list the indicators, then "
        "decide. Choose technique_id ONLY from the knowledge-base IDs shown.\nATTACK TYPES:\n"
        + ATTACK_GLOSSARY + "\nReply with JSON only:\n"
        '{"risk_indicators": ["E# short"], "benign_indicators": ["E# short"], '
        '"verdict": "malicious"|"benign", "attack_type": one of ' + str(ATTACK_TYPES) + ', '
        '"technique_id": "MITRE ATT&CK ID like T1078 or null if benign", "confidence": 0.0-1.0, '
        '"action": one of [' + ACTION_HELP + '], "evidence_ids": ["E1", ...], "rationale": "<= 3 sentences"}'
    )
    user = (f"User profile: {state['profile']}\nDetector signals: {state['alert'].get('signals', [])}\n\n"
            "EVIDENCE:\n" + "\n".join(state["evidence"]) +
            (f"\n\nKNOWLEDGE BASE (retrieved):\n{ctx}" if ctx else "") +
            (f"\n\nYOUR PREVIOUS ANSWER WAS REJECTED BY THE VERIFIER: {state['critique']}\nFix it." if state.get("critique") else ""))
    data, tr = L.call_json(s["analyst_model"], system, user, node="analyst", max_tokens=450)
    hyp = resolve_technique(normalize_hypothesis(data), state.get("retrieved", []))
    out = dict(hypothesis=hyp, attempts=state.get("attempts", 0) + 1, trace=[tr])
    if not state.get("first_hypothesis"):
        out["first_hypothesis"] = hyp
    return out


def normalize_hypothesis(d: dict) -> dict:
    verdict = str(d.get("verdict", "")).lower().strip()
    action = str(d.get("action", "")).lower().strip()
    tid = d.get("technique_id")
    tid = str(tid).strip().upper() if tid and str(tid).lower() not in ("null", "none", "") else None
    try:
        conf = float(d.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    ev = d.get("evidence_ids") or []
    return dict(verdict=verdict, attack_type=str(d.get("attack_type", "")).strip(), technique_id=tid,
                confidence=max(0.0, min(conf, 1.0)), action=action,
                evidence_ids=[str(e) for e in ev] if isinstance(ev, list) else [], rationale=str(d.get("rationale", ""))[:600])


RUNBOOK_FOR_TYPE = {"brute_force": "RB-001", "account_takeover": "RB-002", "privilege_escalation": "RB-003",
                    "insider_exfiltration": "RB-004", "token_abuse": "RB-005", "mfa_fatigue": "RB-006",
                    "session_hijack": "RB-007", "auth_policy_tamper": "RB-008"}


def resolve_technique(h: dict, retrieved: list[dict]) -> dict:
    """Ground the ATT&CK ID in the runbook catalog: keep the LLM's ID if it is (a sub-technique of) one the
    runbook for the chosen attack type lists; otherwise fall back to that runbook's primary technique."""
    if h["verdict"] != "malicious" or h["attack_type"] not in RUNBOOK_FOR_TYPE:
        return h
    from sentinel.rag.retriever import get_kb

    rb = get_kb().by_id[RUNBOOK_FOR_TYPE[h["attack_type"]]]
    llm_tid = h.get("technique_id")
    if llm_tid and llm_tid.split(".")[0] in rb.techniques:
        h["technique_source"] = "llm"
        return h
    h["technique_id_llm"] = llm_tid
    h["technique_id"] = rb.techniques[0]
    h["technique_source"] = f"runbook:{rb.doc_id}"
    return h


def verify(state: SOCState) -> dict:
    """Deterministic guardrails: schema, ATT&CK validity, grounding, verdict/action consistency."""
    s, h = _s(state), state["hypothesis"]
    problems = []
    if h["verdict"] not in ("malicious", "benign"):
        problems.append("verdict must be 'malicious' or 'benign'")
    if h["action"] not in config.ACTIONS:
        problems.append(f"action must be one of {config.ACTIONS}")
    if h["verdict"] == "benign" and h["action"] not in ("none", "monitor"):
        problems.append("a benign verdict cannot trigger a containment action; use none or monitor")
    if h["verdict"] == "malicious":
        if h["action"] in ("none",):
            problems.append("a malicious verdict needs a response action")
        from sentinel.rag.retriever import get_kb

        valid = {d.doc_id for d in get_kb().docs if d.kind == "technique"}
        grounded = {t for r in state.get("retrieved", []) for t in r["techniques"]}
        if not h["technique_id"] or h["technique_id"] not in valid:
            problems.append(f"technique_id '{h['technique_id']}' is not a valid, current MITRE ATT&CK technique")
        elif grounded and h["technique_id"] not in grounded:
            problems.append(f"technique_id '{h['technique_id']}' is not among the retrieved knowledge-base "
                            f"techniques {sorted(grounded)}; pick the best-supported one")
        anomalous = {e.split(" ")[0] for e in state["evidence"] if is_anomalous(e)}
        if not anomalous:
            problems.append("no evidence item shows an anomaly versus the user's baseline; reconsider a benign verdict")
        elif not any(e in anomalous for e in h["evidence_ids"]):
            problems.append(f"the cited evidence is not anomalous; cite the anomalous facts ({sorted(anomalous)}) "
                            "or change the verdict to benign")
    if not s["use_verifier"] or not problems:
        return dict(verified=not problems, critique="")
    return dict(verified=False, critique="; ".join(problems),
                trace=[dict(node="verifier", model="rules", problems=problems)])


def route_after_verify(state: SOCState) -> str:
    s = _s(state)
    if state["verified"] or not s["use_verifier"] or state["attempts"] >= MAX_ANALYST_ATTEMPTS:
        return "action_gate"
    return "analyst"


def action_gate(state: SOCState) -> dict:
    s, h = _s(state), state["hypothesis"]
    action = h["action"] if h["action"] in config.ACTIONS else "monitor"
    if not state.get("verified") and s["use_verifier"]:
        action = "monitor"  # failed verification => no automated containment, escalate to a human
    tier = config.ACTION_TIERS[action]
    # Defense in depth: auto-contain only when two independent signals agree (ML detector fired AND the
    # verified LLM verdict is confident). Everything else waits for a human.
    ml_fired = bool(state["alert"].get("ml_alert", True))
    if tier == "read_only":
        status = "executed"
    elif (tier == "reversible" and state.get("verified") and ml_fired
          and h["confidence"] >= AUTO_CONTAIN_MIN_CONFIDENCE):
        status = "executed"
    else:
        if s["approval"] == "interactive":
            answer = interrupt(dict(question=f"Approve '{action}' on {state['alert']['user_id']}?",
                                    hypothesis=h, tier=tier))
            status = "executed" if str(answer).lower() in ("approve", "approved", "yes", "y") else "rejected"
        else:
            status = "pending_approval"  # batch/eval mode: queue for a human, never auto-execute
    return dict(decision=dict(action=action, tier=tier, status=status, needs_review=not state.get("verified", True)))


def report(state: SOCState) -> dict:
    a, h, d = state["alert"], state["hypothesis"], state["decision"]
    tid = h.get("technique_id")
    link = f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/" if tid else "n/a"
    cited = [f"- {e}" for e in state["evidence"] if e.split(" ")[0] in h["evidence_ids"]] or ["- (none cited)"]
    lines = [
        f"# Incident report: {a['user_id']} (day {a['day']})",
        f"**Verdict:** {h['verdict']} ({h['attack_type']}), confidence {h['confidence']:.2f}",
        f"**MITRE ATT&CK:** {tid or 'n/a'} ({link})",
        f"**Action:** {d['action']} [{d['tier']}], status: **{d['status']}**"
        + (" (verifier failed, routed to human)" if d.get("needs_review") else ""),
        "", "## Rationale", h["rationale"], "", "## Evidence",
        *cited,
        "", "## Investigation trail", f"- Triage plan: {state.get('plan')} ({state.get('triage_reason', '')})",
        f"- Retrieved: {[r['doc_id'] for r in state.get('retrieved', [])]}",
        f"- Analyst attempts: {state.get('attempts')}",
    ]
    return dict(report="\n".join(lines))


# ---------------------------------------------------------------- graph
def build_graph(checkpointer: Any = None):
    g = StateGraph(SOCState)
    g.add_node("triage", triage)
    g.add_node("investigate", investigate)
    g.add_node("collect", collect)
    g.add_node("retrieve", retrieve)
    g.add_node("analyst", analyst)
    g.add_node("verify", verify)
    g.add_node("action_gate", action_gate)
    g.add_node("report", report)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", fan_out, ["investigate"])
    g.add_edge("investigate", "collect")
    g.add_edge("collect", "retrieve")
    g.add_edge("retrieve", "analyst")
    g.add_edge("analyst", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["analyst", "action_gate"])
    g.add_edge("action_gate", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path=None):
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(str(path or config.ARTIFACTS_DIR / "checkpoints.sqlite"), check_same_thread=False)
    return SqliteSaver(conn)


__all__ = ["build_graph", "sqlite_checkpointer", "SOCState", "Command"]
