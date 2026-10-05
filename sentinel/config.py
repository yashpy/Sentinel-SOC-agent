"""Central paths and constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GEN_DIR = DATA_DIR / "generated"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
RUNBOOK_DIR = ROOT / "sentinel" / "rag" / "runbooks"

for _d in (GEN_DIR, ARTIFACTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EVENTS_PATH = GEN_DIR / "events.parquet"
USERS_PATH = GEN_DIR / "users.parquet"
INCIDENTS_PATH = GEN_DIR / "incidents.parquet"
FEATURES_PATH = GEN_DIR / "features.parquet"
ATTACK_STIX_PATH = RAW_DIR / "enterprise-attack.json"

# Columns that carry ground truth. They must never reach the detector or the agents.
LABEL_COLUMNS = ["incident_id", "scenario"]

# Simulation timeline: the first WARMUP_DAYS are attack-free history used only for baselines.
N_DAYS = 44
WARMUP_DAYS = 7
TRAIN_END_DAY = 29  # local days [WARMUP_DAYS, TRAIN_END_DAY] train; (TRAIN_END_DAY, N_DAYS) test

# LLM model specs, overridable by env. "ollama:<model>" runs locally; anything else goes to
# langchain's init_chat_model (e.g. "openai:gpt-4o-mini", "anthropic:claude-sonnet-4-5").
LLM_STRONG = os.getenv("SENTINEL_LLM_STRONG", "ollama:qwen2.5:3b")
LLM_FAST = os.getenv("SENTINEL_LLM_FAST", "ollama:qwen2.5:1.5b")

# Response actions grouped by risk tier.
ACTION_TIERS = {
    "none": "read_only",
    "monitor": "read_only",
    "revoke_sessions": "reversible",
    "reset_password": "reversible",
    "enforce_mfa": "reversible",
    "remove_permission": "destructive",
    "freeze_user": "destructive",
}
ACTIONS = list(ACTION_TIERS)
