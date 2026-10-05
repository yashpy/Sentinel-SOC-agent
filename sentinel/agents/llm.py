"""Provider-agnostic LLM access with JSON output, usage accounting and retries."""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache

from sentinel import config


@lru_cache(maxsize=8)
def get_llm(spec: str, max_tokens: int = 400):
    """spec: 'ollama:<model>' for local models, otherwise any langchain init_chat_model spec."""
    if spec.startswith("ollama:"):
        from langchain_ollama import ChatOllama

        return ChatOllama(model=spec.split(":", 1)[1], temperature=0, format="json", num_ctx=8192,
                          num_predict=max_tokens, keep_alive="30m")
    if spec.startswith("groq:"):
        from langchain_groq import ChatGroq

        model = spec.split(":", 1)[1]
        extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
        return ChatGroq(model=model, temperature=0, max_tokens=max_tokens + 600, max_retries=12,
                        model_kwargs={"response_format": {"type": "json_object"}}, **extra)
    from langchain.chat_models import init_chat_model

    return init_chat_model(spec, temperature=0, max_tokens=max_tokens)


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def call_json(spec: str, system: str, user: str, node: str, max_tokens: int = 400, retries: int = 1) -> tuple[dict, dict]:
    """Returns (parsed_json, trace_record). Retries once on unparseable output."""
    llm = get_llm(spec, max_tokens)
    msgs = [("system", system), ("user", user)]
    t0 = time.perf_counter()
    tin = tout = 0
    data: dict = {}
    for _ in range(retries + 1):
        resp = llm.invoke(msgs)
        um = getattr(resp, "usage_metadata", None) or {}
        tin += um.get("input_tokens", 0)
        tout += um.get("output_tokens", 0)
        data = parse_json(resp.content if isinstance(resp.content, str) else str(resp.content))
        if data:
            break
    return data, dict(node=node, model=spec, latency_s=round(time.perf_counter() - t0, 2),
                      input_tokens=tin, output_tokens=tout, parsed=bool(data))


DEFAULT_STRONG = config.LLM_STRONG
DEFAULT_FAST = config.LLM_FAST
