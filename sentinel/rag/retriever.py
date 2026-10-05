"""Hybrid RAG retriever over MITRE ATT&CK + SOC runbooks.

BM25 (lexical) + bge-small dense embeddings, fused with Reciprocal Rank Fusion, then
re-scored by a MiniLM cross-encoder. Embeddings are cached on disk.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from sentinel import config

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
EMB_CACHE = config.ARTIFACTS_DIR / "kb_embeddings.npy"
DOCS_CACHE = config.ARTIFACTS_DIR / "kb_docs.json"
_STOP = set("a an the of to and or in on for with by from is are was were be been it its this that as at "
            "into their they them his her he she you your we our not no can will may".split())
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Doc:
    doc_id: str
    kind: str  # technique | runbook
    title: str
    text: str
    techniques: list[str]


def _tok(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP]


def _clean(s: str) -> str:
    s = re.sub(r"\(Citation:[^)]*\)", "", s)
    s = re.sub(r"<code>|</code>|\[([^\]]+)\]\([^)]*\)", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


ATTACK_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"


def load_attack_docs() -> list[Doc]:
    if not config.ATTACK_STIX_PATH.exists():  # not committed (46 MB); fetched on first use
        import urllib.request

        config.ATTACK_STIX_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ATTACK_URL, config.ATTACK_STIX_PATH)
    bundle = json.loads(config.ATTACK_STIX_PATH.read_text())
    docs = []
    for o in bundle["objects"]:
        if o.get("type") != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        ext = next((r for r in o.get("external_references", []) if r.get("source_name") == "mitre-attack"), None)
        if not ext:
            continue
        tid = ext["external_id"]
        tactics = ", ".join(p["phase_name"] for p in o.get("kill_chain_phases", []))
        desc = _clean(o.get("description", ""))[:1500]
        docs.append(Doc(tid, "technique", f"{tid} {o['name']}", f"{o['name']}. Tactics: {tactics}. {desc}", [tid]))
    return docs


def load_runbook_docs() -> list[Doc]:
    docs = []
    for path in sorted(config.RUNBOOK_DIR.glob("*.md")):
        for block in re.split(r"^# ", path.read_text(), flags=re.M)[1:]:
            header, _, body = block.partition("\n")
            parts = [p.strip() for p in header.split("|")] + [""]
            rb_id, title, techs = parts[0], parts[1], parts[2]
            docs.append(Doc(rb_id, "runbook", f"{rb_id} {title}", f"{title}. {body.strip()}",
                            [t for t in techs.split() if t.startswith("T")]))
    return docs


class KnowledgeBase:
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self.docs = load_attack_docs() + load_runbook_docs()
        self.by_id = {d.doc_id: d for d in self.docs}
        self.bm25 = BM25Okapi([_tok(d.title + " " + d.text) for d in self.docs])
        self.embedder = TextEmbedding(EMBED_MODEL)
        ids = [d.doc_id for d in self.docs]
        if EMB_CACHE.exists() and DOCS_CACHE.exists() and json.loads(DOCS_CACHE.read_text()) == ids:
            self.emb = np.load(EMB_CACHE)
        else:
            self.emb = np.array(list(self.embedder.passage_embed([d.title + ". " + d.text for d in self.docs])))
            self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)
            np.save(EMB_CACHE, self.emb)
            DOCS_CACHE.write_text(json.dumps(ids))
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(RERANK_MODEL)
        return self._reranker

    def _bm25_rank(self, q: str) -> np.ndarray:
        return np.argsort(-self.bm25.get_scores(_tok(q)))

    def _dense_rank(self, q: str) -> np.ndarray:
        qv = np.array(next(iter(self.embedder.query_embed([q]))))
        return np.argsort(-(self.emb @ (qv / np.linalg.norm(qv))))

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", kind: str | None = None,
               pool: int = 30) -> list[dict]:
        if mode == "bm25":
            order = self._bm25_rank(query)
        elif mode == "dense":
            order = self._dense_rank(query)
        else:
            rrf = np.zeros(len(self.docs))
            for ranking in (self._bm25_rank(query), self._dense_rank(query)):
                rrf[ranking[:200]] += 1.0 / (60 + np.arange(1, 201))
            order = np.argsort(-rrf)
        if kind:
            order = np.array([i for i in order if self.docs[i].kind == kind])
        if mode == "hybrid_rerank":
            cand = order[:pool]
            scores = np.array(list(self.reranker.rerank(query, [self.docs[i].title + ". " + self.docs[i].text
                                                                for i in cand])))
            order = cand[np.argsort(-scores)]
        return [dict(doc_id=self.docs[i].doc_id, kind=self.docs[i].kind, title=self.docs[i].title,
                     techniques=self.docs[i].techniques, text=self.docs[i].text) for i in order[:k]]


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()
