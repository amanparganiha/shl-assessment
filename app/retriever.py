"""Hybrid retrieval over the catalog.

Combines:
  - BM25 (lexical) — nails exact skill/product tokens like 'docker', 'hipaa', 'opq32r'.
  - Dense cosine (semantic) — catches fuzzy intent like 'reliability' -> DSI.
fused with Reciprocal Rank Fusion (rank-based, so no score normalisation needed).

Two domain-aware boosts encode real SHL knowledge rather than relying on the model:
  - ALIASES: expand abbreviations/jargon (OPQ, GSA, SJT, JS, cognitive...) for lexical recall.
  - STAPLES: cross-cutting instruments (OPQ32r personality, Verify G+ cognitive,
    Graduate Scenarios SJT, DSI safety) are always added to the candidate pool so the
    agent can include them for a role even when a skill-only query wouldn't surface them.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .catalog import Catalog, Product
from .config import settings
from .llm import embed_one

logger = logging.getLogger("shl.retriever")

_TOKEN = re.compile(r"[a-z0-9][a-z0-9#+.\-]*")

# Abbreviation / synonym expansion. Keys are matched as whole words (case-insensitive).
ALIASES: dict[str, str] = {
    "opq": "occupational personality questionnaire opq32r",
    "gsa": "global skills assessment",
    "dsi": "dependability safety instrument",
    "mq": "motivation questionnaire",
    "svar": "spoken english svar",
    "sjt": "situational judgment scenarios",
    "js": "javascript",
    "ml": "machine learning",
    "ai": "artificial intelligence machine learning",
    "qa": "quality assurance testing",
    "cognitive": "cognitive ability aptitude reasoning verify general",
    "aptitude": "ability aptitude reasoning verify",
    "reasoning": "ability aptitude reasoning verify",
    "iq": "ability aptitude reasoning general",
    "gma": "general mental ability reasoning verify",
    "personality": "personality behavior occupational questionnaire opq",
    "behavioural": "personality behavior",
    "behavioral": "personality behavior",
    "leadership": "leadership executive director manager",
    "executive": "leadership executive director senior",
    "cxo": "executive leadership director senior",
    "sales": "sales selling commercial",
    "safety": "safety dependability reliability compliance",
    "reliability": "dependability safety reliability",
    "dependability": "dependability safety reliability",
    "contact": "contact center customer service call",
    "callcenter": "contact center customer service call simulation",
    "csr": "customer service contact center",
    "developer": "developer engineer programming software",
    "engineer": "engineer developer programming",
    "frontend": "frontend javascript react angular html css",
    "backend": "backend server java python sql api",
    "fullstack": "full stack frontend backend",
    "graduate": "graduate entry-level campus early-careers",
    "fresher": "graduate entry-level",
    "numerical": "numerical reasoning numeracy verify",
    "verbal": "verbal reasoning comprehension verify",
    "data": "data analysis analytics statistics",
}


# Cross-cutting instruments that the agent should be able to add to any battery.
STAPLE_NAMES: list[str] = [
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
    "Graduate Scenarios",
    "Dependability and Safety Instrument (DSI)",
    "Motivation Questionnaire MQM5",
]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def expand_query(query: str) -> str:
    """Append alias expansions for any recognised tokens (lexical recall aid)."""
    tokens = set(tokenize(query))
    extra: list[str] = []
    for tok in tokens:
        if tok in ALIASES:
            extra.append(ALIASES[tok])
    return query + (" " + " ".join(extra) if extra else "")


class Retriever:
    """Hybrid BM25 + dense retriever with RRF fusion."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.corpus_tokens = [tokenize(p.doc) for p in catalog.products]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.embeddings = catalog.embeddings  # (N, D) normalised or None
        self.staple_ids = self._resolve_staples()
        logger.info(
            "Retriever ready: %d docs, dense=%s, staples=%d",
            len(catalog),
            self.embeddings is not None,
            len(self.staple_ids),
        )

    def _resolve_staples(self) -> list[int]:
        ids: list[int] = []
        for name in STAPLE_NAMES:
            p = self.catalog.find_by_name(name)
            if p:
                ids.append(p.id)
            else:
                logger.warning("Staple not found in catalog: %s", name)
        return ids

    def _bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(scores)[::-1])

    def _dense_ranking(self, query: str) -> list[int] | None:
        if self.embeddings is None:
            return None
        try:
            qv = embed_one(query)  # (D,) normalised
        except Exception as exc:  # noqa: BLE001 - degrade to lexical-only
            logger.warning("Query embedding failed (%s); lexical-only this call", exc)
            return None
        sims = self.embeddings @ qv
        return list(np.argsort(sims)[::-1])

    @staticmethod
    def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return scores

    def retrieve(self, query: str, top_k: int | None = None) -> list[Product]:
        """Return up to top_k candidate products for a (already-built) query string.

        Staples are appended if missing so the agent always *can* add a personality
        / cognitive component. The list is intended for an LLM to choose from, not
        to be returned verbatim.
        """
        top_k = top_k or settings.retrieval_top_k
        if not query.strip():
            # No signal yet: return staples + a few generic items so /chat never breaks.
            base = list(self.staple_ids)
            base += [p.id for p in self.catalog.products[: max(0, top_k - len(base))]]
            seen: set[int] = set()
            return [self.catalog.get(i) for i in base if not (i in seen or seen.add(i))]

        expanded = expand_query(query)
        rankings = [self._bm25_ranking(expanded)]
        dense = self._dense_ranking(expanded)
        if dense is not None:
            rankings.append(dense)

        fused = self._rrf(rankings)
        ranked_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]

        # Ensure staples are present (append, don't displace better matches).
        ordered = list(ranked_ids)
        present = set(ordered)
        for sid in self.staple_ids:
            if sid not in present:
                ordered.append(sid)
                present.add(sid)

        return [self.catalog.get(i) for i in ordered if self.catalog.get(i)]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide singleton, built from the on-disk processed catalog."""
    return Retriever(Catalog.load())
