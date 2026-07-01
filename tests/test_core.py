"""Unit tests for the deterministic core (no network / API key needed).

Covers the parts that must never regress: test_type derivation, the grounding /
schema-enforcement in response assembly, query tokenisation/alias expansion, and
trace parsing.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import _assemble, _build_query, _coerce_ids  # noqa: E402
from app.catalog import Product, derive_test_type  # noqa: E402
from app.retriever import expand_query, tokenize  # noqa: E402
from app.schemas import ChatRequest, Message  # noqa: E402


def _p(pid: int, name: str, test_type: str = "K") -> Product:
    return Product(
        id=pid, entity_id=str(pid), name=name,
        url=f"https://www.shl.com/products/product-catalog/view/{name.lower().replace(' ','-')}/",
        test_type=test_type,
    )


# ---- test_type derivation ----
def test_derive_single():
    assert derive_test_type(["Personality & Behavior"]) == "P"
    assert derive_test_type(["Knowledge & Skills"]) == "K"


def test_derive_multi_preserves_order_and_dedupes():
    assert derive_test_type(["Competencies", "Knowledge & Skills"]) == "C, K"
    assert derive_test_type(["Knowledge & Skills", "Knowledge & Skills"]) == "K"


def test_derive_unknown_key_skipped():
    assert derive_test_type(["Nonexistent Category"]) == ""
    assert derive_test_type([]) == ""


# ---- response assembly: grounding + schema enforcement ----
def _cands():
    ps = [_p(0, "Alpha"), _p(1, "Beta", "P"), _p(2, "Gamma")]
    return {p.id: p for p in ps}


def test_recommend_maps_ids_to_catalog():
    resp = _assemble(
        {"action": "recommend", "reply": "ok", "recommended_ids": [0, 2], "end_of_conversation": False},
        _cands(),
    )
    assert [r.name for r in resp.recommendations] == ["Alpha", "Gamma"]
    assert all(r.url.startswith("https://www.shl.com/") for r in resp.recommendations)


def test_clarify_forces_empty_recs():
    resp = _assemble(
        {"action": "clarify", "reply": "which role?", "recommended_ids": [0, 1], "end_of_conversation": False},
        _cands(),
    )
    assert resp.recommendations == []


def test_refuse_forces_empty_recs():
    resp = _assemble(
        {"action": "refuse", "reply": "out of scope", "recommended_ids": [0], "end_of_conversation": False},
        _cands(),
    )
    assert resp.recommendations == []


def test_invalid_ids_dropped_not_hallucinated():
    resp = _assemble(
        {"action": "recommend", "reply": "ok", "recommended_ids": [0, 999, 2], "end_of_conversation": False},
        _cands(),
    )
    assert [r.name for r in resp.recommendations] == ["Alpha", "Gamma"]


def test_dedupe_and_cap_at_10():
    cands = {i: _p(i, f"P{i}") for i in range(15)}
    resp = _assemble(
        {"action": "recommend", "reply": "ok",
         "recommended_ids": [0, 0, 1] + list(range(2, 15)), "end_of_conversation": False},
        cands,
    )
    names = [r.name for r in resp.recommendations]
    assert len(names) == 10
    assert len(set(names)) == 10


def test_end_of_conversation_requires_recs():
    resp = _assemble(
        {"action": "clarify", "reply": "?", "recommended_ids": [], "end_of_conversation": True},
        _cands(),
    )
    assert resp.end_of_conversation is False


def test_coerce_ids_tolerant():
    assert _coerce_ids([0, "2", 3.0, None, "x"]) == [0, 2, 3]


# ---- retrieval helpers ----
def test_tokenize_keeps_tech_tokens():
    toks = tokenize("C# and .NET with OPQ32r")
    assert "c#" in toks and "opq32r" in toks


def test_expand_query_adds_aliases():
    out = expand_query("need a cognitive test")
    assert "aptitude" in out or "reasoning" in out


# ---- query building & schema leniency ----
def test_build_query_uses_user_turns_latest_first():
    msgs = [Message(role="user", content="hiring java dev"),
            Message(role="assistant", content="what level?"),
            Message(role="user", content="senior, add AWS")]
    q = _build_query(msgs)
    assert q.startswith("senior, add AWS")
    assert "hiring java dev" in q


def test_request_schema_is_lenient():
    req = ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}], "extra": 1})
    assert req.messages[0].content == "hi"
    req2 = ChatRequest.model_validate({})  # missing messages -> default []
    assert req2.messages == []
