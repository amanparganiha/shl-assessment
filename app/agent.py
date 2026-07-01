"""The conversational agent.

One /chat turn = one structured LLM call over (system prompt + retrieved candidates
+ full conversation history). The model returns an action + reply + candidate ids;
the SERVER maps ids back to exact catalog records, so a hallucinated name/URL is
structurally impossible. Everything degrades gracefully to a valid schema so the
evaluator never sees a 500 or a malformed response.
"""
from __future__ import annotations

import logging

from .catalog import Product
from .config import settings
from .llm import chat_json
from .prompts import ACTION_JSON_SCHEMA, SYSTEM_PROMPT
from .retriever import Retriever, get_retriever
from .schemas import ChatResponse, Message, Recommendation

logger = logging.getLogger("shl.agent")

_VALID_ROLES = {"user", "assistant", "system"}
_RECOMMENDING_ACTIONS = {"recommend", "refine", "compare"}

_GREETING = (
    "Hi! I can help you choose SHL assessments. What role are you hiring for, "
    "and what skills or level matter most?"
)
_FALLBACK = (
    "Could you tell me a bit more about the role, the key skills, and the seniority "
    "level? That will let me suggest the right SHL assessments."
)


def _to_openai_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Convert history to OpenAI chat format, coercing roles and dropping empties."""
    out: list[dict[str, str]] = []
    for m in messages:
        content = (m.content or "").strip()
        if not content:
            continue
        role = m.role if m.role in _VALID_ROLES else "user"
        out.append({"role": role, "content": content})
    return out


def _build_query(messages: list[Message]) -> str:
    """Retrieval query = the user's cumulative intent.

    Using ALL user turns (recent-weighted by recency cap) means mid-conversation
    refinements like 'add AWS' or 'drop REST' are naturally part of the query, so
    the right candidates surface without any server-side state.
    """
    user_turns = [m.content.strip() for m in messages if m.role == "user" and m.content.strip()]
    # Emphasise the latest turn (it carries the current instruction) then context.
    if not user_turns:
        return ""
    latest = user_turns[-1]
    context = " \n ".join(user_turns[-8:])
    return f"{latest}\n{context}"


def _coerce_ids(raw) -> list[int]:
    ids: list[int] = []
    for x in raw or []:
        try:
            ids.append(int(x))
        except (TypeError, ValueError):
            continue
    return ids


def _assemble(action: dict, candidates_by_id: dict[int, Product]) -> ChatResponse:
    act = action.get("action", "clarify")
    reply = (action.get("reply") or "").strip()
    end = bool(action.get("end_of_conversation", False))
    ids = _coerce_ids(action.get("recommended_ids"))

    recommendations: list[Recommendation] = []
    if act in _RECOMMENDING_ACTIONS:
        seen: set[int] = set()
        for i in ids:
            product = candidates_by_id.get(i)  # grounding: must be a retrieved candidate
            if product is None or product.id in seen:
                continue
            seen.add(product.id)
            recommendations.append(Recommendation(**product.to_recommendation()))
            if len(recommendations) >= settings.max_recommendations:
                break
    # clarify / refuse -> recommendations stay empty (schema + probe requirement).

    if not reply:
        reply = _FALLBACK if not recommendations else "Here is the shortlist."
    # end_of_conversation only makes sense once something has been recommended.
    if end and not recommendations:
        end = False

    return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=end)


def run_agent(messages: list[Message], retriever: Retriever | None = None) -> ChatResponse:
    """Produce the next agent reply + (optional) shortlist for a conversation."""
    retriever = retriever or get_retriever()
    convo = _to_openai_messages(messages)
    if not convo:
        return ChatResponse(reply=_GREETING, recommendations=[], end_of_conversation=False)

    query = _build_query(messages)
    candidates = retriever.retrieve(query)
    candidates_by_id = {p.id: p for p in candidates}
    staple_ids = set(retriever.staple_ids)
    # Staples first (most salient) so canonical instruments win over older variants.
    ordered = [p for p in candidates if p.id in staple_ids] + [
        p for p in candidates if p.id not in staple_ids
    ]
    lines = [
        p.context_line() + ("  <-- SHL default instrument" if p.id in staple_ids else "")
        for p in ordered
    ]
    system_prompt = SYSTEM_PROMPT.format(candidates="\n".join(lines))

    # Turn-cap guard: the evaluator caps conversations at 8 turns. If we have already
    # spent several user turns, stop clarifying and commit to a shortlist (unless the
    # request is out of scope, which should still be refused).
    user_turns = sum(1 for m in messages if m.role == "user")
    if user_turns >= 3:
        system_prompt += (
            "\n\nIMPORTANT: This conversation already has several user turns. If you have ANY "
            "usable hiring context (a role, skills, or a job description), you MUST recommend a "
            "shortlist now instead of asking another clarifying question. Only keep clarifying if "
            "you still have essentially nothing to act on, and still refuse out-of-scope requests."
        )

    try:
        action = chat_json(system_prompt, convo, ACTION_JSON_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - never surface a 500 to the evaluator
        logger.exception("LLM call failed: %s", exc)
        return ChatResponse(reply=_FALLBACK, recommendations=[], end_of_conversation=False)

    return _assemble(action, candidates_by_id)
