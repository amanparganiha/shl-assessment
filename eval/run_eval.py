"""Recall@10 evaluation via an LLM user-simulator (mirrors SHL's replay harness).

For each trace we seed the first user message from the persona, then an LLM plays
the hiring manager: it answers our agent's questions from the persona facts, says
"no preference" for anything outside them, and accepts once a shortlist appears.
We score the FIRST committed shortlist against the labelled final shortlist, since
the real harness ends the conversation as soon as the agent recommends.

Usage:
    python eval/run_eval.py                  # simulate mode (faithful)
    python eval/run_eval.py --mode deterministic   # feed trace user turns in order (cheap)
    python eval/run_eval.py --trace C4       # single trace
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import run_agent  # noqa: E402
from app.catalog import Catalog  # noqa: E402
from app.llm import chat_text  # noqa: E402
from app.retriever import Retriever  # noqa: E402
from app.schemas import ChatResponse, Message  # noqa: E402
from eval.traces import Trace, load_traces, normalize_url  # noqa: E402

MAX_TURNS = 8

SIM_SYSTEM = """You are role-playing a hiring manager chatting with an SHL assessment recommender.

The ONLY facts you know about your hiring need are:
{facts}

Rules:
- Speak in first person as the hiring manager, briefly (1-2 sentences).
- Answer the recommender's questions truthfully using ONLY the facts above.
- If asked about something not covered by your facts, say you have no strong preference.
- Do NOT invent requirements beyond the facts.
- When the recommender presents a shortlist of specific named assessments, briefly accept it and
  indicate you are done (e.g. "That works, thanks.").
"""


@dataclass
class TraceResult:
    name: str
    recall: float
    n_expected: int
    n_hit: int
    turns_used: int
    shortlist_names: list[str]
    missed_names: list[str]


def _simulate_next_user(facts: str, agent_history: list[Message]) -> str:
    """Ask the simulator LLM for the next hiring-manager message."""
    sim_messages: list[dict[str, str]] = []
    for m in agent_history:
        # From the simulator's POV, the agent's lines are 'user', its own are 'assistant'.
        sim_messages.append(
            {"role": "user" if m.role == "assistant" else "assistant", "content": m.content}
        )
    return chat_text(SIM_SYSTEM.format(facts=facts), sim_messages, max_tokens=120)


def _run_conversation(trace: Trace, catalog: Catalog, retriever: Retriever, mode: str):
    """Drive one conversation; return (first_shortlist: ChatResponse|None, turns_used, history)."""
    history: list[Message] = []
    scripted = list(trace.user_messages)
    user_msg = scripted[0] if scripted else "I need help choosing an assessment."
    scripted_idx = 1

    for turn in range(1, MAX_TURNS + 1):
        history.append(Message(role="user", content=user_msg))
        resp = run_agent(history, retriever=retriever)
        history.append(Message(role="assistant", content=resp.reply))

        if resp.recommendations:  # harness ends at first shortlist
            return resp, turn, history
        if resp.end_of_conversation:
            return resp, turn, history

        # Produce the next user message.
        if mode == "deterministic":
            if scripted_idx < len(scripted):
                user_msg = scripted[scripted_idx]
                scripted_idx += 1
            else:
                user_msg = "No strong preference. Please go ahead and recommend."
        else:  # simulate
            user_msg = _simulate_next_user(trace.facts, history)
            if not user_msg:
                user_msg = "No preference — please recommend."
    return None, MAX_TURNS, history


def evaluate_trace(trace: Trace, catalog: Catalog, retriever: Retriever, mode: str) -> TraceResult:
    resp, turns, _ = _run_conversation(trace, catalog, retriever, mode)
    expected = set(trace.final_shortlist)
    got_urls = [normalize_url(r.url) for r in resp.recommendations] if resp else []
    got_top10 = got_urls[:10]
    hit = expected & set(got_top10)
    recall = len(hit) / len(expected) if expected else 0.0

    def names(urls):
        out = []
        for u in urls:
            p = catalog.by_url(u)
            out.append(p.name if p else u)
        return out

    missed = expected - set(got_top10)
    return TraceResult(
        name=trace.name,
        recall=recall,
        n_expected=len(expected),
        n_hit=len(hit),
        turns_used=turns,
        shortlist_names=[r.name for r in resp.recommendations] if resp else [],
        missed_names=names(missed),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simulate", "deterministic"], default="simulate")
    ap.add_argument("--trace", default=None, help="e.g. C4 (default: all)")
    args = ap.parse_args()

    catalog = Catalog.load()
    retriever = Retriever(catalog)
    traces = load_traces()
    if args.trace:
        traces = [t for t in traces if t.name.lower() == args.trace.lower()]

    print(f"=== Recall@10 eval ({args.mode} mode, {len(traces)} traces) ===\n")
    results: list[TraceResult] = []
    for tr in traces:
        res = evaluate_trace(tr, catalog, retriever, args.mode)
        results.append(res)
        print(f"{res.name:5s} Recall@10={res.recall:.2f} ({res.n_hit}/{res.n_expected}) "
              f"turns={res.turns_used}")
        print(f"      shortlist: {res.shortlist_names}")
        if res.missed_names:
            print(f"      MISSED:    {res.missed_names}")
        print()

    if results:
        mean = sum(r.recall for r in results) / len(results)
        print(f"=== MEAN Recall@10 = {mean:.3f} over {len(results)} traces ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
