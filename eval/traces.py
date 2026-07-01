"""Parse the provided sample conversation traces (C1..C10) into structured objects.

Each trace is a hand-authored ideal conversation. We extract, per turn: the user
message(s), whether the agent recommended, and the recommended catalog URLs. The
FINAL shortlist (last turn containing a table) is the labelled answer for Recall@10.
The concatenation of user turns doubles as the 'persona facts' for the LLM user
simulator in run_eval.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TRACE_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sample_conversations"
    / "GenAI_SampleConversations"
)

_URL_RE = re.compile(r"<(https://www\.shl\.com/products/product-catalog/view/[^>]+)>")
_END_RE = re.compile(r"end_of_conversation.*?\*\*(true|false)\*\*", re.IGNORECASE)


@dataclass
class Turn:
    user: str
    agent_reply: str
    rec_urls: list[str] = field(default_factory=list)
    end: bool = False


@dataclass
class Trace:
    name: str
    turns: list[Turn]

    @property
    def user_messages(self) -> list[str]:
        return [t.user for t in self.turns if t.user.strip()]

    @property
    def facts(self) -> str:
        """All user-provided information, for the user simulator's persona."""
        return "\n".join(f"- {u}" for u in self.user_messages)

    @property
    def final_shortlist(self) -> list[str]:
        """URLs of the last turn that presented a shortlist (the labelled answer)."""
        for turn in reversed(self.turns):
            if turn.rec_urls:
                return [normalize_url(u) for u in turn.rec_urls]
        return []


def normalize_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


def _extract_user(block: str) -> str:
    """Collect blockquote lines under the **User** marker of a turn block."""
    m = re.search(r"\*\*User\*\*(.*?)\*\*Agent\*\*", block, re.DOTALL)
    if not m:
        return ""
    quoted = [
        line.lstrip(">").strip()
        for line in m.group(1).splitlines()
        if line.strip().startswith(">")
    ]
    return " ".join(q for q in quoted if q).strip()


def _extract_agent_reply(block: str) -> str:
    """Prose the agent said (everything after **Agent**, minus the table rows)."""
    m = re.search(r"\*\*Agent\*\*(.*)", block, re.DOTALL)
    if not m:
        return ""
    lines = []
    for line in m.group(1).splitlines():
        s = line.strip()
        if s.startswith("|") or s.startswith("_") or not s:
            continue
        lines.append(s)
    return " ".join(lines).strip()


def parse_trace(path: Path) -> Trace:
    text = path.read_text(encoding="utf-8")
    # Split into turn blocks; keep only those that actually contain a turn.
    blocks = re.split(r"### Turn \d+", text)
    turns: list[Turn] = []
    for block in blocks:
        if "**User**" not in block and "**Agent**" not in block:
            continue
        user = _extract_user(block)
        reply = _extract_agent_reply(block)
        urls = _URL_RE.findall(block)
        end_m = _END_RE.search(block)
        end = bool(end_m and end_m.group(1).lower() == "true")
        if user or urls:
            turns.append(Turn(user=user, agent_reply=reply, rec_urls=urls, end=end))
    return Trace(name=path.stem, turns=turns)


def load_traces(trace_dir: Path | None = None) -> list[Trace]:
    trace_dir = trace_dir or TRACE_DIR
    paths = sorted(
        trace_dir.glob("*.md"),
        key=lambda p: int(re.findall(r"\d+", p.stem)[0]) if re.findall(r"\d+", p.stem) else 0,
    )
    return [parse_trace(p) for p in paths]


if __name__ == "__main__":
    for tr in load_traces():
        print(
            f"{tr.name}: {len(tr.turns)} turns, "
            f"{len(tr.user_messages)} user msgs, "
            f"final shortlist={len(tr.final_shortlist)} items"
        )
