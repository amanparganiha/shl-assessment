"""Catalog domain model and access layer.

Loads the *processed* catalog (produced by scripts/build_index.py) and exposes
fast lookups by internal id, entity_id, and URL. Also owns the single source of
truth for deriving the `test_type` letter(s) from SHL's category `keys`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import settings

logger = logging.getLogger("shl.catalog")

# SHL's published test-type legend: category name -> single letter.
TEST_TYPE_LEGEND: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def derive_test_type(keys: list[str]) -> str:
    """Map category keys to comma-joined letters, preserving order, deduped.

    Example: ["Competencies", "Knowledge & Skills"] -> "C, K".
    Unknown keys are skipped; empty input yields "" (callers may default it).
    """
    letters: list[str] = []
    for k in keys or []:
        letter = TEST_TYPE_LEGEND.get(k.strip())
        if letter and letter not in letters:
            letters.append(letter)
    return ", ".join(letters)


@dataclass
class Product:
    """A single Individual Test Solution from the catalog."""

    id: int
    entity_id: str
    name: str
    url: str
    test_type: str
    keys: list[str] = field(default_factory=list)
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    duration: str = ""
    remote: str = ""
    adaptive: str = ""
    description: str = ""
    doc: str = ""  # retrieval text

    def to_recommendation(self) -> dict[str, str]:
        """The exact shape the API returns in `recommendations`."""
        return {"name": self.name, "url": self.url, "test_type": self.test_type or "K"}

    def languages_summary(self, limit: int = 4) -> str:
        if not self.languages:
            return "—"
        head = ", ".join(self.languages[:limit])
        extra = len(self.languages) - limit
        return f"{head} (+{extra} more)" if extra > 0 else head

    def context_line(self) -> str:
        """Compact one-liner shown to the LLM as a candidate."""
        dur = self.duration or "—"
        jl = ", ".join(self.job_levels[:4]) if self.job_levels else "—"
        desc = (self.description or "").strip().replace("\n", " ")
        if len(desc) > 280:
            desc = desc[:277] + "..."
        return (
            f"[{self.id}] {self.name} | type={self.test_type or '—'} "
            f"| keys={', '.join(self.keys) or '—'} | duration={dur} "
            f"| levels={jl}\n     {desc}"
        )


class Catalog:
    """In-memory catalog with id/entity/url indices and aligned embeddings."""

    def __init__(self, products: list[Product], embeddings: np.ndarray | None = None):
        self.products = products
        self.embeddings = embeddings
        self._by_id = {p.id: p for p in products}
        self._by_entity = {p.entity_id: p for p in products}
        self._by_url = {p.url.rstrip("/").lower(): p for p in products}
        if embeddings is not None and len(embeddings) != len(products):
            raise ValueError(
                f"embeddings ({len(embeddings)}) != products ({len(products)})"
            )

    # ---- dunder helpers ----
    def __len__(self) -> int:
        return len(self.products)

    def __iter__(self):
        return iter(self.products)

    # ---- lookups ----
    def get(self, pid: int) -> Product | None:
        return self._by_id.get(pid)

    def by_entity_id(self, entity_id: str) -> Product | None:
        return self._by_entity.get(entity_id)

    def by_url(self, url: str) -> Product | None:
        return self._by_url.get(url.rstrip("/").lower())

    def find_by_name(self, name: str) -> Product | None:
        target = name.strip().lower()
        for p in self.products:
            if p.name.strip().lower() == target:
                return p
        return None

    # ---- persistence ----
    @classmethod
    def load(cls, catalog_path: Path | None = None, embeddings_path: Path | None = None) -> "Catalog":
        catalog_path = catalog_path or settings.catalog_path
        embeddings_path = embeddings_path or settings.embeddings_path
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Processed catalog not found at {catalog_path}. "
                "Run: python scripts/build_index.py"
            )
        records = json.loads(catalog_path.read_text(encoding="utf-8"))
        products = [Product(**r) for r in records]
        embeddings = None
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            logger.info("Loaded %d embeddings (dim=%d)", len(embeddings), embeddings.shape[1])
        else:
            logger.warning("No embeddings file at %s; retrieval will be lexical-only", embeddings_path)
        logger.info("Loaded catalog: %d products", len(products))
        return cls(products, embeddings)

    def save(self, catalog_path: Path | None = None) -> None:
        catalog_path = catalog_path or settings.catalog_path
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        records = [asdict(p) for p in self.products]
        catalog_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved %d products -> %s", len(records), catalog_path)
