"""Offline preprocessing: raw catalog JSON -> processed catalog + embeddings.

Run once (and whenever the catalog changes):

    python scripts/build_index.py            # normalize + compute embeddings
    python scripts/build_index.py --skip-embeddings   # normalize only (no API key)

Outputs:
    data/processed/catalog.json   normalized records (one per product)
    data/processed/embeddings.npy (N, D) float32, L2-normalised, aligned by id
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Make the `app` package importable when run as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog import Catalog, Product, derive_test_type  # noqa: E402
from app.config import settings  # noqa: E402

_WS = re.compile(r"\s+")


def clean_text(s: str | None) -> str:
    """Collapse whitespace and strip control characters."""
    if not s:
        return ""
    s = s.replace("\xa0", " ")  # non-breaking space -> normal space
    s = "".join(ch if (ch >= "\x20" or ch == "\n") else " " for ch in s)
    return _WS.sub(" ", s).strip()


def load_raw(path: Path) -> list[dict]:
    """Load the raw catalog, tolerating unescaped control chars in strings."""
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw, strict=False)


def build_doc(name: str, keys: list[str], job_levels: list[str], description: str) -> str:
    """Compose the text used for lexical + dense retrieval.

    Name is included twice to give product/skill names extra lexical weight in
    BM25 (e.g. 'Docker', 'HIPAA', 'OPQ32r' should dominate matches).
    """
    parts = [
        name,
        name,
        ("Category: " + ", ".join(keys)) if keys else "",
        ("Suitable for: " + ", ".join(job_levels)) if job_levels else "",
        description,
    ]
    return clean_text(" . ".join(p for p in parts if p))


def normalize(raw: list[dict]) -> list[Product]:
    products: list[Product] = []
    for d in raw:
        name = clean_text(d.get("name"))
        url = (d.get("link") or "").strip()
        if not name or not url:
            continue  # skip unusable rows defensively
        keys = [k.strip() for k in (d.get("keys") or []) if k and k.strip()]
        job_levels = [j.strip() for j in (d.get("job_levels") or []) if j and j.strip()]
        languages = [l.strip() for l in (d.get("languages") or []) if l and l.strip()]
        duration = clean_text(d.get("duration")) or clean_text(d.get("duration_raw"))
        description = clean_text(d.get("description"))
        products.append(
            Product(
                id=len(products),
                entity_id=str(d.get("entity_id") or len(products)),
                name=name,
                url=url,
                test_type=derive_test_type(keys),
                keys=keys,
                job_levels=job_levels,
                languages=languages,
                duration=duration,
                remote=clean_text(d.get("remote")),
                adaptive=clean_text(d.get("adaptive")),
                description=description,
                doc=build_doc(name, keys, job_levels, description),
            )
        )
    return products


def main() -> int:
    parser = argparse.ArgumentParser(description="Build processed catalog + embeddings.")
    parser.add_argument("--skip-embeddings", action="store_true", help="normalize only")
    args = parser.parse_args()

    raw = load_raw(settings.raw_catalog_path)
    print(f"Loaded {len(raw)} raw records from {settings.raw_catalog_path.name}")

    products = normalize(raw)
    print(f"Normalized {len(products)} products")

    # Quick distribution sanity print.
    from collections import Counter

    tt = Counter(p.test_type for p in products)
    print("test_type distribution:", dict(tt.most_common()))
    missing_tt = [p.name for p in products if not p.test_type]
    if missing_tt:
        print(f"WARNING: {len(missing_tt)} products have empty test_type:", missing_tt[:5])

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(products)
    catalog.save()  # writes catalog.json

    if args.skip_embeddings:
        print("Skipped embeddings (--skip-embeddings).")
        return 0

    if not settings.has_api_key:
        print(
            "No API key found (OPENAI_API_KEY). Wrote catalog.json only.\n"
            "Set the key in .env and re-run to compute embeddings."
        )
        return 0

    from app.llm import embed_texts

    print(f"Embedding {len(products)} docs with {settings.embed_model} ...")
    vectors = embed_texts([p.doc for p in products])
    np.save(settings.embeddings_path, vectors)
    print(f"Saved embeddings {vectors.shape} -> {settings.embeddings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
