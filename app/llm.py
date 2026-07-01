"""Provider-agnostic LLM access.

Uses the OpenAI SDK, which also speaks to any OpenAI-compatible endpoint
(Groq, OpenRouter, Gemini's compat layer) by setting *_BASE_URL / *_API_KEY.
That keeps the stack swappable while defaulting to plain OpenAI.

Exposes two primitives:
  - chat_json(): one chat turn returning a JSON object (structured output).
  - embed_texts(): batch embeddings as a float32 numpy array.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import numpy as np
from openai import OpenAI

from .config import settings

logger = logging.getLogger("shl.llm")


@lru_cache(maxsize=1)
def _chat_client() -> OpenAI:
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.request_timeout_s,
        max_retries=settings.max_retries,
    )


@lru_cache(maxsize=1)
def _embed_client() -> OpenAI:
    return OpenAI(
        api_key=settings.embed_api_key,
        base_url=settings.embed_base_url,
        timeout=settings.request_timeout_s,
        max_retries=settings.max_retries,
    )


def chat_json(
    system_prompt: str,
    messages: list[dict[str, str]],
    json_schema: dict[str, Any],
    *,
    temperature: float = 0.1,
    max_tokens: int = 900,
) -> dict[str, Any]:
    """Run one chat completion and return a parsed JSON object.

    Tries OpenAI structured outputs (response_format=json_schema, which *guarantees*
    a schema-valid object). Falls back to json_object mode, then to best-effort
    brace extraction, so non-OpenAI providers still work.

    Raises the underlying exception only if the API call itself fails; callers are
    expected to wrap this so the HTTP layer can degrade gracefully.
    """
    client = _chat_client()
    convo = [{"role": "system", "content": system_prompt}, *messages]

    # 1) Preferred: strict structured output (OpenAI gpt-4o family).
    try:
        resp = client.chat.completions.create(
            model=settings.chat_model,
            messages=convo,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
        )
        return _parse(resp.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001 - provider may not support json_schema
        logger.warning("json_schema mode failed (%s); falling back to json_object", exc)

    # 2) Fallback: json_object mode (schema described in the prompt).
    convo[0] = {
        "role": "system",
        "content": system_prompt
        + "\n\nYou MUST reply with a single JSON object matching this schema:\n"
        + json.dumps(json_schema.get("schema", json_schema), ensure_ascii=False),
    }
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=convo,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return _parse(resp.choices[0].message.content)


def _parse(content: str | None) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating stray text/fences."""
    if not content:
        raise ValueError("empty model response")
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Best-effort: grab the outermost {...} block.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def chat_text(
    system_prompt: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.4,
    max_tokens: int = 300,
) -> str:
    """Plain free-text chat turn (used by the eval user-simulator, not the API)."""
    client = _chat_client()
    convo = [{"role": "system", "content": system_prompt}, *messages]
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=convo,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def embed_texts(texts: list[str], *, batch_size: int = 96) -> np.ndarray:
    """Embed a list of texts, returning an (N, D) float32 array (L2-normalised)."""
    client = _embed_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ")[:8000] for t in texts[i : i + batch_size]]
        resp = client.embeddings.create(model=settings.embed_model, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    arr = np.asarray(vectors, dtype=np.float32)
    # Normalise so dot product == cosine similarity.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_one(text: str) -> np.ndarray:
    """Embed a single string, returning a (D,) normalised float32 vector."""
    return embed_texts([text])[0]
