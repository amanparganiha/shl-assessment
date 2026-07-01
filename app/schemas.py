"""Pydantic models for the public API.

The /chat request/response schema is dictated by SHL's automated evaluator and is
NON-NEGOTIABLE. We keep the *request* side deliberately lenient (defaults + ignore
unknown fields) so a slightly malformed request never produces a 422 that would
count as a schema-compliance failure; we keep the *response* side exact.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """A single turn in the conversation history."""

    model_config = ConfigDict(extra="ignore")

    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    """POST /chat body. Stateless: the full history is sent every call."""

    model_config = ConfigDict(extra="ignore")

    messages: List[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    """One recommended assessment. Fields match the evaluator exactly."""

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """POST /chat response. Schema is exact and must not gain/lose fields."""

    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
