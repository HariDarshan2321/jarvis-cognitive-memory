"""Retrieval result models — what recall() and prime() return."""

from __future__ import annotations

from pydantic import BaseModel

from jarvis.models.memory import Memory


class RecallResult(BaseModel):
    """Result from a recall() query."""

    memories: list[Memory]
    query: str
    total_candidates: int  # how many were considered before top-K


class PrimingResult(BaseModel):
    """Result from a prime() contextual activation."""

    activated_memories: list[Memory]
    briefing: str  # compressed context summary
    confidence: float  # how relevant the priming is (0-1)
    context_signals: dict  # what signals triggered the priming
