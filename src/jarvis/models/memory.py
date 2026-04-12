"""Memory domain models — the core data structures for Jarvis."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemorySubtype(StrEnum):
    # Episodic
    DEBUG_SESSION = "debug_session"
    CODE_REVIEW = "code_review"
    INCIDENT = "incident"
    # Semantic
    ARCHITECTURE = "architecture"
    CONVENTION = "convention"
    DOMAIN = "domain"
    # Procedural
    COMMAND = "command"
    WORKFLOW = "workflow"
    DEBUGGING = "debugging"


def _make_id() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Memory(BaseModel):
    """A single memory unit stored by Jarvis."""

    id: str = Field(default_factory=_make_id)
    type: MemoryType
    subtype: MemorySubtype | None = None
    content: str
    summary: str | None = None
    source: str  # git, linear, session, manual, notion
    source_ref: str | None = None  # commit hash, ticket ID, etc.
    tags: list[str] = Field(default_factory=list)

    # Cognitive properties
    importance: float = 0.5
    confidence: float = 0.8
    decay_half_life_days: float | None = None  # None → use ontology default
    access_count: int = 0

    # Temporal
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime | None = None
    last_accessed: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    superseded_by: str | None = None
    is_active: bool = True
