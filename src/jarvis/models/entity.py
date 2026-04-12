"""Entity and relationship models for the knowledge graph."""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    SERVICE = "service"
    PERSON = "person"
    TICKET = "ticket"
    CONCEPT = "concept"
    TOOL = "tool"
    FILE = "file"


def _make_id() -> str:
    return uuid.uuid4().hex[:16]


class Entity(BaseModel):
    """A named entity that memories reference."""

    id: str = Field(default_factory=_make_id)
    name: str
    entity_type: EntityType
    metadata: dict = Field(default_factory=dict)


class EntityRelation(BaseModel):
    """A directed relationship between two entities."""

    source_id: str
    target_id: str
    relation: str  # depends_on, part_of, caused_by, relates_to
    strength: float = 1.0
    metadata: dict = Field(default_factory=dict)


class MemoryEntity(BaseModel):
    """Links a memory to an entity with a role."""

    memory_id: str
    entity_id: str
    role: str = "subject"  # subject, object, context
