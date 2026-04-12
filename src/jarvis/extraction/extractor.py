"""Memory extraction — convert raw inputs into typed memories with entities."""

from __future__ import annotations

import logging

from jarvis.extraction import ollama_client
from jarvis.extraction.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_SYSTEM,
    SUMMARY_PROMPT,
    SUMMARY_SYSTEM,
)
from jarvis.models.entity import Entity, EntityType

logger = logging.getLogger(__name__)

_ENTITY_TYPE_MAP = {
    "service": EntityType.SERVICE,
    "person": EntityType.PERSON,
    "ticket": EntityType.TICKET,
    "concept": EntityType.CONCEPT,
    "tool": EntityType.TOOL,
    "file": EntityType.FILE,
}


async def extract_entities(text: str) -> list[tuple[Entity, str]]:
    """Extract entities from text. Returns list of (Entity, role)."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
    data = await ollama_client.extract_json(prompt, system=ENTITY_EXTRACTION_SYSTEM)

    results: list[tuple[Entity, str]] = []
    for item in data.get("entities", []):
        name = item.get("name", "").strip()
        etype = _ENTITY_TYPE_MAP.get(item.get("type", ""), EntityType.CONCEPT)
        role = item.get("role", "subject")
        if name:
            results.append((Entity(name=name, entity_type=etype), role))
    return results


async def summarize(text: str) -> str:
    """Generate a concise summary of the text."""
    prompt = SUMMARY_PROMPT.format(text=text)
    return await ollama_client.extract(prompt, system=SUMMARY_SYSTEM)
