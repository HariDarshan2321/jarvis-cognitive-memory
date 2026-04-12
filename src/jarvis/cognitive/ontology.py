"""Developer Memory Ontology — type-specific rules for memory lifecycle.

This is Pillar 1 of Jarvis's novel contribution: not all developer
memories are equal. Each (type, subtype) pair has distinct:
- Decay half-life (how fast it loses importance)
- Default importance
- Creation triggers
"""

from __future__ import annotations

from jarvis.config import DECAY_HALF_LIVES
from jarvis.models.memory import MemorySubtype, MemoryType

# Default importance by subtype
DEFAULT_IMPORTANCE: dict[MemorySubtype | None, float] = {
    # Episodic — moderate, decays
    MemorySubtype.DEBUG_SESSION: 0.5,
    MemorySubtype.CODE_REVIEW: 0.5,
    MemorySubtype.INCIDENT: 0.7,
    # Semantic — high, persists
    MemorySubtype.ARCHITECTURE: 0.7,
    MemorySubtype.CONVENTION: 0.6,
    MemorySubtype.DOMAIN: 0.6,
    # Procedural — high, persists
    MemorySubtype.COMMAND: 0.8,
    MemorySubtype.WORKFLOW: 0.6,
    MemorySubtype.DEBUGGING: 0.6,
    # Default
    None: 0.5,
}


def get_decay_half_life(mem_type: MemoryType, subtype: MemorySubtype | None) -> float:
    """Get the decay half-life in days for a (type, subtype) pair."""
    key = (mem_type.value, subtype.value if subtype else None)
    fallback = (mem_type.value, None)
    return DECAY_HALF_LIVES.get(key, DECAY_HALF_LIVES.get(fallback, 90.0))


def get_default_importance(subtype: MemorySubtype | None) -> float:
    """Get default importance for a memory subtype."""
    return DEFAULT_IMPORTANCE.get(subtype, 0.5)


def infer_subtype(mem_type: MemoryType, content: str, source: str) -> MemorySubtype | None:
    """Heuristically infer a subtype from content and source signals."""
    content_lower = content.lower()

    if mem_type == MemoryType.EPISODIC:
        if source == "git":
            # Commits are typically code reviews or debug sessions
            if any(w in content_lower for w in ["fix", "bug", "debug", "hotfix", "patch"]):
                return MemorySubtype.DEBUG_SESSION
            if any(w in content_lower for w in ["pr", "review", "merge"]):
                return MemorySubtype.CODE_REVIEW
            if any(w in content_lower for w in ["incident", "outage", "alert", "prod"]):
                return MemorySubtype.INCIDENT
            return MemorySubtype.CODE_REVIEW
        if any(w in content_lower for w in ["debug", "investigate", "root cause", "traced"]):
            return MemorySubtype.DEBUG_SESSION
        if any(w in content_lower for w in ["incident", "outage", "p0", "p1"]):
            return MemorySubtype.INCIDENT

    elif mem_type == MemoryType.SEMANTIC:
        if any(w in content_lower for w in ["architecture", "design", "pattern", "layer"]):
            return MemorySubtype.ARCHITECTURE
        if any(w in content_lower for w in ["convention", "rule", "standard", "always", "never"]):
            return MemorySubtype.CONVENTION
        if any(w in content_lower for w in ["domain", "business", "sp type", "customer"]):
            return MemorySubtype.DOMAIN

    elif mem_type == MemoryType.PROCEDURAL:
        if any(w in content_lower for w in ["run", "execute", "command", "cli", "`"]):
            return MemorySubtype.COMMAND
        if any(w in content_lower for w in ["workflow", "process", "step", "pipeline"]):
            return MemorySubtype.WORKFLOW
        if any(w in content_lower for w in ["debug", "troubleshoot", "check", "verify"]):
            return MemorySubtype.DEBUGGING

    return None
