"""Ebbinghaus-inspired forgetting curves for memory importance decay.

This is Pillar 2 of Jarvis's novel contribution: active forgetting
with type-specific decay rates and access reinforcement.

importance(t) = base_importance × access_boost × decay_factor(type, t)

where:
  decay_factor = e^(-ln(2) × t / half_life)
  access_boost = 1 + ln(1 + access_count) × 0.3
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from jarvis.models.memory import Memory


def compute_importance(memory: Memory, now: datetime | None = None) -> float:
    """Compute current importance of a memory using Ebbinghaus decay.

    Higher access count slows decay (reinforcement).
    Type-specific half-life controls decay rate.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    age_days = (now - memory.created_at).total_seconds() / 86400.0
    half_life = memory.decay_half_life_days or 90.0

    # Exponential decay: importance halves every half_life days
    decay = math.exp(-0.693 * age_days / half_life)

    # Access reinforcement: each access logarithmically boosts importance
    access_boost = 1.0 + math.log(1.0 + memory.access_count) * 0.3

    return min(1.0, memory.importance * decay * access_boost)


def compute_recency(memory: Memory, now: datetime | None = None, weight: float = 0.3) -> float:
    """Score based on how recently the memory was accessed or created.

    Returns 0-1, where 1 = accessed just now, decaying exponentially.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    ref = memory.last_accessed or memory.created_at
    days_ago = (now - ref).total_seconds() / 86400.0
    return math.exp(-days_ago * weight)


def should_prune(memory: Memory, threshold: float = 0.05) -> bool:
    """Check if a memory has decayed below the pruning threshold."""
    return compute_importance(memory) < threshold


def find_stale_memories(memories: list[Memory], threshold: float = 0.1) -> list[Memory]:
    """Find memories that have decayed significantly and may need consolidation or pruning."""
    return [m for m in memories if compute_importance(m) < threshold]
