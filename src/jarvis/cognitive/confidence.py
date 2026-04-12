"""Bayesian confidence tracking for memories.

Memories have a confidence score (0-1) that updates when:
- Confirming evidence arrives → confidence increases
- Contradictory evidence arrives → confidence decreases
- Memory is explicitly validated → confidence set to high

Uses a simple Bayesian update model.
"""

from __future__ import annotations

from jarvis.models.memory import Memory


def update_confidence(memory: Memory, evidence_strength: float, is_confirming: bool) -> float:
    """Update memory confidence given new evidence.

    Args:
        memory: The memory to update.
        evidence_strength: How strong the evidence is (0-1).
        is_confirming: True if evidence confirms the memory, False if it contradicts.

    Returns:
        New confidence value.
    """
    prior = memory.confidence
    # Simple Bayesian-inspired update
    if is_confirming:
        # Confirming evidence moves confidence toward 1.0
        new_conf = prior + (1.0 - prior) * evidence_strength * 0.3
    else:
        # Contradictory evidence moves confidence toward 0.0
        new_conf = prior - prior * evidence_strength * 0.4

    memory.confidence = max(0.05, min(0.99, new_conf))
    return memory.confidence


def is_uncertain(memory: Memory, threshold: float = 0.4) -> bool:
    """Check if a memory's confidence is below the uncertainty threshold."""
    return memory.confidence < threshold
