"""Memory consolidation — dedup, merge, promote, and maintain.

Inspired by sleep consolidation in neuroscience: periodic cycles that
reorganize memories, merge duplicates, and promote repeated patterns
into durable semantic knowledge.

Three consolidation levels:
- Daily: dedup near-duplicates, recalculate importance scores
- Weekly: promote episodic clusters to semantic, strengthen accessed memories
- Full: prune stale memories, generate knowledge digest, rebuild importance
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from jarvis.extraction.ollama_client import deep_extract
from jarvis.models.memory import Memory, MemoryType, MemorySubtype
from jarvis.cognitive.forgetting import compute_importance, should_prune

logger = logging.getLogger(__name__)


async def find_duplicates(
    memories: list[Memory], threshold: float = 0.85
) -> list[tuple[Memory, Memory, float]]:
    """Find pairs of memories that are near-duplicates."""
    if len(memories) < 2:
        return []

    embeddings: list[list[float]] = []
    for mem in memories:
        from jarvis.extraction.ollama_client import embed
        vec = await embed(mem.content)
        embeddings.append(vec)

    duplicates: list[tuple[Memory, Memory, float]] = []
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                duplicates.append((memories[i], memories[j], sim))

    return duplicates


async def merge_memories(mem_a: Memory, mem_b: Memory) -> Memory:
    """Merge two similar memories into one, keeping the richer content."""
    if compute_importance(mem_a) >= compute_importance(mem_b):
        base, other = mem_a, mem_b
    else:
        base, other = mem_b, mem_a

    if base.content != other.content:
        base.content = f"{base.content}\n\n[Merged] {other.content}"

    base.tags = list(set(base.tags + other.tags))
    base.importance = min(1.0, base.importance + 0.1)
    base.access_count += other.access_count
    return base


async def consolidate_episodic_to_semantic(
    episodic_memories: list[Memory],
    min_cluster_size: int = 3,
) -> list[Memory]:
    """Promote episodic clusters to semantic knowledge."""
    if len(episodic_memories) < min_cluster_size:
        return []

    tag_groups: dict[str, list[Memory]] = {}
    for mem in episodic_memories:
        for tag in mem.tags:
            tag_groups.setdefault(tag, []).append(mem)

    new_semantic: list[Memory] = []
    promoted_tags: set[str] = set()

    for tag, group in sorted(tag_groups.items(), key=lambda x: len(x[1]), reverse=True):
        if len(group) < min_cluster_size or tag in promoted_tags:
            continue

        contents = "\n".join(f"- {m.content[:200]}" for m in group[:10])
        summary_prompt = (
            f"Consolidate these {len(group)} related developer experiences about '{tag}' "
            f"into a single factual knowledge entry. Focus on patterns, not individual events:\n\n"
            f"{contents}"
        )
        summary = await deep_extract(summary_prompt)

        semantic_mem = Memory(
            type=MemoryType.SEMANTIC,
            content=summary.strip(),
            source="consolidation",
            tags=[tag],
            importance=0.7,
            confidence=0.7,
        )
        new_semantic.append(semantic_mem)
        promoted_tags.add(tag)

    return new_semantic


def recalculate_importance(memories: list[Memory]) -> list[tuple[Memory, float, float]]:
    """Recalculate current importance for all memories.

    Returns list of (memory, old_importance, new_current_importance).
    Useful for identifying memories that have decayed significantly.
    """
    results: list[tuple[Memory, float, float]] = []
    for mem in memories:
        old = mem.importance
        current = compute_importance(mem)
        results.append((mem, old, current))
    return results


def find_reinforcement_candidates(memories: list[Memory]) -> list[Memory]:
    """Find memories that are accessed frequently but have low base importance.

    These should have their base importance boosted — the user clearly values them.
    """
    candidates = []
    for mem in memories:
        if mem.access_count >= 3 and mem.importance < 0.6:
            candidates.append(mem)
    return candidates


def find_conflict_candidates(memories: list[Memory]) -> list[tuple[Memory, Memory]]:
    """Find pairs of memories that might contradict each other.

    Heuristic: same tags + same type but different content.
    """
    conflicts: list[tuple[Memory, Memory]] = []
    by_tag: dict[str, list[Memory]] = {}
    for mem in memories:
        for tag in mem.tags:
            by_tag.setdefault(tag, []).append(mem)

    for tag, group in by_tag.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.type == b.type and a.subtype == b.subtype:
                    # Same type+subtype+tag = potential conflict
                    conflicts.append((a, b))

    return conflicts


async def run_consolidation(
    store,
    scope: str = "daily",
) -> dict:
    """Run a full consolidation cycle.

    Args:
        store: MemoryStore instance.
        scope: 'daily', 'weekly', or 'full'.

    Returns:
        Consolidation report dict.
    """
    all_memories = store.get_all_active(limit=1000)
    report: dict = {
        "scope": scope,
        "total_memories": len(all_memories),
        "actions": [],
    }

    # ── Always: Recalculate importance ──
    importance_data = recalculate_importance(all_memories)
    stale = [m for m, _, current in importance_data if current < 0.1]
    report["stale_count"] = len(stale)

    # Reinforce frequently-accessed memories
    reinforced = find_reinforcement_candidates(all_memories)
    for mem in reinforced:
        old_imp = mem.importance
        mem.importance = min(1.0, mem.importance + 0.15)
        store._sql.update_memory(mem)
        report["actions"].append(
            f"Reinforced {mem.id[:8]} ({old_imp:.2f} → {mem.importance:.2f}, accessed {mem.access_count}x)"
        )
    report["reinforced_count"] = len(reinforced)

    # ── Daily: Dedup ──
    if scope in ("daily", "weekly", "full"):
        recent = sorted(all_memories, key=lambda m: m.created_at, reverse=True)[:50]
        dupes = await find_duplicates(recent)
        report["duplicates_found"] = len(dupes)

        for mem_a, mem_b, sim in dupes:
            merged = await merge_memories(mem_a, mem_b)
            await store.update(merged)
            loser_id = mem_b.id if merged.id == mem_a.id else mem_a.id
            await store.forget(loser_id)
            report["actions"].append(f"Merged {mem_a.id[:8]} + {mem_b.id[:8]} (sim={sim:.2f})")

    # ── Weekly: Promote + detect conflicts ──
    if scope in ("weekly", "full"):
        episodic = [m for m in all_memories if m.type == MemoryType.EPISODIC]
        promoted = await consolidate_episodic_to_semantic(episodic)
        for sem_mem in promoted:
            await store.add(sem_mem)
            report["actions"].append(f"Promoted to semantic: {sem_mem.tags}")
        report["promoted_count"] = len(promoted)

        conflicts = find_conflict_candidates(all_memories)
        report["potential_conflicts"] = len(conflicts)
        for a, b in conflicts[:5]:
            report["actions"].append(
                f"Potential conflict: {a.id[:8]} vs {b.id[:8]} (tag: {set(a.tags) & set(b.tags)})"
            )

    # ── Full: Prune stale ──
    if scope == "full":
        pruned = 0
        for mem in stale:
            await store.forget(mem.id)
            pruned += 1
            report["actions"].append(f"Pruned stale: {mem.id[:8]} (importance < 0.1)")
        report["pruned_count"] = pruned

    return report


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
