"""Jarvis MCP Server — Cognitive Developer Memory System.

Exposes memory tools via MCP protocol for Claude Code integration.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

from jarvis.config import DEFAULT_K
from jarvis.extraction import ollama_client
from jarvis.extraction.extractor import extract_entities, summarize
from jarvis.extraction.prompts import BRIEFING_PROMPT, BRIEFING_SYSTEM
from jarvis.models.entity import MemoryEntity
from jarvis.models.memory import Memory, MemorySubtype, MemoryType
from jarvis.store.memory_store import MemoryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

mcp = FastMCP(
    "Jarvis",
    instructions=(
        "Jarvis is a cognitive developer memory system. "
        "Use 'remember' to store knowledge, 'recall' to search memories, "
        "'brief' for topic summaries, and 'prime' for automatic context activation. "
        "Memories have types (episodic/semantic/procedural) with different decay rates."
    ),
)

_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


# ── remember ─────────────────────────────────────────────────────────────────

@mcp.tool()
async def remember(
    content: str,
    type: str = "episodic",
    subtype: str | None = None,
    source: str = "manual",
    source_ref: str | None = None,
    tags: list[str] | None = None,
    importance: float = 0.5,
) -> str:
    """Store a new memory in Jarvis.

    Args:
        content: The memory content to store.
        type: Memory type — 'episodic' (events), 'semantic' (facts), 'procedural' (how-to).
        subtype: Optional subtype — debug_session, code_review, incident, architecture,
                 convention, domain, command, workflow, debugging.
        source: Where this memory came from — manual, git, linear, session, notion.
        source_ref: Reference ID (commit hash, ticket ID, etc.).
        tags: Tags for categorization.
        importance: Base importance 0.0-1.0 (default 0.5).
    """
    store = _get_store()

    mem = Memory(
        type=MemoryType(type),
        subtype=MemorySubtype(subtype) if subtype else None,
        content=content,
        source=source,
        source_ref=source_ref,
        tags=tags or [],
        importance=importance,
    )

    # Store with embedding (nomic, ~50ms) + fast entity extraction (qwen2.5:1.5b, ~1-2s)
    stored = await store.add(mem)

    # Entity extraction with fast model — won't hang
    try:
        entities = await extract_entities(content)
        for entity, role in entities:
            existing = store.find_entity(entity.name)
            if existing:
                entity = existing
            else:
                await store.add_entity(entity)
            await store.link_memory_entity(
                MemoryEntity(memory_id=stored.id, entity_id=entity.id, role=role)
            )
    except Exception as e:
        logger.warning("Entity extraction failed (non-blocking): %s", e)

    return json.dumps({
        "id": stored.id,
        "type": stored.type.value,
        "subtype": stored.subtype.value if stored.subtype else None,
        "tags": stored.tags,
        "importance": stored.importance,
        "decay_half_life_days": stored.decay_half_life_days,
        "message": "Memory stored successfully.",
    })


# ── recall ───────────────────────────────────────────────────────────────────

@mcp.tool()
async def recall(
    query: str,
    k: int = DEFAULT_K,
    types: list[str] | None = None,
    min_importance: float = 0.0,
) -> str:
    """Search memories by semantic similarity + keyword matching.

    Returns the top-K most relevant memories ranked by relevance, importance, and recency.

    Args:
        query: What to search for (natural language).
        k: Maximum number of memories to return (default 5).
        types: Filter by memory types — ['episodic', 'semantic', 'procedural'].
        min_importance: Minimum importance threshold (0.0-1.0).
    """
    store = _get_store()
    memories, total = await store.recall(query, k=k, types=types, min_importance=min_importance)

    # Cap content at 500 chars per memory to prevent context flooding.
    # Use brief() for compressed summaries, or get() for full content.
    max_content = 500

    results = []
    for mem in memories:
        content = mem.content
        if len(content) > max_content:
            content = content[:max_content] + f"... [truncated, {len(mem.content)} chars total — use get({mem.id}) for full]"
        results.append({
            "id": mem.id,
            "type": mem.type.value,
            "subtype": mem.subtype.value if mem.subtype else None,
            "content": content,
            "source": mem.source,
            "tags": mem.tags,
            "importance": round(store._compute_importance(mem), 3),
            "created_at": mem.created_at.isoformat() if mem.created_at else None,
            "access_count": mem.access_count,
        })

    return json.dumps({
        "memories": results,
        "count": len(results),
        "total_candidates": total,
        "query": query,
    })


# ── update ───────────────────────────────────────────────────────────────────

@mcp.tool()
async def update(memory_id: str, content: str) -> str:
    """Update the content of an existing memory.

    The memory will be re-embedded for future searches.

    Args:
        memory_id: ID of the memory to update.
        content: New content for the memory.
    """
    store = _get_store()
    mem = await store.get(memory_id)
    if mem is None:
        return json.dumps({"error": f"Memory {memory_id} not found."})

    mem.content = content
    mem.summary = None  # will be regenerated
    updated = await store.update(mem)
    return json.dumps({
        "id": updated.id,
        "message": "Memory updated and re-embedded.",
    })


# ── forget ───────────────────────────────────────────────────────────────────

@mcp.tool()
async def forget(memory_id: str, reason: str = "") -> str:
    """Soft-delete a memory (excluded from future searches but kept for audit).

    Args:
        memory_id: ID of the memory to forget.
        reason: Why this memory is being forgotten.
    """
    store = _get_store()
    success = await store.forget(memory_id)
    if success:
        return json.dumps({"message": f"Memory {memory_id} forgotten.", "reason": reason})
    return json.dumps({"error": f"Memory {memory_id} not found."})


# ── brief ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def brief(topic: str, max_tokens: int = 500) -> str:
    """Get a compressed briefing on a topic.

    Retrieves relevant memories and compresses them into a concise summary.
    Much more token-efficient than recall() for getting context.

    Args:
        topic: The topic to get briefed on (e.g., "commitment module", "billing service").
        max_tokens: Maximum length of the briefing (default 500).
    """
    store = _get_store()
    memories, _ = await store.recall(topic, k=8, min_importance=0.1)

    if not memories:
        return json.dumps({"briefing": f"No memories found about '{topic}'.", "memory_count": 0})

    # Build briefing from memory content directly — no LLM call, instant response
    briefing_parts = []
    char_budget = max_tokens * 4  # rough chars-to-tokens
    used = 0
    for m in memories:
        snippet = m.content[:400].replace("\n", " ").strip()
        if used + len(snippet) > char_budget:
            break
        tag_str = f" [{', '.join(m.tags[:3])}]" if m.tags else ""
        briefing_parts.append(f"- ({m.type.value}{tag_str}) {snippet}")
        used += len(snippet)

    return json.dumps({
        "briefing": "\n".join(briefing_parts),
        "memory_count": len(memories),
        "topic": topic,
    })


# ── prime ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def prime(
    file_path: str | None = None,
    git_branch: str | None = None,
    recent_errors: list[str] | None = None,
    mentioned_text: str | None = None,
    cwd: str | None = None,
) -> str:
    """Contextual priming — automatically activate relevant memories from context.

    System 1 retrieval: provide your working context (or just cwd) and Jarvis
    activates relevant memories. If cwd is provided, auto-detects git branch,
    recent commits, and modified files.

    Args:
        file_path: Current file being worked on (e.g., "services/billing/app/...").
        git_branch: Current git branch (e.g., "gc-blo-1774"). Auto-detected from cwd if not set.
        recent_errors: Recent error messages encountered.
        mentioned_text: Any relevant text mentioned in conversation.
        cwd: Working directory — enables auto-detection of git state, branch, modified files.
    """
    from jarvis.cognitive.priming import prime as do_prime

    store = _get_store()
    memories, signals = await do_prime(
        store,
        file_path=file_path,
        git_branch=git_branch,
        recent_errors=recent_errors,
        mentioned_text=mentioned_text,
        cwd=cwd,
    )

    if not memories:
        return json.dumps({
            "briefing": "No relevant memories activated for this context.",
            "activated": [],
            "signals": signals,
            "confidence": 0.0,
        })

    # Build a quick briefing from activated memories
    briefing_parts = []
    for m in memories[:5]:
        prefix = f"[{m.type.value}]"
        briefing_parts.append(f"{prefix} {m.content[:300]}")
    briefing = "\n\n".join(briefing_parts)

    activated = [
        {
            "id": m.id,
            "type": m.type.value,
            "content": m.content[:500],
            "tags": m.tags,
            "source": m.source,
        }
        for m in memories
    ]

    confidence = min(1.0, len(memories) * 0.2)

    return json.dumps({
        "briefing": briefing,
        "activated": activated,
        "signals": signals,
        "confidence": confidence,
    })


# ── consolidate ──────────────────────────────────────────────────────────────

@mcp.tool()
async def consolidate(scope: str = "daily") -> str:
    """Trigger memory consolidation — merge duplicates, promote patterns, prune stale.

    Consolidation levels:
    - 'daily': dedup near-duplicates, reinforce frequently-accessed memories
    - 'weekly': + promote episodic clusters to semantic, detect conflicts
    - 'full': + prune stale memories below importance threshold

    Args:
        scope: Consolidation scope — 'daily', 'weekly', or 'full'.
    """
    from jarvis.cognitive.consolidation import run_consolidation

    store = _get_store()
    report = await run_consolidation(store, scope=scope)
    return json.dumps(report)


# ── reflect ──────────────────────────────────────────────────────────────────

@mcp.tool()
async def reflect(period: str = "week") -> str:
    """Discover patterns from recent work — what have you been focused on?

    Args:
        period: Time period to reflect on — 'day', 'week', 'month'.
    """
    from datetime import timedelta

    store = _get_store()

    days = {"day": 1, "week": 7, "month": 30}.get(period, 7)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    all_memories = store.get_all_active(limit=500)
    recent = [m for m in all_memories if m.created_at >= cutoff]

    if not recent:
        return json.dumps({"reflection": f"No memories from the last {period}.", "count": 0})

    # Analyze tag frequency
    tag_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for mem in recent:
        type_counts[mem.type.value] = type_counts.get(mem.type.value, 0) + 1
        source_counts[mem.source] = source_counts.get(mem.source, 0) + 1
        for tag in mem.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Build reflection summary
    summary_parts = [
        f"In the last {period}, you created {len(recent)} memories.",
        f"Types: {type_counts}",
        f"Sources: {source_counts}",
        f"Top topics: {', '.join(f'{t}({c})' for t, c in top_tags[:5])}",
    ]

    return json.dumps({
        "reflection": " ".join(summary_parts),
        "count": len(recent),
        "top_tags": dict(top_tags),
        "by_type": type_counts,
        "by_source": source_counts,
    })


# ── learn ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def learn(source: str, repo_path: str | None = None, memory_dir: str | None = None) -> str:
    """Ingest knowledge from an external source into Jarvis memory.

    Args:
        source: Source type — 'git' (last commit), 'claude_memory' (import existing memories).
        repo_path: For 'git' source — path to the git repository.
        memory_dir: For 'claude_memory' — path to ~/.claude/.../memory/ directory.
    """
    store = _get_store()

    if source == "git":
        from jarvis.ingestion.git import extract_commit_memory, get_last_commit

        if not repo_path:
            return json.dumps({"error": "repo_path is required for git source."})

        commit = get_last_commit(repo_path)
        if not commit:
            return json.dumps({"error": "Could not read last commit."})

        memory, entities = await extract_commit_memory(commit)
        stored = await store.add(memory)

        for entity, role in entities:
            existing = store.find_entity(entity.name)
            if existing:
                entity = existing
            else:
                await store.add_entity(entity)
            await store.link_memory_entity(
                MemoryEntity(memory_id=stored.id, entity_id=entity.id, role=role)
            )

        return json.dumps({
            "message": f"Learned from commit {commit['hash']}",
            "memory_id": stored.id,
            "tags": stored.tags,
        })

    elif source == "claude_memory":
        from pathlib import Path
        from jarvis.migration.claude_memory import migrate_all

        if not memory_dir:
            # Default Claude memory location
            memory_dir = str(
                Path.home() / ".claude" / "projects"
                / "-Users-darshanthevarmahalingam" / "memory"
            )

        imported = await migrate_all(Path(memory_dir), store)
        return json.dumps({
            "message": f"Imported {len(imported)} memories from Claude memory.",
            "imported_ids": imported,
        })

    return json.dumps({"error": f"Unknown source: {source}. Use 'git' or 'claude_memory'."})


# ── status ───────────────────────────────────────────────────────────────────

@mcp.tool()
async def status() -> str:
    """Get memory system health and statistics."""
    store = _get_store()
    total = store.count(active_only=False)
    active = store.count(active_only=True)

    # Count by type
    type_counts = {}
    for mt in MemoryType:
        mems = store._sql.get_by_type(mt.value, limit=10000)
        type_counts[mt.value] = len(mems)

    return json.dumps({
        "total_memories": total,
        "active_memories": active,
        "forgotten": total - active,
        "by_type": type_counts,
        "vector_count": store._vec.count(),
    })


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Run the Jarvis MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
