"""Unified MemoryStore facade — combines SQLite + LanceDB for hybrid retrieval."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from jarvis.config import (
    DECAY_HALF_LIVES,
    DEFAULT_K,
    DEDUP_THRESHOLD,
    SIMILARITY_THRESHOLD,
    SQLITE_DB,
    TIME_DECAY_WEIGHT,
    VECTOR_DIR,
    ensure_dirs,
)
from jarvis.extraction import ollama_client
from jarvis.models.entity import Entity, EntityRelation, MemoryEntity
from jarvis.models.memory import Memory, MemoryType
from jarvis.store.sqlite_store import SQLiteStore
from jarvis.store.vector_store import VectorStore


class MemoryStore:
    """Unified facade over SQLite (metadata + FTS + graph) and LanceDB (vectors)."""

    def __init__(self) -> None:
        ensure_dirs()
        self._sql = SQLiteStore(SQLITE_DB)
        self._vec = VectorStore(VECTOR_DIR)

    def close(self) -> None:
        self._sql.close()

    # -- Core operations --

    async def add(self, memory: Memory) -> Memory:
        """Store a new memory with embedding."""
        # Set decay half-life from ontology if not explicitly set
        if memory.decay_half_life_days is None:
            key = (memory.type.value, memory.subtype.value if memory.subtype else None)
            fallback_key = (memory.type.value, None)
            memory.decay_half_life_days = DECAY_HALF_LIVES.get(
                key, DECAY_HALF_LIVES.get(fallback_key, 90.0)
            )

        # Generate embedding
        vector = await ollama_client.embed(memory.content)

        # Check for duplicates
        similar = self._vec.search(vector, limit=3)
        for mem_id, distance in similar:
            similarity = 1.0 - distance
            if similarity >= DEDUP_THRESHOLD:
                existing = self._sql.get_memory(mem_id)
                if existing and existing.is_active:
                    # Update existing instead of creating duplicate
                    existing.content = f"{existing.content}\n\n---\n\n{memory.content}"
                    existing.access_count += 1
                    existing.importance = min(1.0, existing.importance + 0.1)
                    await self.update(existing)
                    return existing

        # Store
        self._sql.add_memory(memory)
        self._vec.add(memory.id, vector)
        return memory

    async def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID and mark as accessed."""
        mem = self._sql.get_memory(memory_id)
        if mem:
            self._sql.touch(memory_id)
        return mem

    async def update(self, memory: Memory) -> Memory:
        """Update a memory and re-embed."""
        self._sql.update_memory(memory)
        vector = await ollama_client.embed(memory.content)
        self._vec.update(memory.id, vector)
        return memory

    async def forget(self, memory_id: str) -> bool:
        """Soft-delete a memory."""
        return self._sql.soft_delete(memory_id)

    # -- Retrieval --

    async def recall(
        self,
        query: str,
        k: int = DEFAULT_K,
        types: list[str] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        min_importance: float = 0.0,
    ) -> tuple[list[Memory], int]:
        """Hybrid retrieval: vector similarity + FTS + importance scoring.

        Returns (memories, total_candidates_considered).
        """
        query_vector = await ollama_client.embed(query)

        # Vector search — get more candidates than needed for re-ranking
        candidates_limit = max(k * 5, 30)
        vector_hits = self._vec.search(query_vector, limit=candidates_limit)

        # FTS search
        fts_hits = self._sql.search_fts(query, limit=candidates_limit)
        fts_ids = {m.id for m in fts_hits}

        # Merge candidates
        scored: dict[str, float] = {}
        for mem_id, distance in vector_hits:
            similarity = 1.0 - distance  # cosine distance → similarity
            if similarity >= SIMILARITY_THRESHOLD:
                scored[mem_id] = similarity

        # Boost FTS matches
        for mem in fts_hits:
            scored[mem.id] = scored.get(mem.id, 0.3) + 0.2

        total_candidates = len(scored)

        # Fetch full memories and apply filters + re-ranking
        results: list[tuple[Memory, float]] = []
        for mem_id, base_score in scored.items():
            mem = self._sql.get_memory(mem_id)
            if mem is None or not mem.is_active:
                continue

            # Type filter
            if types and mem.type.value not in types:
                continue

            # Time range filter
            if time_range:
                start, end = time_range
                if mem.created_at < start or mem.created_at > end:
                    continue

            # Compute final score with importance and recency
            importance_score = self._compute_importance(mem)
            if importance_score < min_importance:
                continue

            final_score = base_score * 0.6 + importance_score * 0.2 + self._recency_score(mem) * 0.2
            results.append((mem, final_score))

        # Sort by score, take top-K
        results.sort(key=lambda x: x[1], reverse=True)
        top_memories = [mem for mem, _ in results[:k]]

        # Touch accessed memories
        for mem in top_memories:
            self._sql.touch(mem.id)

        return top_memories, total_candidates

    # -- Entity operations --

    async def add_entity(self, entity: Entity) -> Entity:
        self._sql.add_entity(entity)
        return entity

    def find_entity(self, name: str) -> Entity | None:
        return self._sql.find_entity_by_name(name)

    async def link_memory_entity(self, link: MemoryEntity) -> None:
        self._sql.link_memory_entity(link)

    async def add_relation(self, rel: EntityRelation) -> None:
        self._sql.add_entity_relation(rel)

    def get_related_entities(self, entity_id: str, max_hops: int = 2) -> list[Entity]:
        return self._sql.get_related_entities(entity_id, max_hops)

    def get_memories_for_entity(self, entity_id: str, limit: int = 20) -> list[Memory]:
        return self._sql.get_memories_for_entity(entity_id, limit)

    # -- Stats --

    def count(self, active_only: bool = True) -> int:
        return self._sql.count(active_only)

    def get_all_active(self, limit: int = 1000) -> list[Memory]:
        return self._sql.get_all_active(limit)

    # -- Scoring helpers --

    def _compute_importance(self, mem: Memory) -> float:
        """Compute current importance using Ebbinghaus decay + access reinforcement."""
        now = datetime.now(timezone.utc)
        age_days = (now - mem.created_at).total_seconds() / 86400.0

        half_life = mem.decay_half_life_days or 90.0
        decay = math.exp(-0.693 * age_days / half_life)  # ln(2) ≈ 0.693

        access_boost = 1.0 + math.log(1.0 + mem.access_count) * 0.3

        return min(1.0, mem.importance * decay * access_boost)

    def _recency_score(self, mem: Memory) -> float:
        """Score based on how recently the memory was accessed or created."""
        now = datetime.now(timezone.utc)
        ref = mem.last_accessed or mem.created_at
        days_ago = (now - ref).total_seconds() / 86400.0
        return math.exp(-days_ago * TIME_DECAY_WEIGHT)
