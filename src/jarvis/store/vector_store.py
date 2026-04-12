"""LanceDB vector store for semantic similarity search."""

from __future__ import annotations

import logging
from pathlib import Path

import lancedb
import pyarrow as pa

from jarvis.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Schema for the vectors table
_SCHEMA = pa.schema([
    pa.field("id", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
])


class VectorStore:
    """LanceDB-backed vector store for memory embeddings."""

    def __init__(self, db_path: Path) -> None:
        self._db = lancedb.connect(str(db_path))
        try:
            self._table = self._db.open_table("memories")
        except Exception:
            self._table = self._db.create_table("memories", schema=_SCHEMA)

    def add(self, memory_id: str, vector: list[float]) -> None:
        """Add a vector for a memory."""
        self._table.add([{"id": memory_id, "vector": vector}])

    def delete(self, memory_id: str) -> None:
        """Delete a vector by memory ID."""
        self._table.delete(f"id = '{memory_id}'")

    def update(self, memory_id: str, vector: list[float]) -> None:
        """Update a vector — delete then re-add."""
        self.delete(memory_id)
        self.add(memory_id, vector)

    def search(self, query_vector: list[float], limit: int = 20) -> list[tuple[str, float]]:
        """Search for similar vectors. Returns list of (memory_id, distance)."""
        if self._table.count_rows() == 0:
            return []
        results = (
            self._table.search(query_vector)
            .metric("cosine")
            .limit(limit)
            .to_arrow()
        )
        ids = results.column("id").to_pylist()
        distances = results.column("_distance").to_pylist()
        return list(zip(ids, distances))

    def count(self) -> int:
        return self._table.count_rows()
