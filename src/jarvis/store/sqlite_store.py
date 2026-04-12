"""SQLite storage for memory metadata, entities, relations, and FTS."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from jarvis.models.memory import Memory, MemoryType
from jarvis.models.entity import Entity, EntityRelation, MemoryEntity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    subtype TEXT,
    content TEXT NOT NULL,
    summary TEXT,
    source TEXT NOT NULL,
    source_ref TEXT,
    tags TEXT DEFAULT '[]',
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.8,
    decay_half_life_days REAL,
    access_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    last_accessed TEXT,
    valid_from TEXT,
    valid_until TEXT,
    superseded_by TEXT,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    role TEXT DEFAULT 'subject',
    PRIMARY KEY (memory_id, entity_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS entity_relations (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    PRIMARY KEY (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES entities(id),
    FOREIGN KEY (target_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    content,
    summary,
    tags,
    tokenize='porter unicode61'
);
"""


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class SQLiteStore:
    """SQLite-backed store for memory metadata and entity graph."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.executescript(_FTS_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- Memory CRUD --

    def add_memory(self, mem: Memory) -> None:
        self._conn.execute(
            """INSERT INTO memories
               (id, type, subtype, content, summary, source, source_ref, tags,
                importance, confidence, decay_half_life_days, access_count,
                created_at, updated_at, last_accessed,
                valid_from, valid_until, superseded_by, is_active)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mem.id, mem.type.value, mem.subtype.value if mem.subtype else None,
                mem.content, mem.summary, mem.source, mem.source_ref,
                json.dumps(mem.tags), mem.importance, mem.confidence,
                mem.decay_half_life_days, mem.access_count,
                _dt_to_str(mem.created_at), _dt_to_str(mem.updated_at),
                _dt_to_str(mem.last_accessed), _dt_to_str(mem.valid_from),
                _dt_to_str(mem.valid_until), mem.superseded_by,
                1 if mem.is_active else 0,
            ),
        )
        # FTS index
        self._conn.execute(
            "INSERT INTO memories_fts (id, content, summary, tags) VALUES (?,?,?,?)",
            (mem.id, mem.content, mem.summary or "", json.dumps(mem.tags)),
        )
        self._conn.commit()

    def get_memory(self, memory_id: str) -> Memory | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def update_memory(self, mem: Memory) -> None:
        mem.updated_at = datetime.now(timezone.utc)
        self._conn.execute(
            """UPDATE memories SET
               content=?, summary=?, tags=?, importance=?, confidence=?,
               decay_half_life_days=?, access_count=?, updated_at=?,
               last_accessed=?, valid_until=?, superseded_by=?, is_active=?
               WHERE id=?""",
            (
                mem.content, mem.summary, json.dumps(mem.tags),
                mem.importance, mem.confidence, mem.decay_half_life_days,
                mem.access_count, _dt_to_str(mem.updated_at),
                _dt_to_str(mem.last_accessed), _dt_to_str(mem.valid_until),
                mem.superseded_by, 1 if mem.is_active else 0, mem.id,
            ),
        )
        # Update FTS
        self._conn.execute("DELETE FROM memories_fts WHERE id = ?", (mem.id,))
        self._conn.execute(
            "INSERT INTO memories_fts (id, content, summary, tags) VALUES (?,?,?,?)",
            (mem.id, mem.content, mem.summary or "", json.dumps(mem.tags)),
        )
        self._conn.commit()

    def soft_delete(self, memory_id: str) -> bool:
        cur = self._conn.execute(
            "UPDATE memories SET is_active = 0 WHERE id = ?", (memory_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def touch(self, memory_id: str) -> None:
        """Increment access count and update last_accessed."""
        now = _dt_to_str(datetime.now(timezone.utc))
        self._conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, memory_id),
        )
        self._conn.commit()

    # -- Query --

    def search_fts(self, query: str, limit: int = 20) -> list[Memory]:
        """Full-text search over memory content."""
        rows = self._conn.execute(
            """SELECT m.* FROM memories m
               JOIN memories_fts f ON m.id = f.id
               WHERE memories_fts MATCH ? AND m.is_active = 1
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_by_type(
        self, mem_type: str, limit: int = 50, active_only: bool = True
    ) -> list[Memory]:
        where = "type = ?"
        params: list = [mem_type]
        if active_only:
            where += " AND is_active = 1"
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_by_tags(self, tags: list[str], limit: int = 20) -> list[Memory]:
        """Find memories matching any of the given tags."""
        conditions = " OR ".join(["tags LIKE ?" for _ in tags])
        params = [f'%"{t}"%' for t in tags]
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE ({conditions}) AND is_active = 1 ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_all_active(self, limit: int = 1000) -> list[Memory]:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE is_active = 1 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self, active_only: bool = True) -> int:
        where = "WHERE is_active = 1" if active_only else ""
        return self._conn.execute(f"SELECT COUNT(*) FROM memories {where}").fetchone()[0]

    # -- Entity operations --

    def add_entity(self, entity: Entity) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO entities (id, name, entity_type, metadata) VALUES (?,?,?,?)",
            (entity.id, entity.name, entity.entity_type.value, json.dumps(entity.metadata)),
        )
        self._conn.commit()

    def find_entity_by_name(self, name: str) -> Entity | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
        if row is None:
            return None
        return Entity(id=row[0], name=row[1], entity_type=row[2], metadata=json.loads(row[3]))

    def link_memory_entity(self, link: MemoryEntity) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id, role) VALUES (?,?,?)",
            (link.memory_id, link.entity_id, link.role),
        )
        self._conn.commit()

    def add_entity_relation(self, rel: EntityRelation) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO entity_relations (source_id, target_id, relation, strength, metadata) VALUES (?,?,?,?,?)",
            (rel.source_id, rel.target_id, rel.relation, rel.strength, json.dumps(rel.metadata)),
        )
        self._conn.commit()

    def get_related_entities(self, entity_id: str, max_hops: int = 2) -> list[Entity]:
        """BFS graph traversal up to max_hops from an entity."""
        visited: set[str] = {entity_id}
        frontier = {entity_id}
        for _ in range(max_hops):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = self._conn.execute(
                f"""SELECT DISTINCT target_id FROM entity_relations WHERE source_id IN ({placeholders})
                    UNION
                    SELECT DISTINCT source_id FROM entity_relations WHERE target_id IN ({placeholders})""",
                (*frontier, *frontier),
            ).fetchall()
            next_frontier: set[str] = set()
            for (eid,) in rows:
                if eid not in visited:
                    visited.add(eid)
                    next_frontier.add(eid)
            frontier = next_frontier

        visited.discard(entity_id)
        if not visited:
            return []
        placeholders = ",".join("?" for _ in visited)
        rows = self._conn.execute(
            f"SELECT * FROM entities WHERE id IN ({placeholders})", tuple(visited)
        ).fetchall()
        return [Entity(id=r[0], name=r[1], entity_type=r[2], metadata=json.loads(r[3])) for r in rows]

    def get_memories_for_entity(self, entity_id: str, limit: int = 20) -> list[Memory]:
        """Get memories linked to an entity."""
        rows = self._conn.execute(
            """SELECT m.* FROM memories m
               JOIN memory_entities me ON m.id = me.memory_id
               WHERE me.entity_id = ? AND m.is_active = 1
               ORDER BY m.created_at DESC LIMIT ?""",
            (entity_id, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # -- Internal --

    def _row_to_memory(self, row: tuple) -> Memory:
        return Memory(
            id=row[0],
            type=MemoryType(row[1]),
            subtype=row[2],
            content=row[3],
            summary=row[4],
            source=row[5],
            source_ref=row[6],
            tags=json.loads(row[7]) if row[7] else [],
            importance=row[8],
            confidence=row[9],
            decay_half_life_days=row[10],
            access_count=row[11],
            created_at=_str_to_dt(row[12]),
            updated_at=_str_to_dt(row[13]),
            last_accessed=_str_to_dt(row[14]),
            valid_from=_str_to_dt(row[15]),
            valid_until=_str_to_dt(row[16]),
            superseded_by=row[17],
            is_active=bool(row[18]),
        )
