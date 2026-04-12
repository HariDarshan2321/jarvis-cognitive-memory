"""Tests for SQLite and Vector stores."""

import tempfile
from pathlib import Path

import pytest

from jarvis.models.memory import Memory, MemoryType, MemorySubtype
from jarvis.models.entity import Entity, EntityType, MemoryEntity, EntityRelation
from jarvis.store.sqlite_store import SQLiteStore
from jarvis.store.vector_store import VectorStore


@pytest.fixture
def sql_store(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    yield store
    store.close()


@pytest.fixture
def vec_store(tmp_path):
    return VectorStore(tmp_path / "vectors")


def _make_memory(**kwargs) -> Memory:
    defaults = dict(
        type=MemoryType.EPISODIC,
        content="Test memory content",
        source="manual",
    )
    defaults.update(kwargs)
    return Memory(**defaults)


class TestSQLiteStore:
    def test_add_and_get(self, sql_store):
        mem = _make_memory(content="Billing uses Clean Architecture")
        sql_store.add_memory(mem)
        got = sql_store.get_memory(mem.id)
        assert got is not None
        assert got.content == "Billing uses Clean Architecture"
        assert got.type == MemoryType.EPISODIC

    def test_soft_delete(self, sql_store):
        mem = _make_memory()
        sql_store.add_memory(mem)
        assert sql_store.soft_delete(mem.id)
        got = sql_store.get_memory(mem.id)
        assert got is not None
        assert not got.is_active

    def test_fts_search(self, sql_store):
        sql_store.add_memory(_make_memory(content="Clean Architecture pattern in billing"))
        sql_store.add_memory(_make_memory(content="Python pytest testing framework"))
        results = sql_store.search_fts("Clean Architecture")
        assert len(results) == 1
        assert "Clean Architecture" in results[0].content

    def test_touch_increments_access(self, sql_store):
        mem = _make_memory()
        sql_store.add_memory(mem)
        sql_store.touch(mem.id)
        sql_store.touch(mem.id)
        got = sql_store.get_memory(mem.id)
        assert got.access_count == 2
        assert got.last_accessed is not None

    def test_get_by_type(self, sql_store):
        sql_store.add_memory(_make_memory(type=MemoryType.SEMANTIC, content="fact"))
        sql_store.add_memory(_make_memory(type=MemoryType.EPISODIC, content="event"))
        semantic = sql_store.get_by_type("semantic")
        assert len(semantic) == 1
        assert semantic[0].content == "fact"

    def test_get_by_tags(self, sql_store):
        sql_store.add_memory(_make_memory(tags=["billing", "fix"]))
        sql_store.add_memory(_make_memory(tags=["etl"]))
        results = sql_store.get_by_tags(["billing"])
        assert len(results) == 1

    def test_count(self, sql_store):
        sql_store.add_memory(_make_memory())
        sql_store.add_memory(_make_memory())
        assert sql_store.count() == 2

    def test_entity_crud(self, sql_store):
        entity = Entity(name="billing-service", entity_type=EntityType.SERVICE)
        sql_store.add_entity(entity)
        found = sql_store.find_entity_by_name("billing-service")
        assert found is not None
        assert found.entity_type == EntityType.SERVICE

    def test_entity_relations(self, sql_store):
        e1 = Entity(name="billing", entity_type=EntityType.SERVICE)
        e2 = Entity(name="stripe", entity_type=EntityType.TOOL)
        sql_store.add_entity(e1)
        sql_store.add_entity(e2)
        sql_store.add_entity_relation(
            EntityRelation(source_id=e1.id, target_id=e2.id, relation="depends_on")
        )
        related = sql_store.get_related_entities(e1.id)
        assert len(related) == 1
        assert related[0].name == "stripe"

    def test_memory_entity_link(self, sql_store):
        mem = _make_memory()
        entity = Entity(name="billing", entity_type=EntityType.SERVICE)
        sql_store.add_memory(mem)
        sql_store.add_entity(entity)
        sql_store.link_memory_entity(MemoryEntity(memory_id=mem.id, entity_id=entity.id))
        mems = sql_store.get_memories_for_entity(entity.id)
        assert len(mems) == 1


class TestVectorStore:
    def test_add_and_search(self, vec_store):
        vec_store.add("m1", [0.1] * 768)
        vec_store.add("m2", [0.9] * 768)
        results = vec_store.search([0.1] * 768, limit=2)
        assert len(results) == 2
        assert results[0][0] == "m1"  # closest match

    def test_delete(self, vec_store):
        vec_store.add("m1", [0.1] * 768)
        vec_store.delete("m1")
        assert vec_store.count() == 0

    def test_update(self, vec_store):
        vec_store.add("m1", [0.1] * 768)
        vec_store.update("m1", [0.9] * 768)
        results = vec_store.search([0.9] * 768, limit=1)
        assert results[0][0] == "m1"

    def test_empty_search(self, vec_store):
        results = vec_store.search([0.1] * 768)
        assert results == []
