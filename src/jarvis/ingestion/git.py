"""Git ingestion — extract memories from commits."""

from __future__ import annotations

import json
import logging
import subprocess

from jarvis.extraction import ollama_client
from jarvis.extraction.prompts import COMMIT_EXTRACTION_PROMPT, COMMIT_EXTRACTION_SYSTEM
from jarvis.models.entity import Entity, EntityType, MemoryEntity
from jarvis.models.memory import Memory, MemoryType, MemorySubtype

logger = logging.getLogger(__name__)


def get_last_commit(repo_path: str) -> dict | None:
    """Get the last commit's message, files, and diff stats."""
    try:
        message = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%B"], cwd=repo_path, text=True
        ).strip()

        files = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            cwd=repo_path, text=True,
        ).strip()

        stats = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--stat", "-r", "HEAD"],
            cwd=repo_path, text=True,
        ).strip()

        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_path, text=True
        ).strip()

        return {
            "message": message,
            "files": files,
            "stats": stats,
            "hash": commit_hash,
        }
    except subprocess.CalledProcessError as e:
        logger.error("Failed to get last commit: %s", e)
        return None


async def extract_commit_memory(commit: dict) -> tuple[Memory, list[tuple[Entity, str]]]:
    """Extract a structured memory from a git commit."""
    prompt = COMMIT_EXTRACTION_PROMPT.format(
        message=commit["message"],
        files=commit["files"],
        stats=commit["stats"],
    )

    data = await ollama_client.extract_json(prompt, system=COMMIT_EXTRACTION_SYSTEM)

    # Build memory
    subtype_str = data.get("subtype", "code_review")
    try:
        subtype = MemorySubtype(subtype_str)
    except ValueError:
        subtype = MemorySubtype.CODE_REVIEW

    memory = Memory(
        type=MemoryType.EPISODIC,
        subtype=subtype,
        content=data.get("summary", commit["message"]),
        source="git",
        source_ref=commit["hash"],
        tags=data.get("tags", []),
        importance=data.get("importance", 0.5),
    )

    # Build entities
    entities: list[tuple[Entity, str]] = []
    for ent in data.get("entities", []):
        name = ent.get("name", "")
        etype_str = ent.get("type", "concept")
        try:
            etype = EntityType(etype_str)
        except ValueError:
            etype = EntityType.CONCEPT
        if name:
            entities.append((Entity(name=name, entity_type=etype), "subject"))

    return memory, entities
