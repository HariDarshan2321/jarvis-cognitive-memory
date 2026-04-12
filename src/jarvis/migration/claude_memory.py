"""Migration tool — import existing Claude Code memory files into Jarvis."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from jarvis.cognitive.ontology import infer_subtype
from jarvis.models.memory import Memory, MemoryType, MemorySubtype

logger = logging.getLogger(__name__)

# Map Claude memory types to Jarvis types
_TYPE_MAP = {
    "user": MemoryType.SEMANTIC,
    "feedback": MemoryType.PROCEDURAL,
    "project": MemoryType.EPISODIC,
    "reference": MemoryType.SEMANTIC,
}


def parse_memory_file(file_path: Path) -> Memory | None:
    """Parse a single Claude memory markdown file into a Jarvis Memory."""
    try:
        text = file_path.read_text()
    except OSError as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return None

    # Parse frontmatter
    frontmatter: dict[str, str] = {}
    content_start = 0

    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm_text = text[3:end].strip()
            for line in fm_text.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    frontmatter[key.strip()] = val.strip()
            content_start = end + 3

    content = text[content_start:].strip()
    if not content:
        return None

    # Determine type
    claude_type = frontmatter.get("type", "reference")
    mem_type = _TYPE_MAP.get(claude_type, MemoryType.SEMANTIC)

    # Infer subtype
    subtype = infer_subtype(mem_type, content, "claude_memory")

    # Extract tags from content
    tags: list[str] = []
    # Look for service names, ticket IDs, etc.
    ticket_matches = re.findall(r"BLO-\d+", content)
    tags.extend(ticket_matches)

    for service in ["billing", "cloud-scanner", "etl", "backend-data", "commitment", "user-management"]:
        if service.lower() in content.lower():
            tags.append(service)

    # Determine importance based on type
    importance = 0.6 if claude_type in ("reference", "feedback") else 0.5

    name = frontmatter.get("name", file_path.stem)

    return Memory(
        type=mem_type,
        subtype=subtype,
        content=content,
        summary=frontmatter.get("description"),
        source="claude_memory",
        source_ref=str(file_path.name),
        tags=list(set(tags)),
        importance=importance,
    )


def find_memory_files(memory_dir: Path) -> list[Path]:
    """Find all markdown memory files in the Claude memory directory."""
    if not memory_dir.exists():
        return []
    return [f for f in memory_dir.glob("*.md") if f.name != "MEMORY.md"]


async def migrate_all(memory_dir: Path, store) -> list[str]:
    """Import all Claude memory files into Jarvis.

    Args:
        memory_dir: Path to ~/.claude/projects/.../memory/
        store: MemoryStore instance.

    Returns:
        List of imported memory IDs.
    """
    files = find_memory_files(memory_dir)
    imported: list[str] = []

    for file_path in files:
        memory = parse_memory_file(file_path)
        if memory:
            try:
                stored = await store.add(memory)
                imported.append(stored.id)
                logger.info("Imported %s → %s (%s)", file_path.name, stored.id, stored.type.value)
            except Exception as e:
                logger.warning("Failed to import %s: %s", file_path.name, e)

    return imported
