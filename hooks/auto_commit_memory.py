"""Auto-memorize git commits — called by Claude Code PostToolUse hook.

Runs async: detects the repo from cwd, extracts the last commit, stores as episodic memory.
"""

import asyncio
import os
import subprocess
import sys

from jarvis.models.memory import Memory, MemoryType, MemorySubtype
from jarvis.store.memory_store import MemoryStore


async def memorize_commit():
    # Find the git repo root
    try:
        repo = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return  # not in a git repo

    # Get last commit info
    try:
        message = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%B"], cwd=repo, text=True
        ).strip()
        short_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True
        ).strip()
        files = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            cwd=repo, text=True,
        ).strip()
    except Exception:
        return

    if not message:
        return

    # Build tags from commit message and file paths
    tags = []
    import re
    tickets = re.findall(r"[A-Z]+-\d+", message.upper())
    tags.extend(tickets)

    services = {"billing", "cloud-scanner", "etl", "backend-data", "user-management"}
    for svc in services:
        if svc in files.lower() or svc in message.lower():
            tags.append(svc)

    # Detect subtype from conventional commit
    msg_lower = message.lower()
    if any(w in msg_lower for w in ["fix", "bug", "hotfix"]):
        subtype = MemorySubtype.DEBUG_SESSION
    elif any(w in msg_lower for w in ["feat", "add", "implement"]):
        subtype = MemorySubtype.CODE_REVIEW
    else:
        subtype = MemorySubtype.CODE_REVIEW

    # Store
    store = MemoryStore()
    mem = Memory(
        type=MemoryType.EPISODIC,
        subtype=subtype,
        content=f"Commit {short_hash}: {message}\n\nFiles: {files[:500]}",
        source="git",
        source_ref=short_hash,
        tags=list(set(tags)),
        importance=0.5,
    )
    await store.add(mem)
    store.close()


asyncio.run(memorize_commit())
