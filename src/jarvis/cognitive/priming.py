"""Dual-Process Retrieval — System 1: Contextual Priming.

This is Pillar 3 of Jarvis's novel contribution. Instead of explicit
queries (System 2), priming automatically activates relevant memories
based on context signals: file paths, git branches, error messages.

Smart prime: can auto-detect context from the working directory by
reading git state (branch, recent commits, modified files).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from jarvis.cognitive.forgetting import compute_importance, compute_recency
from jarvis.models.memory import Memory
from jarvis.store.memory_store import MemoryStore


# Service detection from file paths
_SERVICE_PATTERNS: dict[str, list[str]] = {
    "billing": ["billing", "invoice", "payment", "stripe", "sepa"],
    "cloud-scanner": ["cloud-scanner", "commitment", "savings-plan", "rebalance"],
    "etl": ["etl", "airflow", "dag", "pipeline"],
    "backend-data": ["backend-data", "dashboard", "api/routes"],
    "user-management": ["user-management", "auth", "account"],
}

_TICKET_RE = re.compile(r"[A-Z]+-\d+")


def auto_detect_context(cwd: str | None = None) -> dict:
    """Auto-detect development context from the working directory.

    Reads git state: branch, recent commits, modified files, last commit message.
    Returns a dict of detected signals.
    """
    detected: dict = {
        "git_branch": None,
        "recent_commit_messages": [],
        "modified_files": [],
        "cwd": cwd,
    }

    if not cwd:
        return detected

    try:
        # Current branch
        detected["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, text=True, stderr=subprocess.DEVNULL,
        ).strip()

        # Last 3 commit messages (one-line)
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-3"],
            cwd=cwd, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if log:
            detected["recent_commit_messages"] = log.split("\n")

        # Modified/staged files
        status = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=cwd, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if status:
            detected["modified_files"] = status.split("\n")[:10]

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return detected


def extract_context_signals(
    file_path: str | None = None,
    git_branch: str | None = None,
    recent_errors: list[str] | None = None,
    mentioned_text: str | None = None,
    cwd: str | None = None,
    auto_context: dict | None = None,
) -> dict:
    """Extract queryable signals from the current dev context."""
    signals: dict = {
        "services": set(),
        "tickets": set(),
        "keywords": set(),
        "files": set(),
    }

    # Merge auto-detected context
    if auto_context:
        git_branch = git_branch or auto_context.get("git_branch")
        for f in auto_context.get("modified_files", []):
            signals["files"].add(f)
        for msg in auto_context.get("recent_commit_messages", []):
            mentioned_text = (mentioned_text or "") + " " + msg

    all_text = " ".join(filter(None, [file_path, git_branch, mentioned_text] + (recent_errors or [])))

    # Detect services from file paths and modified files
    all_paths = [file_path or ""] + list(signals.get("files", []))
    for path in all_paths:
        for service, patterns in _SERVICE_PATTERNS.items():
            if any(p in path.lower() for p in patterns):
                signals["services"].add(service)

    # Detect tickets from branch names, text, and commit messages
    tickets = _TICKET_RE.findall(all_text.upper())
    signals["tickets"].update(tickets)

    if git_branch:
        tickets_from_branch = _TICKET_RE.findall(git_branch.upper())
        signals["tickets"].update(tickets_from_branch)
        for service, patterns in _SERVICE_PATTERNS.items():
            if any(p in git_branch.lower() for p in patterns):
                signals["services"].add(service)

    # Extract keywords from errors
    if recent_errors:
        error_keywords = {"timeout", "connection", "permission", "denied", "null",
                          "undefined", "missing", "failed", "error", "exception",
                          "traceback", "import", "module", "attribute", "key"}
        for error in recent_errors:
            words = set(error.lower().split())
            signals["keywords"].update(words & error_keywords)

    # Extract keywords from commit messages (actions/topics)
    if mentioned_text:
        action_words = {"fix", "add", "remove", "update", "refactor", "migrate",
                        "debug", "test", "deploy", "revert", "hotfix", "optimize"}
        words = set(mentioned_text.lower().split())
        signals["keywords"].update(words & action_words)

    return {k: list(v) for k, v in signals.items()}


async def prime(
    store: MemoryStore,
    file_path: str | None = None,
    git_branch: str | None = None,
    recent_errors: list[str] | None = None,
    mentioned_text: str | None = None,
    cwd: str | None = None,
    max_memories: int = 5,
) -> tuple[list[Memory], dict]:
    """System 1 retrieval — activate relevant memories from context signals.

    If cwd is provided, auto-detects git branch, recent commits, and modified files.
    Returns (activated_memories, signals_used).
    """
    # Auto-detect context if cwd provided
    auto_context = auto_detect_context(cwd) if cwd else None

    signals = extract_context_signals(
        file_path, git_branch, recent_errors, mentioned_text,
        cwd=cwd, auto_context=auto_context,
    )

    # Build composite query — weight tickets and services higher
    query_parts: list[str] = []
    # Tickets are most specific — put them first
    query_parts.extend(signals.get("tickets", []))
    # Services are next
    query_parts.extend(signals.get("services", []))
    # Then action keywords
    query_parts.extend(signals.get("keywords", []))
    # Fallback: modified file paths give some signal
    if not query_parts:
        query_parts = list(signals.get("files", []))[:3]
    if not query_parts:
        query_parts = [file_path or git_branch or "recent work"]

    query = " ".join(query_parts)

    # Hybrid retrieval
    memories, total = await store.recall(
        query=query,
        k=max_memories * 3,  # over-fetch more for better re-ranking
        min_importance=0.05,
    )

    # Re-rank with context-aware boosting
    boosted: list[tuple[Memory, float]] = []
    for mem in memories:
        base = compute_importance(mem) * 0.3 + compute_recency(mem) * 0.2

        # Service match boost (strong signal)
        for service in signals.get("services", []):
            if service.lower() in mem.content.lower() or service.lower() in " ".join(mem.tags).lower():
                base += 0.25

        # Ticket match boost (strongest signal)
        for ticket in signals.get("tickets", []):
            if ticket.upper() in mem.content.upper() or ticket.upper() in " ".join(mem.tags).upper():
                base += 0.35

        # File overlap boost
        for f in signals.get("files", []):
            if f in mem.content:
                base += 0.15

        # Keyword match boost
        for kw in signals.get("keywords", []):
            if kw.lower() in mem.content.lower():
                base += 0.05

        boosted.append((mem, base))

    boosted.sort(key=lambda x: x[1], reverse=True)
    activated = [m for m, _ in boosted[:max_memories]]

    return activated, signals
