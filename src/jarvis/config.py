"""Jarvis configuration — paths, model names, thresholds."""

from __future__ import annotations

import os
from pathlib import Path

# Paths
DATA_DIR = Path(os.environ.get("JARVIS_DATA_DIR", Path.home() / "Desktop" / "jarvis" / "data"))
SQLITE_DB = DATA_DIR / "jarvis.db"
VECTOR_DIR = DATA_DIR / "vectors"
EXPORT_DIR = DATA_DIR / "exports"

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("JARVIS_EMBED_MODEL", "nomic-embed-text")
FAST_MODEL = os.environ.get("JARVIS_FAST_MODEL", "qwen2.5:1.5b")  # 1GB, instant — entity extraction, tagging
DEEP_MODEL = os.environ.get("JARVIS_DEEP_MODEL", "qwen3-coder:30b")  # 18GB, on-demand — consolidation, reflection
EMBEDDING_DIM = 768

# Retrieval
DEFAULT_K = 5
SIMILARITY_THRESHOLD = 0.3
DEDUP_THRESHOLD = 0.85
TIME_DECAY_WEIGHT = 0.3

# Forgetting — half-life defaults (days) per (type, subtype)
DECAY_HALF_LIVES: dict[tuple[str, str | None], float] = {
    # Episodic
    ("episodic", "debug_session"): 7.0,
    ("episodic", "code_review"): 14.0,
    ("episodic", "incident"): 30.0,
    ("episodic", None): 14.0,
    # Semantic
    ("semantic", "architecture"): 180.0,
    ("semantic", "convention"): 365.0,
    ("semantic", "domain"): 365.0,
    ("semantic", None): 180.0,
    # Procedural
    ("procedural", "command"): 99999.0,  # effectively never
    ("procedural", "workflow"): 180.0,
    ("procedural", "debugging"): 90.0,
    ("procedural", None): 180.0,
}


def ensure_dirs() -> None:
    """Create data directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
