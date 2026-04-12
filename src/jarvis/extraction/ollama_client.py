"""Ollama API client for embeddings and LLM extraction.

Two-model strategy:
- FAST_MODEL (qwen2.5:1.5b, 1GB): entity extraction, tagging — stays loaded, instant
- DEEP_MODEL (qwen3-coder:30b, 18GB): consolidation, reflection — loads on demand
"""

from __future__ import annotations

import json
import logging

import httpx

from jarvis.config import DEEP_MODEL, EMBEDDING_MODEL, FAST_MODEL, OLLAMA_HOST

logger = logging.getLogger(__name__)

_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(base_url=OLLAMA_HOST, timeout=120.0)
    return _http


async def embed(text: str) -> list[float]:
    """Generate an embedding vector. Uses nomic-embed-text (300MB, instant)."""
    if len(text) > 7000:
        text = text[:7000]
    resp = await _client().post(
        "/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


async def fast_extract(prompt: str, system: str = "") -> str:
    """Fast extraction using qwen2.5:1.5b (1GB, ~60 tok/s).

    Use for: entity extraction, tagging, subtype inference, simple JSON output.
    """
    return await _chat(FAST_MODEL, prompt, system, max_tokens=512)


async def fast_extract_json(prompt: str, system: str = "") -> dict:
    """Fast extraction with JSON parsing."""
    raw = await fast_extract(prompt, system)
    return _parse_json(raw)


async def deep_extract(prompt: str, system: str = "") -> str:
    """Deep extraction using qwen3-coder:30b (18GB, loads on demand).

    Use for: consolidation, reflection, complex diff analysis, conflict detection.
    Warning: first call loads 18GB model (~15s), subsequent calls are fast.
    """
    return await _chat(DEEP_MODEL, prompt, system, max_tokens=2048)


async def deep_extract_json(prompt: str, system: str = "") -> dict:
    """Deep extraction with JSON parsing."""
    raw = await deep_extract(prompt, system)
    return _parse_json(raw)


# Keep backward-compatible aliases
async def extract(prompt: str, system: str = "") -> str:
    """Default extraction — uses fast model."""
    return await fast_extract(prompt, system)


async def extract_json(prompt: str, system: str = "") -> dict:
    """Default JSON extraction — uses fast model."""
    return await fast_extract_json(prompt, system)


async def _chat(model: str, prompt: str, system: str, max_tokens: int = 1024) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = await _client().post(
        "/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        },
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = cleaned.find(start_char)
            end = cleaned.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    continue
        logger.warning("Failed to parse JSON from LLM response: %s", raw[:200])
        return {}


async def close() -> None:
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None
