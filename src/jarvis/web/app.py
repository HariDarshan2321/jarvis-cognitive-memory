"""Jarvis Web UI — dashboard + memory browser."""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from jarvis.cognitive.forgetting import compute_importance, compute_recency
from jarvis.store.memory_store import MemoryStore

TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="Jarvis", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def _format_ago(dt: datetime | None) -> str:
    if dt is None:
        return "never"
    now = datetime.now(timezone.utc)
    delta = now - dt
    if delta.days > 30:
        return f"{delta.days // 30}mo ago"
    if delta.days > 0:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    return "just now"


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    store = _get_store()
    all_mems = store.get_all_active(limit=1000)
    total = len(all_mems)
    forgotten = store.count(active_only=False) - total

    # Type counts
    type_counts = {"episodic": 0, "semantic": 0, "procedural": 0}
    source_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}

    # Decay buckets
    decay_buckets = {"strong": 0, "moderate": 0, "fading": 0, "stale": 0}

    for m in all_mems:
        type_counts[m.type.value] = type_counts.get(m.type.value, 0) + 1
        source_counts[m.source] = source_counts.get(m.source, 0) + 1
        for t in m.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

        imp = compute_importance(m)
        if imp >= 0.5:
            decay_buckets["strong"] += 1
        elif imp >= 0.3:
            decay_buckets["moderate"] += 1
        elif imp >= 0.1:
            decay_buckets["fading"] += 1
        else:
            decay_buckets["stale"] += 1

    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    # Recent memories
    recent = sorted(all_mems, key=lambda m: m.created_at, reverse=True)[:10]
    recent_data = [
        {
            "id": m.id,
            "type": m.type.value,
            "subtype": m.subtype.value if m.subtype else "",
            "content_preview": m.content[:120].replace("\n", " "),
            "source": m.source,
            "importance": round(compute_importance(m), 2),
            "created_ago": _format_ago(m.created_at),
            "access_count": m.access_count,
            "tags": m.tags[:5],
        }
        for m in recent
    ]

    return templates.TemplateResponse(request, "dashboard.html", {
        "total": total,
        "forgotten": forgotten,
        "type_counts": type_counts,
        "source_counts": source_counts,
        "decay_buckets": decay_buckets,
        "top_tags": top_tags,
        "recent": recent_data,
    })


# ── Memory Browser ───────────────────────────────────────────────────────────

@app.get("/memories", response_class=HTMLResponse)
async def memories(
    request: Request,
    type: str | None = None,
    source: str | None = None,
    tag: str | None = None,
    q: str | None = None,
):
    store = _get_store()

    if q:
        mems = store._sql.search_fts(q, limit=50)
    elif tag:
        mems = store._sql.get_by_tags([tag], limit=50)
    elif type:
        mems = store._sql.get_by_type(type, limit=50)
    else:
        mems = store.get_all_active(limit=100)

    # Filter by source if specified
    if source:
        mems = [m for m in mems if m.source == source]

    # Sort by importance (current, with decay)
    mems_with_score = [(m, compute_importance(m)) for m in mems]
    mems_with_score.sort(key=lambda x: x[1], reverse=True)

    memory_data = [
        {
            "id": m.id,
            "type": m.type.value,
            "subtype": m.subtype.value if m.subtype else "",
            "content_preview": m.content[:200].replace("\n", " "),
            "content_full": m.content,
            "source": m.source,
            "source_ref": m.source_ref or "",
            "importance": round(score, 2),
            "confidence": round(m.confidence, 2),
            "decay_half_life": m.decay_half_life_days,
            "created_ago": _format_ago(m.created_at),
            "created_at": m.created_at.strftime("%Y-%m-%d") if m.created_at else "",
            "access_count": m.access_count,
            "tags": m.tags,
            "is_stale": score < 0.1,
        }
        for m, score in mems_with_score
    ]

    # Collect filter options
    all_mems = store.get_all_active(limit=1000)
    all_sources = sorted(set(m.source for m in all_mems))
    all_tags = sorted(set(t for m in all_mems for t in m.tags))

    return templates.TemplateResponse(request, "memories.html", {
        "memories": memory_data,
        "count": len(memory_data),
        "filters": {"type": type, "source": source, "tag": tag, "q": q},
        "all_sources": all_sources,
        "all_tags": all_tags,
    })


# ── Memory Detail ────────────────────────────────────────────────────────────

@app.get("/memory/{memory_id}", response_class=HTMLResponse)
async def memory_detail(request: Request, memory_id: str):
    store = _get_store()
    mem = store._sql.get_memory(memory_id)
    if mem is None:
        return HTMLResponse("<h1>Memory not found</h1>", status_code=404)

    score = compute_importance(mem)
    recency = compute_recency(mem)

    return templates.TemplateResponse(request, "memory_detail.html", {
        "memory": {
            "id": mem.id,
            "type": mem.type.value,
            "subtype": mem.subtype.value if mem.subtype else "",
            "content": mem.content,
            "summary": mem.summary,
            "source": mem.source,
            "source_ref": mem.source_ref,
            "tags": mem.tags,
            "importance_base": round(mem.importance, 3),
            "importance_current": round(score, 3),
            "confidence": round(mem.confidence, 3),
            "decay_half_life": mem.decay_half_life_days,
            "access_count": mem.access_count,
            "created_at": mem.created_at.strftime("%Y-%m-%d %H:%M") if mem.created_at else "",
            "last_accessed": _format_ago(mem.last_accessed),
            "recency": round(recency, 3),
            "is_active": mem.is_active,
        },
    })


# ── Forget (POST) ───────────────────────────────────────────────────────────

@app.post("/memory/{memory_id}/forget")
async def forget_memory(memory_id: str):
    store = _get_store()
    store._sql.soft_delete(memory_id)
    return RedirectResponse("/memories", status_code=303)


# ── Search API (for HTMX) ───────────────────────────────────────────────────

@app.get("/search", response_class=HTMLResponse)
async def search_fragment(request: Request, q: str = ""):
    if not q or len(q) < 2:
        return HTMLResponse("")

    store = _get_store()
    memories, total = await store.recall(q, k=10, min_importance=0.0)

    rows = []
    for m in memories:
        rows.append(f"""
        <tr class="border-b border-gray-700 hover:bg-gray-800/50">
            <td class="py-2 px-3"><span class="badge badge-{m.type.value}">{m.type.value}</span></td>
            <td class="py-2 px-3"><a href="/memory/{m.id}" class="text-blue-400 hover:underline">{m.content[:120].replace(chr(10), ' ')}</a></td>
            <td class="py-2 px-3 text-gray-400">{', '.join(m.tags[:3])}</td>
            <td class="py-2 px-3 text-right">{round(compute_importance(m), 2)}</td>
        </tr>""")

    if not rows:
        return HTMLResponse('<tr><td colspan="4" class="py-4 text-center text-gray-500">No memories found</td></tr>')

    return HTMLResponse("".join(rows))
