# jarvis-cognitive-memory

A local-first MCP server that gives AI coding assistants brain-inspired long-term memory with Ebbinghaus forgetting curves, dual-process retrieval, and a developer-specific memory ontology.

Works with Claude Code, Cursor, Windsurf, or any MCP-compatible client.

## Why

Every AI coding session starts with amnesia. Your assistant doesn't remember what you worked on yesterday, what debugging approaches failed, or how your codebase is structured. You waste ~15 minutes re-explaining context every time.

Existing memory systems (Mem0, MemGPT, Zep) are domain-agnostic. They don't understand that a debugging hypothesis should fade in days while an architectural decision should persist for months.

Jarvis fixes this with three ideas from cognitive science:

1. **Developer Memory Ontology** — memories have types (episodic, semantic, procedural) with distinct decay rates. Debug sessions fade in 7 days. Architecture persists 6 months. Commands never expire.

2. **Dual-Process Retrieval** — `recall()` for explicit search (System 2) + `prime()` for automatic context activation from your git branch and open files (System 1). No other memory system auto-activates context without a query.

3. **Ebbinghaus Forgetting Curves** — memories decay exponentially but are reinforced by access. Frequently used knowledge becomes permanent. Abandoned knowledge fades naturally.

## Quick Start

```bash
# Prerequisites
brew install ollama
ollama serve
ollama pull nomic-embed-text    # 274 MB — embeddings
ollama pull qwen2.5:1.5b        # 986 MB — fast entity extraction

# Install
git clone https://github.com/HariDarshan2321/jarvis-cognitive-memory.git
cd jarvis-cognitive-memory
uv sync

# Add to Claude Code (~/.mcp.json)
```

```json
{
  "mcpServers": {
    "jarvis": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/jarvis-cognitive-memory", "jarvis"]
    }
  }
}
```

```bash
# Or run the web UI
uv run jarvis-ui  # http://localhost:7777

# Or use Docker
docker compose up
```

## MCP Tools

| Tool | What It Does |
|---|---|
| `remember` | Store a memory with type, tags, and auto-extracted entities |
| `recall` | Semantic + keyword hybrid search, ranked by relevance x importance x recency |
| `prime` | Auto-activate memories from file path, git branch, recent errors (no query needed) |
| `brief` | Get a compressed topic summary (low context cost) |
| `update` | Modify and re-embed an existing memory |
| `forget` | Soft-delete a memory |
| `consolidate` | Merge duplicates, promote episodic clusters to semantic, prune stale memories |
| `reflect` | Discover patterns from recent work |
| `learn` | Ingest from git commits or import Claude Code memories |
| `status` | Memory system health and statistics |

## Memory Types and Decay

| Type | Subtype | Example | Half-Life |
|---|---|---|---|
| Episodic | debug_session | "Spent 2h on SEPA bug, root cause was missing metadata" | 7 days |
| Episodic | code_review | "PR #549: annual_total_potential was missing GB" | 14 days |
| Episodic | incident | "Prod EventBridge source mismatch" | 30 days |
| Semantic | architecture | "Billing service uses Clean Architecture" | 180 days |
| Semantic | convention | "Use conventional commits, mise for tasks" | 365 days |
| Procedural | command | "`uv run pytest -v`" | Never (until superseded) |
| Procedural | debugging | "Check idempotency token format for ghost SP" | 90 days |

Importance formula:

```
importance(t) = base x e^(-ln2 x t / half_life) x (1 + ln(1 + access_count) x 0.3)
```

## Three-Model Strategy

| Model | Size | Role | Latency |
|---|---|---|---|
| nomic-embed-text | 274 MB | Embeddings (every query) | 63ms |
| qwen2.5:1.5b | 986 MB | Entity extraction (every remember) | 600ms |
| qwen3-coder:30b | 18 GB | Deep analysis (consolidation, on-demand) | ~15s load |

Total always-loaded: **1.3 GB**. The 18 GB model only loads for background maintenance tasks.

## Architecture

```
jarvis-cognitive-memory/
├── src/jarvis/
│   ├── server.py              # MCP server (FastMCP, stdio)
│   ├── config.py              # Models, thresholds, paths
│   ├── models/                # Memory, Entity, Retrieval dataclasses
│   ├── store/                 # SQLite + FTS5, LanceDB vectors, unified facade
│   ├── cognitive/             # Ontology, forgetting curves, consolidation, priming
│   ├── extraction/            # Ollama client (fast + deep), prompts
│   ├── ingestion/             # Git hook handler
│   ├── migration/             # Claude Code memory importer
│   └── web/                   # FastAPI + HTMX dashboard and memory browser
├── tests/                     # 40 tests
├── hooks/                     # Git post-commit hook, auto-commit memory
├── Dockerfile
├── docker-compose.yml
└── JARVIS-EXPLAINER.md        # Full architecture deep-dive and comparison
```

Storage: **SQLite** (metadata, entity graph, FTS5 full-text search) + **LanceDB** (768-dim vector embeddings). Both embedded, zero infrastructure.

## How It Compares

| Feature | Jarvis | Mem0 | MemGPT/Letta | Zep | MemPalace | Claude Code Built-in |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Semantic search | Yes | Yes | Yes | Yes | Yes | No |
| Type-aware decay | **Yes** | No | No | No | No | No |
| Auto-priming from file/branch | **Yes** | No | No | No | No | Partial |
| Developer-specific ontology | **Yes** | No | No | No | Partial | Yes |
| Local-first, zero infra | Yes | No | No | No | Yes | Yes |
| Knowledge graph | Yes | Yes | No | Yes | Yes | No |
| Access reinforcement | **Yes** | No | No | No | No | No |

See [JARVIS-EXPLAINER.md](JARVIS-EXPLAINER.md) for detailed comparison with reasoning.

## Research Foundation

- [AgeMem](https://arxiv.org/abs/2601.01885) (2026) — memory-as-tools, LTM/STM separation
- [A-Mem](https://arxiv.org/abs/2409.11531) (NeurIPS 2025) — self-organizing linked knowledge
- [EverMemOS](https://arxiv.org/abs/2501.13956) (2026) — engram-based consolidation
- Ebbinghaus (1885) — forgetting curves with spaced repetition
- Kahneman — dual-process theory (System 1 / System 2)

## Development

```bash
# Run tests
uv run --extra dev pytest tests/ -v

# Run MCP server locally
uv run jarvis

# Run web UI
uv run jarvis-ui

# Docker
docker compose up
```

## License

MIT
