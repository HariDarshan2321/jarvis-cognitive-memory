# Jarvis — Cognitive Developer Memory System

> A local-first MCP server that gives AI coding assistants a brain-inspired long-term memory. Built for developers who use Claude Code, Cursor, or any MCP-compatible tool.

## The Problem

Every time you start a new AI coding session, your assistant has amnesia. It doesn't know:
- What you worked on yesterday
- How your codebase is structured
- What debugging approaches you already tried
- What architectural decisions were made and why

Developers lose **~23 minutes per context switch**. AI assistants make this worse by requiring you to re-explain everything. Existing memory systems (Mem0, MemGPT, Zep) are domain-agnostic — they don't understand how developers think.

## The Solution

```
┌──────────────────────────────────────────────────┐
│              JARVIS MCP SERVER                    │
│                                                  │
│   ┌────────── MCP Tools ─────────────┐           │
│   │ remember  recall  prime  brief   │           │
│   │ update  forget  consolidate      │           │
│   │ reflect  learn  status           │           │
│   └──────────────────────────────────┘           │
│                                                  │
│   ┌────────── Cognitive Engine ──────┐           │
│   │                                  │           │
│   │  Developer Memory Ontology       │           │
│   │  ┌─────────┬──────────┬────────┐ │           │
│   │  │Episodic │Semantic  │Proced- │ │           │
│   │  │(events) │(facts)   │ural   │ │           │
│   │  └─────────┴──────────┴────────┘ │           │
│   │                                  │           │
│   │  Ebbinghaus Forgetting Curves    │           │
│   │  Dual-Process Retrieval          │           │
│   │  Confidence Tracking             │           │
│   └──────────────────────────────────┘           │
│                                                  │
│   ┌─── Storage ──────────────────────┐           │
│   │  SQLite + FTS5  │  LanceDB      │           │
│   │  (metadata,     │  (768-dim     │           │
│   │   graph, text)  │   vectors)    │           │
│   └──────────────────────────────────┘           │
│                                                  │
│   ┌─── Three-Model Strategy ────────┐           │
│   │  nomic-embed  │ qwen2.5:1.5b   │           │
│   │  (274MB,      │ (986MB,        │           │
│   │   embeddings) │  fast extract) │           │
│   │               │                │           │
│   │  qwen3-coder:30b (18GB,       │           │
│   │  deep analysis — on demand)    │           │
│   └──────────────────────────────────┘           │
│                                                  │
└──────────────┬───────────────────────────────────┘
               │ MCP (stdio)
          ┌────┴────┐
          │ Claude  │   (or Cursor, Windsurf,
          │  Code   │    any MCP client)
          └─────────┘
```

## What Makes This Novel

Three contributions that no existing system combines:

### 1. Developer Memory Ontology

Not all memories are equal. A debugging hypothesis should fade in days. An architectural decision should persist for months. A command you run daily should never expire.

```
┌──────────────┬───────────────────┬──────────────┐
│ Memory Type  │ Example           │ Decay        │
├──────────────┼───────────────────┼──────────────┤
│ Episodic     │ "Fixed SEPA bug"  │ 7-30 days    │
│ (events)     │ "PR #549 merged"  │              │
├──────────────┼───────────────────┼──────────────┤
│ Semantic     │ "Billing uses     │ 180-365 days │
│ (facts)      │  Clean Arch"      │              │
├──────────────┼───────────────────┼──────────────┤
│ Procedural   │ "Run: uv run      │ Never        │
│ (how-to)     │  pytest -v"       │ (until       │
│              │                   │  superseded) │
└──────────────┴───────────────────┴──────────────┘
```

Each memory type has a distinct Ebbinghaus-inspired decay curve:

```
importance(t) = base × e^(-ln2 × t/half_life) × (1 + ln(1+access_count) × 0.3)
```

Memories you keep accessing get reinforced. Memories you never revisit fade naturally.

### 2. Dual-Process Retrieval

Inspired by cognitive science (Kahneman's System 1/System 2):

**System 1 — `prime()` (automatic)**
```
You open services/billing/app/...
→ Jarvis auto-activates billing memories
→ No query needed — context signals (file path, git branch,
  recent errors) trigger memory activation
→ Like how your brain "just knows" relevant context
```

**System 2 — `recall()` (explicit)**
```
You ask: "how does rebalancing work?"
→ Hybrid search: vector similarity + keyword FTS + importance scoring
→ Returns top-K ranked results
→ Deliberate, focused retrieval
```

No existing memory system has automatic contextual priming.

### 3. Three-Model Strategy

Instead of one model doing everything (and hanging your system):

```
┌─────────────────┬──────────┬────────┬──────────────────────┐
│ Model           │ Size     │ Speed  │ Role                 │
├─────────────────┼──────────┼────────┼──────────────────────┤
│ nomic-embed     │ 274 MB   │ 63ms   │ Every query          │
│                 │ (always  │        │ (embed, search)      │
│                 │  loaded) │        │                      │
├─────────────────┼──────────┼────────┼──────────────────────┤
│ qwen2.5:1.5b   │ 986 MB   │ 600ms  │ Every remember       │
│                 │ (always  │        │ (entity extraction)  │
│                 │  loaded) │        │                      │
├─────────────────┼──────────┼────────┼──────────────────────┤
│ qwen3-coder    │ 18 GB    │ ~15s   │ Weekly maintenance   │
│   :30b         │ (on-     │ load   │ (consolidation,      │
│                │  demand) │        │  reflection)         │
└─────────────────┴──────────┴────────┴──────────────────────┘

Total always-loaded: 1.3 GB (on a 24GB machine)
```

## Key Design Decisions

### Why SQLite + LanceDB (not Postgres, not Neo4j)?

```
Decision: Embedded databases, zero infrastructure

WHY:
├── Personal tool → no server to manage
├── SQLite handles metadata, graph traversal (recursive CTEs),
│   and full-text search (FTS5) in a single file
├── LanceDB handles vector search, embedded like SQLite
│   (no server, disk-based, handles millions of vectors)
└── Total infrastructure: 0 running services

TRADEOFF:
├── No concurrent write access (single process)
├── Graph traversal limited to ~3 hops (fine for personal use)
└── Cloud migration = swap SQLite → Postgres + pgvector (same interface)
```

### Why MCP Protocol (not REST API, not library)?

```
Decision: Expose memory as MCP tools over stdio

WHY:
├── Claude Code, Cursor, Windsurf all speak MCP
├── The LLM decides WHEN to remember/recall (memory-as-tools pattern)
├── stdio transport = no HTTP server, no port conflicts
└── One config line in .mcp.json to enable

TRADEOFF:
├── Single client at a time (stdio is 1:1)
└── Web UI needs a separate process (jarvis-ui on port 7777)
```

### Why Ebbinghaus Curves (not TTL, not keep-everything)?

```
Decision: Type-specific exponential decay with access reinforcement

WHY:
├── TTL (time-to-live) is binary — alive or dead, no nuance
├── Keep-everything floods retrieval with stale noise
├── Ebbinghaus (1885) showed memory strength decays exponentially
│   but is reinforced by retrieval — empirically proven
└── Developer knowledge has natural decay: debug sessions fade,
    architecture persists, commands are permanent

TRADEOFF:
├── Half-life values (7d, 180d, etc.) are educated guesses
├── Need real usage data to tune — currently based on cognitive science
└── A memory could decay below threshold before you need it again
    (mitigated by access reinforcement — if you use it, it stays)
```

### Why Local-First (not cloud)?

```
Decision: Everything runs on your Mac, no cloud dependency

WHY:
├── Your code context is sensitive (AWS keys, customer data references)
├── Zero latency — no network round trips for memory ops
├── Works offline, on planes, without internet
└── You control your data completely

TRADEOFF:
├── No cross-device sync (yet — V2 plan: encrypted S3 backup)
├── No team sharing (yet — V3 plan: Postgres + API layer)
└── Backup is manual (cp data/ somewhere)
```

## How Jarvis Compares to Existing Systems

I researched every major AI memory system (verified from their repos and docs, not guessed). Here's what each does and what Jarvis does differently.

### The Landscape

| System | What It Is | Storage | Semantic Search | Auto-Decay | Developer-Aware | Auto-Priming |
|--------|-----------|---------|:-:|:-:|:-:|:-:|
| **[MemPalace](https://github.com/MemPalace/mempalace)** | Local memory with palace metaphor (wings/rooms) | ChromaDB + SQLite | Yes | No | Partial | No |
| **[Mem0](https://github.com/mem0ai/mem0)** | Universal memory layer for AI apps | 24+ vector DBs + Neo4j | Yes | No | No | No |
| **[Letta/MemGPT](https://github.com/letta-ai/letta)** | Agent framework with OS-like memory paging | In-context + vector DB | Yes | Yes (summarization) | No | No |
| **[Zep/Graphiti](https://github.com/getzep/graphiti)** | Temporal knowledge graph engine | Neo4j graph DB | Yes (hybrid) | No (temporal versioning) | No | No |
| **[Cognee](https://github.com/topoteretes/cognee)** | Knowledge engine with 14 retrieval modes | Graph + vector + SQL | Yes | Partial | No | No |
| **Claude Code** | Built-in CLAUDE.md + auto-memory | Markdown files | No | No | Yes | Partial (path rules) |
| **Cursor** | Rules files + Notepads | Local files | No | No | Yes | Rules auto-inject |
| **Windsurf** | Auto-generated workspace memories | Local files (opaque) | Unknown | Unknown | Partial | Relevance-based |
| **Jarvis** | Cognitive developer memory (MCP) | SQLite + LanceDB | Yes | Yes (Ebbinghaus) | Yes | Yes (file + branch) |

### What Each System Is Missing (and Why It Matters)

**MemPalace** (96.6% on LongMemEval — the highest score):
- Stores conversations verbatim with a palace metaphor (wings → rooms → drawers)
- Has 19 MCP tools and Claude Code hooks
- **But:** No automatic decay — facts persist until you manually invalidate them. No contextual priming from file paths or git branches. The AAAK compression dialect actually *hurts* retrieval (84.2% vs 96.6% raw). Closest competitor to Jarvis but still treats all memories equally.

**Mem0** (48K GitHub stars, $24M raised):
- Extracts structured facts from conversations automatically
- Supports 24+ vector DB backends
- **But:** Only 49.0% on LongMemEval (vs MemPalace's 96.6%). No automatic decay — memories persist forever. No developer awareness. Requires explicit `search()` calls. Designed for chatbots and customer support, not coding workflows.

**Letta/MemGPT** (29K stars, the pioneer):
- The agent manages its own memory by paging in/out like OS virtual memory
- Has genuine strategic forgetting (through summarization when context overflows)
- **But:** Every memory operation costs an LLM call (the agent decides what to retrieve). No developer context. No file/branch awareness. Forgetting is incidental (context overflow triggers it), not principled (no Ebbinghaus curves, no type-specific decay).

**Zep/Graphiti** (best temporal model):
- Bi-temporal knowledge graph — tracks when facts became true AND when they were superseded
- Hybrid retrieval: semantic + BM25 + graph traversal
- **But:** Requires Neo4j (heavy infrastructure). No decay — just temporal versioning. No developer awareness. Designed for enterprise knowledge management, not personal coding memory.

**Cognee** (most sophisticated retrieval):
- 14 retrieval modes including graph traversal, chain-of-thought, auto-routing
- Zero-config with embedded defaults (SQLite + LanceDB + KuzuDB)
- **But:** No automatic decay. No developer-specific features. LLM-dependent extraction quality. Conflict resolution not addressed.

**Claude Code built-in memory:**
- Path-scoped rules (`.claude/rules/`) auto-load when you open matching files
- Developer-focused by design
- **But:** No semantic search — memories are loaded by file path, not meaning. 200-line cap on MEMORY.md. No knowledge graph. No vector DB. No decay. Machine-local only.

### What Jarvis Does That Nobody Else Does

Three things combined. Each individual idea exists somewhere, but no system has all three:

**1. Developer Memory Ontology (nobody has this)**

Every system treats all memories as the same type of thing — a "fact" or a "message." Jarvis understands that developer knowledge has structure:

```
A debugging session ("tried X, it failed because Y")
  → should fade in 7 days if never revisited

An architectural fact ("billing uses Clean Architecture")
  → should persist for 6 months

A command ("uv run pytest -v")
  → should NEVER expire until superseded
```

This isn't just labeling. The type determines the decay curve, default importance, and retrieval priority. MemGPT has forgetting, but it's "forget when context overflows" — not "debug sessions decay faster than architecture knowledge."

**2. Dual-Process Retrieval with `prime()` (nobody has this)**

Every other system requires you to explicitly search. Jarvis has two modes:

- **System 2 (explicit):** `recall("how does rebalancing work")` — like everyone else
- **System 1 (automatic):** `prime(cwd="/path/to/repo")` — reads your git branch, modified files, recent commits, and **auto-activates relevant memories without any query**

When you open `services/billing/app/...`, billing memories activate. When your branch is `gc-blo-1774`, ticket BLO-1774 context activates. This is how expert developers' brains work — you don't consciously search for context when you open a familiar file. You just know.

Claude Code's path-scoped rules are the closest — they auto-load when you read matching files. But they're markdown files, not semantically searchable, and they don't know about git branches.

**3. Principled Forgetting (Letta has forgetting, but not principled)**

Letta forgets when context overflows. Jarvis forgets based on Ebbinghaus (1885):

```
importance(t) = base × e^(-ln2 × t/half_life) × (1 + ln(1+access_count) × 0.3)
```

This means:
- Memories you keep accessing get **reinforced** (access_count boosts importance)
- Memories you never revisit **fade** at a rate determined by their type
- A debugging hypothesis that was wrong fades in days
- An architectural truth you reference weekly becomes nearly permanent

No other system has type-aware decay with access reinforcement.

### Honest Gaps (Where Others Are Better)

| System | Better At | Why |
|--------|-----------|-----|
| **MemPalace** | Raw retrieval accuracy (96.6% LongMemEval) | Palace metaphor with structural filtering is clever |
| **Zep/Graphiti** | Temporal reasoning ("what was true on March 5?") | Bi-temporal model is more rigorous than our simple `valid_from/valid_until` |
| **Cognee** | Retrieval flexibility (14 modes) | Chain-of-thought graph traversal, auto-mode selection |
| **Letta** | Agent autonomy | The agent itself manages memory; Jarvis relies on the caller (Claude) |
| **Mem0** | Ecosystem breadth (24+ vector DBs) | Jarvis only supports LanceDB (embedded, by design) |

Jarvis trades retrieval sophistication for **developer-specific intelligence**. It's not trying to be the best general memory system. It's trying to be the best memory system for a developer using an AI coding assistant.

## What It Looks Like in Practice

### Before Jarvis
```
You: "I need to work on BLO-1774"
Claude: *reads MEMORY.md (200 lines, mostly irrelevant)*
        *reads 5-10 code files to understand context*
        *asks you to explain what you did last time*
        ~15 minutes before productive work starts
```

### After Jarvis
```
You: "I need to work on BLO-1774"
Claude: *calls prime(cwd="/path/to/ground-control")*
        → auto-detects branch gc-blo-1774
        → retrieves: ticket context, last session's progress,
          related debugging notes, test commands
        → 200 tokens of precisely relevant context
        ~2 seconds, starts working immediately
```

### Auto-learning
```
You make a commit → PostToolUse hook fires
→ Jarvis extracts: commit message, files changed, tags
→ Stores as episodic memory with entities linked
→ 0.6 seconds, completely invisible

You never manually "taught" Jarvis anything.
It learned from watching you work.
```

## Numbers

| Metric | Value |
|---|---|
| Memories stored | 45 (after initial import) |
| Storage on disk | ~5 MB (SQLite + LanceDB) |
| RAM (always loaded) | 1.3 GB (embed + fast model) |
| `remember()` latency | 0.66s |
| `recall()` latency | <0.2s |
| `prime()` latency | <0.5s |
| Context cost per recall | ~625 tokens (k=5) |
| Tests | 40 passing |
| Source files | 28 Python files |

## Tech Stack

- **Python 3.12+** with uv
- **MCP SDK** (FastMCP, stdio transport)
- **SQLite** + FTS5 (metadata, entities, full-text search)
- **LanceDB** (768-dim vector embeddings)
- **Ollama** (nomic-embed-text + qwen2.5:1.5b + qwen3-coder:30b)
- **FastAPI + HTMX + Tailwind** (web UI)
- **Docker Compose** (Ollama + Jarvis)

## Research Foundation

Built on ideas from:
- **AgeMem** (arXiv:2601.01885) — memory-as-tools, LTM/STM separation
- **A-Mem** (NeurIPS 2025) — self-organizing linked knowledge
- **EverMemOS** (Jan 2026) — engram-based consolidation
- **Ebbinghaus (1885)** — forgetting curves with spaced repetition
- **Kahneman** — dual-process theory (System 1/System 2)

## Getting Started

```bash
# Prerequisites
brew install ollama
ollama serve
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b

# Install
git clone <repo> && cd jarvis
uv sync

# Run MCP server (add to ~/.mcp.json for Claude Code)
uv run jarvis

# Run web UI
uv run jarvis-ui  # http://localhost:7777

# Docker (for others)
docker compose up
```

## License

MIT

---

*Built by Darshan with Claude Code in one session. Named after the AI that never forgets.*
