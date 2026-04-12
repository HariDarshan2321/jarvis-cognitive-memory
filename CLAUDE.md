# Jarvis — Cognitive Developer Memory System

## Using Jarvis Memory (MCP Tools)

When the user is working with you, use Jarvis to persist and retrieve knowledge:

### Auto-behaviors (do these without being asked)
- **On commit**: The PostToolUse hook auto-memorizes commits. No action needed.
- **On session start**: Use `prime(cwd="/path/to/repo")` to load relevant context.
- **When asked about past work**: Use `recall(query)` before reading files.

### Explicit saves (when user says "remember this", "save this", "note this")
- Use `remember(content, type, tags)` to store knowledge.
- Choose the right type: `episodic` for events, `semantic` for facts, `procedural` for how-to.
- Always include relevant tags (service names, ticket IDs).

### When researching or planning
- After completing research, use `remember()` to save key findings as `semantic` memories.
- After agreeing on a plan, save the decision and rationale as `semantic` memory.
- Save debugging solutions as `procedural` memories with the `debugging` subtype.

### Retrieval preference
- Use `recall(query, k=5)` for specific lookups (returns truncated content, low context cost).
- Use `brief(topic)` for topic overviews (even lower context cost).
- Use `prime(cwd=...)` at session start for automatic context loading.

## Development

```bash
# Run tests
uv run --extra dev pytest tests/ -v

# Run MCP server (stdio — started by Claude Code automatically)
uv run jarvis

# Run web UI
uv run jarvis-ui  # http://localhost:7777

# Docker
docker compose up
```

## Architecture
- MCP server: `src/jarvis/server.py` (FastMCP, stdio transport)
- Storage: SQLite + FTS5 (metadata) + LanceDB (vectors)
- Models: nomic-embed-text (274MB, embeddings), qwen2.5:1.5b (986MB, fast extraction), qwen3-coder:30b (18GB, deep analysis)
