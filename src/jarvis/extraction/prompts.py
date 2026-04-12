"""Prompt templates for LLM-based memory extraction."""

ENTITY_EXTRACTION_SYSTEM = """\
You are a knowledge extraction engine for a developer memory system.
Extract entities and their types from the given text.
Return valid JSON only — no explanation, no markdown fences."""

ENTITY_EXTRACTION_PROMPT = """\
Extract named entities from this developer memory. Return JSON:
{{
  "entities": [
    {{"name": "entity name", "type": "service|person|ticket|concept|tool|file", "role": "subject|object|context"}}
  ]
}}

Entity types:
- service: a software service, module, or component (e.g., "billing-service", "cloud-scanner")
- person: a team member or contributor
- ticket: a ticket/issue reference (e.g., "BLO-1774")
- concept: a technical concept (e.g., "Clean Architecture", "EventBridge")
- tool: a development tool (e.g., "uv", "pytest", "Ollama")
- file: a specific file path

Text:
{text}"""

SUMMARY_SYSTEM = """\
You are a concise summarization engine. Compress the given text to under 100 words,
preserving key facts, decisions, and actionable knowledge. No fluff."""

SUMMARY_PROMPT = """\
Summarize this developer memory in under 100 words. Keep technical details, decisions, and key facts:

{text}"""

BRIEFING_SYSTEM = """\
You are a developer context briefing engine. Create a concise briefing that helps
a developer quickly understand relevant context. Focus on: what the system does,
recent activity, known issues, and key commands/workflows."""

BRIEFING_PROMPT = """\
Create a concise briefing (max {max_tokens} tokens) about "{topic}" from these memories:

{memories}

Format: 2-3 short paragraphs. Include specific commands, file paths, or ticket IDs where relevant.
No introduction or preamble — start directly with the content."""

COMMIT_EXTRACTION_SYSTEM = """\
You are a developer memory extraction engine. Analyze a git commit and extract
structured knowledge. Return valid JSON only."""

COMMIT_EXTRACTION_PROMPT = """\
Analyze this git commit and extract a structured memory. Return JSON:
{{
  "summary": "one-line summary of what changed and why",
  "type": "episodic",
  "subtype": "debug_session|code_review|incident",
  "tags": ["tag1", "tag2"],
  "entities": [
    {{"name": "entity", "type": "service|ticket|concept|tool|file"}}
  ],
  "importance": 0.5
}}

Importance guide: bug fix = 0.6, feature = 0.5, refactor = 0.4, docs = 0.3, chore = 0.2

Commit message: {message}
Files changed: {files}
Diff stats: {stats}"""
