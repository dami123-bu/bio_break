---
name: architecture
description: Best practices for Python RAG, MCP, and agentic AI application architecture
---

## Architecture Best Practices

### RAG (Retrieval-Augmented Generation)

**Ingestion**
- Chunk documents at semantic boundaries (paragraphs), not fixed token counts
- Store metadata (source, date, doc type) alongside embeddings — you'll need it for filtering and attribution
- Make ingestion idempotent: re-running should update, not duplicate
- Keep the raw source files separate from the vector store — the store is a derived artifact

**Retrieval**
- Always set a similarity threshold — don't return chunks just because they're the closest match
- Return `k` candidates but re-rank or filter before passing to the LLM
- Log what was retrieved and what was discarded — essential for debugging poisoning attacks
- Never pass raw retrieved text directly to a tool-calling agent; summarize or sanitize first (trust boundary)

**Trust Boundary**
- Treat retrieved content as untrusted user input, not as system instructions
- Use a two-agent pattern: a retrieval agent that only reads, and an action agent that only acts
- The retrieval agent should never have tool access; the action agent should never see raw document text

---

### Agentic Applications (LangChain / ReAct)

**Agent Design**
- Keep tools small and single-purpose — one tool per action, not one tool per workflow
- Validate tool inputs before execution; don't trust the LLM's argument formatting
- Make tools return structured results (dict/dataclass), not free-form strings
- Log every tool call with its inputs and outputs — agents are hard to debug without traces

**Prompt Engineering**
- Separate system instructions from retrieved context using clear delimiters
- Be explicit about what the agent is and is not allowed to do
- Don't embed business logic in the prompt — put it in tool implementations

**State & Memory**
- Prefer stateless agents where possible — explicit inputs, explicit outputs
- If you need memory, make it a first-class component with a clear schema, not a blob appended to the prompt
- Session state should be persisted externally (file, DB), not held in memory

---

### MCP (Model Context Protocol)

**Server Design**
- Each MCP server should expose a single, coherent domain (e.g., one server for LIMS, one for alerts)
- Tools exposed via MCP should be side-effect-free where possible; mutations should be explicit and logged
- Always validate and sanitize inputs at the MCP server boundary — the client is untrusted
- Return structured errors, not exceptions — the agent needs to reason about failures

**Security**
- MCP servers are a privilege boundary: only expose the minimum set of tools needed
- Never expose tools that can modify system state without an explicit confirmation step
- Log all MCP tool invocations with caller identity, inputs, and outputs

---

### General Python Practices

**Module Structure**
- One responsibility per module — don't mix ingestion, retrieval, and agent logic in one file
- Use `__init__.py` to define the public API of a package; keep internals private
- Configuration (API keys, paths, model names) goes in environment variables or a config file, never hardcoded

**Dependencies**
- Pin dependencies in `pyproject.toml`; use a lockfile (`uv.lock`) for reproducibility
- Prefer the official SDKs (Anthropic, OpenAI, LangChain) over raw HTTP calls

**Error Handling**
- Handle errors at system boundaries (tool calls, API calls, DB queries) — not everywhere
- Fail loudly during development; use structured logging in evaluation/experiment runs
- Don't swallow exceptions silently — especially in agent tool implementations
