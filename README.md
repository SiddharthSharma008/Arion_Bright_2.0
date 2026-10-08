# Arion Bright 2.0

An agentic research companion built on LangGraph, with a custom MCP server, RAG over a local vector store, and an evaluation harness.

**Status: in progress.** 21 automated tests pass on Python 3.13 with LangGraph 1.2 and the MCP Python SDK 2.3. They run the full agent graph, RAG retrieval with citations, the MCP server (through a real MCP client) and the HTTP API, using a scripted model so no API key is needed. It has not yet been run end to end against a live model.


This is a rebuild of a 2023 university project. The original was a prompt-engineered conversational agent with two personalities and no tools. This version is what the same idea looks like once you add multi-agent orchestration, real tool use, persistent memory, and the ability to measure whether any of it actually works.

---

## What it demonstrates

| Capability | Where |
|---|---|
| Multi-agent orchestration with conditional routing | `src/arion/agents/graph.py` |
| Tool-calling ReAct loop with a step budget | `src/arion/agents/graph.py` |
| Conversation memory via LangGraph checkpointing | `MemorySaver` in `build_graph()` |
| RAG with source attribution | `src/arion/rag/` |
| A custom MCP server (not just an MCP client) | `src/arion/mcp/server.py` |
| Evaluation harness with pass/fail and latency | `src/arion/eval/` |
| Production API surface | `src/arion/api.py` |

---

## Architecture

```
                    ┌──────────┐
   user message ───▶│  router  │  cheap classification pass
                    └────┬─────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ┌──────────┐         ┌────────────┐
        │  direct  │         │ researcher │◀────┐
        └────┬─────┘         └─────┬──────┘     │
             │                     │            │
             │                     ▼            │
             │               ┌───────────┐      │
             │               │   tools   │──────┘
             │               └───────────┘   loop until done
             │                     │         or step budget hit
             └──────────┬──────────┘
                        ▼
                      answer
```

The router exists so trivial questions ("hello, who are you?") don't burn a tool-calling round trip. It's a deliberate cost and latency decision, not decoration.

The same tool functions in `src/arion/tools/registry.py` are consumed by both the LangGraph agent and the MCP server. One definition, two surfaces, so they can't drift apart.

---

## Setup

Requires Python 3.11+.

```bash
git clone <your-repo-url> arion-bright
cd arion-bright
./scripts/setup.sh
source .venv/bin/activate
```

Add your API key to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Then:

```bash
arion ingest --path data/docs    # load the knowledge base
arion chat                       # start talking to it
arion eval                       # run the eval suite
```

---

## Running in VS Code

Open the folder and VS Code will pick up `.vscode/launch.json`. Five run configurations are preconfigured:

- **Arion: Chat (CLI)**: interactive session
- **Arion: Ingest docs**: load documents
- **Arion: Run evals**: evaluation suite
- **Arion: FastAPI server**: HTTP API on port 8000
- **Arion: MCP server (stdio)**: MCP server for external clients

Set breakpoints in `graph.py` and step through the agent's decisions. That's the fastest way to understand what LangGraph is actually doing.

Recommended extensions are in `.vscode/extensions.json`: VS Code will prompt.

---

## Connecting the MCP server to Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arion-bright": {
      "command": "/absolute/path/to/arion-bright/.venv/bin/python",
      "args": ["-m", "arion.mcp.server"],
      "env": { "PYTHONPATH": "/absolute/path/to/arion-bright/src" }
    }
  }
}
```

Restart Claude Desktop. Your knowledge base and calculator become available as tools inside Claude.

---

## Evaluation

`arion eval` runs the suite in `src/arion/eval/cases.py` and reports pass/fail, latency, and which tools were actually called.

Cases test three things deliberately:

1. **Correctness**: does the calculator return the right number
2. **Honesty**: when the knowledge base has no relevant content, does the agent say so rather than fabricate a citation
3. **Routing efficiency**: does a greeting correctly skip the tool path

Adding a case means adding a dataclass entry, not writing code.

---

## Testing

```bash
pytest -v          # 21 tests, no API key or network required
make lint          # ruff check + format check
```

`tests/test_tools.py` includes adversarial cases confirming the calculator rejects code execution attempts. Any tool that evaluates user-supplied strings needs that test.

---

## API

```bash
make api
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is (1200 * 0.15) + 300?"}'
```

Response includes the answer, the session ID (reuse it to continue with memory), the route taken, and which tools fired.

---

## Known limitations

Being explicit about these because pretending they don't exist is worse than naming them.

- **Chunking is naive.** Fixed-size sliding window with sentence-boundary preference. Semantic or structure-aware chunking would retrieve better.
- **No reranking.** Retrieval returns top-k by cosine similarity with no second-pass reranker.
- **Router is a single LLM call.** A fine-tuned classifier or heuristic pre-filter would be cheaper and faster at volume.
- **Evals are assertion-based, not LLM-judged.** Fine for deterministic cases like arithmetic; insufficient for grading answer quality on open questions.
- **In-memory checkpointing.** `MemorySaver` doesn't survive restart. Postgres or SQLite checkpointing would be needed for anything real.

---

## Roadmap

- [ ] Postgres checkpointer for durable sessions
- [ ] Langfuse tracing wired through every node
- [ ] LLM-as-judge scoring for open-ended eval cases
- [ ] Reranking pass on retrieval
- [ ] Streaming responses over the API

---

## Licence

MIT
