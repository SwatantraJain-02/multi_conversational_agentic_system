# Multi-agent chatbot (LangGraph + MCP)

A chatbot that answers **web**, **database** and **hybrid** questions. A planner/supervisor
routes work to a web-search agent and a text-to-SQL agent; every tool call goes through an MCP
server. The full roadmap is in [docs/plan/implementation_plan.md](docs/plan/implementation_plan.md).

**Status:** the data layer and the DB MCP server are built (roadmap phases 0–2). The web MCP
server, the agent graph, memory and the UI are next.

## What exists

| Part | Where |
|---|---|
| DB MCP server: six read-only tools over the Northwind database, Streamable HTTP | `src/multi_agenti_conversation_system/mcp/servers/db_server/` |
| SQL guardrail: AST-based, single `SELECT` only, row limit, PII detection | `.../guardrails/sql_validator.py` |
| Schema catalog (table docs, PII flags, few-shot SQL examples) | `.../catalog.py`, `data/*.yaml` |
| Typed tool responses shared by the server and (later) the agents | `.../mcp/contracts/db.py` |
| Chat / embedding / Chroma adapters | `.../llm.py`, `.../vectorstore.py` |
| Settings, logging and tracing | `.../config.py`, `.../observability.py` |

## Setup

You need Docker, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com/) (for embeddings).

```bash
cp .env.example .env          # fill in the passwords; see the comments for who reads what
docker compose up -d          # Postgres 14 + Chroma, both bound to 127.0.0.1
scripts/init_db.sh            # databases, Northwind data, read-only role
uv sync
ollama pull nomic-embed-text
uv run python scripts/build_schema_index.py   # fills Chroma; re-run when data/*.yaml changes
uv run db-mcp-server          # http://localhost:5555/mcp
```

Then, in another terminal:

```bash
uv run python scripts/smoke_db_mcp.py   # end-to-end check of the running server
```

To use the server from VS Code, copy `.mcp.json.example` to `.mcp.json` and add your `DB_MCP_TOKEN`.

The chat model (`CHAT_PROVIDER=openrouter` or `ollama`) is only needed by the agent app and
`scripts/smoke_llm.py`; the DB server does not read `OPENROUTER_API_KEY`.

## Server endpoints

| Path | Auth | Meaning |
|---|---|---|
| `/mcp` | bearer token (`DB_MCP_TOKEN`) | the MCP endpoint |
| `/health` | none | liveness: the process is up |
| `/ready` | none | readiness: 200 only when Postgres, Chroma, the schema index and Ollama all answer, else 503 with a per-check boolean |

An empty `DB_MCP_TOKEN` is accepted only on a loopback address.

## Tests and checks

```bash
uv run pytest                    # offline unit tests; needs no .env, network or Docker
uv run pytest -m integration     # needs `docker compose up -d` and a filled .env
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run lint-imports              # architecture rules, see below
```

CI runs the same checks (`.github/workflows/ci.yml`). `-m live` is reserved for tests that call
model APIs; `scripts/smoke_llm.py` is the live check for now.

## Architecture rules

Dependencies point one way, and `import-linter` fails the build if they don't
(`[tool.importlinter]` in `pyproject.toml`):

- **Pure core**: `catalog`, `guardrails` and `mcp.contracts` import no settings, adapters or
  infrastructure, so the agent and the server can both use them and tests need nothing running.
- **Settings are read only at composition roots** (`server.main`, `build_service`, script
  `main()` functions). Everything else receives what it needs as arguments.
- **DB server layers**: `server.py` (transport) → `service.py` (tool logic) → `db.py`,
  `schema_index.py` (adapters). The service is tested with fakes, no database or Chroma.
- **Agents reach tools only through MCP** and never import `mcp.servers`; they parse tool results
  with the models in `mcp/contracts/`.

## Observability

- Every tool call logs one line: `tool_call tool=… ok=… duration_ms=… row_count=… trace_id=…`.
  Row values and SQL text are never logged at INFO; error messages (which can echo SQL) are
  logged only at `LOG_LEVEL=DEBUG`.
- Tracing is off until an exporter is set. To see traces locally:

  ```bash
  docker compose --profile tracing up -d jaeger
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uv run db-mcp-server
  # UI: http://localhost:16686
  ```

  Or set `OTEL_CONSOLE_EXPORT=true` to print spans. Spans carry counts and flags, never SQL or rows.

## Safety layers for the SQL tools

1. `validate_select` accepts exactly one `SELECT` (CTEs allowed), rejects DML/DDL anywhere in the
   tree, dangerous functions and unknown tables, and enforces a row limit.
2. The server runs it again before every query, so nothing the LLM writes runs unchecked.
3. The database role `agent_readonly` is read-only with a 5 s statement timeout, the last line
   of defence if the first two are bypassed.
