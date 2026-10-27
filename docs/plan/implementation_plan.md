# Capstone Implementation Plan — Multi-Agent Chatbot (LangGraph + MCP + Layered Memory)

## Context

The capstone ([docs/Capstone Problem Statement](../docs/)) asks for one chatbot that answers **web**, **database** and **hybrid** questions:
- a **Planner/Supervisor** routes work to a **Web Search Agent** and a **Text-to-SQL Agent**;
- every tool call goes through **MCP servers**;
- memory is layered: short-term, long-term semantic and episodic, a user profile, and a semantic cache;
- it also needs guardrails, human-in-the-loop (HITL) approval, prompt versioning and a repeatable evaluation.

**How we work together:** you're the sole developer. I'm the reviewer and guide, and I don't create, edit or run anything in your project. The one exception: once you approve, I'll copy this plan into `plan/`. Each phase has three parts:
- **Learn:** concepts to understand first. Ask me as many basic questions as you like.
- **Do:** the steps you carry out.
- **Gate:** the definition of done, and what you bring back for me to review (terminal output, a diff, screenshots). We only move on once a gate passes.

**Note:** the problem statement stops at §6, so it has no deliverables or grading section. The evaluation, prompt-versioning, limitations and architecture-diagram work below comes from §3 Objectives and §4 Reference Architecture.

---

## Decisions locked in

| Area | Choice | Notes |
|---|---|---|
| LLM | **OpenRouter** chat model (`CHAT_MODEL`), switchable to **Ollama** with `CHAT_PROVIDER=ollama` | *As built.* The original plan was a local `qwen3`; Phase 0 ended on OpenRouter. `scripts/smoke_llm.py` is the tool-calling / structured-output check to re-run for any new model. Embeddings stay on Ollama. |
| Embeddings | **Ollama** `nomic-embed-text` | Use one embedding model everywhere. Re-calibrate the 0.92 cache threshold for it (Phase 10). |
| Business DB | **Your existing Postgres container**, loaded with **Northwind** (`pthom/northwind_psql`) | Its sales schema suits "internal sales vs market trend" hybrid questions. |
| Vector DB | **Your existing Chroma container** (`localhost:5430`) | It holds **all** embeddings. Your Postgres image has no pgvector, and Chroma is already running. |
| Web search | **Tavily** | Needs `TAVILY_API_KEY` (free tier). |
| UI | **Streamlit** | Async bridging needs care (Phase 11). |

**Storage split (a good point for the report):** Postgres holds relational and state data, and Chroma holds every vector.
- **Postgres DB `northwind`:** business data, accessed only through a read-only role.
- **Postgres DB `agent_memory`:** the LangGraph checkpointer (conversation state) and the user-profile key-value store.
- **Chroma:** schema docs, seed SQL examples, user facts, episodes and the semantic cache, each in its own collection.

## Changeable defaults

| Area | Default | Why |
|---|---|---|
| Python / packaging | Python 3.14 (as built, see `.python-version`), **uv**, `pyproject.toml`, `src/` layout | Fast, gives you a lockfile, current standard |
| MCP server SDK | `mcp` (official, 2.3.x) | Decorator-based tools, with input schemas taken from type hints |
| MCP client | `langchain[mcp]` → `from langchain.mcp import MCPAdapter` | The old `langchain-mcp-adapters` is **archived**. The new namespace is **beta**, so pin its version. |
| MCP transports | **Both servers: Streamable HTTP** | *As built.* The DB server was built as HTTP instead of stdio (Phase 2). A long-lived process owns the connection pool and the Chroma/Ollama clients, can be probed (`/ready`) and traced on its own, and the web server keeps its rate-limiter state (G3). |
| Checkpointer | `AsyncPostgresSaver` (`langgraph-checkpoint-postgres`) | Conversation state survives restarts |
| User profile store | LangGraph `AsyncPostgresStore` **without** a vector index | A plain key-value store namespaced by `user_id`, so it needs no pgvector |
| Vector client | `chromadb.HttpClient` + `langchain-chroma`, embeddings computed by `OllamaEmbeddings` | Keeps one embedding model and skips Chroma's built-in default model |
| SQL parsing | `sqlglot` | AST-based SELECT-only checks and LIMIT injection, safer than regex |
| Page cleaning | `httpx` + `trafilatura` | Robust main-text extraction |
| Re-ranking | `flashrank` (local cross-encoder) | No API key, fast |
| Config | `pydantic-settings` + `.env` | Typed, all in one place |
| Tests / lint | `pytest`, `pytest-asyncio`, `ruff` | |
| Tracing (optional) | LangSmith | It's an external service, so check that your org allows sending traces there |

> **Versions:** pin everything and commit `uv.lock`. Current releases (Oct 2026): langgraph 1.2.x, mcp 2.3.x, langchain-ollama 1.1.x. Let `uv add` resolve versions rather than copying them from blog posts. Your `chromadb` Python client version must match your Chroma server's version (check it in Phase 1).

---

## Target architecture

**One explicit interpretation.** The specialist agents are **LangGraph subgraphs**, reached from the Supervisor by **conditional edges**. Their **low-level tools** (`search_web`, `fetch_page`, `search_schema`, `run_select_query`, …) are served by **MCP servers** and discovered at startup through `MCPAdapter`. This matches §4.2. Say it in your report, because §3 could also be read as "the agents themselves are MCP tools".

**Planner/Supervisor is one *role* built from two nodes.** Both live in `graph/supervisor.py`:
- **`planner`** (LLM, structured output)
  - Runs once at the start of a turn, and again when a step fails (re-planning).
  - Returns a `RouteDecision` with `intent`, `plan`, `next_agent` and `reason`.
- **`supervisor`** (the router)
  - Runs after every agent returns.
  - Deterministic code marks the step done, picks the next pending step, enforces the budget and decides when the turn is finished.
  - It calls the LLM only to judge a doubtful result (empty, error, off-topic). That judgement either accepts the result or triggers a re-plan.

Why split them:
- §5.1 asks to *separate planning from execution*.
- Small local models are more reliable at a single planning call than at re-deciding everything on each hop.
- It saves one LLM call per hop.

```
User (Streamlit) ── user_id + thread_id ──► LangGraph app   (checkpointer auto-restores thread state)
                                            │
 START → contextualize ─► check_cache ──(hit)──────────────────────────────► respond → END
   (standalone question        │ (miss)
    + entity extraction)       ▼
                          load_context  (profile + relevant facts + relevant episodes)
                               ▼
                          planner ──► supervisor ──┬─► web_agent (subgraph) ─┐
                             ▲  (re-plan)  ▲       ├─► sql_agent (subgraph) ─┤
                             └─────────────┴───────┴──── results ◄───────────┘
                                           │ (plan done / budget hit / direct answer)
                                           ▼
                                        finalize (grounded answer + citations / SQL) → respond
                                           ▼
                                        write_memory (facts, episode, learned SQL pair, cache, trim history) → END

 sql_agent:  retrieve_schema → generate_sql → validate → approval_gate[interrupt()] → execute ─┐
                                    ▲──────────── error fed back (max 2 retries) ◄──────────────┘
 web_agent:  search → fetch → chunk → rerank → compress → summarise (with URLs)

 MCP:     web_server (Streamable HTTP: search_web, fetch_page)
          db_server  (Streamable HTTP: list_tables, get_schema, search_schema, get_few_shots, explain_query, run_select_query)
 Storage: Postgres[northwind] (read-only role) · Postgres[agent_memory] (checkpoints + profile KV)
          Chroma (schema_docs, sql_seed_examples, user_facts, episodes, semantic_cache)
```

## Target project structure (built incrementally)

```
multi_conversational_agentic_system/
├── pyproject.toml, uv.lock, .python-version, .env.example, .env (gitignored), .gitignore, README.md
├── docker-compose.yml              # reproduces your Postgres + Chroma containers (Phase 1)
├── db/                             # 01_databases.sql, 02_northwind.sql, 03_roles.sql
├── docs/                           # problem statement, architecture.md, diagram, final report
├── plan/                           # this plan
├── prompts/                        # supervisor/planner_v1.yaml, sql/v1.yaml, web/v1.yaml, memory/v1.yaml ...
├── data/                           # schema_descriptions.yaml, few_shots.yaml
├── scripts/                        # smoke_ollama.py, build_schema_index.py, smoke_mcp.py, calibrate_cache.py
├── src/multi_agenti_conversation_system/
│   ├── config.py                   # per-concern settings classes + cached get_*() getters, nothing read at import
│   ├── catalog.py                  # SchemaCatalog: schema docs, PII flags, few-shots (pure; shared by agent and server)
│   ├── llm.py, prompts.py          # build_chat_model / build_embeddings (provider switch, retries, task prefixes)
│   ├── observability.py            # logging, OpenTelemetry setup, span() helper
│   ├── guardrails/                 # sql_validator.py (shared by agent AND server), input_guard.py
│   ├── mcp/
│   │   ├── contracts/              # typed tool responses shared by servers and agents: db.py, web.py (pure, pydantic only)
│   │   ├── client.py               # MCPAdapter config + tool discovery → registry per server (Phase 4)
│   │   └── servers/
│   │       ├── http_auth.py        # BearerAuth + is_loopback, shared by every server
│   │       ├── db_server/          # server.py (transport) → service.py (logic) → db.py, schema_index.py (adapters)
│   │       └── web_server/         # same shape: server.py, service.py, tavily_client.py, fetch.py, rate_limit.py
│   ├── vectorstore.py              # Chroma HttpClient + collection factory (cosine space)
│   ├── memory/                     # profile.py, facts.py, episodes.py, cache.py, history.py
│   ├── graph/
│   │   ├── state.py                # AgentState, Plan, PlanStep, RouteDecision
│   │   ├── supervisor.py           # planner + supervisor nodes
│   │   ├── agents/                 # sql_agent.py, web_agent.py
│   │   ├── nodes/                  # contextualize.py, check_cache.py, load_context.py, finalize.py, write_memory.py
│   │   └── builder.py
│   └── ui/streamlit_app.py
├── eval/                           # questions.yaml, run_eval.py, results/
└── tests/                          # unit/, integration/
```

## Architecture rules (apply from Phase 3 on)

The Phase 0–2 code was refactored to these rules, and `import-linter` (`[tool.importlinter]` in `pyproject.toml`, run in CI) enforces the ones marked **[enforced]**. Each phase adds its own contracts when it creates its package: a contract that names a missing module fails with "module not found".

**Dependency rules**
1. **Pure core** **[enforced]**: `catalog`, `guardrails` and `mcp/contracts` import no settings, adapters or infrastructure (no `config`, `llm`, `vectorstore`, `psycopg`, `chromadb`, `httpx`, `langchain*`, `mcp` SDK). Both the agent and the servers use them, and tests need nothing running.
2. **Settings are read only at composition roots**: each server's `main` / `build_service`, `graph/builder.py` for the agent app, and script `main()` functions. Everything else receives what it needs as arguments. **[enforced]** for `db.py`, `schema_index.py`, `service.py`, `http_auth.py` and `observability.py`, which must not import `config` at runtime (type-only imports are fine).
3. **Agents reach tools only through MCP.** `graph`, `memory` and `ui` must not import `mcp.servers`; they parse tool results with the models in `mcp/contracts/`. Add this contract in Phase 5.
4. **Servers are independent**: `db_server` and `web_server` must not import each other. Shared code goes in `mcp/servers/http_auth.py` or the pure core. Add this contract in Phase 3.
5. **Layers:** `ui` → `graph` → (`memory`, `mcp.client`) → adapters (`llm`, `vectorstore`) → pure core. **[enforced]** inside `db_server`: `server` → `service` → (`db`, `schema_index`).

**Conventions per area**
- **Every MCP server** (the Phase 3 web server included) copies the DB server's shape:
  - `server.py`: thin tool adapters and routes, no logic;
  - `service.py`: all tool logic, with adapters passed in (Protocols such as `DbReader` / `SchemaSearch` where a test needs a fake);
  - adapters for each outside service, each with `ping()` for readiness;
  - `build_service()` and `create_app(service, token)` as the composition root;
  - `/health` (liveness), `/ready` (every dependency, per-check booleans, 503 when down), bearer auth through `http_auth.py`;
  - typed responses from `mcp/contracts/<server>.py`, each tool returning one model with `ok` / `error`.
- **Graph nodes** take their dependencies from a context object built once in `graph/builder.py` (MCP tool registry, chat model, embeddings, Chroma client, stores). Nodes never call `get_*()` globals.
- **Memory modules** (`facts`, `episodes`, `cache`) take the Chroma client and an `Embeddings` through their constructors, and each owns its collection name and metadata schema.
- **Prompts** live in `prompts/<area>/vN.yaml` and are loaded through `prompts.py`. Graph code holds no inline prompt strings.
- **Models:** build them with `build_chat_model(cfg)` and `build_embeddings(cfg)`. Embedding task prefixes and retry policy belong to the embedder, not to the code that stores or searches vectors.
- **Resilience:** every outbound call has a timeout from settings. Retry only idempotent calls (embeddings), with backoff. Database queries are not retried; the error goes back to the agent.
- **Observability:** one `tool_call` log line per tool call, and child spans through `observability.span()`. Spans and INFO logs carry counts and flags, never SQL text or row values.
- **The UI** imports only `graph.builder` and runs one background event loop (G6).

---

## Phases

### Phase 0 — Environment and tooling
- **Learn:**
  - uv basics: `init`, `add`, `sync`, `run`;
  - why a `src/` layout helps;
  - `.env` vs `.env.example`;
  - how Ollama serves models.
- **Do:**
  1. `git init`, plus a `.gitignore` covering `.env`, `.venv/`, `__pycache__/` and `eval/results/`.
  2. `uv init --package` with Python 3.14. The package is named `multi_agenti_conversation_system`.
  3. Install Ollama, then `ollama pull qwen3:8b` and `ollama pull nomic-embed-text`.
  4. Add the core deps (`langgraph`, `langchain`, `"langchain[mcp]"`, `langchain-ollama`, `pydantic-settings`) and the dev deps (`pytest`, `pytest-asyncio`, `ruff`).
  5. `config.py` holds model names, DB URLs, the Chroma host/port, the Tavily key and the thresholds. `llm.py` holds the model factory functions.
  6. Write `scripts/smoke_ollama.py`. It should check three things:
     - (a) the model emits a tool call for a bound dummy tool;
     - (b) `with_structured_output` works on a small Pydantic model;
     - (c) embedding two sentences gives a sensible cosine similarity.
- **Gate:** your `pyproject.toml`, the `tree` output, the smoke-script output, and your RAM/GPU specs, so we can lock in the chat model.

### Phase 1 — Data layer (using your running containers)
**Current state (checked):**
- **Postgres:** `postgres:latest` (18.6) on `:5432`.
  - Its data sits in an **anonymous volume**.
  - **No pgvector**.
  - Only the default `postgres` database exists.
- **Chroma:** `chromadb/chroma` on host `:5430`, bind-mounted to `~/chroma-data`.
- **Learn:**
  - named vs anonymous Docker volumes;
  - why to pin a Postgres version instead of `:latest` (built with `postgres:14.24`);
  - Postgres roles and grants, `default_transaction_read_only` and `statement_timeout`;
  - Chroma's HTTP client and its distance metrics.
- **Do:**
  1. **Recommended, and cheapest to do now while the DB is empty:** write a `docker-compose.yml` that reproduces both containers. Use these settings:
     - Postgres: pin the version (built with `postgres:14.24`), a **named volume**, and a healthcheck.
     - Chroma: pin the image to the version you have now.

     Then swap your hand-started containers for the compose ones. This makes the project reproducible for whoever grades it. An anonymous volume gets orphaned if the container is ever recreated, and `:latest` can jump to a new major version that won't start on old data.
  2. SQL scripts in `db/`, run with `psql`:
     - create the databases `northwind` and `agent_memory`;
     - load Northwind;
     - create the role **`agent_readonly`** with SELECT only on `northwind.public`, plus `default_transaction_read_only=on` and `statement_timeout=5s`;
     - create the role **`agent_app`** with read-write access to `agent_memory` only.
  3. Optional: shift Northwind's 1996–98 dates forward, so the hybrid "vs market trend" demos are convincing.
  4. Write `data/schema_descriptions.yaml`. Each table gets:
     - a business description;
     - a `domain` tag: `sales`, `customers`, `products`, `hr` or `logistics`;
     - column descriptions, with `pii: true` on personal fields.
  5. Write `data/few_shots.yaml` with 15–20 question → SQL pairs, each tagged with a `domain`.
  6. Chroma check:
     - read the server version (`/api/v2/version`) and pin the matching `chromadb` client;
     - confirm you can create a collection with the **cosine** distance space (Chroma's default is L2, which would break every similarity threshold);
     - remember Chroma returns a *distance*: `similarity = 1 − distance`.
- **Gate:** `psql` output proving `agent_readonly` can SELECT and that DELETE/DROP fail, the two YAML files, and a Chroma round-trip.

### Phase 2 — DB MCP server (Streamable HTTP, built as `db_server/`)
- **Learn:**
  - MCP basics: server, client, tools, transports;
  - how type hints become a tool's input schema;
  - why guardrails also sit *at the server boundary*.
- **Do:**
  1. `guardrails/sql_validator.py` (`sqlglot`) must:
     - accept exactly one SELECT statement (CTEs allowed);
     - reject DDL/DML anywhere in the parse tree;
     - inject or clamp a `LIMIT` of `MAX_ROWS`;
     - optionally enforce a table allowlist.
  2. `scripts/build_schema_index.py` embeds the table/column docs into Chroma collection `schema_docs` and the few-shots into `sql_seed_examples`. Metadata: `table`, `domain`, `pii`.
  3. Tools:
     - `list_tables()`
     - `get_schema(tables)`
     - `search_schema(question, k, domain?)`
     - `get_few_shots(question, k, domain?)`
     - `explain_query(sql)`: returns the planner's row estimate and whether the query touches PII columns
     - `run_select_query(sql, max_rows)`
  4. Every tool validates its input. `run_select_query` connects **only** as `agent_readonly` and re-runs the validator on the server side.
  5. Unit tests for the validator with adversarial inputs: `DELETE`, `; DROP`, comment tricks, `pg_sleep`, a CTE wrapping DML, mixed case.
- **Gate:** passing tests, plus **MCP Inspector** (`npx @modelcontextprotocol/inspector`) output that calls each tool, including a blocked DELETE.

### Phase 3 — Web MCP server (Streamable HTTP)
- **Learn:**
  - the Streamable HTTP transport;
  - token-bucket rate limiting;
  - prompt injection through web content (fetched text is *data*, never instructions);
  - SSRF (server-side request forgery).
- **Do:**
  1. `search_web(query, max_results≤10, recency: day|week|month|year|any)` calls Tavily and returns `{title, url, snippet, published_date}`.
  2. `fetch_page(url)` must:
     - allow only http/https;
     - block private and localhost IPs;
     - use a timeout and a size cap;
     - clean the page with `trafilatura`.
  3. Add an in-process token-bucket rate limiter that returns a structured error when tripped. It only works because the server is long-lived.
  4. Run it as its own process on `localhost:8001/mcp`.
- **Gate:** Inspector output showing:
  - a search;
  - a fetch;
  - a blocked `127.0.0.1` fetch;
  - a burst that trips the rate limit.

### Phase 4 — MCP client and tool discovery
- **Learn:**
  - the `MCPAdapter` lifecycle: by default each tool call opens its own session;
  - holding the adapter context open reuses one connection.
- **Do:**
  1. `mcp/client.py`:
     - builds an `mcpServers` config with `db` and `web` both as URLs (the DB server needs the `Authorization: Bearer <DB_MCP_TOKEN>` header);
     - discovers tools with `list_tools()`;
     - returns a registry grouped by server.
  2. `scripts/smoke_mcp.py` lists the tools and calls one from each server.
  3. Add a trivial new tool to a server and restart. It should be discovered **without changing any graph code**.
  4. **Check auth and reachability:** a call without the token gets 401, and the DB server's `/ready` shows Postgres, Chroma and Ollama reachable from the server process.
  5. **Time the per-call cost (G3):** time 5 consecutive DB calls. HTTP spawns no process, so what remains is MCP session setup per call. If it is slow, hold the adapter context open for the whole turn.
- **Gate:**
  - the smoke-script output;
  - the timing numbers;
  - your own explanation of why both servers are long-lived HTTP processes rather than stdio children.

### Phase 5 — Graph state and skeleton
- **Learn:**
  - `StateGraph`, nodes, edges and conditional edges;
  - reducers (`add_messages`) and subgraphs;
  - `thread_id` and checkpointers;
  - why short-term memory needs *no load node*: the checkpointer restores state automatically.
- **Do:**
  1. `state.py` defines:
     - `PlanStep`: `id`, `agent`, `task`, `status`, `result`.
     - `RouteDecision`:
       - `intent: Literal["web","sql","hybrid","direct"]`, which eval scores;
       - `next_agent: Literal["web","sql","finalize"]`;
       - `reason`;
       - `plan: list[PlanStep]`.
     - `AgentState`:
       - conversation: `messages`, `standalone_question`, `entities`;
       - planning: `intent`, `plan`, `current_step`, `dispatch_count`;
       - results: `citations`, `sql_used`, `cache_hit`;
       - session context: `selected_domain`, `filters`;
       - user and memory: `user_id`, `profile`, `memories`;
       - prompt tracking: `prompt_versions`.
  2. In `builder.py`, wire stub nodes in the shape of the architecture diagram, using `AsyncPostgresSaver` connected as `agent_app`.
     - Run `.setup()` once on the saver, and later on the store.
     - If you pass your own psycopg pool, check the settings the saver requires (autocommit, dict row factory) in the current docs.
  3. Run two turns on the same `thread_id`, restart the app, and confirm the history is still there.
- **Gate:** `get_graph().draw_mermaid()` output and the persistence demo.

### Phase 6 — Text-to-SQL agent subgraph
- **Learn:**
  - few-shot prompting;
  - structured output (`SQLGeneration{sql, explanation, tables_used}`);
  - self-correction loops;
  - `interrupt()` and `Command(resume=...)`.
- **Do:** build these nodes:
  1. `retrieve_schema`:
     - calls MCP `search_schema` and `get_few_shots`, filtered by `selected_domain` (metadata filtering);
     - in Phase 9, merges in the user's **learned SQL pairs** from episodic memory.
  2. `generate_sql`: produces the structured output above.
  3. `validate`: the local validator, which is guardrail layer 1.
  4. `approval_gate`:
     - calls MCP `explain_query`;
     - if the estimated rows exceed a threshold or the query touches PII, calls `interrupt({sql, reason})`;
     - **HITL lives in a graph node. The MCP server cannot pause the graph.**
  5. `execute`: calls MCP `run_select_query`. On error, it loops back to `generate_sql` with the error message, up to **2 retries**.
  6. `explain_result`: plain-language answer, and records `sql_used`.
- **Gate:** 5 questions that cover a self-correction, a HITL approve, a HITL reject and a blocked write.

### Phase 7 — Web Search agent subgraph
- **Learn:** chunking; cross-encoder vs embedding re-ranking; context compression; citation discipline.
- **Do:** build these nodes:
  1. `search`: MCP `search_web`.
  2. `fetch`: MCP `fetch_page` on the top N results.
  3. `chunk`: `RecursiveCharacterTextSplitter`.
  4. `rerank`: `flashrank`, keep the top k.
  5. `compress`: keep chunks within a token budget.
  6. `summarise`: the answer plus `sources: list[url]`.

  The prompt tells the model to ignore any instructions found inside page content.
- **Gate:** 3 answers with real URLs, plus logged chunk counts before and after re-ranking.

### Phase 8 — Planner and Supervisor
- **Learn:** planner vs executor; ReAct; re-planning; iteration budgets.
- **Do:**
  1. `planner` (LLM):
     - takes the standalone question, profile, memories and session context;
     - outputs a `RouteDecision` with `intent` and the step list;
     - a direct answer produces an empty plan that goes straight to `finalize`.
  2. `supervisor` (router), after each agent returns:
     - writes the result into `plan[i]`;
     - if the result is doubtful, asks the LLM for a structured judgement `StepReview{ok, reason}`; a not-ok verdict sends control back to `planner` to re-plan;
     - otherwise dispatches the next pending step, or goes to `finalize`.
  3. Budget: **at most 5 supervisor→agent dispatches per turn**. After that, finalize with a partial answer.
     - This counts dispatches, not raw MCP calls (one SQL step alone makes 4+ MCP calls).
     - Each subgraph keeps its own caps: 2 SQL retries, N fetches.
     - Document this reading of §5.1 in the report.
  4. `finalize`: writes a grounded answer with citations and/or the SQL used, and applies the profile's formatting preferences.
  5. Prompts come from `prompts/supervisor/planner_v1.yaml`, and the version used is recorded in `prompt_versions`.
- **Gate:** one conversation with a web, a SQL, a hybrid (e.g., "our beverage sales vs the global beverage market trend") and a direct question. Show the plan in state for each turn.

### Phase 9 — Memory layers (see the dedicated section below)
- **Learn:** the memory guide below, especially the pairs that are easy to confuse.
- **Do:**
  1. `history.py`: trim or summarise when the message count exceeds N.
  2. `profile.py`: Postgres store, namespace `("profile", user_id)`.
  3. `facts.py`: Chroma `user_facts`.
  4. `episodes.py`: Chroma `episodes`.
  5. Wire `load_context`, `write_memory` and the SQL agent's learned-pairs lookup.
- **Gate:** in a new thread with the same `user_id`, the bot recalls a preference and a fact. A different `user_id` sees neither. Bring the "what is stored where and why" table as a draft for the report.

### Phase 10 — Semantic cache (entity-aware; see the dedicated section below)
- **Do:**
  1. `contextualize` node: a single structured-output call that returns `{standalone_question, entities}`.
  2. `check_cache` / cache write in `write_memory`.
  3. `scripts/calibrate_cache.py`: measure the similarity of paraphrase pairs vs near-miss pairs to choose the threshold.
- **Gate:**
  - a paraphrase hits the cache;
  - `product_id=100` → `150` **misses**, yet the SQL agent reuses the learned pair;
  - the calibration numbers.

### Phase 11 — Streamlit UI with streaming and HITL
- **Learn:** Streamlit's rerun model; `st.session_state`; why calling `asyncio.run()` on every rerun breaks async DB pools and MCP clients.
- **Do:**
  1. Run **one background event-loop thread**, held in `st.cache_resource`, that owns the graph, checkpointer, store, Chroma client and MCP adapter. Submit work to it with `asyncio.run_coroutine_threadsafe`.
  2. The sidebar has `user_id`, `thread_id` (with a "new chat" button) and a domain selector.
  3. Stream node progress with `stream_mode="updates"`, then show the final answer. Token-by-token streaming is a stretch goal.
  4. On an interrupt, show the SQL and the reason with Approve/Reject buttons, then resume with `Command(resume=...)`.
  5. Under each answer, show the citations, the SQL used and a "cache hit" badge.
- **Gate:** a screen recording of a full demo, including a HITL approval.

### Phase 12 — Guardrails hardening and prompt versioning
- **Do:**
  1. Input guard:
     - length limits;
     - a basic prompt-injection/jailbreak check;
     - masking PII in logs.
  2. Every prompt lives at `prompts/<area>/vN.yaml`, with `version`, `date` and `changelog`. Prompt versions are switchable in config, so you can compare them in eval.
  3. Check each guardrail layer:
     - **UI:** input checks;
     - **agent:** the SQL validator;
     - **MCP server:** validator, read-only role, timeout, rate limit, SSRF block;
     - **output:** citations required.
- **Gate:** a checklist with one test that proves each layer.

### Phase 13 — Evaluation
- **Do:**
  1. `eval/questions.yaml` with about 35 cases:
     - 10 web, 10 SQL, 6 hybrid, 4 direct;
     - memory cases;
     - **cache near-miss pairs** that differ only by ID, year or country;
     - adversarial cases.

     Each case has an expected `intent`, and SQL cases have a **gold SQL** query.
  2. `run_eval.py` reports these metrics:
     - intent accuracy;
     - SQL execution accuracy (compare result sets, not SQL text);
     - citation presence;
     - guardrail block rate;
     - cache hit rate and **false-hit rate**;
     - latency at p50/p95;
     - an optional local LLM-as-judge score.
  3. Compare at least 2 prompt versions or models, and save the results.
- **Gate:** the results table and an analysis of the top 3 failure modes.

### Phase 14 — Report, diagram, README
- **Do:**
  - an architecture diagram in which every §4.1 component is identifiable;
  - the memory justification table;
  - the guardrails and HITL design;
  - the eval results;
  - **limitations**: local-model quality, Northwind's small schema, Tavily's coverage, cache staleness, the per-call cost of MCP session setup;
  - README setup: `docker compose up`, `ollama pull`, `uv sync`, build the index, start the web server, `streamlit run`.
- **Gate:** final review of the repo and the report.

---

## Memory architecture guide (answers your Phase 9 question)

### The layers at a glance

| Layer | Question it answers | Storage | Key / scope | Written: where and when | Read: where and when | Lifetime |
|---|---|---|---|---|---|---|
| **Short-term: conversation** | "What have we said so far in *this chat*?" | Postgres checkpointer (`agent_memory`) | `thread_id` | Automatically, after every node | **Automatically** when the graph is invoked with the same `thread_id`. There's no load node. Trimmed or summarised in `write_memory` once it passes N messages. | Life of the thread |
| **Short-term: session** | "What context did the user set *in this chat*?" (domain = sales, region = EU) | Custom fields in graph state, saved by the same checkpointer | `thread_id` | UI domain selector, or the planner noticing "let's focus on sales" | `planner` (context) and `sql_agent.retrieve_schema` (**metadata filter**) | Life of the thread |
| **User profile** | "How does this *person* like answers?" (short, EUR, metric) | Postgres store, key-value, **no search** | `("profile", user_id)` | `write_memory` when the user *explicitly* states a preference, or from UI settings | `load_context`, **always, in full**, at the start of every turn (it's tiny) | Permanent until changed |
| **Long-term semantic (facts)** | "What durable facts do we know about this user or their domain?" ("reports revenue in EUR", "manages the Nordic region") | Chroma `user_facts` | metadata `user_id` | `write_memory`: an LLM distils 0–3 facts per turn and **de-dupes** (similarity > 0.9 → update, don't insert) | `load_context` on a cache miss: **top-k by similarity** to the standalone question | Permanent, deduplicated |
| **Long-term episodic** | "What *happened* before that's relevant?" | Chroma `episodes` | metadata `user_id`, `kind`, `agent`, `date` | (a) `kind=summary`: written by `write_memory` every N turns or when a thread ends. (b) `kind=sql_pair`: written after a **successful, non-rejected** SQL run. | (a) by `load_context`: top-k similar past summaries. (b) by **`sql_agent.retrieve_schema`**, as dynamic few-shots. | Permanent, timestamped |
| **Semantic cache** | "Have we *already answered exactly this*?" | Chroma `semantic_cache` | metadata `route`, `entity_key`, `profile_hash`, `data_version`, `expires_at` | `write_memory`, only for successful, complete, standalone answers | `check_cache`: the **first** retrieval, before any other memory or tools | TTL: web ≈ 24h; DB until `data_version` changes |
| **Seed knowledge** (not "memory", but often confused with it) | "What does the schema look like?" | Chroma `schema_docs`, `sql_seed_examples` | metadata `table`, `domain` | Offline by `build_schema_index.py` | DB MCP server tools `search_schema` / `get_few_shots` | Rebuilt when the schema changes |

### Pairs that are easy to confuse
- **Conversation vs session:** the conversation is *what was said*, a growing list of messages. The session is *settings that apply* to this chat, a few small structured fields. Both die with the thread.
- **Session vs profile:** "focus on sales" lasts for *this chat* only. "I always want short answers" belongs to *the person*, across every chat.
- **Profile vs semantic facts:**
  - The profile has a **fixed schema**, is small and is loaded **in full** every turn.
  - Facts are **open-ended and unbounded**, so only the **relevant** few are retrieved by similarity.
- **Semantic vs episodic:** semantic memory is *what is true*, distilled and timeless. Episodic memory is *what happened*: dated events and worked examples.
- **Episodic SQL pairs vs the cache.** This answers your product-ID question. The cache returns a **finished answer** and skips everything. An episodic SQL pair only helps the model **write the SQL**, and the query still runs against live data.

### When each layer loads during one turn
1. **Invoke** with `thread_id`. The checkpointer restores the conversation and session state, with no code of yours involved.
2. **`contextualize`** reads recent messages and rewrites the question into a standalone form, extracting its entities.
3. **`check_cache`** queries the semantic cache. On a hit, it responds and stops.
4. **`load_context`** reads the profile (full), facts (top-k) and episode summaries (top-k).
5. **`planner`** plans using all of the above, plus the session context.
6. **`sql_agent.retrieve_schema`** reads the seed examples (through MCP) and the learned SQL pairs (episodic), filtered by `selected_domain`.
7. **`finalize`** applies the profile's formatting preferences.
8. **`write_memory`** distils facts, writes the episode, the learned SQL pair and the cache entry, and trims the history. It runs *after* the answer has streamed, so the user doesn't wait for it.

---

## Semantic cache design: the `product_id=100` vs `150` problem (answers your cache question)

**The problem.** "Show details of clothes with product_id 100" and "… product_id 150" embed almost identically, with cosine similarity around 0.97. A naive similarity-only cache would return **product 100's answer for product 150**. That is a *false hit*, and it's worse than a miss because the answer looks right but is wrong.

**The design: similarity is necessary but not sufficient.** A cache hit requires **all** of:
1. **Similarity ≥ threshold** on the *standalone* question, calibrated in Phase 10.
2. **An exact `entity_key` match.** `contextualize` extracts the literals that change the answer: IDs, numbers, years and dates, quoted strings, and named entities such as products, customers and countries. It normalises and sorts them into a key like `product_id=100`. The key is applied as an **exact metadata filter** in the Chroma query, so different IDs can never match.
3. **The same `profile_hash`**, so a user who wants EUR never gets a USD-formatted answer cached for someone else.
4. **Freshness:** `expires_at > now` for web answers, and `data_version` equal to the current value for DB answers.

**What happens in your example:**
- **User A asks about 100.** It's a miss, so the full pipeline runs. This writes a cache entry (`entity_key=product_id=100`) and a **learned SQL pair** in episodic memory.
- **User B asks about 150.** The similarity is high but the `entity_key` differs, so it's a **cache miss**, as intended.
  - The SQL agent then retrieves the product-100 SQL pair as a **few-shot example**. It generates `… WHERE product_id = 150` quickly and correctly, and still runs against live data.
  - You lose the instant answer but keep most of the speed-up, with **zero risk of a wrong answer**.
- **Anyone asks "What are the details for product 100?"** It's a paraphrase with the same entity key, so it's a **cache hit** and returns instantly.

**Never cache:**
- follow-ups that couldn't be made standalone;
- answers that errored, were partial (budget hit) or were rejected through HITL;
- direct chit-chat (cheap anyway).

**Measure it:** the eval includes near-miss pairs, and you report the **false-hit rate** next to the hit rate.

---

## Gotchas (each is revisited in its phase)
- **G1 Ollama structured output:** use `with_structured_output(..., method="json_schema")` and test it in Phase 0. Qwen3 has a "thinking" mode, so make sure `<think>` text never leaks into parsed output.
- **G2 MCP client is beta:** `langchain.mcp` emits a beta warning, so pin its version. The fallback is the archived `langchain-mcp-adapters` (`MultiServerMCPClient`).
- **G3 One session per tool call:** the adapter opens a session per call by default. Both servers are long-lived HTTP processes, so in-memory state (pools, rate limiter, clients) survives; the remaining cost is session setup. Measure it in Phase 4.
- **G4 HITL location:** `interrupt()` only works inside graph nodes, and needs a checkpointer plus a `thread_id`.
- **G5 Guardrails in two places:** in the agent *and* at the MCP server. The read-only role is the last line of defence.
- **G6 Streamlit + async:** use one background event loop (Phase 11).
- **G7 Chroma distance:** create every collection with the cosine space, and convert distance to similarity (`1 − distance`).
- **G8 Northwind dates:** the data covers 1996–98 unless you shift it.
- **G9 Server settings (stdio gotcha, n/a for HTTP):** each MCP server loads its own settings from `.env` (`ENV_FILE` overrides the path), so there is no child-process environment to forward. Use `/ready` to see what a server can reach.
- **G10 Postgres `:latest` + anonymous volume:** pin the image (built with `postgres:14.24`) and use a named volume through compose (Phase 1).

## Verification (end-to-end)
1. Start the stack:
   - `docker compose up -d`;
   - Ollama running;
   - the web MCP server;
   - `uv run streamlit run src/multi_agenti_conversation_system/ui/streamlit_app.py`.
2. In one chat, ask a web, a SQL, a hybrid and a direct question. Check the citations and the SQL shown.
3. A PII query triggers HITL approval. A DELETE attempt is blocked.
4. A paraphrase hits the cache. The `product_id 100 → 150` pair misses.
5. A new thread with the same user recalls a preference and a fact. A different user doesn't.
6. `uv run pytest` passes, and `uv run python eval/run_eval.py` produces the metrics table.

## After approval
I'll copy this plan to `plan/implementation_plan.md` in the project root. It's the only file I'll create. Then you start **Phase 0, step 1**, and come back with questions any time.
