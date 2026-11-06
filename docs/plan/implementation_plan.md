# Capstone Implementation Plan — Multi-Agent Chatbot (LangGraph + MCP + Layered Memory)

> **This file is the single source of truth for the project plan.** Any other plan notes are scratch. When a decision changes, it changes here first.

## Context

The capstone ([docs/Capstone Problem Statement](../docs/)) asks for one chatbot that answers **web**, **database** and **hybrid** questions:
- a **Planner/Supervisor** routes work to a **Web Search Agent** and a **Text-to-SQL Agent**;
- every tool call goes through **MCP servers**;
- memory is layered: short-term, long-term semantic and episodic, a user profile, and a semantic cache;
- it also needs guardrails, human-in-the-loop (HITL) approval, prompt versioning and a repeatable evaluation.

**How we work together:** you're the sole developer and I'm the reviewer and guide. Each phase has three parts:
- **Learn:** concepts to understand first. Ask me as many basic questions as you like.
- **Do:** the steps you carry out.
- **Gate:** the definition of done, and what you bring back for me to review (terminal output, a diff, screenshots). We only move on once a gate passes.

**Note:** the problem statement stops at §6, so it has no deliverables or grading section. The evaluation, prompt-versioning, limitations and architecture-diagram work below comes from §3 Objectives and §4 Reference Architecture.

## Status

| Phase | Status |
|---|---|
| 0 Environment and tooling | **Done.** uv, Python 3.14, `scripts/smoke_llm.py` |
| 1 Data layer | **Done.** Compose, Northwind, roles, date shift, YAML catalog |
| 2 DB MCP server | **Done.** Streamable HTTP + bearer token, validator (unit tests), `smoke_db_mcp.py`, DB integration tests |
| 2b Hybrid retrieval | **Done.** `retrieval/hybrid.py` (tokenize, BM25Index, rrf, hybrid_search), wired into `search_schema` / `get_few_shots`, unit tests, degraded mode verified with Chroma stopped |
| 2c Observability, CI, hygiene | Optional, not started |
| 3–14 | Not started |

---

## Decisions locked in

| Area | Choice | Notes |
|---|---|---|
| LLM (chat) | **OpenRouter** (`CHAT_MODEL` in `.env`), built in `llm.py` | Prompts leave the machine on a hosted provider; say so in the report. A local Ollama model (`qwen3:8b`, fallback `llama3.1:8b`) is the fallback if needed. |
| Embeddings | **Ollama** `nomic-embed-text` | One embedding model everywhere, with nomic's `search_document:` / `search_query:` prefixes. Re-calibrate the cache threshold for it (Phase 10). |
| Orchestration | **LangGraph everywhere it fits** | See the [LangGraph component map](#langgraph-component-map). |
| Business DB | **Postgres** (`postgres:14.24`, compose, named volume), loaded with **Northwind** | Its sales schema suits "internal sales vs market trend" hybrid questions. |
| Vector DB | **Chroma** (`chromadb/chroma:1.0.21`, `localhost:5430`) | Holds **all** vectors: schema index, seed few-shots, user facts, episodes, learned SQL pairs and the semantic cache. No pgvector. |
| Keyword search | **In-process BM25** (`bm25s`, Lucene IDF), fused with dense results by **RRF** | Chroma 1.0.x can't rank by keyword. See Phase 2b. |
| User profile | LangGraph **`AsyncPostgresStore`**, **no vector index** | A key-value store namespaced by `user_id`, so it needs no pgvector. |
| Conversation state | LangGraph **`AsyncPostgresSaver`** (`langgraph-checkpoint-postgres`) | Conversation state survives restarts. |
| MCP transports | **Both servers use Streamable HTTP**: DB `:5555`, Web `:5556`, both with a bearer token | Long-lived processes keep pools and rate-limiter state. |
| Web search | **Tavily** | Needs `TAVILY_API_KEY` (free tier). |
| UI | **Streamlit** | Async bridging needs care (Phase 11). |

**Storage split (a good point for the report):**
- **Postgres DB `northwind`:** business data, accessed only through the read-only role `agent_readonly`.
- **Postgres DB `agent_memory`:** the LangGraph checkpointer (conversation state) and the LangGraph store (user profiles), accessed as `agent_app`.
- **Chroma:** every vector, each in its own collection: `schema_docs`, `sql_seed_examples`, `user_facts`, `episodes`, `semantic_cache`.

## Changeable defaults

| Area | Default | Why |
|---|---|---|
| Python / packaging | Python **3.14**, **uv**, `pyproject.toml`, `src/` layout, package `multi_agenti_conversation_system` | Fast, gives you a lockfile, current standard |
| MCP server SDK | `mcp` (official, 2.3.x) | Decorator-based tools, with input schemas taken from type hints |
| MCP client | `langchain[mcp]` → `from langchain.mcp import MCPAdapter` | The old `langchain-mcp-adapters` is **archived**. The new namespace is **beta**, so pin its version. It turns server *elicitation* into a LangGraph `interrupt()`. |
| SQL parsing | `sqlglot` | AST-based SELECT-only checks and LIMIT injection, safer than regex |
| Keyword ranking | `bm25s` (fallback `rank-bm25` **`BM25Plus`**) | Lucene IDF is never negative on tiny corpora. `BM25Okapi`'s IDF can be (G11). |
| Page cleaning | `httpx` + `trafilatura`, with Tavily Extract as a fallback | Robust main-text extraction |
| Re-ranking | `flashrank` (local cross-encoder, pulls `onnxruntime`) | No API key, fast |
| Chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| Config | `pydantic-settings` + `.env` | Typed, all in one place |
| Tests / lint | `pytest`, `pytest-asyncio`, `ruff` (+ `pyright` in Phase 2c) | |
| Tracing (optional) | OpenTelemetry (Phase 2c), LangSmith optional | LangSmith is an external service, so check that your org allows sending traces there |
| Not used | `langmem` | The memory logic is small and stays explicit and testable |

> **Versions:** pin everything and commit `uv.lock`. Every new dependency above was checked to resolve on Python 3.14 (Oct 2026): `bm25s` 0.3.15, `langgraph-checkpoint-postgres` 3.2.0, `flashrank` 0.2.10 (`onnxruntime` 1.31.1), `trafilatura` 2.3.1, `tavily-python` 0.8.5, `langchain-text-splitters` 1.1.3. The `chromadb` client stays pinned to the server's 1.0.x.

---

## Target architecture

**One explicit interpretation.** The specialist agents are **LangGraph subgraphs**, reached from the Supervisor by **conditional edges** or **`Send`**. Their **low-level tools** (`search_web`, `fetch_page`, `search_schema`, `run_select_query`, …) are served by **MCP servers** and discovered at startup through `MCPAdapter`. This matches §4.2. Say it in your report, because §3 could also be read as "the agents themselves are MCP tools".

**Planner/Supervisor is one *role* built from two nodes.** Both live in `graph/supervisor.py`:
- **`planner`** (LLM, structured output)
  - Runs once at the start of a turn, and again when a step fails (re-planning).
  - Returns a `RouteDecision` with `intent`, `plan`, `next_agent` and `reason`.
- **`supervisor`** (the router)
  - Runs after every agent returns.
  - Deterministic code marks steps done, picks the next pending step(s), enforces the budget and decides when the turn is finished.
  - Steps with no unmet `depends_on` are dispatched **in parallel with `Send`**, e.g. web ∥ sql for a hybrid question.
  - It calls the LLM only to judge a doubtful result (empty, error, off-topic). That judgement either accepts the result or triggers a re-plan.

Why split them:
- §5.1 asks to *separate planning from execution*.
- Small models are more reliable at a single planning call than at re-deciding everything on each hop.
- It saves one LLM call per hop.

```
User (Streamlit) ── user_id + thread_id ──► LangGraph app   (checkpointer auto-restores thread state)
                                            │
 START → contextualize → load_profile → check_cache ──(hit)───────────────────────► respond → END
  (standalone question     (profile_hash      │ (miss)
   + entity extraction)     for cache key)    ▼
                                         load_context  (relevant facts + relevant episodes)
                                              ▼
                                         planner ──► supervisor ──┬─► web_agent (subgraph) ─┐   Send: independent
                                            ▲  (re-plan)  ▲       ├─► sql_agent (subgraph) ─┤   steps run in parallel
                                            └─────────────┴───────┴──── results ◄───────────┘
                                                          │ (plan done / budget hit / direct answer)
                                                          ▼
                                                       finalize (grounded answer + citations / SQL) → respond (streamed)
                                                          ▼
                                                       write_memory ─┬─ write_facts      (parallel branches,
                                                                     ├─ update_profile    each with its own guard)
                                                                     ├─ write_episode
                                                                     ├─ write_sql_pair
                                                                     └─ write_cache
                                                          ▼
                                                       summarize_history (only past the threshold) → END

 sql_agent:  retrieve_schema → generate_sql → validate → approval_gate[interrupt()] → execute → explain_result
                (hybrid: seed + learned)  ▲──────── error fed back (max 2 retries) ◄──────┘
 web_agent:  plan_queries → search[Send ×q] → merge_dedupe → fetch[Send ×url] → chunk → rerank → compress
                   ▲                                                                                │
                   └────────────── assess: not sufficient (max 2 rounds) ◄──────────────────────────┘ → summarise

 MCP:     web_server (HTTP :5556: search_web, fetch_page)
          db_server  (HTTP :5555: list_tables, get_schema, search_schema*, get_few_shots*, explain_query, run_select_query)
                     * hybrid search (plain function): dense + BM25 → RRF
 Storage: Postgres[northwind] (read-only role) · Postgres[agent_memory] (checkpoints + profile store)
          Chroma (schema_docs, sql_seed_examples, user_facts, episodes, semantic_cache)
```

## LangGraph component map

LangGraph is the orchestration layer. It is **not** a vector DB or an MCP client, so "using LangGraph" for those means wrapping them in LangGraph primitives.

| Component | LangGraph primitive | Notes |
|---|---|---|
| Whole app | `StateGraph(AgentState, context_schema=AppContext)` | Nodes take `runtime: Runtime[AppContext]`. The context holds the MCP tool registry, models, the Chroma memory classes and the retriever. This is LangGraph's own dependency injection: nodes never call `get_*()` globals. |
| SQL and web agents | compiled **subgraphs** added as nodes | Streamed with `subgraphs=True`. |
| Independent plan steps | **`Send`** fan-out from `supervisor` | Plan results use a reducer. Each `Send` counts toward the 5-dispatch budget. |
| Conversation memory | `AsyncPostgresSaver` checkpointer | Restored automatically per `thread_id`. |
| History trimming | `summarize_history` node using `RemoveMessage` plus a `summary` state field | `SummarizationMiddleware` only works inside `create_agent`, so we use the plain-graph equivalent. |
| User profile | `AsyncPostgresStore` (no index), read through `runtime.store` | Namespace `("profile", user_id)`. |
| MCP client | `MCPAdapter` → LangChain tools, called as `await tool.ainvoke()` inside nodes | Held open for the app's lifetime. Server elicitation becomes `interrupt()`. |
| Transient failures | `RetryPolicy` on LLM and MCP nodes | Retries only connection errors and timeouts. |
| Duplicate web searches | `CachePolicy(ttl=3600)` on the web `search` node, `compile(cache=InMemoryCache())` | Exact-input dedupe within one process, separate from the semantic cache. Check that the cache reaches the subgraph. |
| HITL | `interrupt()` in `approval_gate`, resumed with `Command(resume=…)` | Optional defense-in-depth: the DB server elicits confirmation for PII queries. |
| Memory writes | `write_memory` fans out into **parallel branches** | `write_facts`, `update_profile`, `write_episode`, `write_sql_pair`, `write_cache`. |
| Memory maintenance | a separate small graph, `memory_maintenance` | Run by `scripts/consolidate_memory.py` (Phase 9). |
| UI streaming | `astream(stream_mode=["updates","messages"], subgraphs=True)` | The answer renders at `finalize`, before memory writes finish. |

**Where LangGraph adds nothing (kept plain):**
- the MCP servers themselves;
- hybrid retrieval (`hybrid_search`): a plain function in the RAG path of `search_schema` / `get_few_shots`, reused agent-side for learned SQL pairs;
- the SQL validator, Tavily client, SSRF guard and rate limiter;
- the Chroma client code.

## Conventions

- **Agents reach tools only through MCP.** `graph`, `memory` and `ui` must not import `mcp.servers`, because servers are separate processes. Agents may import `guardrails` and `catalog` for the layer-1 SQL check.
- **Servers are independent.** `db_server` and `web_server` must not import each other. Shared code (`BearerAuth`, the loopback check) moves to a shared module when the second server needs it.
- **Graph nodes get their dependencies from `Runtime[AppContext]`**, built once in `graph/builder.py`. Nodes never call `get_*()` globals.
- Tools return `{"ok": False, "error": ...}` and never raise.
- Memory classes (`facts`, `episodes`, `sql_pairs`, `cache`) take the Chroma client and `Embeddings` in their constructor. Each owns its collection name and metadata schema.
- Prompts live in `prompts/<area>/vN.yaml` and are loaded through `prompts.py`. Graph code holds no inline prompt strings.
- The UI imports only `graph.builder` and runs one background event loop (G6).

## Target project structure (built incrementally)

```
multi_conversational_agentic_system/
├── pyproject.toml, uv.lock, .python-version, .env.example, .env (gitignored), .gitignore, README.md
├── docker-compose.yml              # Postgres 14.24 + Chroma 1.0.21, named volumes
├── db/                             # 01_databases.sql, 02_northwind.sql, 03_grants.sql, 04_shift_dates.sql
├── docs/                           # problem statement, plan/implementation_plan.md (this file), architecture, report
├── prompts/                        # supervisor/planner_v1.yaml, sql/v1.yaml, web/v1.yaml, memory/v1.yaml ...
├── data/                           # schema_descriptions.yaml, few_shots.yaml
├── scripts/                        # smoke_llm.py, smoke_db_mcp.py, build_schema_index.py, smoke_mcp.py,
│                                   # calibrate_cache.py, consolidate_memory.py, init_db.sh
├── src/multi_agenti_conversation_system/
│   ├── config.py, llm.py, prompts.py, vectorstore.py
│   ├── guardrails/                 # sql_validator.py (shared by agent AND server), input_guard.py
│   ├── retrieval/                  # hybrid.py (tokenize, BM25Index, rrf, hybrid_search)
│   ├── mcp/
│   │   ├── client.py               # MCPAdapter setup + tool registry per server (Phase 4)
│   │   └── servers/
│   │       ├── db_server/          # server.py, db.py, schema_index.py, catalog.py
│   │       └── web_server/         # server.py, tavily_client.py, fetch.py, rate_limit.py
│   ├── memory/                     # profile.py, facts.py, episodes.py, sql_pairs.py, cache.py, history.py
│   ├── graph/
│   │   ├── state.py                # AgentState, Plan, PlanStep, RouteDecision
│   │   ├── context.py              # AppContext (context_schema)
│   │   ├── supervisor.py           # planner + supervisor nodes
│   │   ├── agents/                 # sql_agent.py, web_agent.py
│   │   ├── nodes/                  # contextualize, load_profile, check_cache, load_context, finalize,
│   │   │                           # write_memory (+ branches), summarize_history
│   │   ├── maintenance.py          # memory_maintenance graph
│   │   └── builder.py
│   └── ui/streamlit_app.py         # lives in src/ui/ today; moved into the package in Phase 2c
├── eval/                           # questions.yaml, run_eval.py, eval_retrieval.py, results/
└── tests/                          # unit/, integration/
```

---

## Phases

### Phase 0 — Environment and tooling ✅
- **Learn:**
  - uv basics: `init`, `add`, `sync`, `run`;
  - why a `src/` layout helps;
  - `.env` vs `.env.example`;
  - how Ollama serves models.
- **Do (done):**
  1. git, uv, Python 3.14, core and dev dependencies.
  2. `config.py` and `llm.py`.
  3. `scripts/smoke_llm.py` checks four things:
     - (a) a bound dummy tool gets called;
     - (b) `with_structured_output` works on a small Pydantic model;
     - (c) no `<think>` text leaks into parsed output;
     - (d) the cosine similarity of two embedded sentences makes sense.
- **Gate:** passed.

### Phase 1 — Data layer ✅
- **Learn:**
  - named vs anonymous Docker volumes;
  - pinning image versions;
  - Postgres roles and grants, `default_transaction_read_only` and `statement_timeout`;
  - Chroma's HTTP client and its distance metrics.
- **Do (done):**
  1. `docker-compose.yml` with pinned images, named volumes and a Postgres healthcheck.
  2. `db/` scripts:
     - create the databases `northwind` and `agent_memory`;
     - load Northwind;
     - create `agent_readonly` with SELECT only, read-only transactions and `statement_timeout=5s`;
     - create `agent_app` with read-write access to `agent_memory`;
     - shift the dates forward.
  3. `data/schema_descriptions.yaml`: 14 tables, each with a `domain`, column descriptions and `pii: true` flags.
  4. `data/few_shots.yaml`: 20 question → SQL pairs.
  5. Chroma collections use the **cosine** space, and `similarity = 1 − distance`.
- **Gate:** passed.

### Phase 2 — DB MCP server (Streamable HTTP) ✅
- **Learn:**
  - MCP basics: server, client, tools, transports;
  - how type hints become a tool's input schema;
  - why guardrails also sit *at the server boundary*.
- **Do (done):**
  1. `guardrails/sql_validator.py` (`sqlglot`):
     - accepts exactly one SELECT (CTEs allowed);
     - rejects DDL/DML anywhere in the tree;
     - injects or clamps `LIMIT`;
     - enforces the table allowlist;
     - finds PII columns.
  2. `scripts/build_schema_index.py` builds the `schema_docs` and `sql_seed_examples` collections.
  3. Tools:
     - `list_tables`
     - `get_schema`
     - `search_schema`
     - `get_few_shots`
     - `explain_query`: returns the planner's row estimate and the PII columns touched
     - `run_select_query`: connects as `agent_readonly` and re-validates on the server side
  4. Served over Streamable HTTP on `:5555`, with a bearer token and `/health`. An empty token is allowed only on loopback.
  5. Validator unit tests (adversarial inputs), `scripts/smoke_db_mcp.py`, and integration tests in `tests/integration/test_db.py`.
- **Gate:** passed (smoke script prints ALL CHECKS PASSED, tests green, Inspector demo).

### Phase 2b — Hybrid retrieval: dense + BM25 → RRF (inside the DB MCP server)
- **Learn:**
  - lexical vs semantic retrieval, and why each fails where the other works ("unit_price", IDs, rare terms vs paraphrases);
  - BM25 and IDF;
  - Reciprocal Rank Fusion;
  - why we fuse ranks, not raw scores.
- **Constraint:** chromadb 1.0.21 has no BM25 or sparse ranking. `where_document $contains` filters but doesn't rank. Upgrading doesn't help: tested 2026-10-28, a self-hosted `chromadb/chroma:1.5.9` rejects a BM25 sparse index with "Sparse vector indexing is not enabled in local" (sparse search, `Search()` and `Rrf` are Chroma Cloud only). So the keyword branch is **in-process BM25** over the corpus built from the catalog YAML (14 tables + 20 few-shots), built on first search in each process (server and `build_schema_index.py`) and cached.
- **Corpus sources:**
  - **seed** BM25 corpora come from `catalog.py` (the YAML), never from Chroma, so keyword search survives a Chroma outage;
  - the dense branch is the existing Chroma query in `schema_index.py`;
  - a Chroma `get` builds the BM25 corpus only for **learned SQL pairs** (Phase 6/9), which have no other source.
- **Do:**
  1. **`retrieval/hybrid.py`** (pure):
     - `tokenize()`: lowercase; split on non-alphanumerics, snake_case and camelCase; **keep the whole identifier too** (`unit_price` → `unit_price`, `unit`, `price`); a small stopword list; naive plural strip.
     - `BM25Index(docs, metadatas)` wrapping `bm25s` with Lucene IDF, `log(1 + (N−n+0.5)/(n+0.5))`, which is never negative. **Not `BM25Okapi`** (G11).
     - `rrf(ranked_lists, k=60, weights=None)`: `score(d) = Σ_i w_i / (60 + rank_i(d))`, with 1-based ranks. A list that misses the doc contributes 0. Weights default to 1:1, and ties are broken by dense rank.
  2. **`hybrid_search(dense_fn, keyword_fn, k)`** in `hybrid.py`: a plain function, **no LangGraph**. It runs as a normal step of the RAG path (fetching table info and few-shots): call the dense branch (guarded) and the BM25 branch, then `rrf`, and return the hits plus `degraded`.
  3. **Rules:**
     - **The domain filter applies to both branches:**
       - dense: a Chroma `where` filter (as today);
       - BM25: score over the **full** corpus, then **filter by metadata after scoring**, so the IDF stays stable.
     - **Candidate depth:** the whole filtered corpus when it has 50 docs or fewer, otherwise `max(20, 4k)`. So every hit normally has a dense `similarity`.
     - BM25 hits with a score **≤ 0** are dropped, so a non-match never gets a rank.
     - **What gets indexed:**
       - schema docs: dense and BM25 both use `table_document(t)`;
       - few-shots: dense embeds the **question**; BM25 indexes **question + SQL**, so "freight" or "unit_price" match through the SQL.
     - **Graceful degradation:**
       - the dense branch catches Chroma and Ollama errors and returns `[]` plus `degraded=True`. The tool still answers from BM25 with `"degraded": true`, instead of `ok: false`.
       - In degraded mode `similarity` is `null`, and the build script prints `-`.
       - Collections are fetched lazily, so a Chroma outage at startup doesn't stop the server.
     - **Output contract:** the existing keys stay unchanged, including `similarity`. Each hit gains `rrf_score`, `dense_rank` and `bm25_rank`, and the response gains `degraded`.
  4. Wire it into `schema_index.py` and the two tools in `server.py`, so `search_schema` and `get_few_shots` call `hybrid_search`. Add `bm25s` to `pyproject.toml`.
  5. `build_schema_index.py` prints the dense, BM25 and RRF ranks for its sample questions.
  6. `tests/unit/test_hybrid.py`:
     - tokenizer cases;
     - RRF arithmetic and ties;
     - the BM25 domain filter;
     - **a 2-doc corpus, and a filter that leaves 2 docs, both still returning keyword hits**;
     - the degraded path with a fake dense function.
- **Gate:**
  - the tests are green;
  - the smoke script still passes;
  - the rank printout, for example for "freight cost per shipper", inspected by eye. Pass/fail for quality comes from the recall@k eval in Phase 13.
  - With `docker compose stop chroma`, `search_schema` returns `ok: true, degraded: true`, and `run_select_query` still works.

### Phase 2c — Observability, CI and hygiene (optional, recommended before Phase 14)
- **Logging:** key=value logs per tool call: tool, `duration_ms`, `ok`, `row_count`. Never row values, and full SQL only at DEBUG.
- **Tracing:**
  - an OpenTelemetry `TracerProvider` (OTLP or console, otherwise off);
  - child spans in the adapters, with no SQL text and no row values in their attributes;
  - an optional Jaeger service behind a compose profile.
- **Tooling:**
  - ruff rule sets E, F, I, UP, B, SIM;
  - pyright in basic mode;
  - a GitHub Actions workflow running uv sync, ruff, format check, pyright and unit tests. It must be green on its first run.
- **Hygiene:**
  - delete `tests/test_llm.py` (it calls the live provider when pytest collects it; `smoke_llm.py` covers it) and the placeholder `mcp/mcp_client.py`;
  - add pytest markers `integration` and `live`, and run only unmarked tests by default;
  - declare `psycopg-pool` in `pyproject.toml`: `db.py` imports it, but it is installed by hand today, and `uv sync` removes it;
  - bind compose ports to `127.0.0.1`, add a Chroma healthcheck on `/api/v2/heartbeat`, fix the `restart :` typos;
  - move `src/ui/` into the package and delete `scripts/__init__.py`;
  - point `.vscode` at `.venv`;
  - write the README.

### Phase 3 — Web MCP server (Streamable HTTP)
- **Learn:**
  - token-bucket rate limiting;
  - prompt injection through web content (fetched text is *data*, never instructions);
  - SSRF (server-side request forgery), including DNS rebinding.
- **Do:** follow the DB server pattern: `server.py` with thin tool functions, `BearerAuth`, `/health`, uvicorn, on `:5556`.

  | Tool | Signature | How it works |
  |---|---|---|
  | `search_web` | `(query, max_results=5 [1–10], recency: day\|week\|month\|year\|any = any, topic: general\|news = general, include_domains=None, exclude_domains=None)` | Tavily `search(search_depth="basic")` returns `{ok, results:[{title, url, snippet, published_date, score}]}`. |
  | `fetch_page` | `(url, max_chars=20000)` | 1. Allows http/https only. 2. **Resolves DNS and blocks** private, loopback, link-local and metadata IPs. 3. Follows redirects manually (max 3) and **re-checks every hop**. 4. Uses a 10 s timeout and a 2 MB streamed size cap, and accepts only HTML or plain text. 5. Extracts with trafilatura. If that gives fewer than 200 chars (a JS-heavy page), it **falls back to Tavily Extract**. Returns `{ok, url, final_url, title, published_date, text, truncated}`. The remaining DNS-rebinding risk is documented. |

  - **Rate limiter:** a token bucket per tool, e.g. search 10/min with a burst of 5, and fetch 30/min. When empty, it returns `{ok:false, error:"rate_limited", retry_after_s}`. It only works because the server is long-lived.
- **Gate:** Inspector or smoke output showing:
  - a search;
  - a fetch;
  - a blocked `127.0.0.1` fetch, and a blocked redirect to a private IP;
  - a burst that trips the rate limit.

### Phase 4 — MCP client and tool discovery
- **Learn:**
  - the `MCPAdapter` lifecycle, and why holding it open reuses connections;
  - how MCP elicitation becomes a LangGraph `interrupt()`.
- **Do:**
  1. `mcp/client.py`:
     - connects to both HTTP servers with their bearer headers. A plain URL string can't carry headers, so build a `fastmcp` client or transport with headers and wrap it; check the exact API in the installed `langchain.mcp`.
     - discovers tools with `list_tools()`;
     - returns a registry grouped by server.
  2. The adapter(s) stay open for the app's lifetime, built once in `graph/builder.py` and exposed through `AppContext`.
  3. `scripts/smoke_mcp.py` lists the tools and calls one from each server.
  4. Add a trivial new tool to a server and restart. It should be discovered **without changing any graph code**.
  5. Time 5 consecutive calls with the adapter held open vs opened per call.
- **Gate:**
  - the smoke-script output;
  - the timing numbers;
  - your own explanation of why both servers are long-lived HTTP processes: pools, rate-limiter state, no per-call process spawn.

### Phase 5 — Graph state and skeleton
- **Learn:**
  - `StateGraph`, nodes, edges and conditional edges, `Send`;
  - reducers (`add_messages`, custom ones);
  - subgraphs;
  - `context_schema` / `Runtime`;
  - `thread_id` and checkpointers;
  - why short-term memory needs *no load node*: the checkpointer restores state automatically.
- **Do:**
  1. `state.py` defines:
     - `PlanStep`: `id`, `agent`, `task`, `depends_on`, `status`, `result`.
     - `RouteDecision`:
       - `intent: Literal["web","sql","hybrid","direct"]`, which eval scores;
       - `next_agent: Literal["web","sql","finalize"]`;
       - `reason`;
       - `plan: list[PlanStep]`.
     - `AgentState`:
       - conversation: `messages`, `summary`, `standalone_question`, `entities`;
       - planning: `intent`, `plan` (with a reducer that merges parallel step results), `dispatch_count`;
       - results: `citations`, `sql_used`, `cache_hit`;
       - session context: `selected_domain`, `filters`;
       - user and memory: `user_id`, `profile`, `memories`;
       - prompt tracking: `prompt_versions`.
  2. `context.py` defines `AppContext`: models, the MCP tool registry, memory classes, the retriever, settings-derived limits.
  3. In `builder.py`, wire stub nodes in the shape of the architecture diagram:
     - `StateGraph(AgentState, context_schema=AppContext)`;
     - `AsyncPostgresSaver` and `AsyncPostgresStore`, connected as `agent_app`;
     - run `.setup()` once on each;
     - if you pass your own psycopg pool, check the settings the saver needs (autocommit, dict row factory) in the current docs.
  4. Run two turns on the same `thread_id`, restart the app, and confirm the history is still there.
- **Gate:** `get_graph().draw_mermaid()` output and the persistence demo.

### Phase 6 — Text-to-SQL agent subgraph
- **Learn:**
  - few-shot prompting;
  - structured output (`SQLGeneration{sql, explanation, tables_used}`);
  - self-correction loops;
  - `interrupt()` and `Command(resume=...)`;
  - `RetryPolicy`.
- **Do:** build these nodes:
  1. `retrieve_schema`:
     - calls MCP `search_schema` and `get_few_shots` (hybrid on the server), filtered by `selected_domain`;
     - in Phase 9, runs the **same `hybrid_search` function** over learned SQL pairs (Chroma `episodes`, `kind=sql_pair`), then a **second RRF pass** fuses the seed few-shots with the learned pairs and keeps the top 3.
  2. `generate_sql`: produces the structured output above.
  3. `validate`: the local validator, which is guardrail layer 1.
  4. `approval_gate`:
     - calls MCP `explain_query`;
     - if the estimated rows exceed a threshold or the query touches PII, calls `interrupt({sql, reason})`.
     - **HITL lives in this graph node** because it is deterministic and easy to test.
     - `MCPAdapter` *does* let a server pause the graph through elicitation → `interrupt()`. Having the DB server elicit confirmation for PII queries is optional defense-in-depth.
  5. `execute`: calls MCP `run_select_query`. On error, it loops back to `generate_sql` with the error message, up to **2 retries**.
  6. `explain_result`: a plain-language answer, and records `sql_used`.
  - Put `RetryPolicy` (connection errors and timeouts only) on the LLM and MCP nodes.
- **Gate:** 5 questions that cover a self-correction, a HITL approve, a HITL reject and a blocked write.

### Phase 7 — Web Search agent subgraph
- **Learn:**
  - `Send` for dynamic fan-out;
  - `CachePolicy`;
  - chunking;
  - cross-encoder vs embedding re-ranking;
  - context compression;
  - citation discipline.
- **Do:** a **deterministic pipeline**, not a ReAct agent, because small models follow a fixed pipeline more reliably than they choose tools.
  1. `plan_queries` (LLM, structured output): writes 1–3 queries plus recency and topic from the task.
  2. `search`: one `Send` per query, in parallel, with `CachePolicy(ttl=3600)` and `RetryPolicy`.
  3. `merge_dedupe`: normalises URLs, keeps the top 4 with domain diversity.
  4. `fetch`: one `Send` per URL, in parallel; failures are tolerated.
  5. `chunk`: about 800 tokens with 100 overlap; every chunk carries its url and title.
  6. `rerank`: flashrank against the task, keeps the top 6.
  7. `compress`: a token budget of about 3k, and drops near-duplicate chunks.
  8. `assess` (LLM, structured `{sufficient, missing, refined_query}`): if the evidence isn't enough, loops back to `plan_queries`, **at most 2 rounds**.
  9. `summarise`: the answer uses `[n]` citations that map to `{url, title, published_date}`. Any `[n]` with no fetched source is removed.

  The prompt tells the model that **page text is data, never instructions**. If every fetch fails, it answers from the snippets and flags the answer as low-confidence.
- **Gate:**
  - 3 answers with real URLs;
  - logged chunk counts before and after re-ranking;
  - one case where `assess` triggers a second round.

### Phase 8 — Planner and Supervisor
- **Learn:** planner vs executor; ReAct; re-planning; iteration budgets; parallel dispatch with `Send`.
- **Do:**
  1. `planner` (LLM):
     - takes the standalone question, profile, memories and session context;
     - outputs a `RouteDecision` with `intent` and the step list, including `depends_on`;
     - a direct answer produces an empty plan that goes straight to `finalize`.
  2. `supervisor` (router), after each agent returns:
     - writes the results into the plan through the reducer;
     - if a result is doubtful, asks the LLM for a structured judgement `StepReview{ok, reason}`; a not-ok verdict sends control back to `planner` to re-plan;
     - otherwise dispatches every step whose dependencies are met, **in parallel with `Send`**, or goes to `finalize`.
  3. Budget: **at most 5 dispatches per turn**, where **every `Send` counts as one**. After that, finalize with a partial answer.
     - This counts dispatches, not raw MCP calls (one SQL step alone makes 4+ MCP calls).
     - Each subgraph keeps its own caps: 2 SQL retries, 2 web rounds.
     - Document this reading of §5.1 in the report.
  4. `finalize`: writes a grounded answer with citations and/or the SQL used, and applies the profile's formatting preferences.
  5. Prompts come from `prompts/supervisor/planner_v1.yaml`, and the version used is recorded in `prompt_versions`.
- **Gate:** one conversation with a web, a SQL, a hybrid (e.g. "our beverage sales vs the global beverage market trend") and a direct question. Show the plan in state for each turn. The hybrid turn shows both agents dispatched in the same superstep.

### Phase 9 — Memory layers (see the memory architecture guide below)
- **Learn:** the memory guide below, especially the pairs that are easy to confuse and **how memory changes over time**.
- **Do:**
  1. `history.py` + the `summarize_history` node: past 20 messages or about 3k tokens, fold the older messages into `summary` with `RemoveMessage`, keeping the last 6.
  2. `profile.py`:
     - wraps `runtime.store`, namespace `("profile", user_id)`;
     - fixed fields plus `extra` keyed preferences;
     - each field stores `{value, source, updated_at, history}`, plus `signals`;
     - the confirmation-question flow.
  3. `facts.py`:
     - Chroma `user_facts`, with attribute-keyed facts and temporal metadata;
     - supersede-not-overwrite;
     - time-aware ranking and dated injection.
  4. `episodes.py`: Chroma `episodes`, `kind=summary`, one per thread.
  5. `sql_pairs.py`:
     - Chroma `episodes`, `kind=sql_pair`, `scope=global`;
     - PII-free;
     - deduped by fingerprint;
     - tagged with `schema_version`.
  6. Wire `load_profile`, `load_context`, the parallel `write_memory` branches and the SQL agent's learned-pairs lookup.
  7. Build the `memory_maintenance` graph and `scripts/consolidate_memory.py`.
  8. Unit tests for the time-evolution rules (listed in the guide), using a fake clock.
- **Gate:**
  - In a new thread with the same `user_id`, the bot recalls a preference and a fact. A different `user_id` sees neither.
  - The Python → Java and Nordic → APAC scenarios behave as described in the guide.
  - Bring the "what is stored where and why" table as a draft for the report.

### Phase 10 — Semantic cache (entity-aware; see the dedicated section below)
- **Do:**
  1. `contextualize` node: a single structured-output call that returns `{standalone_question, entities}`.
  2. `check_cache` runs after `load_profile`, because the key includes `profile_hash`. The cache write happens in the `write_cache` branch.
  3. `scripts/calibrate_cache.py`: measure the similarity of paraphrase pairs vs near-miss pairs to choose the threshold.
- **Gate:**
  - a paraphrase hits the cache;
  - `product_id=100` → `150` **misses**, yet the SQL agent reuses the learned pair;
  - changing a profile preference makes the old cached answer miss;
  - the calibration numbers.

### Phase 11 — Streamlit UI with streaming and HITL
- **Learn:** Streamlit's rerun model; `st.session_state`; why calling `asyncio.run()` on every rerun breaks async DB pools and MCP clients.
- **Do:**
  1. Run **one background event-loop thread**, held in `st.cache_resource`, that owns the graph, checkpointer, store, vector store and MCP adapter. Submit work to it with `asyncio.run_coroutine_threadsafe`.
  2. **Serialize runs per `thread_id`** with an `asyncio.Lock` keyed on the thread. Memory writes finish after the answer is shown, so a quick follow-up message would otherwise start a second run on the same thread.
  3. The sidebar has:
     - `user_id`, `thread_id` (with a "new chat" button) and a domain selector;
     - a **memory panel**: the profile with its history, active facts with their dates, and forget/reset buttons.
  4. Stream with `astream(stream_mode=["updates","messages"], subgraphs=True)`. Show node progress, then the final answer. Token-by-token streaming is a stretch goal.
  5. On an interrupt, show the SQL and the reason with Approve/Reject buttons, then resume with `Command(resume=...)`.
  6. Under each answer, show the citations, the SQL used, a "cache hit" badge, and thumbs up/down. A thumbs-down deletes the cache entry or learned pair.
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
     - **memory:** facts come only from the user's own statements, and memory blocks are labelled as context, not instructions;
     - **output:** citations required.
- **Gate:** a checklist with one test that proves each layer.

### Phase 13 — Evaluation
- **Do:**
  1. `eval/questions.yaml` with about 35 cases:
     - 10 web, 10 SQL, 6 hybrid, 4 direct;
     - memory cases, **including preference changes over time**;
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
     - an optional LLM-as-judge score.
  3. `eval_retrieval.py`: extract the gold tables from each gold SQL with sqlglot, and report **recall@k for dense-only vs BM25-only vs RRF**.
  4. Compare at least 2 prompt versions or models, and save the results.
- **Gate:** the results tables and an analysis of the top 3 failure modes.

### Phase 14 — Report, diagram, README
- **Do:**
  - an architecture diagram in which every §4.1 component is identifiable;
  - the LangGraph component map;
  - the memory justification table;
  - the guardrails and HITL design;
  - the eval and retrieval results;
  - **limitations**:
    - model quality;
    - Northwind's small schema;
    - in-process BM25 only suits small corpora;
    - Tavily's coverage;
    - cache staleness;
    - residual DNS-rebinding risk;
    - memory extraction errors;
  - README setup: `docker compose up`, `ollama pull`, `uv sync`, build the index, start both MCP servers, `streamlit run`;
  - optional: Dockerfiles for the MCP servers.
- **Gate:** final review of the repo and the report.

---

## Memory architecture guide

### The layers at a glance

| Layer | What it holds | Storage | How it's written | How it's curated | Who reads it, and when | Prompt slot (budget) |
|---|---|---|---|---|---|---|
| **Short-term: conversation** | messages + running `summary` | checkpointer (`agent_memory`), `thread_id` | automatically, after every superstep | `summarize_history`: past 20 messages or about 3k tokens, older messages become the summary (via `RemoveMessage`), keeping the last 6 | restored **automatically** on invoke with the same `thread_id` (no load node). `contextualize`, `planner` and `finalize` read the summary plus recent messages | "Conversation so far" (~1.5k) |
| **Short-term: session** | `selected_domain`, `filters` | graph state (checkpointed) | UI selector in the invoke input, or the planner's `session_update` ("let's focus on sales") | overwritten; cleared by a new thread or "reset" | `planner` (context); `retrieve_schema` as a **metadata filter** | "Session context" (~50) |
| **User profile** | how this *person* likes answers: fixed fields (`answer_length`, `currency`, `units`, `date_format`, `preferred_domain`) + `extra` keyed preferences (e.g. `code_language`) | LangGraph `AsyncPostgresStore`, `("profile", user_id)`, key `prefs` (no vector search) | `update_profile` branch: explicit lasting statements, UI settings, confirmed questions, or evidence-backed inference (see "over time") | by key; explicit beats inferred; `history` keeps the last 5 values; sidebar view and reset | **`load_profile`**, in full, every turn, **before `check_cache`** (the cache key needs `profile_hash`) | "User preferences" in `planner` and `finalize` (~100) |
| **Long-term semantic (facts)** | durable facts about the user ("manages the APAC region") | Chroma `user_facts`, metadata below | `write_facts`: 0–3 facts per turn, extracted **only from the user's own statements**, never from tool output | attribute-keyed dedupe and contradiction handling; supersede, not overwrite; expiry of temporary facts; consolidation; cap of 200 per user | `load_context` on a cache miss: top 3, time-weighted, similarity ≥ 0.75; bumps `last_used_at` | "Known facts about the user", **dated**, in `planner` and `finalize` (~150) |
| **Long-term episodic: summaries** | what *happened* in a past thread: goal, findings, outcome, date | Chroma `episodes`, `kind=summary`, id = `thread_id` | `write_episode`: every 5 turns and when the thread ends; upserted, one per thread | 90-day retention or a cap of 50 per user; recency-weighted | `load_context`: top 2, similarity ≥ 0.70 | "Relevant past interactions" in `planner` only (~200) |
| **Long-term episodic: SQL pairs** | standalone question → validated SQL, tables, domain | Chroma `episodes`, `kind=sql_pair`, **`scope=global`** (shared across users) | `write_sql_pair`: only after a successful run that HITL didn't reject; **skipped if the SQL touches PII columns** | dedupe by similarity ≥ 0.95 plus a sqlglot-normalised SQL fingerprint; `schema_version`, so pairs are dropped when the schema changes; a thumbs-down deletes | `sql_agent.retrieve_schema`: hybrid search, then RRF with the seed few-shots, top 3 | "Examples" in `generate_sql` (~600) |
| **Semantic cache** | finished answers | Chroma `semantic_cache`: `route`, `entity_key`, `profile_hash`, `data_version`, `expires_at` (epoch int, so `$gt` works), `prompt_version` | `write_cache`: only for complete, successful, standalone answers that weren't direct chit-chat or HITL | TTL (web ≈ 24h, DB until `data_version` changes); expired entries swept on read; a thumbs-down deletes | `check_cache`: exact-filter match **plus** similarity ≥ the calibrated threshold | **never injected**: the answer is returned as-is |
| *Seed knowledge* (not memory, but often confused with it) | schema docs, seed few-shots | Chroma `schema_docs`, `sql_seed_examples` | offline by `build_schema_index.py` | rebuilt when the YAML changes | DB MCP tools `search_schema` / `get_few_shots` (hybrid) | SQL prompt (~1k) |

### Pairs that are easy to confuse
- **Conversation vs session:** the conversation is *what was said*, a growing list of messages. The session is *settings that apply* to this chat, a few small structured fields. Both die with the thread.
- **Session vs profile:** "focus on sales" or "write *this* in Java" lasts for *this chat* only. "I always want short answers" or "from now on, Java" belongs to *the person*, across every chat.
- **Profile vs semantic facts:**
  - The profile holds **preferences about how to answer**. It is keyed, small and loaded **in full** every turn.
  - Facts are **open-ended statements about the user**. They're unbounded, so only the **relevant** few are retrieved by similarity.
- **Semantic vs episodic:** semantic memory is *what is true*, distilled and kept current. Episodic memory is *what happened*: dated events and worked examples.
- **Episodic SQL pairs vs the cache.** The cache returns a **finished answer** and skips everything. An SQL pair only helps the model **write the SQL**, and the query still runs against live data.

### How memory changes over time
**Routing rule.** The write step sorts each user statement into one of three kinds, so nothing is stored twice:
- **preference** (how to answer: length, currency, units, code language) → profile;
- **fact** (who the user is or what they work on) → `user_facts`;
- **one-off** ("just this once in USD", "write *this* in Java") → session only, never saved long-term.

**Profile evolution**
- **Keyed, never by similarity.** Each preference is stored under a normalised key, e.g. "I want code in Python" → `extra.code_language = python`. An update replaces the value for its key.
  - Similarity can't be used here. "Code in Python" and "code in Java" embed almost identically, so similarity dedupe would treat Java as a *re-statement* of Python and keep the wrong value (G12).
- **Each field** stores `{value, source: explicit|ui|inferred, updated_at, history: [last 5]}`. The history supports undo and audit.
- **Explicit beats everything.** A lasting statement ("from now on Java", "I've switched to Java") or a UI change overwrites the key immediately. The old value moves to `history`.
- **Ambiguous conflicts are confirmed.** If a statement is ambiguous ("I want code in Java") **and** conflicts with an existing explicit value, `finalize` answers in Java and asks once: "Should I use Java from now on?" A "yes" on the next turn updates the profile.
- **Inferred drift needs evidence.**
  - Signals such as "shorter please" are counted per key in `signals`, with timestamps.
  - Three or more matching signals, from at least two threads within 30 days, set the value with `source=inferred`.
  - If the inferred value would replace an **explicit** value, it triggers the confirmation question instead of overwriting silently.
  - Older signals are pruned.
- **Downstream effects are automatic.** `profile_hash` changes, so old cache entries stop matching. The next turn's `load_profile` sees the new value.

**Python → Java, worked through:**

| User says | Result |
|---|---|
| "I prefer code in Python" | `code_language = python` (explicit) |
| "From now on, give me Java" | `code_language = java` at once; `python` moves to history |
| "Write *this* one in Java" | this chat uses Java; the profile stays `python` |
| "I want code in Java" (ambiguous) | answers in Java and asks "Should I use Java from now on?" |
| Keeps asking for Java without saying "from now on" | after 3 signals across 2 threads, asks the same question |

**Fact evolution.** Metadata on every fact:

| Field | Meaning |
|---|---|
| `attribute` / `value` | e.g. `work.region` / `APAC` |
| `kind` | `stable` or `temporary` |
| `valid_until` | expiry date, for temporary facts |
| `status` | `active`, `superseded` or `archived` |
| `supersedes` / `superseded_by` | links between an old fact and its replacement |
| `created_at`, `last_confirmed_at`, `last_used_at` | when it was first stated, last re-stated, last retrieved |
| `use_count` | how often it has been retrieved |
| `confidence` | how much to trust it |

- **Attribute keys first, similarity second.**
  - The same attribute with a different value is a **contradiction**, whatever the similarity.
  - The same attribute and value is a **re-statement**.
  - Similarity (≥ 0.90 re-statement; 0.80–0.90 the LLM decides ADD/UPDATE/NOOP) is only a fallback for facts without a clear attribute.
  - This stops "manages Nordic" and "manages APAC" from being merged as duplicates.
- **Re-statement** bumps `last_confirmed_at` and `confidence`. Nothing new is inserted.
- **Contradiction** ("I now manage APAC" vs Nordic):
  - the new fact is inserted, the old one gets `status=superseded`, and the two are linked;
  - nothing is deleted, so history survives;
  - retrieval returns only `status=active`.
- **Temporary facts.** "This quarter I'm preparing the Q3 review" is `temporary`, with a `valid_until` taken from the text, or 30 days if none is given. It is filtered out once expired.
- **Time-aware ranking:** `score = similarity × (0.7 + 0.3 × freshness)`, where `freshness = exp(−age_days / 180)` and age counts from `last_confirmed_at`.
- **Dated injection.** Facts appear as "Manages APAC region (stated 2026-10-02)". Facts unconfirmed for over 180 days are tagged "(may be outdated)", so the LLM can hedge.
- **Consolidation.** The `memory_maintenance` graph runs per user, from `scripts/consolidate_memory.py`, nightly or after 20 new facts. It:
  1. archives expired temporary facts;
  2. clusters active facts with similarity ≥ 0.85, and the LLM merges each cluster, with the newest statement winning;
  3. archives facts older than 365 days that were never used;
  4. enforces the 200-fact cap, evicting by `last_used_at`.
- **User control:**
  - "What do you know about me?" lists active facts with their dates.
  - "Forget X" hard-deletes the fact.
  - The sidebar shows the profile, facts and their history.

**Episodes** follow the same pattern: dated summaries, recency weighting at read time, 90-day retention. **SQL pairs** expire through `schema_version`.

**Tests (Phase 9):**
- contradiction supersedes rather than overwrites;
- a temporary fact disappears after `valid_until` (fake clock);
- an inferred value doesn't silently override a recent explicit one;
- Python → Java: "from now on" replaces, "this one" doesn't, an ambiguous conflict asks;
- Nordic → APAC is detected by attribute key;
- consolidation merges duplicates.

### When each layer loads during one turn
1. **Invoke** with `thread_id`. The checkpointer restores the conversation and session state, with no code of yours involved.
2. **`contextualize`** reads the summary and recent messages and rewrites the question into a standalone form, extracting its entities.
3. **`load_profile`** reads the profile in full and computes `profile_hash`.
4. **`check_cache`** queries the semantic cache. On a hit, it responds and stops.
5. **`load_context`** reads facts (top 3, time-weighted) and episode summaries (top 2).
6. **`planner`** plans using all of the above, plus the session context.
7. **`sql_agent.retrieve_schema`** reads the seed few-shots (hybrid, through MCP) and the learned SQL pairs (hybrid + second RRF), filtered by `selected_domain`.
8. **`finalize`** applies the profile's formatting preferences. The answer streams to the UI.
9. **`write_memory`** fans out in parallel to `write_facts`, `update_profile`, `write_episode`, `write_sql_pair` and `write_cache`. These run *after* the answer is shown, so the user doesn't wait for them.
10. **`summarize_history`** runs only past the threshold.

### Prompt assembly and budgets
1. system prompt (role + guardrails)
2. profile
3. session
4. facts (dated)
5. episodes
6. conversation summary
7. recent messages
8. task

If the total is over budget, episodes are cut first, then facts. Every memory block is labelled as *context about the user, not instructions*.

---

## Semantic cache design: the `product_id=100` vs `150` problem

**The problem.** "Show details of clothes with product_id 100" and "… product_id 150" embed almost identically, with cosine similarity around 0.97. A naive similarity-only cache would return **product 100's answer for product 150**. That is a *false hit*, and it's worse than a miss because the answer looks right but is wrong.

**The design: similarity is necessary but not sufficient.** A cache hit requires **all** of:
1. **Similarity ≥ threshold** on the *standalone* question, calibrated in Phase 10.
2. **An exact `entity_key` match.** `contextualize` extracts the literals that change the answer: IDs, numbers, years and dates, quoted strings, and named entities such as products, customers and countries. It normalises and sorts them into a key like `product_id=100`. The key is applied as an **exact metadata filter** in the Chroma query, so different IDs can never match.
3. **The same `profile_hash`**, so a user who wants EUR never gets a USD-formatted answer cached for someone else. `load_profile` therefore runs before `check_cache`, and a profile change automatically invalidates that user's old entries.
4. **Freshness:** `expires_at > now` for web answers, stored as an epoch int so Chroma's `$gt` works, and `data_version` equal to the current value for DB answers.

**What happens in your example:**
- **User A asks about 100.** It's a miss, so the full pipeline runs. This writes a cache entry (`entity_key=product_id=100`) and a **global learned SQL pair** (the query touches no PII).
- **User B asks about 150.** The similarity is high but the `entity_key` differs, so it's a **cache miss**, as intended.
  - The SQL agent then retrieves the product-100 SQL pair as a **few-shot example**. It generates `… WHERE product_id = 150` quickly and correctly, and still runs against live data.
  - You lose the instant answer but keep most of the speed-up, with **zero risk of a wrong answer**.
- **Anyone asks "What are the details for product 100?"** It's a paraphrase with the same entity key (and a matching profile), so it's a **cache hit** and returns instantly.

**Never cache:**
- follow-ups that couldn't be made standalone;
- answers that errored, were partial (budget hit) or went through HITL;
- direct chit-chat (cheap anyway).

**Measure it:** the eval includes near-miss pairs, and you report the **false-hit rate** next to the hit rate.

---

## Gotchas (each is revisited in its phase)
- **G1 Structured output:** use `with_structured_output(..., method="json_schema")` and test it in Phase 0. With Qwen3 on Ollama, make sure `<think>` text never leaks into parsed output.
- **G2 MCP client is beta:** `langchain.mcp` emits a beta warning, so pin its version. The fallback is the archived `langchain-mcp-adapters` (`MultiServerMCPClient`).
- **G3 One session per tool call:** opening the adapter per call pays the connection cost every time. Hold it open for the app's lifetime (Phase 4). *The DB server is HTTP, so there is no process spawn per call.*
- **G4 HITL location:** `interrupt()` only works inside graph nodes (or through MCP elicitation via `MCPAdapter`), and needs a checkpointer plus a `thread_id`.
- **G5 Guardrails in two places:** in the agent *and* at the MCP server. The read-only role is the last line of defence.
- **G6 Streamlit + async:** use one background event loop (Phase 11).
- **G7 Chroma distance:** create every collection with the cosine space, and convert distance to similarity (`1 − distance`).
- **G8 Northwind dates:** the data covers 1996–98 unless shifted (done in `db/04_shift_dates.sql`).
- **G9 Stdio child environment:** *no longer applies.* Both servers run as their own HTTP processes and read `.env` themselves.
- **G10 Postgres image and volumes:** pinned to `postgres:14.24` with a named volume through compose.
- **G11 BM25 IDF on tiny corpora:** `rank_bm25.BM25Okapi` gives an IDF of 0 or less when a term appears in half the docs or more (2-doc corpora, domain-filtered subsets, a new learned-pairs set), which silently empties the keyword branch. Use `bm25s` (Lucene IDF) or `BM25Plus`, score over the full corpus, then filter.
- **G12 Similar embeddings, different meaning:** "code in Python" vs "code in Java", or "manages Nordic" vs "manages APAC", are near-duplicates to an embedding model. Preferences and facts must be keyed (preference key, fact `attribute`), not deduped by similarity alone.
- **G13 Concurrent runs on one thread:** memory writes run after the answer is shown, so serialize runs per `thread_id` in the UI bridge (Phase 11).

## Verification (end-to-end)
1. Start the stack:
   - `docker compose up -d`;
   - Ollama running;
   - both MCP servers (`/health` returns 200);
   - `uv run streamlit run src/multi_agenti_conversation_system/ui/streamlit_app.py`.
2. In one chat, ask a web, a SQL, a hybrid and a direct question. Check the citations and the SQL shown. The hybrid turn dispatches both agents in parallel.
3. A PII query triggers HITL approval. A DELETE attempt is blocked.
4. Retrieval:
   - `search_schema` hits show `rrf_score`, `dense_rank` and `bm25_rank`.
   - With Chroma stopped, the tool returns `degraded: true` with BM25 hits.
   - The recall@k table compares dense, BM25 and RRF.
5. Cache:
   - a paraphrase hits the cache;
   - the `product_id 100 → 150` pair misses but reuses the learned SQL pair;
   - a profile change makes the old answer miss.
6. Memory:
   - A new thread with the same user recalls a preference and a fact. A different user doesn't.
   - Python → Java: "from now on" switches, "this one" doesn't, an ambiguous request asks.
   - Nordic → APAC: only APAC is recalled, and Nordic is stored as `superseded`.
   - A temporary fact stops appearing after `valid_until`.
7. `uv run pytest` passes (unit by default; `-m integration` with Docker up). `uv run python eval/run_eval.py` produces the metrics table.
