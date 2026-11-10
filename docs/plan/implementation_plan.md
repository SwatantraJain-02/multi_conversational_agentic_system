# Implementation Plan — Multi-Agent Assistant (CS2 + CS3): LangGraph · A2A · MCP · Layered Memory · Langfuse · Evals

> **This file is the single source of truth for the project plan.** Any other plan notes are scratch. When a decision changes, it changes here first.
> The pre-CS3 version of this plan is in git history (commit `171006a`).

## Context

**Two briefs, one system:**
- **CS2** ([Capstone Problem Statement](../Capstone%20Problem%20Statement%20-%20Multi-Agent%20Chatbot%20with%20LangGraph%20and%20MCP.docx)) defines the base system:
  - a Planner/Supervisor routing work to a Web Search agent and a Text-to-SQL agent;
  - every tool call through MCP;
  - layered memory: short-term, long-term semantic and episodic, user profile, semantic cache;
  - guardrails, HITL, prompt versioning, repeatable evaluation.
- **CS3** ([Case Study 3](../Case%20Study%203_%20Enhancing%20the%20Multi-Agent%20Conversational%20Assistant.pdf)) **extends** CS2 and says not to rebuild it (§1, §8). It adds:
  - **A2A** collaboration between agents;
  - **semantic memory** with a full lifecycle;
  - **Langfuse** Prompt-as-a-Service;
  - a structured **evaluation framework**.
  - It also adds deliverables and a grading rubric. CS2 stops at §6.
- **Deadline:** Mon **16 Nov 2026, 3:00 PM**, which is also the evaluation session. Submissions go to SharePoint.

**CS3 minimum demos.** Every phase serves at least one of these:
1. A2A collaboration across 2+ agents.
2. A follow-up that reuses stored semantic memory.
3. A prompt pulled from Langfuse, with its version or label visible in the trace.
4. An eval run over a dataset that explains at least one improvement.
5. A controlled failure showing fallback or graceful degradation.

**How we work:** I implement each phase. You review at every **Gate**, ask questions and steer. We only move on once a gate passes. Each phase has a one-line **Learn** list of the concepts to understand.

## Status

| Phase | Status |
|---|---|
| 0 Environment and tooling | **Done** (on the old stack: Ollama embeddings) |
| 1 Data layer | **Done** (on the old stack: compose Postgres + Chroma). Re-done in Phase A on Aiven + Pinecone |
| 2 DB MCP server | **Done.** Streamable HTTP + bearer token, validator, smoke script, integration tests |
| 2b Hybrid retrieval | **Done** (commit `171006a`). Dense + BM25 → RRF, degraded mode. BM25 moved into Pinecone in Phase A |
| A Cloud migration | **Done** (commit `f9de2a2`). Aiven set up, Pinecone document index (4096-dim), smoke + integration tests green |
| B Web MCP server | **Done (6 Nov).** `search_web` + `fetch_page` with SSRF guard and rate limits; smoke + 37 unit tests green; both MCP servers run with bearer auth ON |
| C Foundations | **Done, awaiting review (10 Nov).** All six steps; `smoke_tracing.py` (17 checks), `smoke_a2a.py` (24), 187 unit tests green |
| D SQL Agent service | **Done, awaiting review (10 Nov).** `smoke_sql_agent.py` passes live over A2A (6 questions, up to 24 s each); 21 + 2 graph tests; 210 unit tests green |
| E Web Agent service | **Done, awaiting review (10 Nov).** `smoke_web_agent.py` passes live over A2A (3 questions + a controlled failure); 43 more unit tests; 264 unit tests green. The `assess` loop was skipped (third in the cut order) |
| F Synthesis Agent + peer delegation | **Done, awaiting review (10 Nov).** `smoke_synthesis.py` passes live with five processes: combined answer without delegation, delegation to the Web agent visible in the Langfuse trace, graceful degradation when the Web agent is down; 13 graph tests; 293 unit tests green |
| G Supervisor + short-term memory | **Done, awaiting review (10 Nov).** `smoke_supervisor.py` passes live with five processes, all checks, on the third run (web, SQL, follow-up, hybrid in parallel, compare, direct, approve and decline, trimming, restart, trace session id, Web agent down); 50 supervisor tests; 344 unit tests green. Re-planning after a failed step was skipped |
| H Long-term memory + semantic cache | **Done, awaiting review (10 Nov).** `smoke_memory.py` passes live, all 21 checks (the first run failed 6: see below). 51 memory tests; `calibrate_cache.py` measured Nemotron |
| I Guardrails | **Done, awaiting review (10 Nov).** One test per layer (`test_guardrails.py`) + in-graph tests; `smoke_guardrails.py` live, 9 of 9 |
| J Streamlit UI | **Done except the screen recording, awaiting review (10 Nov).** `scripts/smoke_ui.py` drives the real page with Streamlit's `AppTest` against the live services: all 16 checks pass. Not yet looked at in a browser |
| K Evals | **Harness written (10 Nov); first run invalid.** 39-case dataset, scorers (unit-tested), judge, Langfuse experiment runner, CSV and summary export. The first baseline hit the OpenRouter free daily quota after 23 cases (the last 16 failed as "no valid plan"): the runner now checks the model first and flags such cases. Re-run pending the quota reset |
| L Deliverables | README, demo guide and presentation outline written; diagram and evaluation report not started |

---

## Decisions locked in

| Area | Choice | Notes |
|---|---|---|
| Infra | **No Docker.** Every stateful service is a managed cloud service. | The demo needs a stable network. Say so in the report. |
| LLM (chat) | **OpenRouter** (`CHAT_MODEL`), built in `llm.py` | Prompts leave the machine. Say so in the report. |
| LLM (judge) | **OpenRouter** `JUDGE_MODEL`, a **different** model from the generator | Avoids self-preference bias in evals. |
| Embeddings | **OpenRouter** `nvidia/nemotron-3.5-embed-8b:free` through **`langchain-openai` `OpenAIEmbeddings`** (OpenAI-compatible API) | **4096 dims** (measured). `query: ` / `passage: ` prefixes. No `input_type` or `dimensions` parameter. One text per request (G18). |
| Business DB + memory DB | **Aiven Postgres**: one service holding the `northwind` and `agent_memory` databases | `sslmode=require`. Roles `agent_readonly` (northwind, read-only) and `agent_app` (agent_memory). |
| Vector DB | **Pinecone** serverless **document index** `capstone` (Documents API) | Schema: `keywords` (BM25 full-text string field) + `embedding` (`dense_vector`, cosine). Every other field is filterable metadata. One namespace per corpus: `schema_docs`, `sql_seed_examples`, `user_facts`, `episodes`, `semantic_cache`. |
| Hybrid retrieval | **Pinecone BM25 search + Pinecone dense search, fused client-side by RRF** (k=60), per Pinecone's hybrid-search guide | Both searches run in parallel. `keywords` holds our `tokenize()` output, so `unit_price` also matches `unit` and `price`. If the embedding API fails, results are BM25-only (`degraded: true`). If Pinecone is down, the tool returns `ok: false`. Replaced the in-process `bm25s` index on 6 Nov. |
| User profile | LangGraph **`AsyncPostgresStore`**, no vector index | Namespace `("profile", user_id)`. |
| Conversation state | LangGraph **`AsyncPostgresSaver`** | Used by the supervisor (`thread_id`) and the SQL agent (`thread_id = A2A contextId`). |
| Agent ↔ agent | **A2A protocol** (`a2a-sdk` 1.3, protobuf types, protocol 1.0): agent cards, JSON-RPC over HTTP, task states. Agents are Starlette apps behind `BearerAuth` (`A2A_TOKEN`; the card and `/health` stay public) | The evaluators expect the real protocol. Each agent is its own service, built with LangGraph. |
| Agent → tools | **MCP**, Streamable HTTP + bearer token: DB `:5555`, Web `:5556` | Agents call MCP tools. The supervisor calls agents over A2A (see Report notes). |
| Prompts + tracing | **Langfuse Cloud** | Prompts with labels `development` / `production` and a local YAML fallback. Traces cover every service. Evals run as Langfuse datasets/experiments. |
| Web search | **Tavily** | `TAVILY_API_KEY` (free tier). |
| UI | **Streamlit** | The supervisor runs in-process with the UI. |
| Guardrails | **Custom, layered** (Phase I) | Regex PII recognizers. Presidio only if spaCy installs on Python 3.14. |

**Storage split (for the report):**
- **Aiven `northwind`:** business data, reached only through `agent_readonly` by the DB MCP server.
- **Aiven `agent_memory`:** checkpoints for the supervisor and SQL agent, plus the profile store, as `agent_app`.
- **Pinecone:** every vector, one namespace each.
- **Langfuse:** prompts, traces, datasets and scores. Nothing here is memory.

## Changeable defaults

| Area | Default | Why |
|---|---|---|
| Python / packaging | Python **3.14**, **uv**, `src/` layout, package `multi_agenti_conversation_system` | Already in place. |
| A2A SDK | `a2a-sdk` (pinned) | The official Python SDK. Its API changes between releases (G19). |
| MCP server / client | `mcp` 2.3.x / `langchain[mcp]` `MCPAdapter` | As before (G2). |
| Embeddings client | `PrefixedOpenAIEmbeddings(OpenAIEmbeddings)` (`embeddings.py`) | Adds the prefixes, an LRU query cache and a 429 counter. `check_embedding_ctx_length=False` sends raw text, and `encoding_format=float`. |
| SQL parsing | `sqlglot` | AST-based checks. |
| Page cleaning / re-ranking / chunking | `trafilatura` (+ Tavily Extract fallback) / `flashrank` / `langchain-text-splitters` | Local, with no extra keys. |
| DB bootstrap | `scripts/init_db.py` (psycopg) | There's no `psql` binary locally. |
| Config | `pydantic-settings` + `.env` | Already in place. |
| Tests / lint | `pytest` (markers `integration`, `live`), `pytest-asyncio`, `ruff` | Unit tests by default. |

---

## Target architecture

Diagrams (HLD + LLD, Mermaid): [`docs/architecture/hld_lld.md`](../architecture/hld_lld.md). This plan stays the source of truth.

```
Streamlit UI ──► Supervisor (LangGraph, in-process with the UI) ──── A2A client ────┐
   memory: Aiven (checkpoints, profile) + Pinecone (facts, episodes, cache)        │  A2A: agent cards,
   prompts + traces: Langfuse                                                      │  JSON-RPC/HTTP, task states
       ┌──────────────────────────────┬────────────────────────────────────────────┤
       ▼                              ▼                                            ▼
 SQL Agent :7001 (LangGraph)    Web Agent :7002 (LangGraph)          Synthesis Agent :7003 (LangGraph)
  learned SQL pairs (Pinecone)        │                               │ peer A2A delegation ──► Web Agent
       │ MCP                          │ MCP                           │ (always, when it has no external data)
       ▼                              ▼
 DB MCP :5555                   Web MCP :5556 (Tavily, fetch_page)
  Aiven northwind (read-only) · Pinecone schema_docs / sql_seed_examples (BM25 + dense, RRF client-side)
```

**Supervisor graph** (LangGraph):
```
START → input_guard → contextualize → load_profile → check_cache ─(hit)─► respond → END
                                                         │ (miss)
                                                    load_context → planner ⇄ supervisor ─► dispatch[Send ×steps] ─► supervisor
                                                                                 │              │ (SQL step input-required)
                                                                                 │              ▼
                                                                                 │         hitl_gate[interrupt] → resume_dispatch ─► supervisor
                                                                                 ▼ (plan done / budget / direct)
                                                    finalize → output_guard → respond → write_memory[parallel branches] → summarize_history → END
```
- **`planner`** (LLM, structured output) returns a `RouteDecision` with `intent`, `plan[PlanStep{id, agent, skill, task, depends_on}]` and `reason`. It runs once per turn, and again when a step fails.
- **`supervisor`** is deterministic. It marks steps done, dispatches every step whose dependencies are met (in parallel with `Send`), enforces a **budget of 5 dispatches per turn**, and calls the LLM only to judge a doubtful result.
- **Plan shapes:**
  - web: `[web]`;
  - sql: `[sql]`;
  - hybrid: `[sql ∥ web] → synthesis`;
  - direct: `[]`;
  - "internal vs market" comparisons: `sql → synthesis`, where Synthesis delegates to Web itself. This is the **delegation demo**.
- **`finalize`** only applies the profile's formatting to the answer that Synthesis or the single agent wrote. It never writes a competing answer.

**A2A design:**
- **Discovery:** `AGENT_URLS` in config. The registry fetches each `/.well-known/agent-card.json` at startup and on refresh. The planner's capability list is **rendered from the cards**, so a new agent needs no change to the planner code.
- **Contract** (`a2a/contracts.py`, Pydantic, validated on both ends). **Encoding decision (10 Nov):** each model travels as one A2A `Part` holding its JSON text, media type `application/json`, **not** a `DataPart`. A DataPart is a protobuf `Value` that stores every number as a double, so SQL rows would come back with `product_id` 100 as `100.0` (measured, and covered by a unit test). The JSON text round-trips exactly.
  - `TaskRequest{task_id, parent_task_id, sender, skill, task, data, context{profile, memories, session}, constraints{timeout_s, max_rows}}`; `data` is the skill-specific input (e.g. Synthesis takes `internal` / `external` results)
  - `ResumeRequest{sender, value}`: the answer to an `input-required` task (becomes `Command(resume=value)`)
  - `trace_ctx` is **not** in these models: it travels in the A2A **message metadata**, so a resume message (a decision, not a `TaskRequest`) keeps the caller's trace too
  - `TaskResult{status (completed|input_required|failed), answer, data, citations, sql, error{code, message, retryable}, agent, prompt_versions, a2a_task_id, context_id}`; the client fills the two ids so a result can be resumed
  - Error codes: `timeout`, `unavailable`, `invalid_output`, `bad_request`, `failed`, `internal`
  - The agent card carries the JSON schemas in an A2A **extension** (`urn:multi-agent-capstone:a2a-contracts:v1`) next to its skills. Together these are the "agent contracts" CS3 asks for.
  - Size caps, checked before parsing: task text 4,000 chars, one contract part 1,000,000 chars.
- **Failure handling** (client wrapper), each mapped to a `TaskResult.error.code`:
  - timeout: `asyncio.wait_for`;
  - unavailable: card fetch or connection fails, so the agent is marked down;
  - invalid output: Pydantic fails, so retry once;
  - the A2A `failed` state.
  - The supervisor then re-plans, or finalizes a partial answer that names the missing piece.
- **Peer delegation (deterministic):** if Synthesis gets no external data, it **always** resolves the Web Agent's card and calls it directly. An LLM-decided trigger is too unreliable for a live demo.
- **HITL across A2A:** three supervisor nodes, because resuming re-runs the interrupted node (G22).
  1. `dispatch` sends the task. If the SQL agent ends it in `input-required` with `{sql, reason, est_rows, pii_cols}`, it stores `{a2a_task_id, context_id, payload}` in state.
  2. `hitl_gate` only calls `interrupt(payload)`.
  3. `resume_dispatch` sends the approve or reject decision on the same `contextId`.
  - The SQL agent resumes from its own checkpointer.
  - **Known limitation:** the A2A task store is in memory, so restarting the SQL agent mid-approval loses the task.
- **Tracing:** `trace_ctx` travels in the A2A message metadata, so every agent's spans nest under the supervisor's Langfuse trace (G20).

**Agent graphs** (each served by the generic `a2a/server.py` executor):
- **SQL:** `retrieve_schema` (MCP hybrid + learned pairs → second RRF) → `generate_sql` → `validate` → `approval_gate` (input-required) → `execute` (≤ 2 self-correction retries) → `explain_result` → `write_sql_pair`.
- **Web:** `plan_queries` → `search[Send ×q]` → `merge_dedupe` → `fetch[Send ×url]` → `chunk` → `rerank` → `compress` → `assess` (≤ 2 rounds) → `summarise` with `[n]` citations.
- **Synthesis:** `check_inputs` → (`delegate_web` via A2A when there's no external data) → `synthesise`: a grounded comparison that keeps both sets of citations and the SQL used.

**Ports:**
- DB MCP 5555, Web MCP 5556;
- SQL / Web / Synthesis agents 7001 / 7002 / 7003 (echo test agent 7009);
- Streamlit 8501.
- `scripts/run_all.py` starts everything and waits for `/health` and the agent cards.

## LangGraph component map

| Component | Primitive | Notes |
|---|---|---|
| Supervisor and each agent | `StateGraph(..., context_schema=AppContext)` | Nodes take `runtime: Runtime[AppContext]`, which holds the models, MCP tools, A2A registry, memory classes and prompt registry. |
| Parallel plan steps / web search / fetch | `Send` fan-out | Each supervisor `Send` counts toward the 5-dispatch budget. |
| Conversation memory | `AsyncPostgresSaver` | Supervisor: `thread_id`. SQL agent: `contextId`. |
| History trimming | `summarize_history` with `RemoveMessage` + `summary` | |
| User profile | `AsyncPostgresStore` via `runtime.store` | |
| HITL | `interrupt()` in `hitl_gate`, resumed with `Command(resume=…)` | Bridged to A2A `input-required`. |
| Transient failures | `RetryPolicy` on LLM / MCP / A2A nodes | Connection errors and timeouts only. |
| Memory writes | `write_memory` → parallel branches | `write_facts`, `update_profile`, `write_episode`, `write_cache`. Learned pairs are written in the SQL agent. |
| Memory maintenance | small `memory_maintenance` graph | Should-tier (Phase H). |

**Kept plain (no LangGraph):**
- the MCP servers;
- `hybrid_search`;
- the SQL validator;
- the Tavily client;
- the SSRF guard;
- the rate limiter;
- the Pinecone wrapper;
- the A2A client wrapper.

## Conventions
- **Agents reach tools only through MCP. The supervisor reaches agents only through A2A.** `supervisor/` never imports `agents/`. Agents never import each other; they use A2A.
- **Each server and agent is its own process.** Shared code lives in `a2a/`, `mcp/common.py`, `guardrails/`, `retrieval/` and `memory/`.
- **Nodes get their dependencies from `Runtime[AppContext]`.** They never call `get_*()` globals.
- **Tools and agents never raise across a boundary.** MCP returns `{"ok": False, "error": …}`. A2A returns `TaskResult.error`.
- **No inline prompt strings.** Every prompt comes from `prompts.get_prompt(name)`, which uses Langfuse first with `prompts/<name>.yaml` as the fallback. The version used is recorded in state and linked to its Langfuse generation.
- **No secrets in logs or traces; PII is reduced, not eliminated.** The Langfuse mask hook (`guardrails/pii.py`) redacts configured secrets and credential shapes, e-mails, SSNs, Luhn-valid card numbers and phone numbers that have separators. It does **not** catch names, addresses or bare digit runs (e.g. Northwind's `contact_name`). Keep PII-flagged columns out of span inputs and outputs (Phase D, below).

## Target project structure

```
multi_conversational_agentic_system/
├── pyproject.toml, uv.lock, .env.example, README.md
├── db/                      # 02_northwind.sql, 03_grants.sql, 04_shift_dates.sql (roles + databases are created by scripts/init_db.py)
├── data/                    # schema_descriptions.yaml, few_shots.yaml
├── prompts/                 # <area>/<name>.yaml: local fallback and seed for Langfuse
├── docs/                    # both briefs, plan/implementation_plan.md, architecture, report
├── scripts/                 # init_db.py, build_schema_index.py, sync_prompts.py, sync_eval_dataset.py, run_all.py,
│                            # calibrate_cache.py, smoke_llm.py, smoke_db_mcp.py, smoke_web_mcp.py, smoke_a2a.py
├── src/multi_agenti_conversation_system/
│   ├── config.py, llm.py, embeddings.py, vectorstore.py, prompts.py, tracing.py
│   ├── guardrails/          # sql_validator.py, input_guard.py, pii.py, output_guard.py
│   ├── retrieval/           # hybrid.py
│   ├── mcp/                 # client.py, common.py (BearerAuth, loopback), servers/db_server/, servers/web_server/
│   ├── a2a/                 # contracts.py, server.py (LangGraph → AgentExecutor), client.py (registry, timeouts)
│   ├── agents/              # sql/, web/, synthesis/  (graph.py + app.py each)
│   ├── supervisor/          # state.py, context.py, graph.py, nodes/
│   ├── memory/              # profile.py, facts.py, episodes.py, sql_pairs.py, cache.py, history.py
│   └── ui/streamlit_app.py
├── eval/                    # dataset.yaml, run_eval.py, scorers.py, judge.py, results/
└── tests/                   # unit/, integration/
```

---

## Phases

Must = needed for a CS2 or CS3 demo. Should = cut if behind. **Cut order if behind:**
1. the consolidation graph;
2. inferred-drift signals;
3. the web `assess` loop;
4. the recall@k retrieval eval;
5. token streaming;
6. `CachePolicy`.

None of these is a CS3 minimum demo.

### Done: Phases 0, 1, 2, 2b
- **0:** uv, Python 3.14, `config.py`, `llm.py`, `smoke_llm.py`.
- **1:** Northwind with shifted dates; roles `agent_readonly` (SELECT only, read-only transactions, `statement_timeout=5s`) and `agent_app`; `data/schema_descriptions.yaml` (14 tables, domains, `pii` flags); `data/few_shots.yaml` (20 pairs).
- **2:** `guardrails/sql_validator.py` (one SELECT, no DDL/DML, LIMIT clamp, table allowlist, PII columns). DB MCP tools `list_tables`, `get_schema`, `search_schema`, `get_few_shots`, `explain_query`, `run_select_query`, over Streamable HTTP `:5555` with a bearer token and `/health`.
- **2b:** `retrieval/hybrid.py`: `tokenize`, `rrf` (k=60), `candidate_depth`, `hybrid_search`.
  - When the dense branch is down, the tool returns BM25-only results with `degraded: true`.
  - Phase A replaced the in-process `BM25Index` with Pinecone's BM25 field.
  - The output contract adds `rrf_score`, `dense_rank` and `bm25_rank`.

### Phase A — Cloud migration (6–7 Nov) [Must] — gate passed, awaiting review
- **Learn:** Pinecone document indexes (schema, BM25 field, dense field, filters) and eventual consistency; Aiven roles and SSL; embedding dimensions; OpenRouter rate limits.
- **Done:**
  1. Accounts and keys are in `.env`.
  2. Dependencies: the 3.14 wheels were checked, then the new packages were pinned:
     - `pinecone` 10.3, `langfuse` 5.2, `a2a-sdk` 1.3;
     - `langchain-openai` 1.7, `flashrank`, `tavily-python`, `trafilatura`;
     - `langgraph-checkpoint-postgres`, `psycopg-pool`.
     Removed: `chromadb`, `langchain-chroma`, `langchain-ollama` and `bm25s`. `pinecone-text` has no 3.14 wheels.
  3. `embeddings.py` provides `PrefixedOpenAIEmbeddings`, built on `langchain-openai` and pointed at OpenRouter:
     - prefixes, LRU query cache, 429 counter;
     - `dimension()` is checked against the index.
  4. `vectorstore.py`:
     - creates the document index on first use, with schema `keywords` (FTS, stemming and stop words off, because our tokenizer does both) and `embedding` (`dense_vector`, measured dimension, cosine);
     - helpers: `upsert`, `keyword_search`, `dense_search`, `list_ids`, `delete`, `eq_filter`.
  5. `schema_index.py`:
     - Pinecone BM25 search and dense search, run in parallel and fused by `hybrid_search`;
     - hit metadata and candidate depth still come from the catalog YAML;
     - stale ids are removed after each rebuild.
  6. `scripts/init_db.py` (psycopg, idempotent):
     - roles and databases;
     - `GRANT agent_app TO avnadmin` (G17);
     - Northwind, then the date shift, then the grants, in one transaction.
  7. `config.py` and `.env.example`: Aiven, Pinecone, `LANGFUSE_BASE_URL`, `PROMPT_LABEL`, `JUDGE_MODEL`, the embedding settings, `TAVILY_API_KEY`.
  8. Deleted `docker-compose.yml`, `scripts/init_db.sh`, `scripts/chroma_check.py`, `tests/test_llm.py` and `db/01_databases.sql`. Added the pytest markers.
  9. Updated `smoke_llm.py` (adds a query→document check and a burst test), `smoke_db_mcp.py`, `build_schema_index.py` and the integration tests.
- **Gate results (6 Nov):**
  - `smoke_llm.py`: ALL CHECKS PASSED. Dimension is **4096**. A burst of 20 queries took 20.7 s with 0 × 429.
  - `init_db.py`: 830 orders (2024-07-04 .. 2026-05-06), `read_only=on`. A re-run changes nothing.
  - `build_schema_index.py`: both namespaces built, with dense and BM25 ranks on every sample question.
  - `smoke_db_mcp.py`: ALL CHECKS PASSED. `pytest -m integration`: 8 passed. Unit tests: 97 passed.
  - Bad OpenRouter key: `degraded: true` with BM25 hits. Bad Pinecone key: `ok: false` ("schema search unavailable"), while `run_select_query` still works.
- **Open:** whether to buy OpenRouter credits depends on the daily cap once evals run (G18).

### Phase B — Web MCP server (7 Nov) [Must]
- **Learn:** token-bucket rate limiting; fetched text is *data*, never instructions; SSRF and DNS rebinding.
- **Do:**
  - Follow the DB server pattern on `:5556`, and move `BearerAuth` and the loopback check into `mcp/common.py`.
  - `search_web(query, max_results 1–10, recency, topic, include_domains, exclude_domains)` calls Tavily and returns `{ok, results:[{title, url, snippet, published_date, score}]}`.
  - `fetch_page(url, max_chars=20000)`:
    - http and https only;
    - resolves DNS and blocks private, loopback, link-local and metadata IPs;
    - follows at most 3 redirects, re-checking every hop;
    - 10 s timeout and a 2 MB cap;
    - extracts with trafilatura, falling back to Tavily Extract under 200 chars.
  - Rate limits per tool: search 10/min with a burst of 5, fetch 30/min. When exhausted it returns `{ok:false, error:"rate_limited", retry_after_s}`.
- **Gate:** `smoke_web_mcp.py` shows a search, a fetch, a blocked `127.0.0.1`, a blocked redirect to a private IP, and a tripped rate limit.
- **Gate results (6 Nov):**
  - **`smoke_web_mcp.py`: ALL CHECKS PASSED.**
    - Searches return real URLs; `recency` + `topic=news` work.
    - A search result fetched through trafilatura (3,000 chars, truncated).
    - Blocked: `127.0.0.1:5555`, `169.254.169.254`, `localhost`, `127.0.0.1.nip.io` (a public DNS name for loopback), and `file://`.
    - A burst of 15 fetches → 10 `rate_limited`, with `retry_after_s`.
  - **The redirect-to-private-IP check is proven in the unit tests** (`tests/unit/test_web_fetch.py`, mock transport). Public redirectors such as httpbin refuse to redirect to loopback themselves (HTTP 403), so they can't demonstrate it live.
  - `BearerAuth` and the loopback check moved to `mcp/common.py`, and the DB smoke still passes (401 checks included). Unit tests: 134 passed.
  - **Auth decision:** static bearer tokens (`DB_MCP_TOKEN`, `WEB_MCP_TOKEN`, 32-byte random) on every service, A2A agents included. Re-verified with auth on: both servers answer 401 without a token, DB smoke + Web smoke + 8 integration tests pass. Not OAuth 2.1: every caller is one trusted internal service, so there is no per-user identity. State this as a limitation in the report.
- **Files:**
  - `mcp/common.py`;
  - `mcp/servers/web_server/{server,fetch,tavily_client,rate_limit}.py`;
  - `scripts/smoke_web_mcp.py`, `tests/unit/test_web_fetch.py`;
  - `WEB_MCP_*` and `WEB_*_PER_MIN/BURST` settings.

### Phase C — Foundations: MCP client, Langfuse, A2A plumbing (8 Nov) [Must]
- **Learn:** A2A agent cards, tasks, task states and parts; Langfuse prompts, labels and versions; trace context propagation.
- **Do:**
  1. `mcp/client.py`: `MCPAdapter` with bearer headers (G2). Discovers tools with `list_tools()`, groups them per server, and is held open for the app's lifetime (G3).
  2. **Done.** `prompts.py`: `get_prompt(name) -> Prompt{template, version, label, source, tag, link}`; `source` is `langfuse`, `fallback` (Langfuse configured but didn't answer) or `local` (tracing off). `compile()` is the one strict `{{variable}}` renderer for Langfuse and YAML alike; `type: chat` is supported.
     - Langfuse `get_prompt(name, label=PROMPT_LABEL, cache_ttl_seconds=…, fallback=<local yaml>)`;
     - templates use variables only;
     - links each generation via `langfuse_prompt` metadata.
  3. **Done.** `prompts/<area>/<name>.yaml` (first one: `plumbing/echo-reply`) plus `scripts/sync_prompts.py`: pushes with label `development`, skips unchanged prompts (idempotent), never sets `production`. Promotion to `production` is a manual step in the Langfuse UI: the controlled-promotion step.
     - Prompt names with a slash must be URL-encoded as one path segment when calling the raw `api.prompts` client; `get_prompt` does it itself. An unencoded slash lands on Langfuse's website 404 page, not the API.
     - **`PROMPT_LABEL` defaults to `production`, but the sync script pushes `development`.** Until a version is promoted, fetching `production` gives `source=fallback`. Set `PROMPT_LABEL=development` in `.env` while developing; `smoke_tracing.py` passes the label explicitly.
  4. **Done.** `tracing.py`: a lazily built Langfuse client with the mask (`guardrails/pii.py`: secrets by value and by shape, e-mail, SSN, Luhn-checked cards, phones with separators; order dates, ids, prices and IPs survive), `observation()` (no-op when `TRACING_ENABLED` is false or the keys are empty), `current_trace_ctx()` / `valid_trace_ctx()` (a malformed context is ignored, never fatal). The A2A client opens an `a2a.call:<agent>` span and injects its context; the executor opens `agent:<name>` as its child.
     - **Agent shutdown:** spans are flushed from the app's *lifespan* shutdown. Code after `uvicorn.run()` never runs on SIGTERM, because uvicorn re-raises the signal and the process dies (first gate run lost every agent span this way).
     - The a2a-sdk emits its own OpenTelemetry spans; Langfuse's default filter drops them, so they don't clutter traces.
     - Unit tests set `TRACING_ENABLED=false` in `tests/conftest.py`, so `pytest` never ships traces.
  5. **Done (10 Nov).** `a2a/contracts.py`, `a2a/server.py` (`GraphAgentExecutor`: compiled graph ↔ A2A task, `thread_id = contextId`, node progress → `working` updates, interrupt → `input-required`, any crash → a clean `internal` failure) and `a2a/client.py` (`AgentRegistry`: card discovery, one `wait_for` per call with best-effort cancel, retry-once on invalid output for fresh calls only, a restarted agent is re-detected).
  6. **Done (10 Nov).** `agents/echo/app.py` (:7009) proves the plumbing. `scripts/smoke_a2a.py` runs it as a subprocess (24 checks, including an agent that starts after the registry, `trace_ctx` on call and resume, and the original request on resume); `tests/unit/test_a2a_contracts.py` covers the contracts.
     - **SDK facts observed** (a2a-sdk 1.3.0): the executor must enqueue the `Task` itself on a fresh message; an uncaught exception reaches the caller as a bare `InternalError` carrying the exception text, so the executor catches everything; `input-required` ends the stream, and a follow-up with the `task_id` calls `execute()` again with `current_task` in that state; data-part numbers are floats.
     - Done in the order 1, 5, 6, then 2–4, because A2A is the riskiest piece and Phases D–F depend on it.
- **Gate results (10 Nov), all passed:**
  - `smoke_a2a.py` (24 checks): card discovery, an agent that starts after the registry, a call, interrupt → resume approve and reject, crash / bad skill / timeout as clean failures, 401 without a token, killed agent → `unavailable` and recovery, `trace_ctx` on call and resume.
  - `smoke_tracing.py` (17 checks), read back from Langfuse Cloud: prompt fetched with `version=1` and label `development`; a bad key → `fallback` in under 2 s; empty keys → `local`; a missing label → `fallback`; the trace is `smoke-root > a2a.call:echo > agent:echo > echo-reply` for the call, the pause and the resume (4 + 4 + 3 spans); generations are linked to the prompt name and version; `TaskResult.prompt_versions` carries `plumbing/echo-reply@v1`; an e-mail in a task is masked in Langfuse.
  - The original gate wording, for reference:
  - the echo agent is discovered from its card and called;
  - its span nests under the caller's trace;
  - a prompt fetch shows its version and label, and a bad Langfuse key uses the local fallback;
  - a killed agent gives `error.code = unavailable`, not a crash.

### Phase D — SQL Agent service (8–9 Nov) [Must]
- **Learn:** few-shot prompting; structured output `SQLGeneration{sql, explanation, tables_used}`; self-correction loops; HITL over A2A.
- **Do:**
  - The SQL graph above, on `:7001`, with its card skill `answer_sql_question`.
  - `approval_check` calls MCP `explain_query`. If the query touches PII, or its estimated rows reach `SQL_APPROVAL_ROWS`, the task ends in `input-required` (`hitl_gate` holds only the `interrupt()`, G22).
  - **Why the row trigger is `>= MAX_ROWS`, not a bigger number (measured):** the DB server clamps every query to `LIMIT MAX_ROWS` *before* EXPLAIN, so the estimate never exceeds 100. `SQL_APPROVAL_ROWS` (default 100) therefore means "the result would be cut off". A threshold above `MAX_ROWS` would be dead code. **Refined (10 Nov, found by the live Phase F run):** even `>= MAX_ROWS` was too noisy: the estimate for a join with an aggregate ("revenue by year") also hits the clamp while three rows come back, which paused a harmless query for approval. The row trigger now applies only to queries that do not collapse rows (no `GROUP BY`, no aggregate in the output; `guardrails.sql_validator.is_aggregate_query`, window functions don't count). A raw list query is the case where the estimate is reliable and the cut-off is real. An aggregate over PII columns still asks (the PII trigger is separate).
  - Learned pairs are read from Pinecone `episodes` (`kind=sql_pair`) and fused with the seed few-shots by a second RRF. They're written only after a successful run that HITL didn't reject and that touches no PII.
  - `RetryPolicy` on the LLM and MCP nodes.
  - **Tracing hygiene:** never put rows from PII-flagged columns into span input or output (use `find_pii_columns` to drop them), because the mask can't catch names and addresses. Cache `pii.configured_secrets()` before it runs on thousands of strings per span (it calls `settings.model_dump()` every time).
- **Built (10 Nov):** `agents/sql/{state,db_tools,context,nodes,graph,app}.py`, `memory/sql_pairs.py`, `prompts/sql/{generate,explain}.yaml` (chat prompts, Langfuse `sql/generate` v1 and `sql/explain` v1), `scripts/smoke_sql_agent.py`, `tests/unit/test_sql_agent.py`.
  - **Graph:** `retrieve_schema → generate_sql → validate → approval_check → [hitl_gate] → execute → explain_result → write_sql_pair`. A failure in validate, EXPLAIN or execute goes back to `generate_sql` with the error and the failed SQL, at most `SQL_MAX_RETRIES` (2) times; then the task fails.
  - **Approval binds to one exact SQL string** (`approved_sql`). A corrected query is a different string, so it passes `approval_check` again and asks again; `execute` also refuses an unapproved query that needs approval (tested).
  - **Result shapes (Phase G depends on these):** a HITL decline is `completed` with `data.rejected=true`, and a refusal (e.g. "delete all orders") is `completed` with `data.refused=true`, so the supervisor doesn't re-plan around a decision. `failed` is kept for real failures: retries exhausted (`failed`), a dependency down (`unavailable`, via `AgentError`).
  - **Dependencies come from `Runtime[SqlContext]`** (db tools, pair store, `generate`, `summarise`), so tests run the real graph with fakes. DB tools reach the server only through MCP; the agent never imports `db_server`. `schema_version` is a hash of `get_schema` for all tables.
  - **Learned pairs** (`memory/sql_pairs.py`): Pinecone `episodes`, `kind=sql_pair`, `scope=global`, hybrid search fused with the seed few-shots by a second RRF. The id is the fingerprint of the SQL with literals blanked (sqlglot), so "product 100" and "product 150" share one entry; only pairs for the current `schema_version` are returned. Never written for PII, any query that needed approval (a declined one, or an approved large result), a refusal, an empty result, or when `EVAL_MODE` is on. A failed write never fails the answer.
  - **DB outage vs. SQL mistake:** the server reports a database error as `ClassName: text`. Connection-level errors (the `OperationalError` / `InterfaceError` family, a pool timeout) become `unavailable` and are never shown to the model; SQL errors and `QueryCanceled` (statement timeout) go back to it for a corrected query (tested with the server's real reply shape).
  - **For Phases G and K:** a raw, unbounded list query can still ask for approval, and a PII query always does. The supervisor and the eval harness must handle `input_required` on any SQL step; log how often it happens in Phase K.
- **Trace hygiene:** when the result touched PII columns, the `explain-result` generation's input and output are replaced by a placeholder (the mask can't catch names). Verified in Langfuse: no contact names in any trace (unit-tested too). `pii.configured_secrets()` is cached.
  - **Checkpointer:** `AsyncPostgresSaver` on Aiven `agent_memory` over a pool of at most 2 connections, checked on checkout; `thread_id` = A2A `contextId`. The generic app got a `startup` hook so an agent can open async resources, and the executor passes a runtime `context` to the graph.
  - `RetryPolicy` (one retry) only on transport-level `AgentError`, so retries don't multiply with the chat client's own four.
- **Gate results (10 Nov), `smoke_sql_agent.py`, all passed with a real model (own DB MCP server on :15555 and agent on :17001):**
  - plain question: completed, rows and SQL returned, prompt versions `@v1`, pair learned and then found by search (and deleted afterwards);
  - PII question: pauses with the SQL and `customers.contact_name`; approved → rows; **not** learned; declined → `completed`, `rejected=true`, no rows;
  - "Delete all orders placed in 2024": refused, `orders` count unchanged;
  - single-customer lookup: pauses, flags `customers.phone`;
  - a harder revenue-per-category question: right first time (`attempts=1`).
  - **Self-correction was not seen live** (the model got every query right first time); the loop is proven deterministically in the graph tests (a failed EXPLAIN, an execution error, retries running out, a validator-blocked `DELETE`). We did not add a fault-injection switch to the agent.
  - Gate wording, for reference:
- **Gate:** five questions over A2A covering a self-correction, a HITL approve, a HITL reject, a blocked write, and a PII query.

### Phase E — Web Agent service (9 Nov) [Must; `assess` is Should]
- **Learn:** `Send` fan-out; chunking; cross-encoder re-ranking; context compression; citation discipline.
- **Do:**
  - The Web graph above, on `:7002`, with its card skill `answer_web_question`.
  - Chunks are about 800 tokens with 100 overlap. flashrank keeps the top 6. Compression has a budget of about 3k tokens.
  - Any `[n]` with no fetched source is removed. If every fetch fails, the agent answers from the snippets and flags low confidence.
- **Built (10 Nov):** `agents/web/{state,text,web_tools,context,nodes,graph,app}.py`, `agents/llm_calls.py` (shared structured/text model calls, also used by the SQL agent), `prompts/web/{plan,summarise}.yaml` (Langfuse `web/plan` v1, `web/summarise` v1), `scripts/smoke_web_agent.py`, `tests/unit/test_web_{text,agent}.py`. `tracing.generation()` (shared traced, prompt-linked LLM call) moved out of the SQL nodes.
  - **Graph:** `plan_queries → search_one ×N → merge_dedupe → fetch_one ×M → chunk → rerank → compress → summarise` (`no_results` when there is nothing to read). `search_one` and `fetch_one` fan out with `Send`.
  - **Budget:** at most 3 queries and 5 pages per task, because the Web MCP server rate-limits per process (search bursts at 5, fetch at 10). A rate-limited reply is **data, never an error and never shown to the model**: a wait of up to 3 s is tolerated once, otherwise that search or page is skipped and counted (`data.rate_limited`). If every search is rate limited or fails, the task fails as `unavailable`.
  - **Reading:** the same page found by several queries is read once (URL normalised: no fragment, tracking params or trailing slash) and ranks higher for it (RRF over the per-query rankings). Chunks of 3,200 characters (about 800 tokens) with 400 overlap; flashrank (`ms-marco-TinyBERT-L-2-v2`, 3 MB, downloaded on first use into `.cache/flashrank`) keeps the top 6; the context is cut to 12,000 characters (about 3k tokens).
  - **Citations are checked in code:** any `[n]` without a source is removed (`[1, 9]` keeps the 1); the prompt is not trusted to do it. If no page can be read, the answer comes from the search snippets with a visible note and `confidence=low`. Search snippets have their own, lower minimum length than page chunks (a unit test caught the shared threshold turning a short snippet into "no results").
  - **Per-task isolation:** the accumulating lists are reset with `Overwrite([])` in `build_input`; a plain `[]` would be *added* to a previous task's results in the same A2A context (measured, tested).
  - **Prompt-injection stance:** fetched text only enters the prompt inside the numbered sources block, and the prompt says to treat it as data. Not a guarantee; Phase I adds the output guard.
  - **Checkpointer:** in memory. The agent never pauses for a human, so there is nothing to resume after a restart, and it saves an Aiven connection (G16).
- **Gate results (10 Nov), `smoke_web_agent.py`, all passed with a real model and real Tavily (own Web MCP on :15556, agent on :17002):**
  - 3 questions answered in 12–28 s with real https URLs; every cited URL is one the agent used; every `[n]` in the answer has a citation;
  - chunk counts before → after re-ranking: 10 → 6, 27 → 6, 17 → 6; 3–5 pages read per question; no rate limiting hit;
  - every prompt version `@v1` from Langfuse;
  - controlled failure: with the Web MCP server stopped, the call returns `unavailable` (not a crash).
  - **Known limitation:** `confidence` only checks that citations survived. On question 3 the answer honestly said "the sources do not contain recent news" yet `confidence` was `high`. Phase K's groundedness and completeness scores are the check for this; don't rely on `confidence` alone.
  - Gate wording, for reference:
- **Gate:** three answers with real URLs, and the chunk counts logged before and after re-ranking.

### Phase F — Synthesis Agent + peer delegation (10 Nov) [Must]
- **Learn:** agent-to-agent delegation vs supervisor routing; why delegation needs a deterministic trigger.
- **Do:** the Synthesis graph above, on `:7003`, with its card skill `synthesise_findings`.
  - It accepts `{question, internal: TaskResult?, external: TaskResult?}`.
  - If `external` is missing, it delegates to the Web Agent over A2A, with the parent `task_id` and `trace_ctx`.
- **Built (10 Nov):** `agents/synthesis/{state,context,nodes,graph,app}.py`, `SynthesisData` in `a2a/contracts.py`, `prompts/synthesis/{external-question,compare}.yaml` (Langfuse v1 each), `scripts/smoke_synthesis.py`, `tests/unit/test_synthesis_agent.py`.
  - **Input contract:** the question is `TaskRequest.task`; `data` is `SynthesisData{internal: TaskResult?, external: TaskResult?, external_question: str?}`, published in the agent card's contract extension. It lives in `a2a/contracts.py` because the supervisor must build it without importing an agent. A malformed one is `bad_request`.
  - **Graph:** `check_inputs → [derive_external_question → delegate_web] → compose`. **The trigger is deterministic:** no *usable* external result (missing, failed, or empty) means it asks the Web agent, always; with one it never does (tested both ways). A failed external result counts as missing and is re-fetched.
  - **Delegation** goes through its own A2A client (`AgentRegistry` knowing only the Web agent's address; agents never import each other), with `parent_task_id` = its own task id. The call runs inside its trace span, so Langfuse shows `agent:synthesis > a2a.call:web > agent:web`. An LLM step rewrites the user's question into its external side only (falls back to the original question).
  - **Graceful degradation:** if the Web agent is down or times out, the task still *completes*: `data.partial=true`, `data.missing=["external"]`, and the answer opens with a note naming what could not be included. It only fails (`unavailable`) when nothing at all is usable.
  - **Output:** one grounded answer; internal figures marked "(internal data)" and uncited; external claims cited with the Web agent's own `[n]`, checked in code against its citation list (a `[9]` with no source is removed); the SQL used is kept; the prompt versions of all three agents are reported together. Rows from PII columns never reach a trace (same rule as the SQL agent).
  - **Timing:** a delegated run took about 60 s (the Web agent alone is 12–40 s). The supervisor must call Synthesis with `timeout_s` of at least 150; the delegation inside has its own 90 s limit.
- **Gate results (10 Nov), `smoke_synthesis.py`, all passed live (DB MCP, Web MCP, SQL, Web and Synthesis agents as subprocesses on spare ports; `smoke_a2a.py` remains the echo-agent plumbing check):**
  - (a) given the SQL and Web results: one combined answer, `delegated=false`, the SQL kept, citations all from the Web agent's list, all three agents' prompt versions `@vN`;
  - (b) given only the SQL result: it delegated on its own, with an external question about the outside world only, and got a full answer with web citations;
  - the trace read back from Langfuse: exactly two Synthesis → Web calls (runs b and c, not a), and the Web agent's span nests under the delegated call;
  - (c) with the Web agent stopped: completed, `partial=true`, the gap named in the answer, the SQL still kept.
  - **Found while running it:** the SQL agent paused a harmless aggregate query for approval (see the refinement under Phase D); fixed and unit-tested. Regression-run afterwards: `smoke_sql_agent.py`, `smoke_tracing.py`, `smoke_a2a.py` all pass.
  - Gate wording, for reference:
- **Gate:** `scripts/smoke_a2a.py`, a scripted driver because the supervisor comes in Phase G.
  - Run (a) passes both the SQL and Web results to Synthesis and gets one combined answer back.
  - Run (b) passes only the SQL result. Synthesis delegates to Web on its own, and the Langfuse trace shows the delegation edge.

### Phase G — Supervisor graph + short-term memory (10–11 Nov) [Must]
- **Learn:** `StateGraph`, reducers, `Send`, checkpointers, `interrupt()` re-run semantics (G22); planning vs execution; iteration budgets.
- **Do:**
  - `supervisor/state.py`: `PlanStep`, `RouteDecision`, and `AgentState` with these fields:
    - `messages`, `summary`, `standalone_question`, `entities`;
    - `intent`, `plan` (with a reducer), `dispatch_count`;
    - `pending_hitl`;
    - `citations`, `sql_used`, `cache_hit`;
    - `selected_domain`, `filters`;
    - `user_id`, `profile`, `memories`;
    - `prompt_versions`, `agents_used`.
  - Nodes: `planner`, `supervisor`, `dispatch`, `hitl_gate`, `resume_dispatch`, `finalize` and `summarize_history` (past 20 messages or ~3k tokens, keeping the last 6).
  - Checkpointer and store on Aiven, with small pools (G16).
- **Gate:**
  - one conversation covering web, SQL, hybrid and direct questions, with the plan shown in state for each turn;
  - the hybrid turn dispatches SQL ∥ Web in one superstep;
  - history survives an app restart;
  - with the Web Agent stopped, the answer is partial but graceful.
- **Built (10 Nov):** `supervisor/{state,plan,context,nodes,graph,service}.py`, `prompts/supervisor/{plan,direct,summarise}.yaml`, `scripts/smoke_supervisor.py`, `tests/unit/test_supervisor{,_plan}.py`. The supervisor never imports `agents/`: it learns what exists from the agent cards (`AgentRegistry`).
  - **Flow:** `planner → supervisor → run_step ×N (Send, parallel) → supervisor → … → finalize → summarize_history`. The planner is an LLM with structured output and one repair attempt; `validate_plan` then checks agents and skills against the live cards, `depends_on` only pointing to earlier steps, at most 4 steps, and that `synthesis` depends on `sql`/`web`. The `supervisor` node is deterministic code (ready steps, dispatch budget of 5, stuck detection).
  - **Plan shapes:** `sql`, `web`, `compare` (sql → synthesis, Synthesis fetches the web side itself), `hybrid` (sql ∥ web → synthesis), `direct` (no agent: greetings, or a follow-up the conversation already answers). Each step's task is a standalone question, because agents never see the chat.
  - **HITL:** a `needs_approval` result pauses the turn (`interrupt()` in a node that does nothing else, G22). `Supervisor.resume()` forwards the decision to the SQL agent, bound to the exact SQL string. A decline is a completed result, so there is no re-plan.
  - **Short-term memory:** messages and a rolling summary are checkpointed per thread in Aiven (`AsyncPostgresSaver`, pool of 2). Past 20 messages or ~3k tokens it summarises the older ones and keeps the last 6. Per-thread locks; a Langfuse session id and a `turn` span cover each turn, with `a2a.call:*` spans under it.
  - **Per-skill timeouts:** sql/web 120 s, synthesis 240 s (a delegated run takes about 60 s).
  - **Answer:** composed in code from the step results (it names every gap and marks the answer `partial`), not re-written by a model, so citations and SQL stay exact.
- **Deviation:** re-planning after a failed step was deliberately not built. A failed step gives a partial answer that names the gap, which covers demo 5; a second planning pass would add latency and a new failure mode for little gain.
- **Gate results (10 Nov), `smoke_supervisor.py` (live; DB MCP, Web MCP, SQL, Web, Synthesis on spare ports). The first run failed 15 checks (planner schema, below), the second 19 of 21 (one follow-up turn), the third passed all:**
  - web, SQL, hybrid (SQL ∥ Web: time intervals overlap), compare (through Synthesis) and direct questions all took the planned route; prompt versions in each turn are real Langfuse versions (`supervisor/plan@v2`, …);
  - a PII question paused for approval; approve ran the query, decline said it was not run; the trimmed history became a summary; a new supervisor on the same thread kept the messages and summary and answered a follow-up from them; the turn trace carries the session id and the agent calls hang under the turn span;
  - with the Web agent stopped, a web question said so plainly, and a compare question was answered from our own data, flagged `partial`, with the gap named.
  - The second run's two failures were both turn 3: "And only the ones in Germany and France?" was answered correctly from the earlier result (`direct`, which the planner prompt allows), so it never produced a standalone SQL task. Turn 3 is now "And how many orders did the customers in Germany place?", which the history cannot answer, and the original strict check (a SQL step naming Germany and orders) passed on the third run.
  - **Found while running it:** the first run failed 15 checks. The free planner model left the optional `plan` field out of its JSON and wrote a very long `reason`. Fix: `plan` is required in the schema, `reason` is "one short sentence", and the prompt says `plan` is always present (`supervisor/plan` v2). Lesson for the report: with a free model, a field with a default can silently vanish; make structured-output fields required.


### Phase H — Long-term memory + semantic cache (11–12 Nov) [Must core; Should: inferred drift, consolidation]
- **Learn:** the memory guide below, especially the easy-to-confuse pairs and how memory changes over time.
- **Do:**
  - `profile.py`, `facts.py`, `episodes.py`, `sql_pairs.py`, `cache.py`, `history.py`, per the guide, on Pinecone namespaces.
  - The `contextualize` node returns `{standalone_question, entities}`.
  - `load_profile`, `check_cache` and `load_context` are wired in, along with the `write_memory` branches.
  - Agents receive only the relevant top-k memories in `TaskRequest.context`.
  - `scripts/calibrate_cache.py` re-calibrates the threshold for Nemotron.
  - Tests use a fake clock and poll after upserts (G14).
- **Gate:**
  - a new thread with the same `user_id` recalls a fact and a preference; a different user sees neither;
  - Python → Java and Nordic → APAC behave as the guide says;
  - cache: a paraphrase hits, `product_id` 100 → 150 misses but reuses the learned pair, and a profile change misses;
  - the "semantic memory vs short-term state vs cache vs transcript" table is drafted for the report.
- **Scope cut (10 Nov, advisor-reviewed):** Must-core only. Not built, and listed as Should: inferred preference drift from repeated signals, the confirm-once flow for an ambiguous conflicting statement, similarity fallback for facts without an attribute, consolidation, the 200-fact cap, episode retention. Episodes are minimal: one summary per thread, every 5 turns and from `Supervisor.end_thread`.
- **Built (10 Nov):** `memory/{entities,profile,facts,cache,episodes}.py`, `supervisor/memory_nodes.py`, `open_memory()` (checkpointer and `AsyncPostgresStore` on ONE pool: no extra Aiven connections), `vectorstore.{fetch,update,delete_where,build_filter}`, prompts `supervisor/{contextualize,extract-memory}` (Langfuse v1), `scripts/{calibrate_cache,smoke_memory}.py`, `tests/unit/{test_memory,test_supervisor_memory,memory_fakes}.py` (51 tests, in-memory fakes of Pinecone and the embedder).
  - **Graph:** `contextualize → load_profile → check_cache → (hit) → write_memory`, or `(miss) → load_context → planner → … → finalize | direct_answer → write_memory → summarize_history`. `write_memory` is one node: learn (one model call, skipped unless the message talks about the user), cache write, episode write. Each part catches its own errors: **memory never fails a turn** (a failed read is an empty read; an unknown profile switches the cache off for that turn).
  - **Entity key:** regex over the question (numbers, years, dates, quoted strings, capitalised words that do not open a sentence), merged with the model's entity list from `contextualize`, normalised and sorted. Regexes also run on the original question, so a "150" the free model drops still changes the key. No history means no `contextualize` call at all. Always a string, `""` for none.
  - **Filters that cannot silently widen:** `vectorstore.build_filter` raises on a `None` (the old `eq_filter` drops it, which would remove the `entity_key` clause and match every cached entry: the product 100 vs 150 false hit). A unit test asserts every lookup filter carries all five keys.
  - **Cache hit =** similarity ≥ `CACHE_THRESHOLD` AND exact `entity_key`, `profile_hash`, `prompt_version`, `data_version` AND `expires_at > now`. `prompt_version` is a hash of the tags of the seven answer-writing prompts as the agents fetch them now. **Decision: `DATA_VERSION` is a setting bumped by hand** when the business data changes; web answers expire after 24 h, database answers after 7 days as a backstop. Never cached: partial answers, anything that went through approval, direct chit-chat, a follow-up that could not be made standalone, a turn that changed the profile. `EVAL_MODE` switches the cache off both ways (the Phase G smoke sets it).
  - **Profile:** keyed, `{value, source, updated_at, history[last 5]}`, replaced by key (never by similarity: "Python" and "Java" embed alike). `profile_hash` covers values only and exists for an empty profile.
  - **Facts:** one active fact per (user, attribute) at a deterministic id, so "what does this statement mean?" is a fetch by id and works straight after a write. Same value: re-statement. Other value: the old fact is kept as `superseded` (linked both ways), the new one is active. Every query filters `user_id` and `status=active` and `expires_at > now` (a temporary fact's `valid_until`; stable facts use a far-future date so one range clause covers both). Ranking `similarity × (0.7 + 0.3 × freshness)`; injected dated, "may be outdated" after 180 days.
  - **Agents** get only the profile and the top-3 facts in `TaskRequest.context` (the SQL/Web/Synthesis agents do not use them yet; the planner and the direct answer do).
  - **User control:** `Supervisor.what_i_know(user_id)`; `FactStore.forget` and `ProfileStore.reset` for the UI.
- **Calibration (`scripts/calibrate_cache.py`, Nemotron):** paraphrases 0.79–0.96; different questions about the same entity up to 0.85 ("Average order value" vs "Total order value"); near misses (product 100 vs 150) only 0.67, not the ~0.97 assumed above. `CACHE_THRESHOLD` default is **0.87**: a few paraphrases miss, no different question hits. The entity key remains the second guard (2025 vs 2024 scores 0.76).
- **Gate results (10 Nov), `smoke_memory.py`, live (real model, Pinecone, Aiven, Langfuse; the DB MCP and SQL agent as subprocesses; unique users per run, everything deleted at the end), all 21 checks passed:**
  - a lasting preference and a fact were learned from "From now on please give me code examples in Java. Also, I manage the APAC region."; a **new thread of the same user** recalled both; a different user saw neither and had no stored memory;
  - Python → Java replaced the value with `python` in its history; Nordic → APAC left one active region fact (APAC) and the Nordic one as superseded history;
  - cache: a paraphrase hit and called no agent; product 20 missed where product 10 was cached, its SQL was about 20, and the product 10 query had been learned as a SQL pair (reused for 20); a profile change made the old cached answer unusable;
  - an ended conversation was recalled as an episode for its user and for no other, and not for its own thread.
  - **Found by the first run (6 of 21 failed):** (1) **Nemotron scores are far lower than the thresholds assumed:** a relevant fact scored 0.52 against the 0.75 floor, so no fact was ever recalled. `calibrate_cache.py` now measures facts too: relevant 0.21–0.62, unrelated at most 0.12, so `FACTS_MIN_SIMILARITY` is 0.18 and `EPISODES_MIN_SIMILARITY` 0.25. (2) A fetch **by filter** lags behind a write by a few seconds, like a search (G14); a fetch **by id** does not, which is why attribute-keyed supersede works at once. The smoke polls the filter reads. (3) Two bugs in the smoke itself: the cache poll threshold 0.99 can never be met because the query prefix keeps even identical text below it, and product 100 does not exist in Northwind (so no SQL pair is learned from an empty result).
  - Regression: `smoke_supervisor.py` (Phase G) re-run with the memory nodes in the graph and `EVAL_MODE` on (cache off) passes all checks.


### Phase I — Guardrails (12 Nov) [Must]
- **Learn:** defence in depth; why the read-only DB role is the last line.
- **Do:**

  | Layer | Guard |
  |---|---|
  | Input | `input_guard.py`: length limits, injection/jailbreak heuristics plus an LLM classifier (Langfuse prompt `guard/input`), PII flagging |
  | Agent / MCP | SQL validator in the agent and the server, read-only role, timeouts, web rate limits, SSRF block (all done by now) |
  | HITL | large or PII queries go through `input-required` (Phase D) |
  | A2A boundary | Pydantic validation and size caps on every request and result |
  | Memory | write-gate before any Pinecone write: PII check plus "stable fact from the user's own words?" check. Memory blocks are labelled as context, not instructions |
  | Output | `output_guard.py`: PII redaction, citations stripped when they have no fetched source |
  | Traces | Langfuse mask hook |

- **Gate:** a checklist with one passing test per layer.
- **Built (10 Nov):** `guardrails/{input_guard,output_guard,memory_gate}.py`, `supervisor/guard_nodes.py`, prompt `guard/input` (Langfuse v1), `scripts/smoke_guardrails.py`, `tests/unit/{test_guardrails,test_supervisor_guards}.py`.
  - **Input guard** is the first graph node: `START → input_guard → (blocked) refuse → END`, else `contextualize`. Order of checks: empty/over-long (4000 chars, the same cap as an A2A task) are blocked; NULs stripped; regex heuristics for known injection shapes; only a heuristic hit calls the model classifier (`guard/input`: safe | injection | jailbreak | exfiltration), so ordinary messages cost no model call; PII (email, phone, Luhn card, SSN) is flagged, never blocked. A blocked message reaches no memory, planner or agent, and the blocked exchange is removed from the conversation so it cannot influence later planning.
  - **Decision: a classifier outage allows a heuristic hit but flags it** (no learning that turn). Failing closed would refuse every message containing "ignore" whenever the free model is down; the layers behind (read-only validator and DB role, approval, A2A contracts) bound what an injected instruction could do.
  - **Memory write-gate:** a flagged or PII-bearing message teaches nothing; each proposed preference or fact must contain no personal data or secret and must be grounded in the user's own words (each significant word of the value appears in the message), so a model that infers "the question mentions APAC, so they manage APAC" is rejected. Extraction only ever reads the user's message, never tool output, so a web page cannot poison memory.
  - **Output guard** runs in the supervisor's answer builder, so the shown text, the checkpoint and the cache all hold the guarded answer: secrets always redacted; PII redacted unless a human approved the query that fetched it (the approval gate is the PII control; regexes catch the machine-readable kinds only, not names); a `[n]` marker with no citation is removed.
  - **Traces:** the Langfuse mask (Phase C). **A2A boundary:** Pydantic models with `extra=forbid` and size caps (Phase C). **Agent/MCP and HITL:** Phases B and D.
- **Gate checklist (one passing test per layer, `tests/unit/test_guardrails.py`, whose docstring maps each layer to its tests):**

  | Layer | Test |
  |---|---|
  | Input | `test_input_known_attack_shapes_are_blocked_when_the_classifier_agrees`, `test_input_limits_and_cleaning`, `test_input_pii_is_flagged_not_blocked` |
  | Agent / MCP | `test_layer_sql_validator_rejects_anything_but_one_select` |
  | HITL | `test_sql_agent.py::test_pii_pauses_for_approval_then_runs_after_a_yes` |
  | A2A boundary | `test_layer_a2a_requests_are_validated_and_capped` |
  | Memory | `test_memory_gate_only_stores_what_the_user_said`, `test_memory_gate_never_stores_personal_data_or_secrets` |
  | Output | `test_output_secrets_are_always_redacted`, `test_output_pii_is_redacted_unless_a_human_approved_it`, `test_output_a_citation_marker_without_a_source_is_removed` |
  | Traces | `test_layer_traces_the_langfuse_mask_redacts_nested_text` |
  | In the graph | `test_supervisor_guards.py` (blocked turn reaches nothing; flagged turn learns nothing; ungrounded fact rejected; answer redacted) |
- **Live gate (`smoke_guardrails.py`, 10 Nov):** all 9 checks passed: 4 of 4 attacks blocked by the real classifier, 2 of 2 business questions that trip a heuristic allowed, an ordinary question made no classifier call, and a blocked turn through the real supervisor called no agent and left nothing in the conversation.


### Phase J — Streamlit UI (12–13 Nov) [Must]
- **Learn:** Streamlit's rerun model; why `asyncio.run()` on every rerun breaks async pools (G6).
- **Do:**
  - One background event-loop thread in `st.cache_resource`, and a lock per `thread_id` (G13).
  - Sidebar: user, thread ("new chat") and domain, plus a memory panel (profile with history, active facts with dates, forget/reset).
  - Node-progress streaming, and Approve/Reject on a HITL interrupt.
  - Under each answer: citations, SQL used, a cache-hit badge, the agents used, prompt versions, and thumbs up/down. A thumbs-down deletes the cache entry or learned pair.
- **Gate:** a screen recording of the five CS3 demo flows.
- **Built (10 Nov):** `src/ui/streamlit_app.py`, `scripts/run_all.py`, `Supervisor.{what_i_know, forget_fact, reset_profile, feedback}` (unit-tested with fakes), `trace_ctx` on `ask/resume`.
  - **One event loop in a daemon thread** (`Backend`, kept by `st.cache_resource`) with the supervisor open on it for the whole process; the script submits coroutines to it and polls a queue for node events to show progress (G6). The supervisor serialises turns of a thread (G13).
  - Sidebar: user (changing it starts a new chat, because memory is per user), domain filter (any, sales, customers, products, hr, logistics), thread id, **New chat** (calls `end_thread`, which writes the conversation's episode), and the memory panel (preferences with the value each replaced, active facts with dates, forget and reset buttons).
  - Each answer: badges (cached answer, blocked, partial, agents used, route), Sources, SQL used, "What I learned this turn", prompt versions and Langfuse trace id, and 👍/👎. A 👎 deletes the cache entry for that question and forgets the SQL pair learned from it (shared by all users); both are a Langfuse score on the turn's trace.
  - A turn that needs approval shows the SQL, the PII columns and Approve / Reject buttons; the supervisor resumes the SQL agent's own waiting task.
  - `run_all.py [--ui]` starts both MCP servers and the three agents, waits for each `/health`, reuses anything already running, and stops what it started on Ctrl-C.
- **Gate results (10 Nov), `scripts/smoke_ui.py`** (`AppTest` runs the real page code in-process; needs `run_all.py` running; unique user, everything deleted at the end), all 16 checks passed: the page starts clean; a SQL question shows its answer, SQL, prompt versions and 👍/👎; a PII question shows the approval card with the SQL and the `customers.phone` column, and Approve runs it; a stated preference and fact appear in the memory panel (the fact dated; the poll waits out the fetch-by-filter lag); **New chat** starts empty and the next question still recalls the region; a thumbs-down is acknowledged. The first run failed one check, a bug in the check itself (it read the first code block on the page, which was the previous question's SQL).
- **Still to do:** open it in a browser (`run_all.py --ui`), look at the layout, and record the five demo flows. `AppTest` proves the logic, not the appearance.


### Phase K — Evals (13–14 Nov) [Must]
- **Learn:** deterministic checks vs LLM-as-judge; rubric design; dataset coverage; why runs must be isolated (G23).
- **Do:**
  1. `eval/dataset.yaml`, about 40 cases:
     - normal: 10;
     - complex: 6;
     - hybrid: 6;
     - ambiguous: 5;
     - failure: 8 (injection, DROP, PII request, agent down, timeout, off-topic);
     - memory reuse: 3 two-turn cases;
     - cache near-miss: 2.
     Each case records `expected_intent`, `expected_agents`, plus `gold_sql` or `required_facts`.
  2. `scripts/sync_eval_dataset.py` uploads the dataset to a Langfuse dataset.
  3. `eval/run_eval.py --label development|production --run-name …` runs a Langfuse experiment, so each item is linked to its trace and prompt versions. Concurrency is 1–2 (G18).
     - **Run isolation:** each run uses its own `run_id` suffix on the `semantic_cache`, `user_facts` and `episodes` namespaces, sets `EVAL_MODE` (no learned-pair writes), and uses fresh user and thread ids.
  4. `eval/scorers.py` (deterministic):
     - `route_correct`, `agents_correct`;
     - `a2a_schema_valid`;
     - `sql_executes`, `result_matches_gold` (compares result sets, not SQL text);
     - `citations_present`, `guardrail_blocked`;
     - `latency_ms`.
  5. `eval/judge.py`: `JUDGE_MODEL` with the Langfuse prompt `eval/judge`. It scores relevance, correctness, groundedness, completeness and clarity from 1 to 5, each with a reason, and the scores are posted to Langfuse.
  6. Export to CSV plus `results/summary.md`. Run before/after on one change (e.g. planner v1 → v2) and write the failure analysis.
- **Gate:**
  - two comparable runs in Langfuse;
  - the CSV;
  - the top 3 failure modes, and the improvement made because of them.
- **Built (10 Nov), no run yet:** `src/multi_agenti_conversation_system/evals/{dataset.yaml,dataset,scorers,judge,stack,run_eval}.py` (inside the package, so scripts can import it; the plan said `eval/`), `scripts/sync_eval_dataset.py`, prompt `eval/judge` (Langfuse v1), `tests/unit/test_eval.py` (18 tests, no model).
  - **Dataset, 39 cases:** normal 10 (9 SQL with gold queries, 1 web), complex 6, hybrid 6, ambiguous 5, failure 7 (2 injections, DROP, DELETE, PII request, off-topic, "password of the database"), memory 3 (two-turn), cache 2 (a near miss that must not hit, a paraphrase that must). "Agent down" and "timeout" are not in it because they need a process killed mid-run: the Phase G and F smoke scripts cover them.
  - **Scorers (pure, `None` when not applicable):** `route_correct` (one intent or a list), `agents_correct`, `a2a_schema_valid` (every agent result validates as a `TaskResult`), `sql_executes`, **`result_matches_gold`** (compares result sets, not SQL text: row order ignored, numbers within a small tolerance, the agent's rows may carry extra columns), `required_facts`, `citations_present`, `guardrail_blocked`, `refused` (a destructive request refused and nothing executed), `approval_requested`, `cache_behaviour`, `no_agents_called`, `answered`, `no_secret_leak`, `latency_ms`. Gold queries are validated by the same SQL validator in a unit test.
  - **Judge** (`JUDGE_MODEL` or the chat model, temperature 0, prompt `eval/judge`): relevance, correctness, groundedness, completeness, clarity, 1–5 each with a reason. A failed judgement is skipped, not scored 0.
  - **Runner:** starts the DB MCP, Web MCP and the three agents on spare ports (25555…27003) with `EVAL_MODE` and the run's prompt label, runs the cases one at a time as a Langfuse experiment over the Langfuse dataset (`--local` uses the YAML), and links each turn's trace to the experiment item (`trace_ctx`). Per case: a fresh user and thread, `setup` turns for the memory cases, a primed cache entry for the cache cases (`EVAL_MODE` is off for those turns only), the approval request answered with a decline; anything the case wrote is deleted after it (facts, profile, cache entries). **Run isolation is by unique user id and cleanup, not by a namespace suffix per run** (the plan's idea): simpler, and equivalent for what the cases touch.
  - **Output:** `results/<run-name>.csv` (one row per case, every score) and `results/<run-name>.md` (mean per metric, per category, the failed checks).
- **Gate (to do):** two comparable runs (e.g. planner v1 vs v2) in Langfuse, the CSV, and the top 3 failure modes with the improvement made because of them.


### Phase L — Deliverables (15–16 Nov) [Must]
- **Do:**
  - architecture and flow diagram (LangGraph, MCP, A2A exchange, memory, Langfuse, eval flow);
  - README (prerequisites, env vars, `init_db.py`, `build_schema_index.py`, `sync_prompts.py`, `run_all.py`, `streamlit run`, assumptions, **what is simulated**);
  - demo evidence for the five minimum demos;
  - evaluation report (CSV + summary);
  - presentation (problem, design choices, implementation, demo, results, limitations, next steps).
  - No secrets in code, screenshots or traces.
- **Gate:** final repo and report review.

---

## Memory architecture guide

### The layers at a glance

| Layer | What it holds | Storage | How it's written | How it's curated | Who reads it, and when | Prompt slot (budget) |
|---|---|---|---|---|---|---|
| **Short-term: conversation** | messages + running `summary` | checkpointer (`agent_memory`), `thread_id` | automatically, after every superstep | `summarize_history`: past 20 messages or about 3k tokens, older messages become the summary (via `RemoveMessage`), keeping the last 6 | restored **automatically** on invoke with the same `thread_id` (no load node). `contextualize`, `planner` and `finalize` read the summary plus recent messages | "Conversation so far" (~1.5k) |
| **Short-term: session** | `selected_domain`, `filters` | graph state (checkpointed) | UI selector in the invoke input, or the planner's `session_update` ("let's focus on sales") | overwritten; cleared by a new thread or "reset" | `planner` (context); `retrieve_schema` as a **metadata filter** | "Session context" (~50) |
| **User profile** | how this *person* likes answers: fixed fields (`answer_length`, `currency`, `units`, `date_format`, `preferred_domain`) + `extra` keyed preferences (e.g. `code_language`) | LangGraph `AsyncPostgresStore`, `("profile", user_id)`, key `prefs` (no vector search) | `update_profile` branch: explicit lasting statements, UI settings, confirmed questions, or evidence-backed inference (see "over time") | by key; explicit beats inferred; `history` keeps the last 5 values; sidebar view and reset | **`load_profile`**, in full, every turn, **before `check_cache`** (the cache key needs `profile_hash`) | "User preferences" in `planner` and `finalize` (~100) |
| **Long-term semantic (facts)** | durable facts about the user ("manages the APAC region") | Pinecone ns `user_facts`, metadata below | `write_facts`: 0–3 facts per turn, extracted **only from the user's own statements**, never from tool output | attribute-keyed dedupe and contradiction handling; supersede, not overwrite; expiry of temporary facts; consolidation; cap of 200 per user | `load_context` on a cache miss: top 3, time-weighted, similarity ≥ 0.75; bumps `last_used_at` | "Known facts about the user", **dated**, in `planner` and `finalize` (~150) |
| **Long-term episodic: summaries** | what *happened* in a past thread: goal, findings, outcome, date | Pinecone ns `episodes`, `kind=summary`, id = `thread_id` | `write_episode`: every 5 turns and when the thread ends; upserted, one per thread | 90-day retention or a cap of 50 per user; recency-weighted | `load_context`: top 2, similarity ≥ 0.70 | "Relevant past interactions" in `planner` only (~200) |
| **Long-term episodic: SQL pairs** | standalone question → validated SQL, tables, domain | Pinecone ns `episodes`, `kind=sql_pair`, **`scope=global`** (shared across users) | `write_sql_pair`: only after a successful run that HITL didn't reject; **skipped if the SQL touches PII columns** | dedupe by similarity ≥ 0.95 plus a sqlglot-normalised SQL fingerprint; `schema_version`, so pairs are dropped when the schema changes; a thumbs-down deletes | SQL Agent `retrieve_schema`: hybrid search, then RRF with the seed few-shots, top 3 | "Examples" in `generate_sql` (~600) |
| **Semantic cache** | finished answers | Pinecone ns `semantic_cache`: `route`, `entity_key`, `profile_hash`, `data_version`, `expires_at` (epoch int, so `$gt` works), `prompt_version` | `write_cache`: only for complete, successful, standalone answers that weren't direct chit-chat or HITL | TTL (web ≈ 24h, DB until `data_version` changes); expired entries swept on read; a thumbs-down deletes | `check_cache`: exact-filter match (incl. `prompt_version`) **plus** similarity ≥ the calibrated threshold | **never injected**: the answer is returned as-is |
| *Seed knowledge* (not memory, but often confused with it) | schema docs, seed few-shots | Pinecone ns `schema_docs`, `sql_seed_examples` | offline by `build_schema_index.py` | rebuilt when the YAML changes | DB MCP tools `search_schema` / `get_few_shots` (hybrid) | SQL prompt (~1k) |

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

**Tests (Phase H):**
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
7. **SQL Agent `retrieve_schema`** reads the seed few-shots (hybrid, through MCP) and the learned SQL pairs (hybrid + second RRF), filtered by `selected_domain`.
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
1. **Similarity ≥ threshold** on the *standalone* question, calibrated in Phase H.
2. **An exact `entity_key` match.** `contextualize` extracts the literals that change the answer: IDs, numbers, years and dates, quoted strings, and named entities such as products, customers and countries. It normalises and sorts them into a key like `product_id=100`. The key is applied as an **exact metadata filter** in the Pinecone query, so different IDs can never match.
3. **The same `profile_hash`**, so a user who wants EUR never gets a USD-formatted answer cached for someone else. `load_profile` therefore runs before `check_cache`, and a profile change automatically invalidates that user's old entries.
4. **Freshness:** `expires_at > now` for web answers, stored as an epoch int so Pinecone's `$gt` works, and `data_version` equal to the current value for DB answers.
5. **The same `prompt_version`** of the answer-writing prompts, so a promoted prompt in Langfuse never serves answers produced by the old one (and eval runs stay comparable, see Phase K).

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


---

## CS3 semantic-memory checklist (where each §3.2 requirement lives)

| CS3 §3.2 requirement | Where it's met |
|---|---|
| What to store as semantic memory | Routing rule: preference → profile, fact → `user_facts`, one-off → session only. Validated SQL pairs and business definitions are reusable knowledge. |
| Embeddings in a vector store | Nemotron embeddings, Pinecone namespaces `user_facts`, `episodes` |
| Retrieve by similarity before responding | `load_context` (top 3 facts, top 2 episodes, thresholds), before `planner` |
| Inject only relevant memory | Thresholds, prompt-slot budgets, and agents receive only the top-k in `TaskRequest.context` |
| Duplicates, conflicts, outdated facts, updates | Attribute-keyed re-statement and contradiction (supersede), temporary `valid_until`, freshness ranking, "(may be outdated)" tags, consolidation |
| Later conversation reuses earlier info | Phase H gate: a new thread with the same `user_id` |

**Semantic memory vs everything else (report table):**
- **Semantic memory** is distilled, durable *knowledge*, retrieved by meaning.
- **Short-term state** is *this thread's* messages, plan and session fields. It's restored by thread id.
- **Semantic cache** holds *finished answers* returned verbatim, under strict filters. It's never injected as context.
- **Raw transcripts** are the full checkpointed history. They're never retrieved by similarity, only summarised into episodes.

---

## Gotchas (each is revisited in its phase)
- **G1 Structured output:** use `with_structured_output(..., method="json_schema")`. Make sure reasoning text never leaks into parsed output, and re-test whenever `CHAT_MODEL` changes.
- **G2 MCP client is beta:** `langchain.mcp` emits a beta warning, so pin its version. The fallback is the archived `langchain-mcp-adapters` (`MultiServerMCPClient`).
- **G3 One session per tool call:** opening the adapter per call pays the connection cost every time. Hold it open for the process's lifetime.
- **G4 HITL location:** `interrupt()` only works inside graph nodes, and needs a checkpointer plus a `thread_id`.
- **G5 Guardrails in two places:** in the agent *and* at the MCP server. The read-only role is the last line of defence.
- **G6 Streamlit + async:** use one background event loop (Phase J).
- **G8 Northwind dates:** the data covers 1996–98 unless shifted (`db/04_shift_dates.sql`).
- **G11 BM25 on tiny corpora:** keyword hits must not vanish on 2-doc or domain-filtered corpora. This was the reason for Lucene IDF in 2b. Pinecone now computes BM25, so Phase K's retrieval eval rechecks it on filtered queries.
- **G12 Similar embeddings, different meaning:** "code in Python" vs "code in Java", or "Nordic" vs "APAC", are near-duplicates to an embedding model. Preferences and facts are keyed, not deduped by similarity alone.
- **G13 Concurrent runs on one thread:** memory writes finish after the answer is shown, so serialize runs per `thread_id` in the UI bridge (Phase J).
- **G14 Pinecone eventual consistency:** an upsert isn't instantly queryable. Recall tests and demos poll before they assert.
- **G15 Pinecone document index:** no `null` field values. Filters use `$eq`, `$in`, `$gt` and so on. Timestamps are epoch ints. Dense scores are already cosine similarities. The schema (the `keywords` and `embedding` fields) can't change after creation: a new embedding model means a new index.
- **G16 Aiven connection cap:** the pools of the DB MCP server, the supervisor's saver and store, and the SQL agent's saver share one small limit. Keep `max_size` at 2–3 each.
- **G17 Aiven `avnadmin` isn't a superuser:** `CREATE DATABASE … OWNER agent_app` may need `GRANT agent_app TO avnadmin` first.
- **G18 OpenRouter `:free` limits:** per-minute and per-day caps on chat and embeddings. The free Nemotron endpoint also fails ("service inference failed", HTTP 502) on batches of real documents, so `EMBED_BATCH_SIZE=1`. Cache query embeddings and keep eval concurrency low. Buy credits if the evals are blocked.
- **G19 `a2a-sdk` API churn:** pin the version and check the class names against the installed release.
- **G20 Trace propagation:** without explicit `trace_ctx` in the A2A metadata, each agent's spans become separate Langfuse traces.
- **G21 Python 3.14 wheels:** check every new dependency before writing code (Phase A step 2).
- **G22 Re-run on resume:** `interrupt()` re-runs its whole node when resumed. Any side effect (an A2A send, a memory write) must sit in an earlier node.
- **G23 Eval contamination:** a shared cache and shared learned pairs across runs hide the effect of a prompt change. Isolate runs (Phase K).

## Verification (end-to-end: the CS3 minimum demos)
1. **Start:**
   - `uv run python scripts/run_all.py`: every `/health` endpoint and agent card responds;
   - then `uv run streamlit run src/multi_agenti_conversation_system/ui/streamlit_app.py`.
2. **A2A:**
   - A hybrid question: the trace shows supervisor → SQL ∥ Web → Synthesis.
   - The "our beverage sales vs the global beverage market trend" question routes `sql → synthesis`, and Synthesis delegates to Web directly. Check that it does this on every run.
3. **Semantic memory:**
   - A fact stated in thread 1 is reused in a new thread for the same user.
   - Another user doesn't see it.
   - Nordic → APAC supersedes.
4. **Langfuse prompts:**
   - The trace shows the prompt name, version and label.
   - Edit the planner prompt in Langfuse, promote it, and behaviour changes with no code change.
   - A bad Langfuse key falls back to the local YAML.
5. **Evals:** `uv run python eval/run_eval.py` produces two comparable Langfuse runs, the CSV, and the improvement write-up.
6. **Controlled failure:**
   - With the Web Agent stopped, the answer is partial but graceful.
   - A DROP attempt is blocked.
   - A PII query triggers HITL.
   - A prompt-injection input is refused.
   - A bad OpenRouter key degrades retrieval to BM25 (`degraded: true`). A bad Pinecone key gives `ok: false`, and SQL still runs.
7. **Tests:** `uv run pytest` (unit tests by default), plus `-m integration` against Aiven + Pinecone.

## Report notes (collect as we go)
- **MCP vs A2A:** CS2 §5.4 says every tool call goes through MCP. Here, agents call their tools through MCP, while the supervisor reaches agents through A2A. CS3 §4 step 5 sanctions this: "agents collaborate through the A2A mechanism and use existing MCP tools".
- **What is simulated:**
  - the A2A task store is in memory;
  - Northwind stands in for real business data;
  - free-tier cloud services.
- **Limitations:**
  - the trace mask is regex-based: it misses names, addresses and bare digit runs;
  - model quality;
  - Northwind's small schema;
  - retrieval depends on a single provider (Pinecone serves both BM25 and dense search);
  - Tavily's coverage;
  - cache staleness;
  - residual DNS-rebinding risk;
  - memory extraction errors;
  - cloud and network dependency;
  - free-tier rate limits.
