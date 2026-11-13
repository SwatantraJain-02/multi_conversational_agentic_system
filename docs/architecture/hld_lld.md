# HLD and LLD: Multi-Agent Conversational Assistant (CS2 + CS3)

Diagrams for understanding the system. **The plan (`docs/plan/implementation_plan.md`) is the source of truth.** If the two disagree, the plan wins and this file gets updated.

Legend: green = built and tested, yellow = planned (phase in brackets).

---

# Part 1: High-level design (HLD)

## 1.1 System architecture

What runs where, and how the pieces talk.

```mermaid
flowchart TB
    user([User]) --> ui["Streamlit UI :8501 [J]"]
    ui <--> sup["Supervisor: LangGraph, in-process with the UI [G, H]<br/>plans, dispatches, owns memory"]

    subgraph agents["A2A agent services (each its own process, LangGraph inside)"]
        direction LR
        sqlA["SQL Agent :7001 [D]"]
        synA["Synthesis Agent :7003 [F]"]
        webA["Web Agent :7002 [E]"]
    end

    sup -->|"A2A"| sqlA
    sup -->|"A2A"| synA
    sup -->|"A2A"| webA
    synA -->|"A2A peer delegation"| webA

    subgraph tools["MCP tool servers (Streamable HTTP + bearer token)"]
        direction LR
        dbm["DB MCP :5555<br/>schema search, few-shots, SQL"]
        webm["Web MCP :5556<br/>search_web, fetch_page"]
    end

    sqlA -->|"MCP"| dbm
    webA -->|"MCP"| webm

    subgraph data["Data stores (managed cloud)"]
        direction LR
        nw[("Aiven Postgres: northwind<br/>business data, read-only")]
        pc[("Pinecone 'capstone'<br/>BM25 + dense, 5 namespaces")]
        mem[("Aiven Postgres: agent_memory<br/>checkpoints + profile")]
        tv["Tavily<br/>search + extract"]
    end

    dbm --> nw
    dbm --> pc
    webm --> tv

    subgraph shared["Used by every service (managed cloud)"]
        direction LR
        orr["OpenRouter: chat, judge, embeddings"]
        lf["Langfuse: prompts, traces, evals"]
    end

    sup -.->|"checkpoints, profile"| mem
    sup -.->|"facts, episodes, cache"| pc
    sqlA -.->|"checkpoints"| mem
    sqlA -.->|"learned SQL pairs"| pc

    classDef built fill:#d4edda,stroke:#2e7d32,color:#000
    classDef planned fill:#fff3cd,stroke:#b8860b,color:#000
    class dbm,webm built
    class ui,sup,sqlA,webA,synA planned
```

Not drawn, to keep it readable: every service calls OpenRouter (LLM, embeddings) and Langfuse (prompts, traces). Dotted lines are memory; solid lines are calls.

**The two protocols, in one line each:**
- **A2A** is how *agents talk to agents*: the supervisor delegates tasks, and Synthesis delegates to Web.
- **MCP** is how *agents use tools*: databases and the web are only reached through MCP servers.

## 1.2 Storage split

| Store | Holds | Who writes | Who reads |
|---|---|---|---|
| Aiven `northwind` | business data (orders, products, customers…) | nobody (read-only) | DB MCP server, as `agent_readonly` |
| Aiven `agent_memory` | conversation checkpoints, user profile | supervisor, SQL agent (`agent_app`) | same |
| Pinecone `schema_docs`, `sql_seed_examples` | seed knowledge (not memory) | `build_schema_index.py`, offline | DB MCP server |
| Pinecone `user_facts`, `episodes`, `semantic_cache` | long-term memory and cache | supervisor (facts, episodes, cache), SQL agent (SQL pairs) | same |
| Langfuse | prompts, traces, datasets, scores | `sync_prompts.py`, every service, evals | every service (prompts), you (UI) |

## 1.3 One hybrid question, end to end

"How did our beverage sales do last quarter compared with the market trend?"

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as Streamlit UI
    participant S as Supervisor
    participant SQL as SQL Agent
    participant W as Web Agent
    participant Y as Synthesis Agent
    participant DB as DB MCP
    participant WM as Web MCP
    participant LF as Langfuse

    U->>UI: question
    UI->>S: invoke(thread_id, user_id)
    Note over S: input guard, contextualize,<br/>load profile, cache check (miss),<br/>load facts + episodes
    S->>S: planner gives plan [sql || web] then synthesis
    par parallel A2A tasks
        S->>SQL: TaskRequest (skill answer_sql_question)
        SQL->>DB: search_schema, get_few_shots, run_select_query
        DB-->>SQL: rows
        SQL-->>S: TaskResult (answer, data, sql)
    and
        S->>W: TaskRequest (skill answer_web_question)
        W->>WM: search_web, fetch_page
        WM-->>W: pages
        W-->>S: TaskResult (answer, citations)
    end
    S->>Y: TaskRequest with both results in context
    Y-->>S: TaskResult (combined, grounded answer)
    Note over S: finalize (profile formatting), output guard
    S-->>UI: answer + citations + SQL + agents used
    Note over S: after responding: write facts, profile,<br/>episode, cache (parallel)
    S-)LF: one trace: all agent spans nested under it
```

## 1.4 What each CS3 requirement maps to

| CS3 requirement | Where it lives |
|---|---|
| Real A2A collaboration | supervisor → agents, plus Synthesis → Web peer delegation (§2.3–2.6) |
| Semantic memory lifecycle | facts with supersede, expiry, consolidation; cache with entity keys (§2.8) |
| Prompt-as-a-service | Langfuse prompts with labels, local YAML fallback (§2.10) |
| Evaluation framework | Langfuse datasets + experiments, deterministic scorers + LLM judge (§2.11) |
| Guardrails (CS2 + CS3) | seven layers along the request path (§2.9) |

---

# Part 2: Low-level design (LLD)

## 2.1 Supervisor graph [G, H, I] (built)

```mermaid
flowchart TD
    S([START]) --> ig["input_guard<br/>limits, heuristics, classifier, PII flag"]
    ig -->|"blocked"| rf["refuse<br/>(the exchange leaves the conversation)"]
    rf --> E([END])
    ig --> ctx["contextualize<br/>standalone question + entity_key<br/>(no model call on a first turn)"]
    ctx --> lp["load_profile<br/>profile + profile_hash"]
    lp --> cc{"check_cache"}
    cc -->|"hit"| wm
    cc -->|"miss"| lc["load_context<br/>top 3 facts, top 2 episodes"]
    lc --> pl["planner (LLM)<br/>RouteDecision, one repair attempt"]
    pl --> sv{"supervisor<br/>deterministic"}
    sv -->|"steps ready (Send, parallel)"| rs["run_step<br/>one A2A task per step"]
    rs --> sv
    sv -->|"a step is input-required"| hg["hitl_gate<br/>interrupt(payload)"]
    hg -->|"user approves / rejects"| rd["resume_dispatch<br/>same A2A task"]
    rd --> sv
    sv -->|"plan done / budget of 5 dispatches"| fin["finalize<br/>composed in code, gaps named,<br/>output guard applied"]
    sv -->|"direct"| da["direct_answer<br/>output guard applied"]
    fin & da --> wm["write_memory<br/>learn (write-gate) + cache + episode,<br/>each part isolated"]
    wm --> sh["summarize_history<br/>only past threshold"]
    sh --> E
```

There is **no re-planning after a failed step** (decided in Phase G): the gap is named in a partial answer instead. The output guard is not a node of its own: it runs inside the function that builds every answer, so the text shown, checkpointed and cached is the guarded one.

**Why three HITL nodes:** when a run resumes after `interrupt()`, LangGraph re-runs that whole node. If `dispatch` also interrupted, resuming would send the SQL task a second time (G22).

**Plan shapes the planner can return:**

| Route | Plan |
|---|---|
| web | `[web]` |
| sql | `[sql]` |
| hybrid | `[sql ∥ web] → synthesis` |
| direct (chit-chat) | `[]` |
| internal vs market | `sql → synthesis`, and Synthesis fetches web data itself (the delegation demo) |

## 2.2 Supervisor state [G, H, I] (built, `supervisor/state.py`)

```mermaid
classDiagram
    class AgentState {
        +list messages  (kept across turns)
        +str summary  (kept across turns)
        +int turns  (kept across turns)
        -- this turn: input --
        +str question
        +str user_id
        +str thread_id
        +str selected_domain
        -- input guard --
        +bool guard_blocked
        +bool guard_flagged
        +list guard_reasons
        +list input_pii
        -- memory --
        +str standalone_question
        +str entity_key
        +bool standalone_ok
        +bool memory_ok
        +dict profile
        +str profile_hash
        +list facts
        +list episodes
        +bool cache_hit
        +list memory_changes
        -- plan and results --
        +str intent
        +list plan
        +str plan_error
        +list ready
        +int dispatch_count
        +dict results  (merge reducer)
        +dict timings  (merge reducer)
        +dict hitl_decision
        +list approvals
        -- answer --
        +str answer
        +list citations
        +str sql_used
        +list agents_used
        +dict prompt_versions
        +bool partial
    }
    class RouteDecision {
        +str intent
        +list~PlanStep~ plan
        +str reason
    }
    class PlanStep {
        +str id
        +str agent
        +str skill
        +str task
        +list~str~ depends_on
    }
    AgentState --> RouteDecision
    RouteDecision --> PlanStep
```

`graph.turn_input` resets everything marked "this turn" on each new turn (the accumulating fields with `Overwrite`); `messages`, `summary` and `turns` are restored from the checkpoint by the thread id.

## 2.3 A2A contract (built)

`a2a/contracts.py`. Each model travels as ONE A2A `Part` holding its JSON text (media type `application/json`), validated by Pydantic on **both** ends. Not a `DataPart`: that is a protobuf `Value`, which turns every number into a float (`product_id` 100 becomes 100.0). `trace_ctx` is not in these models: it travels in the A2A message metadata, so a resume keeps the trace. The card publishes the JSON schemas in an A2A extension.

The diagram below predates the final model; `a2a/contracts.py` is authoritative (it adds `ResumeRequest`, `TaskRequest.data`, `TaskResult.a2a_task_id` / `context_id`, and the error codes `bad_request` and `internal`).

```mermaid
classDiagram
    class TaskRequest {
        +str task_id
        +str parent_task_id
        +dict trace_ctx
        +str sender
        +str skill
        +str task
        +TaskContext context
        +Constraints constraints
    }
    class TaskContext {
        +dict profile
        +list memories
        +dict session
    }
    class Constraints {
        +int timeout_s
        +int max_rows
    }
    class TaskResult {
        +str status
        +str answer
        +dict data
        +list citations
        +str sql
        +TaskError error
        +str agent
        +dict prompt_versions
    }
    class TaskError {
        +str code
        +str message
        +bool retryable
    }
    TaskRequest --> TaskContext
    TaskRequest --> Constraints
    TaskResult --> TaskError
```

`TaskError.code` is one of `timeout`, `unavailable`, `invalid_output`, `failed`, `blocked`, `rejected`.

**A2A task states, as our agents use them:**

```mermaid
stateDiagram-v2
    state "input-required" as IR
    [*] --> submitted
    submitted --> working
    working --> working : node progress updates
    working --> completed : TaskResult ok
    working --> failed : error inside the agent
    working --> IR : SQL needs approval (HITL)
    IR --> working : approve sent on same contextId
    IR --> completed : reject (status rejected)
    completed --> [*]
    failed --> [*]
```

## 2.4 A2A plumbing: discovery and failure handling (built)

```mermaid
flowchart TD
    subgraph startup["At startup"]
        cfg["AGENT_URLS in config"] --> fetch["GET /.well-known/agent-card.json"]
        fetch --> reg["Registry: name → card, skills, status"]
        reg --> caps["Planner capability list<br/>rendered from the cards"]
    end

    subgraph call["Each call: a2a/client.py"]
        send["send TaskRequest"] --> wait{"wait_for(timeout_s)"}
        wait -->|"timed out"| et["error timeout"]
        wait -->|"connection refused"| eu["error unavailable<br/>mark agent down"]
        wait -->|"reply"| val{"Pydantic: valid TaskResult?"}
        val -->|"no, first time"| send
        val -->|"no, second time"| ei["error invalid_output"]
        val -->|"yes, state failed"| ef["error failed"]
        val -->|"yes"| ok["TaskResult"]
    end

    et & eu & ei & ef --> sup["Supervisor: re-plan,<br/>or partial answer naming what is missing"]
```

The server side (`a2a/server.py`) is one generic executor. It wraps any compiled LangGraph graph, turns node progress into `working` updates, and turns an `interrupt()` into `input-required`. Every agent reuses it.

## 2.5 HITL across A2A [D, G]

```mermaid
sequenceDiagram
    actor U as User
    participant S as Supervisor
    participant SQL as SQL Agent
    participant CP as SQL checkpointer (Aiven)

    S->>SQL: dispatch: TaskRequest
    SQL->>SQL: generate + validate SQL
    SQL->>CP: save graph state (thread_id = contextId)
    SQL-->>S: state input-required {sql, reason, est_rows, pii_cols}
    Note over S: dispatch stores pending_hitl in state
    S->>S: hitl_gate: interrupt(payload)
    S-->>U: Approve / Reject buttons
    U->>S: Command(resume = approve)
    S->>SQL: resume_dispatch: decision on same contextId
    SQL->>CP: load graph state
    SQL->>SQL: execute, explain
    SQL-->>S: completed: TaskResult
```

**Known limitation:** the A2A task store is in memory. If the SQL agent restarts mid-approval, the task is lost, even though its LangGraph checkpoint survives.

## 2.6 Agent graphs [D, E, F]

### SQL Agent :7001, skill `answer_sql_question`

```mermaid
flowchart TD
    S([START]) --> rs["retrieve_schema<br/>MCP search_schema + get_few_shots<br/>+ learned pairs → second RRF"]
    rs --> gs["generate_sql (LLM)<br/>SQLGeneration{sql, explanation, tables_used}"]
    gs --> va{"validate<br/>(sqlglot validator)"}
    va -->|"write / DDL / forbidden table"| bl["TaskResult: blocked"]
    va -->|"fixable error, retries left"| gs
    va -->|"ok"| ag{"approval_gate<br/>MCP explain_query"}
    ag -->|"too many rows or PII"| ir["input-required<br/>(interrupt)"]
    ir -->|"rejected"| rj["TaskResult: rejected"]
    ir -->|"approved"| ex
    ag -->|"fine"| ex["execute<br/>MCP run_select_query"]
    ex -->|"DB error, at most 2 retries"| gs
    ex --> er["explain_result (LLM)"]
    er --> wp["write_sql_pair<br/>skipped for PII, HITL-rejected, EVAL_MODE"]
    wp --> E([END])
    bl & rj --> E
```

### Web Agent :7002, skill `answer_web_question`

```mermaid
flowchart TD
    S([START]) --> pq["plan_queries (LLM)"]
    pq --> se["search (Send per query)<br/>MCP search_web"]
    se --> md["merge_dedupe"]
    md --> fe["fetch (Send per URL)<br/>MCP fetch_page"]
    fe --> ch["chunk<br/>~800 tokens, 100 overlap"]
    ch --> rr["rerank<br/>flashrank, keep top 6"]
    rr --> co["compress<br/>~3k-token budget"]
    co --> as{"assess<br/>(Should tier)"}
    as -->|"gaps, rounds < 2"| pq
    as -->|"enough"| su["summarise with [n] citations<br/>strip [n] with no fetched source"]
    fe -->|"every fetch failed"| su
    su --> E([END])
```

If every fetch fails, the agent answers from the search snippets and flags low confidence.

### Synthesis Agent :7003, skill `synthesise`

```mermaid
flowchart TD
    S([START]) --> ci{"check_inputs<br/>external data present?"}
    ci -->|"yes"| sy["synthesise (LLM)<br/>keep both sets of citations + SQL"]
    ci -->|"no (always, deterministic)"| dw["delegate_web<br/>discover Web card, call over A2A"]
    dw --> sy
    sy --> E([END])
```

The delegation rule is deterministic so the demo takes the same path every run. An LLM-decided trigger is too unreliable live.

## 2.7 MCP layer: tools, auth and guards (built)

```mermaid
flowchart LR
    subgraph client["mcp/client.py (built)"]
        om["open_mcp(['db','web'])<br/>one MCPAdapter per server<br/>held open for the caller's lifetime"]
    end

    subgraph dbs["DB MCP :5555 (built)"]
        a1["BearerAuth<br/>(/health exempt)"]
        t1["list_tables, get_schema<br/>search_schema, get_few_shots<br/>explain_query, run_select_query"]
        g1["sql_validator: one SELECT,<br/>no DDL/DML, LIMIT clamp,<br/>table allowlist, PII columns"]
        r1["role agent_readonly<br/>statement_timeout 5s"]
        a1 --> t1 --> g1 --> r1
    end

    subgraph wbs["Web MCP :5556 (built)"]
        a2["BearerAuth"]
        t2["search_web, fetch_page"]
        g2["token-bucket rate limits<br/>SSRF guard: public IPs only,<br/>each redirect re-checked (max 3),<br/>10s timeout, 2MB cap"]
        x2["trafilatura,<br/>Tavily Extract fallback"]
        a2 --> t2 --> g2 --> x2
    end

    om -->|"Authorization: Bearer DB_MCP_TOKEN"| a1
    om -->|"Authorization: Bearer WEB_MCP_TOKEN"| a2
    r1 --> nw[("northwind")]
    t1 --> pc[("Pinecone")]
    x2 --> tv["Tavily"]

    classDef built fill:#d4edda,stroke:#2e7d32,color:#000
    class om,a1,t1,g1,r1,a2,t2,g2,x2 built
```

Tools never raise across the boundary. They return `{"ok": false, "error": "..."}`, so the agent can read the message and adapt.

## 2.8 Hybrid retrieval (built)

Used by `search_schema` and `get_few_shots`, and reused for learned SQL pairs.

```mermaid
flowchart TD
    q["query"] --> tok["tokenize()<br/>unit_price → unit, price"]
    q --> emb["embed with 'query: ' prefix<br/>Nemotron, 4096 dims, LRU cache"]
    tok --> kw["Pinecone BM25 search<br/>on field 'keywords'"]
    emb --> dn["Pinecone dense search<br/>on field 'embedding' (cosine)"]
    kw --> rrf["RRF fusion, k = 60<br/>(client-side)"]
    dn --> rrf
    rrf --> res["top-k hits"]
    emb -.->|"embedding API fails"| deg["BM25 only, degraded: true"]
    kw -.->|"Pinecone down"| err["ok: false"]

    classDef built fill:#d4edda,stroke:#2e7d32,color:#000
    class tok,emb,kw,dn,rrf,res built
```

The two searches run in parallel.

## 2.9 Memory [G, H]

### Reads and writes during one turn

```mermaid
flowchart LR
    subgraph reads["Read (before answering)"]
        r0["checkpointer: messages, summary, session<br/>(automatic, by thread_id)"]
        r1["load_profile → Postgres store ('profile', user_id)"]
        r2["check_cache → Pinecone semantic_cache"]
        r3["load_context → Pinecone user_facts (top 3)<br/>+ episodes kind=summary (top 2)"]
        r4["SQL agent retrieve_schema →<br/>schema_docs, sql_seed_examples,<br/>episodes kind=sql_pair"]
        r0 --> r1 --> r2 --> r3 --> r4
    end
    subgraph writes["Write (after answering, in parallel)"]
        w1["write_facts → user_facts<br/>0-3 facts, user statements only"]
        w2["update_profile → Postgres store<br/>keyed, explicit beats inferred"]
        w3["write_episode → episodes<br/>every 5 turns + thread end"]
        w4["write_cache → semantic_cache<br/>complete, standalone answers only"]
        w5["write_sql_pair (SQL agent) → episodes<br/>no PII, not rejected"]
    end
    reads --> ans(["answer shown"]) --> writes
```

**Where a statement goes:** a preference ("always short answers") goes to the profile, a fact ("I manage APAC") to `user_facts`, and a one-off ("this once in USD") to the session only.

### Fact lifecycle (supersede, never overwrite)

```mermaid
stateDiagram-v2
    [*] --> active : new fact extracted
    active --> active : re-statement (same attribute + value), bump confidence
    active --> superseded : contradiction (same attribute, new value), new fact linked
    active --> archived : temporary fact expired / unused 365 days / over 200 cap
    active --> [*] : user says "forget X" (hard delete)
    superseded --> archived : consolidation
```

Only `active` facts are retrieved. They're ranked by `similarity × (0.7 + 0.3 × freshness)` and injected with their date, e.g. "Manages APAC region (stated 2026-10-02)".

### Semantic cache hit test (the `product_id` 100 vs 150 problem)

```mermaid
flowchart TD
    q["standalone question + entity_key<br/>e.g. 150"] --> f["Pinecone query on semantic_cache, exact filters<br/>(none may be missing: a None is an error):<br/>entity_key, profile_hash, prompt_version,<br/>data_version, expires_at > now"]
    f --> s{"best similarity ≥ CACHE_THRESHOLD (0.87, calibrated)?"}
    s -->|"yes"| hit["HIT: return cached answer as-is"]
    s -->|"no"| miss["MISS: run full pipeline"]
    f -->|"no candidates (e.g. entry is for product_id=100)"| miss
    miss --> el{"complete, standalone,<br/>not HITL, not chit-chat?"}
    el -->|"yes"| wr["write_cache"]
```

Similarity alone would return product 100's answer for product 150. The exact `entity_key` filter makes that impossible.

## 2.10 Guardrail layers along the request path [I] (all built)

```mermaid
flowchart LR
    in["User input"] --> g1["1 Input guard<br/>length, injection/jailbreak<br/>heuristics + LLM classifier, PII"]
    g1 --> g2["2 A2A boundary<br/>Pydantic validation, size caps"]
    g2 --> g3["3 MCP tools (built)<br/>bearer auth, SQL validator,<br/>read-only role, SSRF, rate limits"]
    g3 --> g4["4 HITL<br/>large or PII queries need approval"]
    g4 --> g5["5 Memory write-gate<br/>PII check, 'stable fact?' check"]
    g5 --> g6["6 Output guard<br/>PII redaction,<br/>strip unsupported citations"]
    g6 --> out["Answer"]
    g7["7 Trace masking<br/>Langfuse mask hook:<br/>PII + secrets"] -.-> g1 & g2 & g3 & g4 & g5 & g6

    classDef built fill:#d4edda,stroke:#2e7d32,color:#000
    class g1,g2,g3,g4,g5,g6,g7 built
```

## 2.11 Prompts and tracing through Langfuse (built)

### Prompt lifecycle

```mermaid
flowchart LR
    y["prompts/area/name.yaml<br/>(in git)"] -->|"sync_prompts.py"| dev["Langfuse prompt<br/>label: development"]
    dev -->|"manual promotion in Langfuse UI"| prod["label: production"]
    subgraph runtime["At runtime: prompts.get_prompt(name)"]
        g["Langfuse get_prompt(name, label = PROMPT_LABEL)<br/>cached for cache_ttl"]
        fb["local YAML fallback<br/>(Langfuse down or bad key)"]
        g -.->|"fails"| fb
    end
    prod --> g
    dev --> g
    g --> out["(template, version, label, source)<br/>version recorded in state,<br/>linked to the LLM generation"]
    fb --> out
```

Changing behaviour means editing the prompt in Langfuse and promoting it. No code change or redeploy.

### Trace propagation across services

```mermaid
sequenceDiagram
    participant S as Supervisor
    participant A as Any agent
    participant M as MCP server
    participant LF as Langfuse

    S->>LF: start root trace for the turn
    S->>A: A2A message, metadata.trace_ctx = trace id + parent span
    A->>LF: agent spans, parented to the supervisor's span
    A->>M: MCP tool call
    A->>LF: tool span + LLM generations (with prompt version)
    Note over LF: one trace per turn, all services nested (G20)
```

## 2.12 Evaluation pipeline [K] (built, not yet run)

```mermaid
flowchart TD
    ds["evals/dataset.yaml (39 cases)<br/>normal, complex, hybrid, ambiguous,<br/>failure, memory reuse, cache near-miss"] -->|"sync_eval_dataset.py"| lds["Langfuse dataset"]
    lds --> run["run_eval.py --label development | production"]
    run --> iso["Run isolation per run:<br/>services on spare ports with EVAL_MODE (no learned pairs),<br/>fresh user + thread ids per case,<br/>what a case wrote is deleted after it"]
    iso --> sys["Supervisor + agents<br/>(each item's turn nests under its experiment trace)"]
    sys --> det["Deterministic scorers:<br/>route, agents, A2A schema valid,<br/>SQL runs, result matches gold,<br/>citations, guardrail blocked, latency"]
    sys --> jdg["LLM judge (JUDGE_MODEL, different model)<br/>relevance, correctness, groundedness,<br/>completeness, clarity: 1-5 + reason"]
    det & jdg --> sc["Scores in Langfuse<br/>(linked to traces + prompt versions)"]
    sc --> rep["CSV + summary markdown<br/>before/after comparison, top 3 failure modes"]
```

## 2.13 Code layout and import rules

```mermaid
flowchart TD
    ui["src/ui/streamlit_app.py"] --> sup["supervisor/"]
    sup --> mem["memory/"]
    sup --> a2ac["a2a/client.py"]
    ag["agents/sql, web, synthesis"] --> a2as["a2a/server.py"]
    ag --> mcpc["mcp/client.py"]
    ag --> a2ac
    a2ac & a2as --> con["a2a/contracts.py"]
    mcpc -.->|"HTTP only"| srv["mcp/servers/db_server, web_server"]
    srv --> ret["retrieval/hybrid.py"]
    srv --> gr["guardrails/"]
    sup --> gr
    sup & ag & srv --> base["config, llm, embeddings,<br/>vectorstore, prompts, tracing"]

    sup -. "never imports" .-x ag
    ag -. "never import each other" .-x ag
```

- **The supervisor reaches agents only over A2A.** Agents reach tools only over MCP. Each server and agent is its own process.
- **Nodes get their dependencies from `Runtime[AppContext]`** (models, MCP tools, A2A registry, memory, prompts), never from globals.

## 2.14 Processes and ports

| Process | Port | Phase | Status |
|---|---|---|---|
| DB MCP server | 5555 | 2, 2b, A | built |
| Web MCP server | 5556 | B | built |
| Echo test agent (A2A) | 7009 | C | built (plumbing test only) |
| SQL Agent (A2A) | 7001 | D | built |
| Web Agent (A2A) | 7002 | E | built |
| Synthesis Agent (A2A) | 7003 | F | built |
| Supervisor (library, `supervisor/service.py`; runs in the UI process) | none | G | built |
| Streamlit UI | 8501 | J | written, not yet tried in a browser |

`scripts/run_all.py` starts everything and waits for each `/health` and agent card.
