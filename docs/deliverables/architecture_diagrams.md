# Architecture and flow diagrams

Three diagrams: the system, one turn through the supervisor, and the evaluation flow. They are
Mermaid, so they render on GitHub and in VS Code's Markdown preview. For a PDF or slide image,
paste a block into <https://mermaid.live> and export, or print this page to PDF from the
preview. The longer design (state, contracts, memory, guard layers) is in
[docs/architecture/hld_lld.md](../architecture/hld_lld.md).

## 1. System architecture

Solid lines are calls; dotted lines are storage and shared services. MCP is how an agent
reaches a tool; A2A is how an agent reaches another agent.

```mermaid
flowchart TB
    user([User]) --> ui["Streamlit UI :8501"]
    ui <--> sup["Supervisor: LangGraph, in-process with the UI<br/>guards, plans, dispatches, owns memory"]

    subgraph agents["A2A agent services (own process, LangGraph inside, agent card + contracts)"]
        direction LR
        sqlA["SQL Agent :7001"]
        synA["Synthesis Agent :7003"]
        webA["Web Agent :7002"]
    end

    sup -->|"A2A task"| sqlA
    sup -->|"A2A task"| synA
    sup -->|"A2A task"| webA
    synA -->|"A2A peer delegation"| webA

    subgraph tools["MCP tool servers (Streamable HTTP + bearer token)"]
        direction LR
        dbm["DB MCP :5555<br/>schema search, few-shots, SELECT"]
        webm["Web MCP :5556<br/>search_web, fetch_page"]
    end

    sqlA -->|"MCP"| dbm
    webA -->|"MCP"| webm

    subgraph data["Managed cloud stores"]
        direction LR
        nw[("Aiven Postgres<br/>northwind, read-only role")]
        pc[("Pinecone<br/>schema docs, SQL pairs,<br/>facts, episodes, cache")]
        mem[("Aiven Postgres<br/>checkpoints + profile")]
        tv["Tavily<br/>search + extract"]
    end

    dbm --> nw
    dbm --> pc
    webm --> tv
    sup -.->|"checkpoints, profile"| mem
    sup -.->|"facts, episodes, cache"| pc
    sqlA -.->|"checkpoints"| mem
    sqlA -.->|"learned SQL pairs"| pc

    subgraph shared["Used by every service"]
        direction LR
        orr["OpenRouter<br/>chat, judge, embeddings"]
        lf["Langfuse<br/>prompts, traces, experiments"]
    end

    sup -.-> shared
    agents -.-> shared
```

## 2. One turn through the supervisor

```mermaid
flowchart LR
    q([User message]) --> ig{input guard}
    ig -- blocked --> ref[refuse]
    ig -- ok --> ctx[contextualize] --> prof[load profile] --> cache{semantic cache}
    cache -- "hit: same entity, profile,<br/>prompt and data version" --> wm
    cache -- miss --> rec[recall facts + episodes] --> plan[planner]
    plan -- direct --> da[direct answer] --> wm
    plan -- "plan validated in code" --> disp[dispatch steps]
    disp -- "A2A, parallel Send" --> ag["SQL / Web / Synthesis agents"]
    ag -- "PII columns or large result" --> hitl{{"human approval<br/>interrupt()"}} --> disp
    ag --> fin["finalize + output guard"]
    fin --> wm["write memory<br/>(write-gate: facts, cache, episode)"]
    wm --> sum[summarize history] --> a([Answer + sources + trace id])
    ref --> a
```

## 3. Evaluation flow

```mermaid
flowchart LR
    yaml["dataset.yaml<br/>39 cases"] --> sync["sync_eval_dataset.py"] --> ds[("Langfuse dataset")]
    ds --> run["run_eval.py<br/>Langfuse experiment, one case at a time"]
    run --> stack["throwaway stack on spare ports<br/>DB MCP, Web MCP, 3 agents, EVAL_MODE"]
    stack --> turn["supervisor turn per case<br/>(fresh user + thread, traced)"]
    turn --> det["deterministic scorers<br/>route, agents, contract, SQL result vs gold,<br/>citations, guardrails, cache, latency"]
    turn --> judge["LLM judge<br/>relevance, correctness, groundedness,<br/>completeness, clarity"]
    det --> scores[("scores on each trace")]
    judge --> scores
    scores --> out["results/run.csv + run.md<br/>compare two runs in Langfuse"]
```
