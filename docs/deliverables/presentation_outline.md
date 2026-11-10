# Presentation outline

About 12 slides, 15 minutes plus questions. Each slide: its message, what is on it, and where
the evidence is. Numbers marked **(fill)** come from the evaluation run and are filled in
after it.

## 1. The problem
- Message: business users ask questions about company data and about the outside world, often
  in the same breath, and they expect the assistant to remember them and not to be unsafe.
- On the slide: three example questions (SQL, web, hybrid); the four demands: correct,
  grounded, remembers, safe.

## 2. What we built, in one picture
- Message: a supervisor plans, three agents answer, memory and guards wrap the turn.
- On the slide: the flow diagram from the README (supervisor, A2A agents, MCP servers,
  memory stores, Langfuse). Source: [docs/architecture/hld_lld.md](../architecture/hld_lld.md).

## 3. Three protocols, three jobs
- MCP: an agent reaches a tool (database, web search, page fetch).
- A2A: an agent reaches another agent (supervisor to agents; Synthesis to Web).
- LangGraph: the logic inside each agent and the supervisor (fan-out, `interrupt`, retries,
  checkpoints).
- Design rule to show: the supervisor never imports an agent; it only knows agent cards.

## 4. Design choices and why
- Real A2A (a2a-sdk) with typed contracts in the agent card, not function calls dressed up.
- The model proposes, code decides: plan validation, SQL AST validator, approval bound to
  the exact SQL, filters that raise instead of silently dropping a clause.
- Tools and agents never raise across a boundary; they return an error result, so a failure
  becomes a partial answer, not a crash.
- Prompts live in Langfuse (labels, versions) with a local fallback.

## 5. Memory
- Layers: conversation (checkpoint), profile (keyed), facts (one active per attribute,
  superseded not overwritten, expiry), episodes (one per thread), semantic cache.
- Cache is safe by construction: similarity is not enough; the entity, profile hash, prompt
  version and data version must also match.
- Evidence: `smoke_memory.py`, 21 checks live.

## 6. Safety in layers
- Input guard, SQL validator (in the agent and again in the DB server, plus a read-only DB
  role), human approval, A2A size caps, memory write-gate, output redaction, trace masking.
- Evidence: `smoke_guardrails.py` 9 of 9; the approval card in the demo.

## 7. Demo (live or recorded, about 5 minutes)
Order from [demo_evidence.md](demo_evidence.md): hybrid with A2A delegation, memory across a
new chat, prompt update with a new version in the trace, the approval card and a refused
injection, a cached answer and a thumbs-down.

## 8. Observability
- One Langfuse trace per turn, agents nested under it through the A2A metadata, each
  generation linked to its prompt version, PII masked before it leaves the process.
- On the slide: a trace screenshot showing the delegation edge Synthesis to Web.

## 9. How we evaluated
- 39 cases: normal 10, complex 6, hybrid 6, ambiguous 5, failure 7, memory 3, cache 2.
- Deterministic checks (route, agents, contract validity, SQL executes, result set equals
  the gold query's, citations, guardrails, cache behaviour) plus an LLM judge on five 1 to 5
  criteria. Each run is a Langfuse experiment linked to traces and prompt versions.

## 10. Results
- Table: score per metric and per category for the baseline and for the run after the
  change. **(fill)**
- Top three failure modes and the change made because of them. **(fill)**
- Note what the first baseline taught us: a rate-limited model looked like a quality
  failure, so the runner now checks the model first and flags such cases.

## 11. Limits, stated plainly
- Free chat model: slow, rate limited, a daily quota; a paid model changes only `CHAT_MODEL`.
- Northwind is a public sample dataset; one static token per service group, no real login.
- A2A task store is in memory: a pending approval does not survive an agent restart.
- Not built: inferred preference drift, confirm-once conflict flow, fact consolidation,
  fact cap, episode retention, re-planning after a failed step.

## 12. Next steps
- Durable A2A task store; per-user auth; re-planning; consolidation and retention for
  memory; per-run namespaces for evals; a larger dataset with agent-down cases in the runner.

## Before presenting
- No secrets in any screenshot (keys, tokens, connection strings, `.env`).
- Have the recorded demo as a fallback; the free model can be slow or rate limited on the day.
