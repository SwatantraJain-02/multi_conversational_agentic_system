# Evaluation run `planner-v3`

- prompt label: `development`
- cases: 39
- Langfuse run: https://jp.cloud.langfuse.com/project/cmw4wx8gc001cad0ekx9c90xe/experiments/results?baseline=71f7ef51ecbf93f7

## Mean score per metric (over the cases it applies to)

| metric | n | mean |
|---|---|---|
| route_correct | 31 | 1.00 |
| agents_correct | 31 | 1.00 |
| sql_executes | 25 | 1.00 |
| cache_behaviour | 2 | 1.00 |
| answered | 39 | 1.00 |
| no_secret_leak | 39 | 1.00 |
| latency_ms | 39 | 41839.44 |
| judge_relevance | 39 | 4.46 |
| judge_correctness | 39 | 3.49 |
| judge_groundedness | 39 | 3.23 |
| judge_completeness | 39 | 4.44 |
| judge_clarity | 39 | 4.90 |
| a2a_schema_valid | 29 | 1.00 |
| result_matches_gold | 16 | 0.94 |
| required_facts | 3 | 1.00 |
| no_agents_called | 1 | 1.00 |
| approval_requested | 1 | 1.00 |
| refused | 2 | 1.00 |
| guardrail_blocked | 2 | 1.00 |
| citations_present | 7 | 1.00 |

## By category (mean of the 0/1 checks)

| category | cases | checks passed |
|---|---|---|
| cache | 2 | 1.00 |
| memory | 3 | 1.00 |
| failure | 7 | 1.00 |
| ambiguous | 5 | 1.00 |
| hybrid | 6 | 1.00 |
| complex | 6 | 1.00 |
| normal | 10 | 0.99 |

## Failed checks

- `n07`: result_matches_gold
