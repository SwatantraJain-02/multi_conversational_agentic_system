# Evaluation run `baseline-v1`

- prompt label: `development`
- cases: 39
- Langfuse run: https://jp.cloud.langfuse.com/project/cmw4wx8gc001cad0ekx9c90xe/experiments/results?baseline=1eac9c4e91d8202e

## Mean score per metric (over the cases it applies to)

| metric | n | mean |
|---|---|---|
| route_correct | 31 | 1.00 |
| agents_correct | 31 | 1.00 |
| sql_executes | 25 | 1.00 |
| cache_behaviour | 2 | 1.00 |
| answered | 39 | 1.00 |
| no_secret_leak | 39 | 1.00 |
| latency_ms | 39 | 25020.79 |
| judge_relevance | 39 | 4.64 |
| judge_correctness | 39 | 3.44 |
| judge_groundedness | 39 | 3.23 |
| judge_completeness | 39 | 4.49 |
| judge_clarity | 39 | 4.87 |
| a2a_schema_valid | 26 | 1.00 |
| result_matches_gold | 16 | 1.00 |
| required_facts | 3 | 1.00 |
| no_agents_called | 1 | 1.00 |
| approval_requested | 1 | 1.00 |
| refused | 2 | 0.00 |
| guardrail_blocked | 2 | 1.00 |
| citations_present | 7 | 1.00 |

## By category (mean of the 0/1 checks)

| category | cases | checks passed |
|---|---|---|
| cache | 2 | 1.00 |
| memory | 3 | 1.00 |
| failure | 7 | 0.93 |
| ambiguous | 5 | 1.00 |
| hybrid | 6 | 1.00 |
| complex | 6 | 1.00 |
| normal | 10 | 1.00 |

## Failed checks

- `f04`: refused
- `f03`: refused
