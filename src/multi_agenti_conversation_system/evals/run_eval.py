"""Run the evaluation as a Langfuse experiment.

    uv run python -m multi_agenti_conversation_system.evals.run_eval \\
        --label development --run-name planner-v2

    options: --category normal|complex|hybrid|ambiguous|failure|memory|cache
             --limit N        first N cases (a quick check)
             --no-judge       deterministic scores only
             --local          use eval/dataset.yaml directly (no Langfuse dataset, no run link)

For each case: a fresh user and thread, the real supervisor and real agents (started here on
spare ports with `EVAL_MODE`, so no learned SQL pairs are written, G23), the deterministic
scorers of `scorers.py`, and the LLM judge. Each item is linked to its Langfuse trace and to
the prompt versions it ran with. Cases run one at a time: the free model is rate limited
(G18), and the cache cases switch `EVAL_MODE` for their own turns.

Isolation: user ids carry the run id; facts, profiles and cache entries a case creates are
deleted when the case ends, so two runs never see each other's memory, and a run does not
leave the shared stores dirty.

Output: `results/<run-name>.csv` (one row per case, every score) and
`results/<run-name>.md` (means overall and per category, the failures).
"""

import argparse
import asyncio
import csv
import logging
import re
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psycopg
from langfuse import Evaluation

from multi_agenti_conversation_system import tracing, vectorstore
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.evals.dataset import (
    DATASET_NAME,
    expected_of,
    input_of,
    load_cases,
)
from multi_agenti_conversation_system.evals.judge import Judge, evidence_text
from multi_agenti_conversation_system.evals.scorers import score_item
from multi_agenti_conversation_system.evals.stack import eval_stack
from multi_agenti_conversation_system.supervisor.service import open_supervisor

logger = logging.getLogger(__name__)
RESULTS = Path("results")
POLL_S, POLL_EVERY_S = 90, 3


class Runner:
    """The supervisor on one event loop in a thread (like the UI), called from sync code."""

    def __init__(self, urls: list[str], judge: Judge | None) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.judge, self.supervisor, self.run_id = judge, None, uuid.uuid4().hex[:6]
        self._ready, self._error = threading.Event(), None
        asyncio.run_coroutine_threadsafe(self._serve(urls), self.loop)
        if not self._ready.wait(timeout=120) or self._error:
            raise RuntimeError(f"supervisor did not start: {self._error}")

    async def _serve(self, urls: list[str]) -> None:
        try:
            async with open_supervisor(urls) as supervisor:
                self.supervisor = supervisor
                self._ready.set()
                await asyncio.Event().wait()
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()

    def run(self, coro) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def submit(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    # ---------------------------------------------------------------- one case
    async def _poll(self, probe) -> Any:
        end = time.monotonic() + POLL_S
        while time.monotonic() < end:
            if value := await probe():
                return value
            await asyncio.sleep(POLL_EVERY_S)
        return None

    async def case(
        self, inp: dict[str, Any], trace_ctx: dict[str, str]
    ) -> dict[str, Any]:
        sup, ctx = self.supervisor, self.supervisor.context
        user = f"eval-{self.run_id}-{inp['id']}"
        cache_case = inp["category"] == "cache"
        cleanup_texts = [t for t in (inp.get("prime"), inp["question"]) if t]
        settings.EVAL_MODE = not cache_case  # the cache is off in EVAL_MODE
        try:
            for i, message in enumerate(inp.get("setup", [])):
                await sup.ask(f"{user}-setup{i}", user, message, trace_ctx=trace_ctx)
            if inp.get("setup"):  # a fresh write is not instantly searchable (G14)
                await self._poll(lambda: _learned(ctx, user, inp["question"]))
            if inp.get("prime"):
                await sup.ask(f"{user}-prime", user, inp["prime"], trace_ctx=trace_ctx)
                await self._poll(lambda: _cached(sup, inp["prime"]))
            started = time.monotonic()
            turn = await sup.ask(
                f"{user}-main", user, inp["question"], trace_ctx=trace_ctx
            )
            approval = False
            if (
                turn.status == "needs_approval"
            ):  # the eval user declines: nothing leaves the DB
                approval = True
                turn = await sup.resume(
                    f"{user}-main", approved=False, user_id=user, trace_ctx=trace_ctx
                )
            output = asdict(turn)
            output.update(
                approval_requested=approval,
                latency_ms=round((time.monotonic() - started) * 1000),
                error=None,
            )
            return output
        except Exception as exc:
            logger.exception("case %s crashed", inp["id"])
            return {
                "status": "error",
                "answer": "",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": None,
            }
        finally:
            settings.EVAL_MODE = True
            await _cleanup(ctx, user, cleanup_texts)


async def _learned(ctx, user: str, question: str) -> bool:
    if await ctx.profiles.load(user):
        return True
    return bool(await ctx.facts.recall(user, question))


async def _cached(sup, question: str) -> bool:
    from multi_agenti_conversation_system.memory.entities import entity_key
    from multi_agenti_conversation_system.memory.profile import profile_hash
    from multi_agenti_conversation_system.supervisor.memory_nodes import (
        current_prompt_version,
    )

    return bool(
        await sup.context.cache.lookup(
            question,
            entity_key(question),
            profile_hash({}),
            await current_prompt_version(),
            threshold=0.8,
        )
    )


async def _cleanup(ctx, user: str, texts: list[str]) -> None:
    try:
        await ctx.facts.delete_user(user)
        await ctx.profiles.reset(user)
        for text in texts:
            await asyncio.to_thread(
                vectorstore.delete_where,
                vectorstore.SEMANTIC_CACHE,
                {"text": {"$eq": text}},
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("eval cleanup failed: %s", exc)


# --------------------------------------------------------------------------- scoring
def gold_executor():
    """Run a query on the read-only role, for `sql_executes` and `result_matches_gold`."""

    def execute(sql: str) -> list[tuple]:
        with psycopg.connect(settings.northwind_conninfo, autocommit=True) as conn:
            conn.execute("SET statement_timeout = '15s'")
            return conn.execute(sql.rstrip().rstrip(";")).fetchall()

    return execute


def make_evaluators(runner: Runner, judge: Judge | None):
    execute = gold_executor()

    def deterministic(*, input, output, expected_output, metadata=None, **_):
        return [
            Evaluation(name=name, value=value, comment=comment or None)
            for name, value, comment in score_item(
                expected_output or {}, output or {}, execute
            )
        ]

    def judged(*, input, output, expected_output, metadata=None, **_):
        if judge is None or not output or not output.get("answer"):
            return []
        notes = (expected_output or {}).get("required_facts") or []
        scores = runner.run(
            judge.judge(
                input["question"],
                output["answer"],
                output.get("citations") or [],
                ", ".join(notes),
                evidence_text(output.get("results")),
            )
        )
        return [
            Evaluation(name=name, value=value, comment=reason)
            for name, value, reason in (scores or [])
        ]

    return [deterministic, judged]


# -------------------------------------------------------------------------- export
def export(result, run_name: str, label: str) -> tuple[Path, Path]:
    RESULTS.mkdir(exist_ok=True)
    rows, names = [], []
    for item in result.item_results:
        inp = item.item.input if hasattr(item.item, "input") else item.item["input"]
        out = item.output or {}
        scores = {e.name: e.value for e in item.evaluations}
        names += [n for n in scores if n not in names]
        rows.append(
            {
                "case": inp["id"],
                "category": inp["category"],
                "question": inp["question"],
                "intent": out.get("intent", ""),
                "agents": "+".join(out.get("agents_used") or []),
                "status": out.get("status", ""),
                "cache_hit": out.get("cache_hit", ""),
                "answer": (out.get("answer") or "").replace("\n", " ")[:400],
                **scores,
            }
        )
    base = [
        "case",
        "category",
        "question",
        "intent",
        "agents",
        "status",
        "cache_hit",
        "answer",
    ]
    csv_path = RESULTS / f"{run_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base + names)
        writer.writeheader()
        writer.writerows(rows)
    md_path = RESULTS / f"{run_name}.md"
    md_path.write_text(
        summary(rows, names, run_name, label, getattr(result, "dataset_run_url", None)),
        encoding="utf-8",
    )
    return csv_path, md_path


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.2f}" if values else "-"


def summary(
    rows: list[dict], names: list[str], run_name: str, label: str, url: str | None
) -> str:
    lines = [
        f"# Evaluation run `{run_name}`",
        "",
        f"- prompt label: `{label}`",
        f"- cases: {len(rows)}",
    ]
    if url:
        lines.append(f"- Langfuse run: {url}")
    lines += [
        "",
        "## Mean score per metric (over the cases it applies to)",
        "",
        "| metric | n | mean |",
        "|---|---|---|",
    ]
    for name in names:
        vals = [float(r[name]) for r in rows if r.get(name) not in (None, "")]
        lines.append(f"| {name} | {len(vals)} | {_mean(vals)} |")
    lines += [
        "",
        "## By category (mean of the 0/1 checks)",
        "",
        "| category | cases | checks passed |",
        "|---|---|---|",
    ]
    binary = [n for n in names if not n.startswith("judge_") and n != "latency_ms"]
    for cat in dict.fromkeys(r["category"] for r in rows):
        group = [r for r in rows if r["category"] == cat]
        vals = [
            float(r[n]) for r in group for n in binary if r.get(n) not in (None, "")
        ]
        lines.append(f"| {cat} | {len(group)} | {_mean(vals)} |")
    broken = [r["case"] for r in rows if MODEL_FAILED in (r.get("answer") or "")]
    if broken:
        lines += [
            "",
            (
                "> **Not a quality result:** the model call failed (rate limit or outage) "
                f"in {len(broken)} case(s): {', '.join(broken)}. Re-run them before reading "
                "the scores."
            ),
        ]
    lines += ["", "## Failed checks", ""]
    failed = [
        (r["case"], n)
        for r in rows
        for n in binary
        if r.get(n) not in (None, "") and float(r[n]) < 1
    ]
    lines += [f"- `{c}`: {n}" for c, n in failed] or ["- none"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main
MODEL_FAILED = "the model call failed"  # the planner's marker (supervisor/nodes.py)


def preflight() -> None:
    """One tiny model call before anything starts: a rate-limited or unreachable model would
    otherwise turn every case into a failure that looks like a quality problem."""
    from multi_agenti_conversation_system.llm import get_chat_model

    try:
        asyncio.run(get_chat_model().ainvoke("Reply with the single word OK."))
    except Exception as exc:
        raise SystemExit(
            f"the chat model is not usable right now ({type(exc).__name__}: {str(exc)[:200]}). "
            "Nothing was run. Fix that (quota, key, network), then start again."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--label", default="development", help="prompt label for every agent"
    )
    parser.add_argument("--run-name", default=f"run-{time.strftime('%Y%m%d-%H%M')}")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.run_name)

    settings.PROMPT_LABEL, settings.EVAL_MODE = args.label, True
    cases = load_cases(args.category, args.limit)
    preflight()
    print(f"{len(cases)} case(s), prompt label {args.label!r}, run {run_name!r}")
    with eval_stack(args.label) as stack:
        settings.A2A_TOKEN = stack["token"]
        judge = None if args.no_judge else Judge()
        runner = Runner(stack["urls"].split(","), judge)
        client = tracing.get_langfuse()

        def task(*, item, **_):
            inp = item["input"] if isinstance(item, dict) else item.input
            ctx = tracing.current_trace_ctx()
            print(f"  {inp['id']:4} {inp['question'][:70]}")
            return runner.run(runner.case(inp, ctx))

        if args.local or client is None:
            data = [
                {"input": input_of(c), "expected_output": expected_of(c)} for c in cases
            ]
        else:
            wanted = {c["id"] for c in cases}
            data = [
                i
                for i in client.get_dataset(DATASET_NAME).items
                if (i.metadata or {}).get("case") in wanted
            ]
            if not data:
                raise SystemExit(
                    "the Langfuse dataset is empty: run scripts/sync_eval_dataset.py first"
                )
        if client is None:
            raise SystemExit(
                "Langfuse is required for an experiment (keys empty or TRACING_ENABLED=false)"
            )
        result = client.run_experiment(
            name=DATASET_NAME,
            run_name=run_name,
            description=f"prompt label {args.label}",
            data=data,
            task=task,
            evaluators=make_evaluators(runner, judge),
            max_concurrency=1,
            metadata={"prompt_label": args.label},
        )
        csv_path, md_path = export(result, run_name, args.label)
        print(f"\nwrote {csv_path} and {md_path}")
        if getattr(result, "dataset_run_url", None):
            print(f"Langfuse run: {result.dataset_run_url}")
    tracing.shutdown()


if __name__ == "__main__":
    main()
