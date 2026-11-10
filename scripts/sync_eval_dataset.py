"""Upload evals/dataset.yaml to a Langfuse dataset, so experiment runs link to their items.

    uv run python scripts/sync_eval_dataset.py            # create or update the items
    uv run python scripts/sync_eval_dataset.py --dry-run

Idempotent: an item is identified by the case id (`id=` of the Langfuse item), so re-running
updates it in place instead of adding a duplicate.
"""

import argparse
import sys

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.evals.dataset import (
    DATASET_NAME,
    expected_of,
    input_of,
    load_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = load_cases()
    if args.dry_run:
        print(f"would upload {len(cases)} items to dataset {DATASET_NAME!r}")
        return
    client = tracing.get_langfuse()
    if client is None:
        sys.exit("Langfuse is not configured (keys empty or TRACING_ENABLED=false)")
    client.create_dataset(
        name=DATASET_NAME,
        description="Multi-agent assistant evaluation: normal, complex, hybrid, ambiguous, failure, memory and cache cases.",
    )
    for case in cases:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=f"{DATASET_NAME}-{case['id']}",
            input=input_of(case),
            expected_output=expected_of(case),
            metadata={"category": case["category"], "case": case["id"]},
        )
    client.flush()
    print(f"uploaded {len(cases)} items to dataset {DATASET_NAME!r}")


if __name__ == "__main__":
    main()
