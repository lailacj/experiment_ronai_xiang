#!/usr/bin/env python3
"""Aggregate Ronai & Xiang human responses by item and condition.

This script reads the original trial-level human response CSVs from
ronai_xiang_data/, applies the participant exclusions used in the original
analysis_codes.R file, and writes one derived item-level table for joining to
model scores.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "ronai_xiang_data"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "human_model_analysis"
    / "human_response_rates_by_item_condition.csv"
)

CONDITION_ORDER = {
    "ESI": 0,
    "Eweak": 1,
    "Estrong": 2,
    "Eonly": 3,
    "Eonlystrong": 4,
}

INPUT_FILES = [
    ("experiment_1", "Experiment_1_data.csv"),
    ("experiment_2", "Experiment_2_data.csv"),
    ("experiment_3", "Experiment_3_data.csv"),
    ("experiment_4", "Experiment_4_data.csv"),
]

# These are the participant exclusions in ronai_xiang_data/analysis_codes.R.
EXCLUDED_PARTICIPANTS = {
    "Experiment_1_data.csv": {
        "8a26c1e8a936b01376d0d52c692c9c67",
        "c84d34ee565c598d9fd817e3f6090e52",
    },
    "Experiment_3_data.csv": {
        "1dd030bd145fd71c59f6e2377f6241de",
    },
}


@dataclass
class GroupStats:
    experiment: str
    condition: str
    item_id: int
    n: int = 0
    sum_response: int = 0

    @property
    def response_rate(self) -> float:
        if self.n == 0:
            return float("nan")
        return self.sum_response / self.n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Ronai & Xiang human responses by item and condition."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path. Default: {OUTPUT_PATH}",
    )
    return parser.parse_args()


def parse_response(value: str, path: Path, row_number: int) -> int:
    try:
        response = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse Response={value!r} in {path} row {row_number}"
        ) from exc

    if response not in {0, 1}:
        raise ValueError(
            f"Expected binary Response in {path} row {row_number}, got {response}"
        )
    return response


def aggregate() -> tuple[list[GroupStats], dict[str, dict[str, int]]]:
    groups: dict[tuple[str, int], GroupStats] = {}
    run_summary: dict[str, dict[str, int]] = {}

    for experiment, filename in INPUT_FILES:
        path = DATA_DIR / filename
        excluded = EXCLUDED_PARTICIPANTS.get(filename, set())
        before_rows = 0
        after_rows = 0
        excluded_rows = 0
        before_participants: set[str] = set()
        after_participants: set[str] = set()

        with path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"Participant", "Condition", "Item", "Response"}
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"{path} is missing required column(s): {missing}")

            for row_number, row in enumerate(reader, start=2):
                before_rows += 1
                participant = row["Participant"]
                before_participants.add(participant)

                if participant in excluded:
                    excluded_rows += 1
                    continue

                condition = row["Condition"]
                item_id = int(row["Item"])
                response = parse_response(row["Response"], path, row_number)

                key = (condition, item_id)
                if key not in groups:
                    groups[key] = GroupStats(
                        experiment=experiment,
                        condition=condition,
                        item_id=item_id,
                    )

                groups[key].n += 1
                groups[key].sum_response += response
                after_rows += 1
                after_participants.add(participant)

        run_summary[filename] = {
            "before_rows": before_rows,
            "after_rows": after_rows,
            "excluded_rows": excluded_rows,
            "before_participants": len(before_participants),
            "after_participants": len(after_participants),
            "excluded_participants": len(excluded),
        }

    sorted_groups = sorted(
        groups.values(),
        key=lambda group: (
            CONDITION_ORDER.get(group.condition, 999),
            group.item_id,
        ),
    )
    return sorted_groups, run_summary


def write_output(groups: list[GroupStats], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "experiment",
                "condition",
                "item_id",
                "N",
                "sum_response",
                "response_rate",
            ],
        )
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    "experiment": group.experiment,
                    "condition": group.condition,
                    "item_id": group.item_id,
                    "N": group.n,
                    "sum_response": group.sum_response,
                    "response_rate": f"{group.response_rate:.10f}",
                }
            )


def print_summary(
    groups: list[GroupStats],
    run_summary: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    print("Input summary:")
    for filename, summary in run_summary.items():
        print(
            f"  {filename}: "
            f"{summary['before_rows']} rows / "
            f"{summary['before_participants']} participants before exclusions; "
            f"{summary['after_rows']} rows / "
            f"{summary['after_participants']} participants after exclusions"
        )

    print("\nAggregated rows by condition:")
    by_condition: dict[str, list[GroupStats]] = defaultdict(list)
    for group in groups:
        by_condition[group.condition].append(group)

    for condition in sorted(
        by_condition,
        key=lambda value: CONDITION_ORDER.get(value, 999),
    ):
        condition_groups = by_condition[condition]
        ns = [group.n for group in condition_groups]
        mean_rate = sum(group.response_rate for group in condition_groups) / len(
            condition_groups
        )
        print(
            f"  {condition}: {len(condition_groups)} items, "
            f"N range {min(ns)}-{max(ns)}, "
            f"mean response_rate {mean_rate:.3f}"
        )

    print(f"\nWrote {len(groups)} rows to {output_path}")


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    groups, run_summary = aggregate()
    write_output(groups, output_path)
    print_summary(groups, run_summary, output_path)


if __name__ == "__main__":
    main()
