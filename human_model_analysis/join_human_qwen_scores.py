#!/usr/bin/env python3
"""Join item-level human response rates to Qwen alternative scores.

The output is one row per experiment/condition/item_id. Model scores are
widened so weaker and stronger alternative log probabilities live side by side.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMAN_INPUT = (
    PROJECT_ROOT
    / "human_model_analysis"
    / "human_response_rates_by_item_condition.csv"
)
DEFAULT_MODEL_INPUT = (
    PROJECT_ROOT
    / "model_scores"
    / "qwen2_7b_scored_alternatives.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "human_model_analysis"
    / "human_qwen_item_condition_joined.csv"
)

CONDITION_ORDER = {
    "ESI": 0,
    "Eweak": 1,
    "Estrong": 2,
    "Eonly": 3,
    "Eonlystrong": 4,
}

OUTPUT_COLUMNS = [
    "experiment",
    "condition",
    "item_id",
    "N",
    "sum_response",
    "response_rate",
    "human_log_response_rate",
    "weaker_lemma",
    "weaker_surface",
    "stronger_lemma",
    "stronger_surface_guess",
    "prompt_text",
    "context_text",
    "suffix_text",
    "weaker_target_surface",
    "stronger_target_surface",
    "weaker_word_logprob",
    "stronger_word_logprob",
    "weaker_candidate_plus_suffix_logprob",
    "stronger_candidate_plus_suffix_logprob",
    "word_logprob_stronger_minus_weaker",
    "candidate_plus_suffix_logprob_stronger_minus_weaker",
    "weaker_word_n_tokens",
    "stronger_word_n_tokens",
    "weaker_candidate_plus_suffix_n_tokens",
    "stronger_candidate_plus_suffix_n_tokens",
    "model_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join human response rates to Qwen alternative scores."
    )
    parser.add_argument(
        "--human-input",
        type=Path,
        default=DEFAULT_HUMAN_INPUT,
        help=f"Human aggregate CSV. Default: {DEFAULT_HUMAN_INPUT}",
    )
    parser.add_argument(
        "--model-input",
        type=Path,
        default=DEFAULT_MODEL_INPUT,
        help=f"Qwen scored alternatives CSV. Default: {DEFAULT_MODEL_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Joined output CSV. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_human_rows(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    human_rows: dict[tuple[str, str, int], dict[str, str]] = {}

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "experiment",
            "condition",
            "item_id",
            "N",
            "sum_response",
            "response_rate",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            key = (row["experiment"], row["condition"], int(row["item_id"]))
            if key in human_rows:
                raise ValueError(f"Duplicate human key {key} in {path} row {row_number}")
            human_rows[key] = row

    return human_rows


def read_model_rows(path: Path) -> dict[tuple[str, str, int], dict[str, dict[str, str]]]:
    model_rows: dict[tuple[str, str, int], dict[str, dict[str, str]]] = {}

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "experiment",
            "condition",
            "item_id",
            "target_type",
            "target_surface",
            "word_logprob",
            "candidate_plus_suffix_logprob",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            target_type = row["target_type"]
            if target_type not in {"weaker", "stronger"}:
                raise ValueError(
                    f"Unexpected target_type={target_type!r} in {path} row {row_number}"
                )

            key = (row["experiment"], row["condition"], int(row["item_id"]))
            if key not in model_rows:
                model_rows[key] = {}
            if target_type in model_rows[key]:
                raise ValueError(
                    f"Duplicate model key {key} target_type={target_type!r} "
                    f"in {path} row {row_number}"
                )
            model_rows[key][target_type] = row

    return model_rows


def format_float(value: float) -> str:
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.10f}"


def log_response_rate(response_rate_text: str) -> str:
    response_rate = float(response_rate_text)
    if response_rate < 0 or response_rate > 1:
        raise ValueError(f"Expected response_rate in [0, 1], got {response_rate}")
    if response_rate == 0:
        return "-inf"
    return format_float(math.log(response_rate))


def difference(stronger_text: str, weaker_text: str) -> str:
    return format_float(float(stronger_text) - float(weaker_text))


def build_joined_rows(
    human_rows: dict[tuple[str, str, int], dict[str, str]],
    model_rows: dict[tuple[str, str, int], dict[str, dict[str, str]]],
) -> list[dict[str, str]]:
    joined_rows: list[dict[str, str]] = []
    missing_model_keys = []

    for key, human in human_rows.items():
        model_pair = model_rows.get(key)
        if model_pair is None or {"weaker", "stronger"} - set(model_pair):
            missing_model_keys.append(key)
            continue

        weaker = model_pair["weaker"]
        stronger = model_pair["stronger"]

        joined_rows.append(
            {
                "experiment": human["experiment"],
                "condition": human["condition"],
                "item_id": human["item_id"],
                "N": human["N"],
                "sum_response": human["sum_response"],
                "response_rate": human["response_rate"],
                "human_log_response_rate": log_response_rate(human["response_rate"]),
                "weaker_lemma": weaker["weaker_lemma"],
                "weaker_surface": weaker["weaker_surface"],
                "stronger_lemma": weaker["stronger_lemma"],
                "stronger_surface_guess": weaker["stronger_surface_guess"],
                "prompt_text": stronger["prompt_text"],
                "context_text": stronger["context_text"],
                "suffix_text": stronger["suffix_text"],
                "weaker_target_surface": weaker["target_surface"],
                "stronger_target_surface": stronger["target_surface"],
                "weaker_word_logprob": weaker["word_logprob"],
                "stronger_word_logprob": stronger["word_logprob"],
                "weaker_candidate_plus_suffix_logprob": weaker[
                    "candidate_plus_suffix_logprob"
                ],
                "stronger_candidate_plus_suffix_logprob": stronger[
                    "candidate_plus_suffix_logprob"
                ],
                "word_logprob_stronger_minus_weaker": difference(
                    stronger["word_logprob"], weaker["word_logprob"]
                ),
                "candidate_plus_suffix_logprob_stronger_minus_weaker": difference(
                    stronger["candidate_plus_suffix_logprob"],
                    weaker["candidate_plus_suffix_logprob"],
                ),
                "weaker_word_n_tokens": weaker["word_n_tokens"],
                "stronger_word_n_tokens": stronger["word_n_tokens"],
                "weaker_candidate_plus_suffix_n_tokens": weaker[
                    "candidate_plus_suffix_n_tokens"
                ],
                "stronger_candidate_plus_suffix_n_tokens": stronger[
                    "candidate_plus_suffix_n_tokens"
                ],
                "model_name": stronger["model_name"],
            }
        )

    if missing_model_keys:
        examples = ", ".join(map(str, missing_model_keys[:5]))
        raise ValueError(
            f"Missing weaker/stronger model scores for {len(missing_model_keys)} "
            f"human row(s). Examples: {examples}"
        )

    return sorted(
        joined_rows,
        key=lambda row: (
            row["experiment"],
            CONDITION_ORDER.get(row["condition"], 999),
            int(row["item_id"]),
        ),
    )


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    by_experiment = Counter(row["experiment"] for row in rows)
    print("Joined rows by experiment:")
    for experiment in sorted(by_experiment):
        print(f"  {experiment}: {by_experiment[experiment]} rows")

    print(f"\nWrote {len(rows)} rows to {output_path}")


def main() -> None:
    args = parse_args()
    human_input = project_path(args.human_input)
    model_input = project_path(args.model_input)
    output = project_path(args.output)

    human_rows = read_human_rows(human_input)
    model_rows = read_model_rows(model_input)
    joined_rows = build_joined_rows(human_rows, model_rows)
    write_rows(joined_rows, output)
    print_summary(joined_rows, output)


if __name__ == "__main__":
    main()
