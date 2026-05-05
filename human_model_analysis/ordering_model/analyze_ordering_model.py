#!/usr/bin/env python3
"""Analyze a simple query-trigger ordering model.

The model compares Qwen's log probability for the stronger alternative
("query") against the weaker alternative ("trigger"). A positive score means
the query was more likely than the trigger:

    log P(query) - log P(trigger) > 0

The human comparison asks whether this ordering tracks item-level rates of
participants negating or excluding the stronger alternative.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "human_model_analysis"
    / "human_qwen_item_condition_joined.csv"
)
DEFAULT_ITEM_OUTPUT = MODEL_DIR / "ordering_model_item_condition.csv"
DEFAULT_SUMMARY_OUTPUT = MODEL_DIR / "ordering_model_binary_summary.csv"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "plots"

TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []

EXPERIMENTS = [
    ("experiment_1", "Experiment 1"),
    ("experiment_2", "Experiment 2"),
    ("experiment_3", "Experiment 3"),
    ("experiment_4", "Experiment 4"),
]

CONDITION_LABELS = {
    "ESI": "SI",
    "Eweak": "Weak QUD",
    "Estrong": "Strong QUD",
    "Eonly": "Only",
    "Eonlystrong": "Strong QUD + only",
}

CONDITION_COLORS = {
    "ESI": "#2f6f9f",
    "Eweak": "#8f6bb8",
    "Estrong": "#c45a52",
    "Eonly": "#3c8c5a",
    "Eonlystrong": "#d28a2e",
}

CONDITION_ORDER = {
    "ESI": 0,
    "Eweak": 1,
    "Estrong": 2,
    "Eonly": 3,
    "Eonlystrong": 4,
}

SCORE_TYPES = {
    "word": {
        "query_col": "stronger_word_logprob",
        "trigger_col": "weaker_word_logprob",
        "label": "word",
        "x_label": "log P(query/stronger) - log P(trigger/weaker)",
    },
    "candidate-plus-suffix": {
        "query_col": "stronger_candidate_plus_suffix_logprob",
        "trigger_col": "weaker_candidate_plus_suffix_logprob",
        "label": "candidate_plus_suffix",
        "x_label": (
            "log P(query/stronger + suffix) - "
            "log P(trigger/weaker + suffix)"
        ),
    },
}

ITEM_OUTPUT_COLUMNS = [
    "experiment",
    "condition",
    "item_id",
    "N",
    "response_rate",
    "trigger_lemma",
    "query_lemma",
    "trigger_surface",
    "query_surface",
    "trigger_logprob",
    "query_logprob",
    "ordering_score",
    "query_more_likely_than_trigger",
    "score_type",
    "model_name",
]

SUMMARY_OUTPUT_COLUMNS = [
    "scope",
    "experiment",
    "condition",
    "n_items",
    "n_query_more_likely_than_trigger",
    "n_query_not_more_likely_than_trigger",
    "proportion_query_more_likely_than_trigger",
    "mean_response_query_more_likely_than_trigger",
    "mean_response_query_not_more_likely_than_trigger",
    "mean_response_difference_query_more_minus_not_more",
    "mean_ordering_score",
    "pearson_r",
    "spearman_rho",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether query-trigger logprob ordering predicts humans."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Joined human/Qwen CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--item-output",
        type=Path,
        default=None,
        help=(
            f"Output item-level CSV. Default: {DEFAULT_ITEM_OUTPUT} for word scores; "
            "score-type-specific path otherwise."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=(
            f"Output binary summary CSV. Default: {DEFAULT_SUMMARY_OUTPUT} for "
            "word scores; score-type-specific path otherwise."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output plot directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--score-type",
        choices=sorted(SCORE_TYPES),
        default="word",
        help=(
            "Which Qwen scores to compare. Default: word. "
            "Use candidate-plus-suffix to include the suffix in both scores."
        ),
    )
    return parser.parse_args()


def default_item_output_path(score_type: str) -> Path:
    if score_type == "word":
        return DEFAULT_ITEM_OUTPUT
    score_label = SCORE_TYPES[score_type]["label"]
    return (
        MODEL_DIR / f"ordering_model_{score_label}_item_condition.csv"
    )


def default_summary_output_path(score_type: str) -> Path:
    if score_type == "word":
        return DEFAULT_SUMMARY_OUTPUT
    score_label = SCORE_TYPES[score_type]["label"]
    return (
        MODEL_DIR / f"ordering_model_{score_label}_binary_summary.csv"
    )


def default_plot_paths(score_type: str, output_dir: Path) -> tuple[Path, Path]:
    if score_type == "word":
        return (
            output_dir / "qwen_ordering_model_scatter_by_experiment.png",
            output_dir / "qwen_ordering_model_binary_split_by_experiment.png",
        )

    score_label = SCORE_TYPES[score_type]["label"]
    return (
        output_dir / f"qwen_ordering_model_{score_label}_scatter_by_experiment.png",
        output_dir / f"qwen_ordering_model_{score_label}_binary_split_by_experiment.png",
    )


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def import_pyplot(output_dir: Path):
    """Import matplotlib after pointing cache/temp directories at scratch space."""
    temp_root = tempfile.TemporaryDirectory(prefix="ronai_xiang_ordering_")
    TEMP_DIRS.append(temp_root)
    scratch_dir = Path(temp_root.name)
    cache_dir = scratch_dir / "cache"
    temp_dir = scratch_dir / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("TMPDIR", str(temp_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def finite_float(value: str) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{value:.10f}"


def normalized_label(value: str) -> str:
    return " ".join(value.strip().lower().split())


def read_ordering_rows(
    input_path: Path,
    score_type: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    score_config = SCORE_TYPES[score_type]
    query_col = score_config["query_col"]
    trigger_col = score_config["trigger_col"]
    rows: list[dict[str, str]] = []
    counts = {
        "input_rows": 0,
        "output_rows": 0,
        "excluded_same_query_trigger": 0,
        "excluded_nonfinite": 0,
    }

    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "experiment",
            "condition",
            "item_id",
            "N",
            "response_rate",
            "weaker_lemma",
            "stronger_lemma",
            "weaker_target_surface",
            "stronger_target_surface",
            "model_name",
            query_col,
            trigger_col,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{input_path} is missing column(s): {', '.join(sorted(missing))}"
            )

        for row in reader:
            counts["input_rows"] += 1
            trigger_surface = row["weaker_target_surface"]
            query_surface = row["stronger_target_surface"]
            trigger_lemma = row["weaker_lemma"]
            query_lemma = row["stronger_lemma"]

            if (
                normalized_label(trigger_surface) == normalized_label(query_surface)
                or normalized_label(trigger_lemma) == normalized_label(query_lemma)
            ):
                counts["excluded_same_query_trigger"] += 1
                continue

            response_rate = finite_float(row["response_rate"])
            trigger_logprob = finite_float(row[trigger_col])
            query_logprob = finite_float(row[query_col])
            if (
                response_rate is None
                or trigger_logprob is None
                or query_logprob is None
            ):
                counts["excluded_nonfinite"] += 1
                continue

            ordering_score = query_logprob - trigger_logprob
            query_more_likely = ordering_score > 0
            rows.append(
                {
                    "experiment": row["experiment"],
                    "condition": row["condition"],
                    "item_id": row["item_id"],
                    "N": row["N"],
                    "response_rate": format_float(response_rate),
                    "trigger_lemma": trigger_lemma,
                    "query_lemma": query_lemma,
                    "trigger_surface": trigger_surface,
                    "query_surface": query_surface,
                    "trigger_logprob": format_float(trigger_logprob),
                    "query_logprob": format_float(query_logprob),
                    "ordering_score": format_float(ordering_score),
                    "query_more_likely_than_trigger": str(query_more_likely).lower(),
                    "score_type": score_config["label"],
                    "model_name": row["model_name"],
                }
            )
            counts["output_rows"] += 1

    return sorted(rows, key=row_sort_key), counts


def row_sort_key(row: dict[str, str]) -> tuple[str, int, int]:
    return (
        row["experiment"],
        CONDITION_ORDER.get(row["condition"], 999),
        int(row["item_id"]),
    )


def write_csv(
    rows: list[dict[str, str]],
    output_path: Path,
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def pearson_from_values(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    sum_xx = sum(x * x for x in centered_x)
    sum_yy = sum(y * y for y in centered_y)
    if sum_xx == 0 or sum_yy == 0:
        return None

    return sum(x * y for x, y in zip(centered_x, centered_y)) / math.sqrt(
        sum_xx * sum_yy
    )


def average_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    sorted_pairs = sorted(enumerate(values), key=lambda pair: pair[1])
    index = 0

    while index < len(sorted_pairs):
        tie_end = index + 1
        while (
            tie_end < len(sorted_pairs)
            and sorted_pairs[tie_end][1] == sorted_pairs[index][1]
        ):
            tie_end += 1

        average_rank = (index + 1 + tie_end) / 2
        for original_index, _value in sorted_pairs[index:tie_end]:
            ranks[original_index] = average_rank
        index = tie_end

    return ranks


def spearman_from_values(xs: list[float], ys: list[float]) -> float | None:
    return pearson_from_values(average_ranks(xs), average_ranks(ys))


def numeric_column(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def summarize_group(
    rows: list[dict[str, str]],
    *,
    scope: str,
    experiment: str,
    condition: str,
) -> dict[str, str]:
    query_more_rows = [
        row for row in rows if row["query_more_likely_than_trigger"] == "true"
    ]
    query_not_more_rows = [
        row for row in rows if row["query_more_likely_than_trigger"] == "false"
    ]

    query_more_mean = mean(numeric_column(query_more_rows, "response_rate"))
    query_not_more_mean = mean(numeric_column(query_not_more_rows, "response_rate"))
    if query_more_mean is None or query_not_more_mean is None:
        mean_difference = None
    else:
        mean_difference = query_more_mean - query_not_more_mean

    ordering_scores = numeric_column(rows, "ordering_score")
    response_rates = numeric_column(rows, "response_rate")
    pearson = pearson_from_values(ordering_scores, response_rates)
    spearman = spearman_from_values(ordering_scores, response_rates)

    return {
        "scope": scope,
        "experiment": experiment,
        "condition": condition,
        "n_items": str(len(rows)),
        "n_query_more_likely_than_trigger": str(len(query_more_rows)),
        "n_query_not_more_likely_than_trigger": str(len(query_not_more_rows)),
        "proportion_query_more_likely_than_trigger": format_float(
            len(query_more_rows) / len(rows) if rows else None
        ),
        "mean_response_query_more_likely_than_trigger": format_float(
            query_more_mean
        ),
        "mean_response_query_not_more_likely_than_trigger": format_float(
            query_not_more_mean
        ),
        "mean_response_difference_query_more_minus_not_more": format_float(
            mean_difference
        ),
        "mean_ordering_score": format_float(mean(ordering_scores)),
        "pearson_r": format_float(pearson),
        "spearman_rho": format_float(spearman),
    }


def build_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary_rows = [
        summarize_group(
            rows,
            scope="overall",
            experiment="all",
            condition="all",
        )
    ]

    by_experiment: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_condition: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment"]].append(row)
        by_condition[(row["experiment"], row["condition"])].append(row)

    for experiment, _label in EXPERIMENTS:
        experiment_rows = by_experiment.get(experiment, [])
        if experiment_rows:
            summary_rows.append(
                summarize_group(
                    experiment_rows,
                    scope="experiment",
                    experiment=experiment,
                    condition="all",
                )
            )

    for key in sorted(
        by_condition,
        key=lambda value: (
            value[0],
            CONDITION_ORDER.get(value[1], 999),
        ),
    ):
        experiment, condition = key
        summary_rows.append(
            summarize_group(
                by_condition[key],
                scope="condition",
                experiment=experiment,
                condition=condition,
            )
        )

    return summary_rows


def padded_limits(values: list[float], minimum_padding: float) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    padding = max((high - low) * 0.06, minimum_padding)
    return low - padding, high + padding


def regression_line(rows: list[dict[str, str]]) -> tuple[float, float] | None:
    if len(rows) < 2:
        return None

    xs = numeric_column(rows, "ordering_score")
    ys = numeric_column(rows, "response_rate")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sum_xx = sum((x - mean_x) ** 2 for x in xs)
    if sum_xx == 0:
        return None

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum_xx
    intercept = mean_y - slope * mean_x
    return slope, intercept


def add_correlation_text(axis, rows: list[dict[str, str]]) -> None:
    xs = numeric_column(rows, "ordering_score")
    ys = numeric_column(rows, "response_rate")
    pearson = pearson_from_values(xs, ys)
    spearman = spearman_from_values(xs, ys)
    pearson_text = "undefined" if pearson is None else f"{pearson:.2f}"
    spearman_text = "undefined" if spearman is None else f"{spearman:.2f}"

    axis.text(
        0.04,
        0.94,
        (
            f"Pearson r = {pearson_text}\n"
            f"Spearman rho = {spearman_text}\n"
            f"n = {len(rows)}"
        ),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#c7c7c7",
            "alpha": 0.88,
        },
    )


def add_condition_points(axis, rows: list[dict[str, str]]) -> None:
    conditions = sorted(
        {row["condition"] for row in rows},
        key=lambda condition: CONDITION_ORDER.get(condition, 999),
    )
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        axis.scatter(
            numeric_column(condition_rows, "ordering_score"),
            numeric_column(condition_rows, "response_rate"),
            s=42,
            alpha=0.82,
            color=CONDITION_COLORS.get(condition, "#595959"),
            edgecolors="white",
            linewidths=0.45,
            label=CONDITION_LABELS.get(condition, condition),
        )


def add_trend_line(axis, rows: list[dict[str, str]]) -> None:
    fit = regression_line(rows)
    if fit is None:
        return

    slope, intercept = fit
    xs = numeric_column(rows, "ordering_score")
    x_min = min(xs)
    x_max = max(xs)
    axis.plot(
        [x_min, x_max],
        [slope * x_min + intercept, slope * x_max + intercept],
        color="#2d2d2d",
        linewidth=1.3,
        alpha=0.8,
    )


def rows_by_experiment(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)
    return grouped


def make_scatter_plot(
    plt,
    rows: list[dict[str, str]],
    output_path: Path,
    score_type: str,
) -> None:
    grouped_rows = rows_by_experiment(rows)
    all_x = numeric_column(rows, "ordering_score")
    x_low, x_high = padded_limits(all_x + [0.0], 0.5)
    y_limits = (-0.04, 1.04)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(12, 8.2),
    )
    axes_flat = list(axes.flat)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1, hspace=0.22)

    for axis, (experiment, experiment_label) in zip(axes_flat, EXPERIMENTS):
        experiment_rows = grouped_rows.get(experiment, [])
        axis.set_title(experiment_label, fontsize=13, pad=8)
        axis.set_xlim(x_low, x_high)
        axis.set_ylim(*y_limits)
        axis.grid(color="#e6e6e6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.axvline(0, color="#737373", linestyle="--", linewidth=1.0)

        if not experiment_rows:
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            continue

        add_condition_points(axis, experiment_rows)
        add_trend_line(axis, experiment_rows)
        add_correlation_text(axis, experiment_rows)

        conditions = sorted(
            {row["condition"] for row in experiment_rows},
            key=lambda condition: CONDITION_ORDER.get(condition, 999),
        )
        if len(conditions) > 1:
            axis.legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle("Ordering Model: Human Responses by Query-Trigger Logprob", fontsize=15)
    fig.text(0.5, 0.035, SCORE_TYPES[score_type]["x_label"], ha="center", fontsize=11)
    fig.text(
        0.02,
        0.5,
        "Human response probability",
        va="center",
        rotation=90,
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def binary_label(row: dict[str, str]) -> str:
    if row["query_more_likely_than_trigger"] == "true":
        return "query > trigger"
    return "query <= trigger"


def jitter_for_index(index: int) -> float:
    return ((index % 11) - 5) * 0.018


def add_binary_points(axis, rows: list[dict[str, str]]) -> None:
    conditions = sorted(
        {row["condition"] for row in rows},
        key=lambda condition: CONDITION_ORDER.get(condition, 999),
    )
    category_x = {"query <= trigger": 0.0, "query > trigger": 1.0}

    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        for category in ["query <= trigger", "query > trigger"]:
            category_rows = [
                row for row in condition_rows if binary_label(row) == category
            ]
            axis.scatter(
                [
                    category_x[category] + jitter_for_index(index)
                    for index, _row in enumerate(category_rows)
                ],
                numeric_column(category_rows, "response_rate"),
                s=38,
                alpha=0.72,
                color=CONDITION_COLORS.get(condition, "#595959"),
                edgecolors="white",
                linewidths=0.4,
                label=CONDITION_LABELS.get(condition, condition),
            )


def add_mean_markers(axis, rows: list[dict[str, str]]) -> None:
    category_x = {"query <= trigger": 0.0, "query > trigger": 1.0}
    for category in ["query <= trigger", "query > trigger"]:
        category_rows = [row for row in rows if binary_label(row) == category]
        category_mean = mean(numeric_column(category_rows, "response_rate"))
        if category_mean is None:
            continue

        x = category_x[category]
        axis.hlines(
            category_mean,
            x - 0.22,
            x + 0.22,
            color="#2d2d2d",
            linewidth=2.2,
        )
        if category_mean > 0.88:
            label_y = category_mean - 0.035
            vertical_alignment = "top"
        else:
            label_y = category_mean + 0.035
            vertical_alignment = "bottom"
        axis.text(
            x,
            label_y,
            f"mean={category_mean:.2f}",
            ha="center",
            va=vertical_alignment,
            fontsize=8.5,
            color="#2d2d2d",
        )


def add_binary_summary_text(axis, rows: list[dict[str, str]]) -> None:
    query_more_rows = [
        row for row in rows if row["query_more_likely_than_trigger"] == "true"
    ]
    query_not_more_rows = [
        row for row in rows if row["query_more_likely_than_trigger"] == "false"
    ]
    mean_more = mean(numeric_column(query_more_rows, "response_rate"))
    mean_not_more = mean(numeric_column(query_not_more_rows, "response_rate"))
    if mean_more is None or mean_not_more is None:
        difference_text = "undefined"
    else:
        difference_text = f"{mean_more - mean_not_more:.2f}"

    all_response_mean = mean(numeric_column(rows, "response_rate"))
    if all_response_mean is not None and all_response_mean > 0.68:
        y_position = 0.06
        vertical_alignment = "bottom"
    else:
        y_position = 0.94
        vertical_alignment = "top"

    axis.text(
        0.04,
        y_position,
        (
            f"mean diff = {difference_text}\n"
            f"n <= 0: {len(query_not_more_rows)}\n"
            f"n > 0: {len(query_more_rows)}"
        ),
        transform=axis.transAxes,
        va=vertical_alignment,
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#c7c7c7",
            "alpha": 0.88,
        },
    )


def make_binary_plot(
    plt,
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    grouped_rows = rows_by_experiment(rows)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(12, 8.2),
    )
    axes_flat = list(axes.flat)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.12, hspace=0.22)

    for axis, (experiment, experiment_label) in zip(axes_flat, EXPERIMENTS):
        experiment_rows = grouped_rows.get(experiment, [])
        axis.set_title(experiment_label, fontsize=13, pad=8)
        axis.set_xlim(-0.5, 1.5)
        axis.set_ylim(-0.04, 1.04)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["query <= trigger", "query > trigger"])

        if not experiment_rows:
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            continue

        add_binary_points(axis, experiment_rows)
        add_mean_markers(axis, experiment_rows)
        add_binary_summary_text(axis, experiment_rows)

        handles, labels = axis.get_legend_handles_labels()
        deduplicated = dict(zip(labels, handles))
        if len(deduplicated) > 1:
            axis.legend(
                deduplicated.values(),
                deduplicated.keys(),
                frameon=False,
                fontsize=9,
                loc="lower left",
            )

    fig.suptitle("Ordering Model: Binary Query-Trigger Prediction", fontsize=15)
    fig.text(
        0.5,
        0.045,
        "Binary model prediction from log P(query) - log P(trigger)",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.02,
        0.5,
        "Human response probability",
        va="center",
        rotation=90,
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def print_summary(
    counts: dict[str, int],
    summary_rows: list[dict[str, str]],
    item_output: Path,
    summary_output: Path,
    scatter_output: Path,
    binary_output: Path,
) -> None:
    print("Input summary:")
    print(f"  read rows: {counts['input_rows']}")
    print(f"  kept rows: {counts['output_rows']}")
    print(
        "  excluded same query/trigger: "
        f"{counts['excluded_same_query_trigger']}"
    )
    print(f"  excluded non-finite rows: {counts['excluded_nonfinite']}")

    print("\nOverall ordering summary:")
    overall = summary_rows[0]
    print(
        "  Pearson r="
        f"{overall['pearson_r']}, Spearman rho={overall['spearman_rho']}"
    )
    print(
        "  mean response difference query>trigger minus query<=trigger="
        f"{overall['mean_response_difference_query_more_minus_not_more']}"
    )

    print("\nWrote:")
    print(f"  {item_output}")
    print(f"  {summary_output}")
    print(f"  {scatter_output}")
    print(f"  {binary_output}")


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    item_output = args.item_output or default_item_output_path(args.score_type)
    summary_output = args.summary_output or default_summary_output_path(args.score_type)
    item_output = project_path(item_output)
    summary_output = project_path(summary_output)
    output_dir = project_path(args.output_dir)

    item_rows, counts = read_ordering_rows(input_path, args.score_type)
    if not item_rows:
        raise ValueError("No item rows were available for the ordering analysis.")

    summary_rows = build_summary_rows(item_rows)
    scatter_output, binary_output = default_plot_paths(args.score_type, output_dir)

    write_csv(item_rows, item_output, ITEM_OUTPUT_COLUMNS)
    write_csv(summary_rows, summary_output, SUMMARY_OUTPUT_COLUMNS)

    plt = import_pyplot(output_dir)
    make_scatter_plot(plt, item_rows, scatter_output, args.score_type)
    make_binary_plot(plt, item_rows, binary_output)

    print_summary(
        counts,
        summary_rows,
        item_output,
        summary_output,
        scatter_output,
        binary_output,
    )


if __name__ == "__main__":
    main()
