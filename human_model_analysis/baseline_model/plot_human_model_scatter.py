#!/usr/bin/env python3
"""Scatter item-level human response rates against Qwen scores."""

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
DEFAULT_HUMAN_PROBABILITY_QWEN_LOGPROB_OUTPUT = (
    MODEL_DIR
    / "plots"
    / "qwen_human_response_scatter_by_experiment.png"
)
DEFAULT_HUMAN_PROBABILITY_QWEN_PROBABILITY_OUTPUT = (
    MODEL_DIR
    / "plots"
    / "qwen_probability_human_response_scatter_by_experiment.png"
)
DEFAULT_HUMAN_LOGPROB_QWEN_LOGPROB_OUTPUT = (
    MODEL_DIR
    / "plots"
    / "qwen_logprob_human_logprob_scatter_by_experiment.png"
)
DEFAULT_SCORE_COLUMN = "stronger_word_logprob"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make PNG scatter plots comparing item-level human response "
            "probabilities with Qwen log probabilities."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Joined human/Qwen CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Default is chosen from the x/y transform combination."
        ),
    )
    parser.add_argument(
        "--score-column",
        default=DEFAULT_SCORE_COLUMN,
        help=(
            "Model score column for the x-axis. Default: stronger_word_logprob. "
            "Use stronger_candidate_plus_suffix_logprob for candidate+suffix scores."
        ),
    )
    parser.add_argument(
        "--x-transform",
        choices=["logprob", "probability"],
        default="logprob",
        help=(
            "Transform applied to the score-column before plotting. "
            "Use probability to plot exp(logprob). Default: logprob."
        ),
    )
    parser.add_argument(
        "--y-transform",
        choices=["probability", "logprob"],
        default="probability",
        help=(
            "Transform applied to human response_rate before plotting. "
            "Use logprob to plot log(response_rate). Default: probability."
        ),
    )
    return parser.parse_args()


def default_output_path(x_transform: str, y_transform: str) -> Path:
    if y_transform == "probability" and x_transform == "logprob":
        return DEFAULT_HUMAN_PROBABILITY_QWEN_LOGPROB_OUTPUT
    if y_transform == "probability" and x_transform == "probability":
        return DEFAULT_HUMAN_PROBABILITY_QWEN_PROBABILITY_OUTPUT
    if y_transform == "logprob" and x_transform == "logprob":
        return DEFAULT_HUMAN_LOGPROB_QWEN_LOGPROB_OUTPUT
    return (
        MODEL_DIR
        / "plots"
        / f"qwen_{x_transform}_human_{y_transform}_scatter_by_experiment.png"
    )


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def import_pyplot(output_path: Path):
    """Import matplotlib after pointing cache/temp directories at scratch space."""
    temp_root = tempfile.TemporaryDirectory(prefix="ronai_xiang_plot_")
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


def read_rows(input_path: Path, score_column: str) -> list[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "experiment",
            "condition",
            "item_id",
            "response_rate",
            score_column,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{input_path} is missing column(s): {', '.join(sorted(missing))}"
            )
        return list(reader)


def finite_float(value: str) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def rows_by_experiment(
    rows: list[dict[str, str]],
    score_column: str,
    x_transform: str,
    y_transform: str,
) -> dict[str, list[dict[str, float | int | str]]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)

    for row in rows:
        x = finite_float(row[score_column])
        y = finite_float(row["response_rate"])
        if x is None or y is None:
            continue
        if x_transform == "probability":
            x = math.exp(x)
        if y_transform == "logprob":
            if y <= 0:
                continue
            y = math.log(y)
        grouped[row["experiment"]].append(
            {
                "experiment": row["experiment"],
                "condition": row["condition"],
                "item_id": int(row["item_id"]),
                "x": x,
                "y": y,
            }
        )

    return grouped


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


def pearson_r(points: list[dict[str, float | int | str]]) -> float | None:
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    return pearson_from_values(xs, ys)


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


def spearman_r(points: list[dict[str, float | int | str]]) -> float | None:
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    return pearson_from_values(average_ranks(xs), average_ranks(ys))


def regression_line(
    points: list[dict[str, float | int | str]],
) -> tuple[float, float] | None:
    if len(points) < 2:
        return None

    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sum_xx = sum((x - mean_x) ** 2 for x in xs)
    if sum_xx == 0:
        return None

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum_xx
    intercept = mean_y - slope * mean_x
    return slope, intercept


def padded_limits(values: list[float], minimum_padding: float) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    padding = max((high - low) * 0.06, minimum_padding)
    return low - padding, high + padding


def score_label(score_column: str) -> str:
    labels = {
        "stronger_word_logprob": "Qwen log P(stronger alternative | context)",
        "stronger_candidate_plus_suffix_logprob": (
            "Qwen log P(stronger alternative + suffix | context)"
        ),
        "weaker_word_logprob": "Qwen log P(weaker alternative | context)",
        "weaker_candidate_plus_suffix_logprob": (
            "Qwen log P(weaker alternative + suffix | context)"
        ),
        "word_logprob_stronger_minus_weaker": (
            "Qwen logprob difference: stronger - weaker"
        ),
        "candidate_plus_suffix_logprob_stronger_minus_weaker": (
            "Qwen candidate+suffix logprob difference: stronger - weaker"
        ),
    }
    return labels.get(score_column, score_column)


def probability_score_label(score_column: str) -> str:
    labels = {
        "stronger_word_logprob": "Qwen P(stronger alternative | context)",
        "stronger_candidate_plus_suffix_logprob": (
            "Qwen P(stronger alternative + suffix | context)"
        ),
        "weaker_word_logprob": "Qwen P(weaker alternative | context)",
        "weaker_candidate_plus_suffix_logprob": (
            "Qwen P(weaker alternative + suffix | context)"
        ),
    }
    if score_column in labels:
        return labels[score_column]
    return f"exp({score_column})"


def x_axis_label(score_column: str, x_transform: str) -> str:
    if x_transform == "probability":
        return probability_score_label(score_column)
    return score_label(score_column)


def y_axis_label(y_transform: str) -> str:
    if y_transform == "logprob":
        return "Human log response probability"
    return "Human response probability"


def figure_title(x_transform: str, y_transform: str) -> str:
    x_name = "Qwen Probability" if x_transform == "probability" else "Qwen Log Probability"
    y_name = (
        "Human Log Probability"
        if y_transform == "logprob"
        else "Human Response Probability"
    )
    return f"{y_name} vs. {x_name}"


def add_points(axis, points: list[dict[str, float | int | str]]) -> None:
    conditions = sorted(
        {str(point["condition"]) for point in points},
        key=lambda condition: CONDITION_ORDER.get(condition, 999),
    )
    for condition in conditions:
        condition_points = [
            point for point in points if str(point["condition"]) == condition
        ]
        axis.scatter(
            [float(point["x"]) for point in condition_points],
            [float(point["y"]) for point in condition_points],
            s=42,
            alpha=0.82,
            color=CONDITION_COLORS.get(condition, "#595959"),
            edgecolors="white",
            linewidths=0.45,
            label=CONDITION_LABELS.get(condition, condition),
        )


def add_trend_line(axis, points: list[dict[str, float | int | str]]) -> None:
    fit = regression_line(points)
    if fit is None:
        return

    slope, intercept = fit
    x_values = [float(point["x"]) for point in points]
    x_min = min(x_values)
    x_max = max(x_values)
    axis.plot(
        [x_min, x_max],
        [slope * x_min + intercept, slope * x_max + intercept],
        color="#2d2d2d",
        linewidth=1.3,
        alpha=0.8,
    )


def add_correlation_text(
    axis,
    points: list[dict[str, float | int | str]],
) -> dict[str, float | None]:
    pearson = pearson_r(points)
    spearman = spearman_r(points)
    pearson_text = "undefined" if pearson is None else f"{pearson:.2f}"
    spearman_text = "undefined" if spearman is None else f"{spearman:.2f}"
    axis.text(
        0.04,
        0.94,
        (
            f"Pearson r = {pearson_text}\n"
            f"Spearman rho = {spearman_text}\n"
            f"n = {len(points)}"
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
    return {"pearson": pearson, "spearman": spearman}


def make_plot(
    plt,
    grouped_rows: dict[str, list[dict[str, float | int | str]]],
    output_path: Path,
    score_column: str,
    x_transform: str,
    y_transform: str,
) -> dict[str, dict[str, float | None]]:
    all_points = [
        point
        for experiment, _label in EXPERIMENTS
        for point in grouped_rows.get(experiment, [])
    ]
    if not all_points:
        raise ValueError("No finite rows were available to plot.")

    minimum_padding = 0.01 if x_transform == "probability" else 0.5
    x_limits = padded_limits(
        [float(point["x"]) for point in all_points],
        minimum_padding,
    )
    if x_transform == "probability":
        x_limits = (max(0.0, x_limits[0]), min(1.0, x_limits[1]))
    if y_transform == "logprob":
        y_low, _y_high = padded_limits(
            [float(point["y"]) for point in all_points],
            0.2,
        )
        y_limits = (y_low, 0.05)
    else:
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

    correlations: dict[str, dict[str, float | None]] = {}
    for axis, (experiment, experiment_label) in zip(axes_flat, EXPERIMENTS):
        points = grouped_rows.get(experiment, [])
        axis.set_title(experiment_label, fontsize=13, pad=8)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(color="#e6e6e6", linewidth=0.8)
        axis.set_axisbelow(True)

        if not points:
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            correlations[experiment] = {"pearson": None, "spearman": None}
            continue

        add_points(axis, points)
        add_trend_line(axis, points)
        correlations[experiment] = add_correlation_text(axis, points)

        conditions = sorted(
            {str(point["condition"]) for point in points},
            key=lambda condition: CONDITION_ORDER.get(condition, 999),
        )
        if len(conditions) > 1:
            axis.legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle(figure_title(x_transform, y_transform), fontsize=15)
    fig.text(
        0.5,
        0.035,
        x_axis_label(score_column, x_transform),
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.02,
        0.5,
        y_axis_label(y_transform),
        va="center",
        rotation=90,
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return correlations


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    output_path = args.output
    if output_path is None:
        output_path = default_output_path(args.x_transform, args.y_transform)
    output_path = project_path(output_path)

    rows = read_rows(input_path, args.score_column)
    grouped_rows = rows_by_experiment(
        rows,
        args.score_column,
        args.x_transform,
        args.y_transform,
    )
    plt = import_pyplot(output_path)
    correlations = make_plot(
        plt,
        grouped_rows,
        output_path,
        args.score_column,
        args.x_transform,
        args.y_transform,
    )

    print(f"Wrote {output_path}")
    print("Correlations:")
    for experiment, _label in EXPERIMENTS:
        experiment_correlations = correlations.get(
            experiment,
            {"pearson": None, "spearman": None},
        )
        pearson = experiment_correlations["pearson"]
        spearman = experiment_correlations["spearman"]
        pearson_text = "undefined" if pearson is None else f"{pearson:.3f}"
        spearman_text = "undefined" if spearman is None else f"{spearman:.3f}"
        n = len(grouped_rows.get(experiment, []))
        print(
            f"  {experiment}: "
            f"Pearson r={pearson_text}, Spearman rho={spearman_text}, n={n}"
        )


if __name__ == "__main__":
    main()
