#!/usr/bin/env python3
"""Plot Experiment 2 weak-to-strong QUD changes.

Each figure shows paired item-level values for the Weak QUD and Strong QUD
conditions. Thin colored lines are individual scalar items; the black line is
the item-level condition mean.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import math
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "human_model_analysis"
    / "human_qwen_item_condition_joined.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "human_model_analysis" / "qud_change_plots"
TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []

CONDITIONS = ["Eweak", "Estrong"]
CONDITION_LABELS = {
    "Eweak": "Weak QUD",
    "Estrong": "Strong QUD",
}

PLOT_CONFIGS = {
    "human": {
        "filename": "experiment_2_human_response_rate_qud_change.png",
        "title": "Experiment 2 Human Responses",
        "metric_label": "Human response rate",
        "y_label": "Response rate",
        "caption": (
            "Response rate is the proportion of participants who interpreted "
            "the weaker answer as excluding the stronger alternative."
        ),
        "zero_line": False,
    },
    "baseline": {
        "filename": "experiment_2_baseline_stronger_logprob_qud_change.png",
        "title": "Experiment 2 Baseline Model",
        "metric_label": "Qwen log P(stronger alternative | context)",
        "y_label": "Baseline prediction score",
        "caption": (
            "The baseline score is Qwen's log probability for the stronger "
            "alternative in context; higher values predict more negation of "
            "the stronger alternative."
        ),
        "zero_line": False,
    },
    "ordering": {
        "filename": "experiment_2_ordering_score_qud_change.png",
        "title": "Experiment 2 Ordering Model",
        "metric_label": "log P(stronger) - log P(weaker)",
        "y_label": "Ordering prediction score",
        "caption": (
            "The ordering score compares the stronger query to the weaker "
            "trigger; values above zero mean the stronger alternative is more "
            "likely than the weaker alternative."
        ),
        "zero_line": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make paired Weak QUD vs Strong QUD plots for Experiment 2."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Joined human/Qwen CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output plot directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--baseline-score-column",
        default="stronger_word_logprob",
        help=(
            "Column to use for the baseline model plot. "
            "Default: stronger_word_logprob."
        ),
    )
    parser.add_argument(
        "--ordering-score-column",
        default="word_logprob_stronger_minus_weaker",
        help=(
            "Column to use for the ordering model plot. "
            "Default: word_logprob_stronger_minus_weaker."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Plot file formats to write. Default: png",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def import_pyplot(output_dir: Path):
    """Import matplotlib after pointing cache/temp directories at scratch space."""
    temp_root = tempfile.TemporaryDirectory(prefix="ronai_xiang_qud_change_")
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
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def read_experiment_2_rows(
    input_path: Path,
    baseline_score_column: str,
    ordering_score_column: str,
) -> list[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "experiment",
            "condition",
            "item_id",
            "response_rate",
            "weaker_lemma",
            "stronger_lemma",
            baseline_score_column,
            ordering_score_column,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{input_path} is missing column(s): {', '.join(sorted(missing))}"
            )

        return [
            row
            for row in reader
            if row["experiment"] == "experiment_2"
            and row["condition"] in CONDITIONS
        ]


def scale_label(row: dict[str, str]) -> str:
    return f"{row['weaker_lemma']}/{row['stronger_lemma']}"


def build_item_pairs(
    rows: list[dict[str, str]],
    value_column: str,
) -> tuple[list[dict[str, object]], list[str]]:
    by_item: dict[int, dict[str, dict[str, str]]] = {}

    for row in rows:
        value = finite_float(row[value_column])
        if value is None:
            continue
        item_id = int(row["item_id"])
        by_item.setdefault(item_id, {})[row["condition"]] = row

    pairs: list[dict[str, object]] = []
    for item_id in sorted(by_item):
        item_rows = by_item[item_id]
        if not all(condition in item_rows for condition in CONDITIONS):
            continue

        weak = item_rows["Eweak"]
        strong = item_rows["Estrong"]
        pairs.append(
            {
                "item_id": item_id,
                "scale": scale_label(weak),
                "values": {
                    "Eweak": float(weak[value_column]),
                    "Estrong": float(strong[value_column]),
                },
            }
        )

    missing_conditions = []
    for item_id in sorted(by_item):
        item_rows = by_item[item_id]
        for condition in CONDITIONS:
            if condition not in item_rows:
                missing_conditions.append(f"item {item_id} {condition}")

    return pairs, missing_conditions


def item_color(item_id: int) -> tuple[float, float, float]:
    hue = ((item_id - 1) * 0.61803398875) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.62, 0.72)


def padded_limits(values: list[float], *, include_zero: bool = False) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if low == high:
        padding = max(abs(low) * 0.08, 0.1)
    else:
        padding = max((high - low) * 0.08, 0.04)
    return low - padding, high + padding


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def add_mean_overlay(axis, pairs: list[dict[str, object]]) -> tuple[float, float]:
    weak_mean = mean(
        [float(pair["values"]["Eweak"]) for pair in pairs]  # type: ignore[index]
    )
    strong_mean = mean(
        [float(pair["values"]["Estrong"]) for pair in pairs]  # type: ignore[index]
    )
    axis.plot(
        [0, 1],
        [weak_mean, strong_mean],
        color="#111111",
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="#111111",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=5,
        label="Item mean",
    )
    return weak_mean, strong_mean


def draw_paired_plot(
    plt,
    pairs: list[dict[str, object]],
    output_stem: Path,
    *,
    title: str,
    y_label: str,
    metric_label: str,
    caption: str,
    zero_line: bool,
    formats: list[str],
) -> tuple[float, float, float, list[Path]]:
    if not pairs:
        raise ValueError(f"No complete item pairs were available for {output_stem}.")

    all_values = [
        float(pair["values"][condition])  # type: ignore[index]
        for pair in pairs
        for condition in CONDITIONS
    ]
    y_limits = (-0.04, 1.04) if y_label == "Response rate" else padded_limits(
        all_values,
        include_zero=zero_line,
    )

    fig, axis = plt.subplots(figsize=(6.6, 6.1))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.87, bottom=0.23)

    for pair in pairs:
        item_id = int(pair["item_id"])
        values = pair["values"]  # type: ignore[assignment]
        y_values = [float(values["Eweak"]), float(values["Estrong"])]
        color = item_color(item_id)
        axis.plot(
            [0, 1],
            y_values,
            color=color,
            linewidth=1.15,
            alpha=0.5,
            zorder=2,
        )
        axis.scatter(
            [0, 1],
            y_values,
            color=color,
            s=26,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    if zero_line:
        axis.axhline(
            0,
            color="#6f6f6f",
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            alpha=0.85,
            zorder=1,
        )

    weak_mean, strong_mean = add_mean_overlay(axis, pairs)

    axis.set_xlim(-0.18, 1.18)
    axis.set_ylim(*y_limits)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([CONDITION_LABELS[condition] for condition in CONDITIONS])
    axis.set_ylabel(y_label)
    axis.set_title(title, fontsize=14, pad=12)
    axis.grid(axis="y", color="#e4e4e4", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")

    axis.text(
        0.5,
        -0.18,
        f"{metric_label}. {caption}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
        wrap=True,
    )

    written_paths = []
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        output_path = output_stem.with_suffix(f".{file_format}")
        fig.savefig(output_path, dpi=220)
        written_paths.append(output_path)
    plt.close(fig)

    return weak_mean, strong_mean, strong_mean - weak_mean, written_paths


def print_summary(
    plot_name: str,
    pairs: list[dict[str, object]],
    weak_mean: float,
    strong_mean: float,
    difference: float,
    paths: list[Path],
) -> None:
    n_positive = sum(
        1
        for pair in pairs
        if float(pair["values"]["Estrong"])  # type: ignore[index]
        > float(pair["values"]["Eweak"])  # type: ignore[index]
    )
    n_negative = sum(
        1
        for pair in pairs
        if float(pair["values"]["Estrong"])  # type: ignore[index]
        < float(pair["values"]["Eweak"])  # type: ignore[index]
    )
    print(f"{plot_name}:")
    print(f"  n complete item pairs: {len(pairs)}")
    print(f"  Weak QUD mean: {weak_mean:.4f}")
    print(f"  Strong QUD mean: {strong_mean:.4f}")
    print(f"  Strong - weak mean change: {difference:.4f}")
    print(f"  Items increasing/decreasing: {n_positive}/{n_negative}")
    for path in paths:
        print(f"  Wrote {path}")


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    output_dir = project_path(args.output_dir)

    rows = read_experiment_2_rows(
        input_path,
        args.baseline_score_column,
        args.ordering_score_column,
    )
    plt = import_pyplot(output_dir)

    plot_columns = {
        "human": "response_rate",
        "baseline": args.baseline_score_column,
        "ordering": args.ordering_score_column,
    }

    for plot_name in ["human", "baseline", "ordering"]:
        config = PLOT_CONFIGS[plot_name]
        value_column = plot_columns[plot_name]
        pairs, missing_conditions = build_item_pairs(rows, value_column)
        if missing_conditions:
            examples = ", ".join(missing_conditions[:5])
            print(
                f"Warning: skipped {len(missing_conditions)} incomplete "
                f"{plot_name} value(s), including {examples}"
            )

        output_stem = output_dir / Path(str(config["filename"])).with_suffix("")
        weak_mean, strong_mean, difference, paths = draw_paired_plot(
            plt,
            pairs,
            output_stem,
            title=str(config["title"]),
            y_label=str(config["y_label"]),
            metric_label=str(config["metric_label"]),
            caption=str(config["caption"]),
            zero_line=bool(config["zero_line"]),
            formats=args.formats,
        )
        print_summary(
            plot_name,
            pairs,
            weak_mean,
            strong_mean,
            difference,
            paths,
        )


if __name__ == "__main__":
    main()
