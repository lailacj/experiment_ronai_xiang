#!/usr/bin/env python3
"""Plot Experiment 1 SI to Experiment 2 Weak QUD changes.

Each figure shows paired item-level values for the SI condition from
Experiment 1 and the Weak QUD condition from Experiment 2. Thin colored lines
are individual scalar items; the black line is the item-level condition mean.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import math
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "human_model_analysis"
DEFAULT_INPUT = ANALYSIS_ROOT / "human_qwen_item_condition_joined.csv"
TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []

COMPARISON_POINTS = [
    {
        "key": "SI",
        "experiment": "experiment_1",
        "condition": "ESI",
        "label": "Experiment 1\nSI",
    },
    {
        "key": "WEAK_QUD",
        "experiment": "experiment_2",
        "condition": "Eweak",
        "label": "Experiment 2\nWeak QUD",
    },
]
POINT_KEYS = [str(point["key"]) for point in COMPARISON_POINTS]

PLOT_CONFIGS = {
    "human": {
        "filename": "si_vs_weak_qud_human_response_rate_change.png",
        "title": "SI vs. Weak QUD Human Responses",
        "metric_label": "Human response rate",
        "y_label": "Response rate",
        "caption": (
            "Response rate is the proportion of participants who interpreted "
            "the weaker answer as excluding the stronger alternative; lower "
            "Weak QUD values indicate implicature suppression."
        ),
        "zero_line": False,
    },
    "baseline": {
        "filename": "si_vs_weak_qud_baseline_stronger_logprob_change.png",
        "title": "SI vs. Weak QUD Baseline Model",
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
        "filename": "si_vs_weak_qud_ordering_score_change.png",
        "title": "SI vs. Weak QUD Ordering Model",
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

SCORE_FAMILIES = {
    "word": {
        "folder": "word_logprob",
        "baseline_score_column": "stronger_word_logprob",
        "ordering_score_column": "word_logprob_stronger_minus_weaker",
        "baseline_metric_label": "Qwen log P(stronger alternative | context)",
        "baseline_caption": (
            "The baseline score is Qwen's log probability for the stronger "
            "alternative in context; higher values predict more negation of "
            "the stronger alternative."
        ),
        "ordering_metric_label": "log P(stronger) - log P(weaker)",
        "ordering_caption": (
            "The ordering score compares the stronger query to the weaker "
            "trigger; values above zero mean the stronger alternative is more "
            "likely than the weaker alternative."
        ),
    },
    "candidate-plus-suffix": {
        "folder": "candidate_plus_suffix_logprob",
        "baseline_score_column": "stronger_candidate_plus_suffix_logprob",
        "ordering_score_column": (
            "candidate_plus_suffix_logprob_stronger_minus_weaker"
        ),
        "baseline_metric_label": (
            "Qwen log P(stronger alternative + suffix | context)"
        ),
        "baseline_caption": (
            "The baseline score is Qwen's log probability for the stronger "
            "alternative plus suffix in context; higher values predict more "
            "negation of the stronger alternative."
        ),
        "ordering_metric_label": (
            "log P(stronger + suffix) - log P(weaker + suffix)"
        ),
        "ordering_caption": (
            "The ordering score compares the stronger query-plus-suffix to "
            "the weaker trigger-plus-suffix; values above zero mean the "
            "stronger continuation is more likely than the weaker continuation."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make paired Experiment 1 SI vs Experiment 2 Weak QUD plots."
        )
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
        default=None,
        help="Output plot directory. Default comes from --score-family.",
    )
    parser.add_argument(
        "--score-family",
        choices=sorted(SCORE_FAMILIES),
        default="word",
        help=(
            "Score family for default columns and output directories. "
            "Default: word."
        ),
    )
    parser.add_argument(
        "--baseline-score-column",
        default=None,
        help=(
            "Column to use for the baseline model plot. "
            "Default comes from --score-family."
        ),
    )
    parser.add_argument(
        "--ordering-score-column",
        default=None,
        help=(
            "Column to use for the ordering model plot. "
            "Default comes from --score-family."
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


def default_output_dir(score_family: str) -> Path:
    score_folder = str(SCORE_FAMILIES[score_family]["folder"])
    return ANALYSIS_ROOT / score_folder / "qud_change_plots"


def default_baseline_score_column(score_family: str) -> str:
    return str(SCORE_FAMILIES[score_family]["baseline_score_column"])


def default_ordering_score_column(score_family: str) -> str:
    return str(SCORE_FAMILIES[score_family]["ordering_score_column"])


def plot_configs_for_score_family(score_family: str) -> dict[str, dict[str, object]]:
    configs = {name: dict(config) for name, config in PLOT_CONFIGS.items()}
    score_config = SCORE_FAMILIES[score_family]
    configs["baseline"]["metric_label"] = score_config["baseline_metric_label"]
    configs["baseline"]["caption"] = score_config["baseline_caption"]
    configs["ordering"]["metric_label"] = score_config["ordering_metric_label"]
    configs["ordering"]["caption"] = score_config["ordering_caption"]
    return configs


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def import_pyplot(output_dir: Path):
    """Import matplotlib after pointing cache/temp directories at scratch space."""
    temp_root = tempfile.TemporaryDirectory(prefix="ronai_xiang_si_weak_qud_")
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


def read_comparison_rows(
    input_path: Path,
    baseline_score_column: str,
    ordering_score_column: str,
) -> list[dict[str, str]]:
    comparison_set = {
        (str(point["experiment"]), str(point["condition"]))
        for point in COMPARISON_POINTS
    }

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
            if (row["experiment"], row["condition"]) in comparison_set
        ]


def comparison_key(row: dict[str, str]) -> str | None:
    for point in COMPARISON_POINTS:
        if (
            row["experiment"] == point["experiment"]
            and row["condition"] == point["condition"]
        ):
            return str(point["key"])
    return None


def scale_label(row: dict[str, str]) -> str:
    return f"{row['weaker_lemma']}/{row['stronger_lemma']}"


def build_item_pairs(
    rows: list[dict[str, str]],
    value_column: str,
) -> tuple[list[dict[str, object]], list[str]]:
    by_item: dict[int, dict[str, dict[str, str]]] = {}

    for row in rows:
        value = finite_float(row[value_column])
        key = comparison_key(row)
        if value is None or key is None:
            continue
        item_id = int(row["item_id"])
        by_item.setdefault(item_id, {})[key] = row

    pairs: list[dict[str, object]] = []
    for item_id in sorted(by_item):
        item_rows = by_item[item_id]
        if not all(key in item_rows for key in POINT_KEYS):
            continue

        left = item_rows[POINT_KEYS[0]]
        right = item_rows[POINT_KEYS[1]]
        pairs.append(
            {
                "item_id": item_id,
                "scale": scale_label(left),
                "values": {
                    POINT_KEYS[0]: float(left[value_column]),
                    POINT_KEYS[1]: float(right[value_column]),
                },
            }
        )

    missing_points = []
    for item_id in sorted(by_item):
        item_rows = by_item[item_id]
        for point in COMPARISON_POINTS:
            key = str(point["key"])
            if key not in item_rows:
                missing_points.append(
                    f"item {item_id} {point['experiment']} {point['condition']}"
                )

    return pairs, missing_points


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
    left_mean = mean(
        [float(pair["values"][POINT_KEYS[0]]) for pair in pairs]  # type: ignore[index]
    )
    right_mean = mean(
        [float(pair["values"][POINT_KEYS[1]]) for pair in pairs]  # type: ignore[index]
    )
    axis.plot(
        [0, 1],
        [left_mean, right_mean],
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
    return left_mean, right_mean


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
        float(pair["values"][key])  # type: ignore[index]
        for pair in pairs
        for key in POINT_KEYS
    ]
    y_limits = (-0.04, 1.04) if y_label == "Response rate" else padded_limits(
        all_values,
        include_zero=zero_line,
    )

    fig, axis = plt.subplots(figsize=(6.6, 6.1))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.87, bottom=0.24)

    for pair in pairs:
        item_id = int(pair["item_id"])
        values = pair["values"]  # type: ignore[assignment]
        y_values = [float(values[POINT_KEYS[0]]), float(values[POINT_KEYS[1]])]
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

    left_mean, right_mean = add_mean_overlay(axis, pairs)

    axis.set_xlim(-0.18, 1.18)
    axis.set_ylim(*y_limits)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([str(point["label"]) for point in COMPARISON_POINTS])
    axis.set_ylabel(y_label)
    axis.set_title(title, fontsize=14, pad=12)
    axis.grid(axis="y", color="#e4e4e4", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")

    axis.text(
        0.5,
        -0.19,
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

    return left_mean, right_mean, right_mean - left_mean, written_paths


def print_summary(
    plot_name: str,
    pairs: list[dict[str, object]],
    left_mean: float,
    right_mean: float,
    difference: float,
    paths: list[Path],
) -> None:
    n_increasing = sum(
        1
        for pair in pairs
        if float(pair["values"][POINT_KEYS[1]])  # type: ignore[index]
        > float(pair["values"][POINT_KEYS[0]])  # type: ignore[index]
    )
    n_decreasing = sum(
        1
        for pair in pairs
        if float(pair["values"][POINT_KEYS[1]])  # type: ignore[index]
        < float(pair["values"][POINT_KEYS[0]])  # type: ignore[index]
    )
    left_label = str(COMPARISON_POINTS[0]["label"]).replace("\n", " ")
    right_label = str(COMPARISON_POINTS[1]["label"]).replace("\n", " ")
    print(f"{plot_name}:")
    print(f"  n complete item pairs: {len(pairs)}")
    print(f"  {left_label} mean: {left_mean:.4f}")
    print(f"  {right_label} mean: {right_mean:.4f}")
    print(f"  Weak QUD - SI mean change: {difference:.4f}")
    print(f"  Items increasing/decreasing: {n_increasing}/{n_decreasing}")
    for path in paths:
        print(f"  Wrote {path}")


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    output_dir = args.output_dir or default_output_dir(args.score_family)
    output_dir = project_path(output_dir)
    baseline_score_column = (
        args.baseline_score_column
        or default_baseline_score_column(args.score_family)
    )
    ordering_score_column = (
        args.ordering_score_column
        or default_ordering_score_column(args.score_family)
    )
    plot_configs = plot_configs_for_score_family(args.score_family)

    rows = read_comparison_rows(
        input_path,
        baseline_score_column,
        ordering_score_column,
    )
    plt = import_pyplot(output_dir)

    plot_columns = {
        "human": "response_rate",
        "baseline": baseline_score_column,
        "ordering": ordering_score_column,
    }

    for plot_name in ["human", "baseline", "ordering"]:
        config = plot_configs[plot_name]
        value_column = plot_columns[plot_name]
        pairs, missing_points = build_item_pairs(rows, value_column)
        if missing_points:
            examples = ", ".join(missing_points[:5])
            print(
                f"Warning: skipped {len(missing_points)} incomplete "
                f"{plot_name} value(s), including {examples}"
            )

        output_stem = output_dir / Path(str(config["filename"])).with_suffix("")
        left_mean, right_mean, difference, paths = draw_paired_plot(
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
            left_mean,
            right_mean,
            difference,
            paths,
        )


if __name__ == "__main__":
    main()
