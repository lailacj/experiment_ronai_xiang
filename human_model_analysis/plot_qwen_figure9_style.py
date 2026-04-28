#!/usr/bin/env python3
"""Plot Qwen scores in a Figure-9-like layout.

The plots use one bar per scale in each condition. Scales are ordered by the
Experiment 1 human SI rate, matching the ordering used in the Ronai & Xiang
Figure 9 plotting code.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "human_model_analysis"
    / "human_qwen_item_condition_joined.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "human_model_analysis" / "plots"

CONDITIONS = [
    ("ESI", "SI"),
    ("Eweak", "Weak QUD"),
    ("Estrong", "Strong QUD"),
    ("Eonly", "Only"),
    ("Eonlystrong", "QUD+only"),
]

DEFAULT_SCORE_COLUMN = "stronger_word_logprob"
BAR_COLOR = "#595959"
GRID_COLOR = "#e6e6e6"
STRIP_COLOR = "#d9d9d9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make Figure-9-like Qwen bar plots by condition and scale."
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
        "--score-column",
        default=DEFAULT_SCORE_COLUMN,
        help=(
            "Model score column to plot. Default: stronger_word_logprob. "
            "Use stronger_candidate_plus_suffix_logprob for candidate+suffix scores."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Plot file formats to write. Default: png pdf",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def import_pyplot(output_dir: Path):
    """Import matplotlib after pointing cache/temp directories at output_dir."""
    cache_dir = output_dir / ".matplotlib"
    temp_dir = output_dir / ".tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("TMPDIR", str(temp_dir))

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
            "weaker_lemma",
            "stronger_lemma",
            score_column,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{input_path} is missing column(s): {', '.join(sorted(missing))}"
            )
        return list(reader)


def scale_label(row: dict[str, str]) -> str:
    return f"{row['weaker_lemma']}/{row['stronger_lemma']}"


def get_scale_order(rows: list[dict[str, str]]) -> tuple[list[int], dict[int, str]]:
    esi_rows = [row for row in rows if row["condition"] == "ESI"]
    if not esi_rows:
        raise ValueError("Cannot order scales because no ESI rows were found.")

    sorted_esi_rows = sorted(
        esi_rows,
        key=lambda row: (float(row["response_rate"]), int(row["item_id"])),
    )
    item_order = [int(row["item_id"]) for row in sorted_esi_rows]
    labels_by_item = {int(row["item_id"]): scale_label(row) for row in rows}

    if len(item_order) != 60:
        raise ValueError(f"Expected 60 ESI items for scale order, found {len(item_order)}")

    return item_order, labels_by_item


def build_score_lookup(
    rows: list[dict[str, str]],
    score_column: str,
) -> dict[tuple[str, int], float]:
    lookup: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (row["condition"], int(row["item_id"]))
        if key in lookup:
            raise ValueError(f"Duplicate condition/item score for {key}")
        lookup[key] = float(row[score_column])
    return lookup


def check_complete_scores(
    score_lookup: dict[tuple[str, int], float],
    item_order: list[int],
) -> None:
    missing = []
    for condition, _label in CONDITIONS:
        for item_id in item_order:
            if (condition, item_id) not in score_lookup:
                missing.append((condition, item_id))

    if missing:
        examples = ", ".join(map(str, missing[:5]))
        raise ValueError(f"Missing {len(missing)} condition/item score(s): {examples}")


def minmax_scale(values: list[float]) -> tuple[list[float], float, float]:
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        raise ValueError("Cannot min-max scale scores because all values are identical.")
    scaled = [
        100.0 * (value - min_value) / (max_value - min_value)
        for value in values
    ]
    return scaled, min_value, max_value


def add_condition_strip(axis, label: str) -> None:
    axis.text(
        1.004,
        0.5,
        label,
        transform=axis.transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=12,
        bbox={
            "boxstyle": "square,pad=0.45",
            "facecolor": STRIP_COLOR,
            "edgecolor": "#8c8c8c",
            "linewidth": 0.7,
        },
    )


def make_plot(
    plt,
    item_order: list[int],
    labels_by_item: dict[int, str],
    score_lookup: dict[tuple[str, int], float],
    output_path: Path,
    y_label: str,
    title: str,
    raw_logprob: bool,
) -> None:
    x_positions = list(range(len(item_order)))
    x_labels = [labels_by_item[item_id] for item_id in item_order]
    all_values = [
        score_lookup[(condition, item_id)]
        for condition, _label in CONDITIONS
        for item_id in item_order
    ]

    if raw_logprob:
        plot_lookup = score_lookup
        y_min = min(all_values)
        margin = max(0.5, abs(y_min) * 0.04)
        y_limits = (y_min - margin, 0)
    else:
        scaled_values, raw_min, raw_max = minmax_scale(all_values)
        keys = [
            (condition, item_id)
            for condition, _label in CONDITIONS
            for item_id in item_order
        ]
        plot_lookup = dict(zip(keys, scaled_values))
        y_limits = (100, 0)

    fig, axes = plt.subplots(
        nrows=len(CONDITIONS),
        ncols=1,
        sharex=True,
        figsize=(20, 13),
    )
    fig.subplots_adjust(left=0.07, right=0.94, top=0.91, bottom=0.27, hspace=0.08)

    for axis, (condition, condition_label) in zip(axes, CONDITIONS):
        values = [plot_lookup[(condition, item_id)] for item_id in item_order]
        axis.bar(x_positions, values, width=0.78, color=BAR_COLOR)
        axis.set_xlim(-0.7, len(item_order) - 0.3)
        axis.set_ylim(*y_limits)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", labelsize=9)
        add_condition_strip(axis, condition_label)

        if raw_logprob:
            axis.axhline(0, color="#2f2f2f", linewidth=0.8)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_labels, rotation=90, ha="center", fontsize=7)
    axes[-1].tick_params(axis="x", length=0)

    fig.suptitle(title, fontsize=16)
    fig.text(0.5, 0.04, "Scales ordered by Experiment 1 human SI rate", ha="center")
    fig.text(0.015, 0.57, y_label, va="center", rotation=90, fontsize=12)

    if not raw_logprob:
        fig.text(
            0.5,
            0.015,
            (
                "0-100 scale is global min-max scaling over all plotted "
                f"Qwen logprobs: min={raw_min:.3f}, max={raw_max:.3f}"
            ),
            ha="center",
            fontsize=9,
        )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_path, args.score_column)
    item_order, labels_by_item = get_scale_order(rows)
    score_lookup = build_score_lookup(rows, args.score_column)
    check_complete_scores(score_lookup, item_order)

    plt = import_pyplot(output_dir)

    raw_stem = f"qwen_{args.score_column}_figure9_style_raw_logprob"
    scaled_stem = f"qwen_{args.score_column}_figure9_style_minmax_0_100"
    written_paths = []

    for fmt in args.formats:
        raw_path = output_dir / f"{raw_stem}.{fmt}"
        make_plot(
            plt=plt,
            item_order=item_order,
            labels_by_item=labels_by_item,
            score_lookup=score_lookup,
            output_path=raw_path,
            y_label="Qwen log P(stronger alternative as next word)",
            title="Qwen Stronger-Alternative Scores by Scale",
            raw_logprob=True,
        )
        written_paths.append(raw_path)

        scaled_path = output_dir / f"{scaled_stem}.{fmt}"
        make_plot(
            plt=plt,
            item_order=item_order,
            labels_by_item=labels_by_item,
            score_lookup=score_lookup,
            output_path=scaled_path,
            y_label="Qwen stronger-alternative score, min-max scaled to 0-100",
            title="Qwen Stronger-Alternative Scores by Scale, 0-100 Visual Scale",
            raw_logprob=False,
        )
        written_paths.append(scaled_path)

    print("Wrote plots:")
    for path in written_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
