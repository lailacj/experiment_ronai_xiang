#!/usr/bin/env python3
"""Analyze Qwen exact-strong scores for the Hu et al. scalar benchmark.

This script makes the first-pass figures and tables for the paper section that
conceptually replicates Hu et al. (2023): stronger alternatives that are more
expected in context should produce higher human scalar-inference rates.

The script intentionally avoids plotting dependencies such as matplotlib. It
writes PDF figures directly with ReportLab, plus CSV/LaTeX summary tables.
"""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor, black
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "model_scores" / "qwen2_7b_exact_strong_scores.csv"
ANALYSIS_PATH = PROJECT_ROOT / "analysis" / "hu_qwen_exact_strong_analysis.csv"
FIGURE_DIR = PROJECT_ROOT / "figures"
TABLE_DIR = PROJECT_ROOT / "tables"

DATASET_ORDER = [
    "Gotzner et al. (2018)",
    "Pankratz & van Tiel (2021)",
    "Ronai & Xiang (2022)",
    "van Tiel et al. (2016)",
]
DATASET_COLORS = {
    "Gotzner et al. (2018)": "#4C78A8",
    "Pankratz & van Tiel (2021)": "#F58518",
    "Ronai & Xiang (2022)": "#54A24B",
    "van Tiel et al. (2016)": "#B279A2",
}
BIN_COLORS = {
    "low": "#9ecae1",
    "mid": "#fdae6b",
    "high": "#74c476",
}


def ensure_dirs() -> None:
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    for col in ["human_si_rate", "target_logprob", "target_n_tokens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["human_si_rate", "target_logprob"]).copy()
    df["target_prob"] = np.exp(df["target_logprob"])
    df["target_logprob_rank"] = df["target_logprob"].rank(method="average")
    df["human_si_rank"] = df["human_si_rate"].rank(method="average")
    return df


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    tmp = pd.DataFrame({"x": list(x), "y": list(y)}).dropna()
    if len(tmp) < 3:
        return float("nan")
    return pearson(tmp["x"].rank(method="average"), tmp["y"].rank(method="average"))


def bootstrap_slope_ci(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    controls: list[str] | None = None,
    n_boot: int = 5000,
    seed: int = 1,
) -> tuple[float, float, float, float]:
    """OLS slope and bootstrap 95% CI for x_col, optionally with categorical controls."""
    controls = controls or []
    y = df[y_col].to_numpy(dtype=float)
    X_parts = [np.ones(len(df)), df[x_col].to_numpy(dtype=float)]
    for col in controls:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
        for dummy_col in dummies.columns:
            X_parts.append(dummies[dummy_col].to_numpy(dtype=float))
    X = np.column_stack(X_parts)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        try:
            b = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
            slopes.append(float(b[1]))
        except np.linalg.LinAlgError:
            continue
    lo, hi = np.percentile(slopes, [2.5, 97.5]) if slopes else (float("nan"), float("nan"))
    return float(beta[1]), float(lo), float(hi), r2


def summarize_correlations(df: pd.DataFrame) -> None:
    rows = []
    groups: list[tuple[str, str | None, pd.DataFrame]] = [("overall", None, df)]
    groups.extend(("dataset", label, g) for label, g in df.groupby("dataset_label"))
    groups.extend(("pos", label, g) for label, g in df.groupby("pos"))
    groups.extend(("target_n_tokens", str(label), g) for label, g in df.groupby("target_n_tokens"))

    for group_type, group_value, g in groups:
        rows.append(
            {
                "group_type": group_type,
                "group_value": group_value or "all",
                "n": len(g),
                "mean_human_si_rate": g["human_si_rate"].mean(),
                "mean_target_logprob": g["target_logprob"].mean(),
                "pearson_logprob": pearson(g["target_logprob"], g["human_si_rate"]),
                "spearman_logprob": spearman(g["target_logprob"], g["human_si_rate"]),
                "pearson_prob": pearson(g["target_prob"], g["human_si_rate"]),
                "spearman_prob": spearman(g["target_prob"], g["human_si_rate"]),
            }
        )

    pd.DataFrame(rows).to_csv(TABLE_DIR / "hu_qwen_correlation_summary.csv", index=False)

    regression_rows = []
    for controls in [[], ["dataset"], ["dataset", "pos"], ["dataset", "pos", "target_n_tokens"]]:
        slope, lo, hi, r2 = bootstrap_slope_ci(
            df,
            x_col="target_logprob",
            y_col="human_si_rate",
            controls=controls,
        )
        regression_rows.append(
            {
                "predictor": "target_logprob",
                "controls": "+".join(controls) if controls else "none",
                "slope": slope,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "r2": r2,
            }
        )
        slope, lo, hi, r2 = bootstrap_slope_ci(
            df,
            x_col="target_prob",
            y_col="human_si_rate",
            controls=controls,
        )
        regression_rows.append(
            {
                "predictor": "target_prob",
                "controls": "+".join(controls) if controls else "none",
                "slope": slope,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "r2": r2,
            }
        )
    pd.DataFrame(regression_rows).to_csv(TABLE_DIR / "hu_qwen_regression_summary.csv", index=False)


def make_bins(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    binned = df.copy()
    binned["logprob_quintile"] = pd.qcut(
        binned["target_logprob"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop"
    )
    quintiles = (
        binned.groupby("logprob_quintile", observed=True)
        .agg(
            n=("human_si_rate", "size"),
            mean_target_logprob=("target_logprob", "mean"),
            min_target_logprob=("target_logprob", "min"),
            max_target_logprob=("target_logprob", "max"),
            mean_human_si_rate=("human_si_rate", "mean"),
            sem_human_si_rate=("human_si_rate", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
        )
        .reset_index()
    )
    quintiles.to_csv(TABLE_DIR / "hu_qwen_logprob_quintiles.csv", index=False)

    rows = []
    for dataset_label, g in df.groupby("dataset_label"):
        tertile_labels = ["low", "mid", "high"]
        tmp = g.copy()
        tmp["expectedness_tertile"] = pd.qcut(
            tmp["target_logprob"], 3, labels=tertile_labels, duplicates="drop"
        )
        for bin_name, h in tmp.groupby("expectedness_tertile", observed=True):
            rows.append(
                {
                    "dataset_label": dataset_label,
                    "expectedness_tertile": str(bin_name),
                    "n": len(h),
                    "mean_target_logprob": h["target_logprob"].mean(),
                    "mean_human_si_rate": h["human_si_rate"].mean(),
                    "sem_human_si_rate": h["human_si_rate"].std(ddof=1) / math.sqrt(len(h)),
                }
            )
    tertiles = pd.DataFrame(rows)
    tertiles.to_csv(TABLE_DIR / "hu_qwen_within_dataset_tertiles.csv", index=False)
    return quintiles, tertiles


def lin_map(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) / (src_max - src_min) * (dst_max - dst_min)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;} .title{font-size:16px;font-weight:bold;} .axis{stroke:#333;stroke-width:1.2;} .grid{stroke:#ddd;stroke-width:0.8;} .small{font-size:10px;} .label{font-size:13px;}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def add_axes(lines: list[str], x0: int, y0: int, plot_w: int, plot_h: int) -> None:
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}"/>')


def add_y_grid(lines: list[str], x0: int, y0: int, plot_w: int, plot_h: int, ticks: Iterable[float]) -> None:
    for t in ticks:
        y = lin_map(t, 0, 1, y0 + plot_h, y0)
        lines.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text class="small" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{t:.1f}</text>')


def write_scatter(df: pd.DataFrame) -> None:
    width, height = 820, 560
    x0, y0, plot_w, plot_h = 75, 55, 560, 410
    x_min = math.floor(df["target_logprob"].min())
    x_max = math.ceil(df["target_logprob"].max())
    y_min, y_max = 0, 1
    lines = svg_header(width, height)
    lines.append('<text class="title" x="75" y="28">Qwen expectedness of stronger alternatives predicts scalar inference rates</text>')
    add_y_grid(lines, x0, y0, plot_w, plot_h, [0, .2, .4, .6, .8, 1.0])
    for t in range(int(x_min), int(x_max) + 1, 2):
        x = lin_map(t, x_min, x_max, x0, x0 + plot_w)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + plot_h}"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{y0 + plot_h + 18}" text-anchor="middle">{t}</text>')
    add_axes(lines, x0, y0, plot_w, plot_h)

    for dataset_label, g in df.groupby("dataset_label"):
        color = DATASET_COLORS.get(dataset_label, "#777")
        for _, row in g.iterrows():
            x = lin_map(row["target_logprob"], x_min, x_max, x0, x0 + plot_w)
            y = lin_map(row["human_si_rate"], y_min, y_max, y0 + plot_h, y0)
            title = escape(f'{row["scale_id"]}: human={row["human_si_rate"]:.2f}, logp={row["target_logprob"]:.2f}')
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}" opacity="0.65"><title>{title}</title></circle>')

    slope, intercept = np.polyfit(df["target_logprob"], df["human_si_rate"], 1)
    x_a, x_b = x_min, x_max
    y_a, y_b = slope * x_a + intercept, slope * x_b + intercept
    lines.append(
        f'<line x1="{lin_map(x_a, x_min, x_max, x0, x0 + plot_w):.1f}" y1="{lin_map(y_a, y_min, y_max, y0 + plot_h, y0):.1f}" '
        f'x2="{lin_map(x_b, x_min, x_max, x0, x0 + plot_w):.1f}" y2="{lin_map(y_b, y_min, y_max, y0 + plot_h, y0):.1f}" stroke="#111" stroke-width="2.2"/>'
    )

    lines.append(f'<text class="label" x="{x0 + plot_w / 2}" y="{height - 45}" text-anchor="middle">Qwen log P(stronger alternative | context)</text>')
    lines.append(f'<text class="label" transform="translate(18 {y0 + plot_h / 2}) rotate(-90)" text-anchor="middle">Human scalar inference rate</text>')
    pear = pearson(df["target_logprob"], df["human_si_rate"])
    spear = spearman(df["target_logprob"], df["human_si_rate"])
    lines.append(f'<text x="{x0 + 12}" y="{y0 + 20}">Pearson r = {pear:.2f}; Spearman ρ = {spear:.2f}; n = {len(df)}</text>')

    lx, ly = 660, 90
    lines.append(f'<text class="label" x="{lx}" y="{ly - 22}">Source dataset</text>')
    for i, label in enumerate(DATASET_ORDER):
        y = ly + i * 22
        lines.append(f'<circle cx="{lx}" cy="{y}" r="5" fill="{DATASET_COLORS[label]}" opacity="0.75"/>')
        lines.append(f'<text x="{lx + 12}" y="{y + 4}">{escape(label)}</text>')
    lines.append('</svg>')
    (FIGURE_DIR / "hu_qwen_scatter.svg").write_text("\n".join(lines), encoding="utf-8")


def write_quintile_plot(quintiles: pd.DataFrame) -> None:
    width, height = 700, 500
    x0, y0, plot_w, plot_h = 80, 55, 540, 345
    x_min = float(quintiles["mean_target_logprob"].min()) - .5
    x_max = float(quintiles["mean_target_logprob"].max()) + .5
    lines = svg_header(width, height)
    lines.append('<text class="title" x="80" y="28">Binned summary of Hu-style scalar items</text>')
    add_y_grid(lines, x0, y0, plot_w, plot_h, [0, .2, .4, .6, .8, 1.0])
    add_axes(lines, x0, y0, plot_w, plot_h)
    points = []
    for _, row in quintiles.iterrows():
        x = lin_map(row["mean_target_logprob"], x_min, x_max, x0, x0 + plot_w)
        y = lin_map(row["mean_human_si_rate"], 0, 1, y0 + plot_h, y0)
        err = row["sem_human_si_rate"] * plot_h
        points.append((x, y))
        lines.append(f'<line x1="{x:.1f}" y1="{y - err:.1f}" x2="{x:.1f}" y2="{y + err:.1f}" stroke="#111"/>')
        lines.append(f'<line x1="{x - 5:.1f}" y1="{y - err:.1f}" x2="{x + 5:.1f}" y2="{y - err:.1f}" stroke="#111"/>')
        lines.append(f'<line x1="{x - 5:.1f}" y1="{y + err:.1f}" x2="{x + 5:.1f}" y2="{y + err:.1f}" stroke="#111"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#111"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle">n={int(row["n"])}</text>')
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#111" stroke-width="1.5"/>')
    for t in np.linspace(math.floor(x_min), math.ceil(x_max), 6):
        x = lin_map(float(t), x_min, x_max, x0, x0 + plot_w)
        lines.append(f'<text class="small" x="{x:.1f}" y="{y0 + plot_h + 18}" text-anchor="middle">{t:.1f}</text>')
    lines.append(f'<text class="label" x="{x0 + plot_w / 2}" y="{height - 45}" text-anchor="middle">Mean Qwen log P(stronger alternative | context) in bin</text>')
    lines.append(f'<text class="label" transform="translate(18 {y0 + plot_h / 2}) rotate(-90)" text-anchor="middle">Mean human scalar inference rate</text>')
    lines.append('</svg>')
    (FIGURE_DIR / "hu_qwen_logprob_quintiles.svg").write_text("\n".join(lines), encoding="utf-8")


def write_tertile_plot(tertiles: pd.DataFrame) -> None:
    width, height = 850, 520
    x0, y0, plot_w, plot_h = 85, 60, 690, 335
    lines = svg_header(width, height)
    lines.append('<text class="title" x="85" y="30">Within-dataset bins of Qwen expectedness for stronger alternatives</text>')
    add_y_grid(lines, x0, y0, plot_w, plot_h, [0, .2, .4, .6, .8, 1.0])
    add_axes(lines, x0, y0, plot_w, plot_h)
    group_w = plot_w / len(DATASET_ORDER)
    bar_w = group_w * 0.22
    offsets = {"low": -bar_w * 1.15, "mid": 0, "high": bar_w * 1.15}
    for i, label in enumerate(DATASET_ORDER):
        center = x0 + group_w * (i + .5)
        sub = tertiles[tertiles["dataset_label"] == label].set_index("expectedness_tertile")
        for bin_name in ["low", "mid", "high"]:
            if bin_name not in sub.index:
                continue
            row = sub.loc[bin_name]
            x = center + offsets[bin_name]
            y = lin_map(row["mean_human_si_rate"], 0, 1, y0 + plot_h, y0)
            h = y0 + plot_h - y
            color = BIN_COLORS[bin_name]
            lines.append(f'<rect x="{x - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.9"/>')
            err = row["sem_human_si_rate"] * plot_h
            lines.append(f'<line x1="{x:.1f}" y1="{y - err:.1f}" x2="{x:.1f}" y2="{y + err:.1f}" stroke="#111"/>')
            lines.append(f'<line x1="{x - 4:.1f}" y1="{y - err:.1f}" x2="{x + 4:.1f}" y2="{y - err:.1f}" stroke="#111"/>')
            lines.append(f'<line x1="{x - 4:.1f}" y1="{y + err:.1f}" x2="{x + 4:.1f}" y2="{y + err:.1f}" stroke="#111"/>')
        display_label = label.replace("Pankratz & van Tiel", "Pankratz &\nvan Tiel").replace("van Tiel et al.", "van Tiel\net al.")
        parts = display_label.split("\n")
        for j, part in enumerate(parts):
            lines.append(f'<text class="small" x="{center:.1f}" y="{y0 + plot_h + 22 + j*13}" text-anchor="middle">{escape(part)}</text>')
    lx, ly = 635, 92
    lines.append(f'<text class="label" x="{lx}" y="{ly - 24}">Qwen expectedness</text>')
    for i, bin_name in enumerate(["low", "mid", "high"]):
        y = ly + i * 22
        lines.append(f'<rect x="{lx}" y="{y - 10}" width="12" height="12" fill="{BIN_COLORS[bin_name]}"/>')
        lines.append(f'<text x="{lx + 18}" y="{y}">{bin_name}</text>')
    lines.append(f'<text class="label" transform="translate(18 {y0 + plot_h / 2}) rotate(-90)" text-anchor="middle">Mean human scalar inference rate</text>')
    lines.append('</svg>')
    (FIGURE_DIR / "hu_qwen_within_dataset_tertiles.svg").write_text("\n".join(lines), encoding="utf-8")



def pdf_color(hex_color: str):
    return HexColor(hex_color)


def draw_pdf_y_grid(c: canvas.Canvas, x0: float, y0: float, plot_w: float, plot_h: float, ticks: Iterable[float]) -> None:
    c.setStrokeColorRGB(0.86, 0.86, 0.86)
    c.setLineWidth(0.6)
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    for tick in ticks:
        y = lin_map(tick, 0, 1, y0, y0 + plot_h)
        c.line(x0, y, x0 + plot_w, y)
        c.drawRightString(x0 - 8, y - 3, f"{tick:.1f}")


def draw_pdf_axes(c: canvas.Canvas, x0: float, y0: float, plot_w: float, plot_h: float) -> None:
    c.setStrokeColorRGB(0.15, 0.15, 0.15)
    c.setLineWidth(1.1)
    c.line(x0, y0, x0, y0 + plot_h)
    c.line(x0, y0, x0 + plot_w, y0)


def draw_pdf_y_label(c: canvas.Canvas, text: str, x: float, y: float) -> None:
    c.saveState()
    c.translate(x, y)
    c.rotate(90)
    c.setFont("Helvetica", 11)
    c.setFillColor(black)
    c.drawCentredString(0, 0, text)
    c.restoreState()


def write_scatter_pdf(df: pd.DataFrame) -> None:
    width, height = 820, 560
    x0, y0, plot_w, plot_h = 75, 90, 560, 390
    x_min = math.floor(df["target_logprob"].min())
    x_max = math.ceil(df["target_logprob"].max())
    y_min, y_max = 0, 1
    out = FIGURE_DIR / "hu_qwen_scatter.pdf"
    c = canvas.Canvas(str(out), pagesize=(width, height))

    c.setFont("Helvetica-Bold", 15)
    c.drawString(x0, height - 30, "Qwen expectedness of stronger alternatives predicts scalar inference rates")
    draw_pdf_y_grid(c, x0, y0, plot_w, plot_h, [0, .2, .4, .6, .8, 1.0])
    c.setStrokeColorRGB(0.86, 0.86, 0.86)
    c.setLineWidth(0.6)
    c.setFont("Helvetica", 9)
    for tick in range(int(x_min), int(x_max) + 1, 2):
        x = lin_map(tick, x_min, x_max, x0, x0 + plot_w)
        c.line(x, y0, x, y0 + plot_h)
        c.drawCentredString(x, y0 - 18, str(tick))
    draw_pdf_axes(c, x0, y0, plot_w, plot_h)

    try:
        c.setFillAlpha(0.68)
    except AttributeError:
        pass
    for dataset_label, g in df.groupby("dataset_label"):
        c.setFillColor(pdf_color(DATASET_COLORS.get(dataset_label, "#777777")))
        for _, row in g.iterrows():
            x = lin_map(row["target_logprob"], x_min, x_max, x0, x0 + plot_w)
            y = lin_map(row["human_si_rate"], y_min, y_max, y0, y0 + plot_h)
            c.circle(x, y, 3.2, stroke=0, fill=1)
    try:
        c.setFillAlpha(1)
    except AttributeError:
        pass

    slope, intercept = np.polyfit(df["target_logprob"], df["human_si_rate"], 1)
    x_a, x_b = x_min, x_max
    y_a, y_b = slope * x_a + intercept, slope * x_b + intercept
    c.setStrokeColorRGB(0.05, 0.05, 0.05)
    c.setLineWidth(2.0)
    c.line(
        lin_map(x_a, x_min, x_max, x0, x0 + plot_w),
        lin_map(y_a, y_min, y_max, y0, y0 + plot_h),
        lin_map(x_b, x_min, x_max, x0, x0 + plot_w),
        lin_map(y_b, y_min, y_max, y0, y0 + plot_h),
    )

    pear = pearson(df["target_logprob"], df["human_si_rate"])
    spear = spearman(df["target_logprob"], df["human_si_rate"])
    c.setFont("Helvetica", 11)
    c.setFillColor(black)
    c.drawString(x0 + 12, y0 + plot_h - 20, f"Pearson r = {pear:.2f}; Spearman rho = {spear:.2f}; n = {len(df)}")
    c.setFont("Helvetica", 11)
    c.drawCentredString(x0 + plot_w / 2, 38, "Qwen log P(stronger alternative | context)")
    draw_pdf_y_label(c, "Human scalar inference rate", 22, y0 + plot_h / 2)

    lx, ly = 660, height - 95
    c.setFont("Helvetica", 11)
    c.drawString(lx, ly + 22, "Source dataset")
    for i, label in enumerate(DATASET_ORDER):
        y = ly - i * 22
        c.setFillColor(pdf_color(DATASET_COLORS[label]))
        c.circle(lx + 5, y + 3, 4.5, stroke=0, fill=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 9)
        c.drawString(lx + 17, y, label)
    c.showPage()
    c.save()


def write_quintile_plot_pdf(quintiles: pd.DataFrame) -> None:
    width, height = 700, 500
    x0, y0, plot_w, plot_h = 80, 85, 540, 330
    x_min = float(quintiles["mean_target_logprob"].min()) - .5
    x_max = float(quintiles["mean_target_logprob"].max()) + .5
    out = FIGURE_DIR / "hu_qwen_logprob_quintiles.pdf"
    c = canvas.Canvas(str(out), pagesize=(width, height))

    c.setFont("Helvetica-Bold", 15)
    c.drawString(x0, height - 30, "Binned summary of Hu-style scalar items")
    draw_pdf_y_grid(c, x0, y0, plot_w, plot_h, [0, .2, .4, .6, .8, 1.0])
    draw_pdf_axes(c, x0, y0, plot_w, plot_h)

    points = []
    c.setStrokeColor(black)
    c.setFillColor(black)
    c.setLineWidth(1.0)
    c.setFont("Helvetica", 8)
    for _, row in quintiles.iterrows():
        x = lin_map(row["mean_target_logprob"], x_min, x_max, x0, x0 + plot_w)
        y = lin_map(row["mean_human_si_rate"], 0, 1, y0, y0 + plot_h)
        err = row["sem_human_si_rate"] * plot_h
        points.append((x, y))
        c.line(x, y - err, x, y + err)
        c.line(x - 5, y - err, x + 5, y - err)
        c.line(x - 5, y + err, x + 5, y + err)
        c.circle(x, y, 5.0, stroke=0, fill=1)
        c.drawCentredString(x, y + 13, f"n={int(row['n'])}")
    c.setLineWidth(1.4)
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        c.line(x1, y1, x2, y2)
    c.setFont("Helvetica", 9)
    for tick in np.linspace(math.floor(x_min), math.ceil(x_max), 6):
        x = lin_map(float(tick), x_min, x_max, x0, x0 + plot_w)
        c.drawCentredString(x, y0 - 18, f"{tick:.1f}")
    c.setFont("Helvetica", 11)
    c.drawCentredString(x0 + plot_w / 2, 35, "Mean Qwen log P(stronger alternative | context) in bin")
    draw_pdf_y_label(c, "Mean human scalar inference rate", 22, y0 + plot_h / 2)
    c.showPage()
    c.save()


def write_tertile_plot_pdf(tertiles: pd.DataFrame) -> None:
    width, height = 850, 520
    x0, y0, plot_w, plot_h = 85, 100, 690, 320
    out = FIGURE_DIR / "hu_qwen_within_dataset_tertiles.pdf"
    c = canvas.Canvas(str(out), pagesize=(width, height))

    c.setFont("Helvetica-Bold", 15)
    c.drawString(x0, height - 30, "Within-dataset bins of Qwen expectedness for stronger alternatives")
    draw_pdf_y_grid(c, x0, y0, plot_w, plot_h, [0, .2, .4, .6, .8, 1.0])
    draw_pdf_axes(c, x0, y0, plot_w, plot_h)

    group_w = plot_w / len(DATASET_ORDER)
    bar_w = group_w * 0.22
    offsets = {"low": -bar_w * 1.15, "mid": 0, "high": bar_w * 1.15}
    for i, label in enumerate(DATASET_ORDER):
        center = x0 + group_w * (i + .5)
        sub = tertiles[tertiles["dataset_label"] == label].set_index("expectedness_tertile")
        for bin_name in ["low", "mid", "high"]:
            if bin_name not in sub.index:
                continue
            row = sub.loc[bin_name]
            x = center + offsets[bin_name]
            y = lin_map(row["mean_human_si_rate"], 0, 1, y0, y0 + plot_h)
            h = y - y0
            c.setFillColor(pdf_color(BIN_COLORS[bin_name]))
            c.rect(x - bar_w / 2, y0, bar_w, h, stroke=0, fill=1)
            err = row["sem_human_si_rate"] * plot_h
            c.setStrokeColor(black)
            c.setLineWidth(0.8)
            c.line(x, y - err, x, y + err)
            c.line(x - 4, y - err, x + 4, y - err)
            c.line(x - 4, y + err, x + 4, y + err)
        display_label = label.replace("Pankratz & van Tiel", "Pankratz &\nvan Tiel").replace("van Tiel et al.", "van Tiel\net al.")
        c.setFillColor(black)
        c.setFont("Helvetica", 8.5)
        for j, part in enumerate(display_label.split("\n")):
            c.drawCentredString(center, y0 - 20 - j * 12, part)

    lx, ly = 638, height - 90
    c.setFont("Helvetica", 11)
    c.setFillColor(black)
    c.drawString(lx, ly + 24, "Qwen expectedness")
    for i, bin_name in enumerate(["low", "mid", "high"]):
        y = ly - i * 22
        c.setFillColor(pdf_color(BIN_COLORS[bin_name]))
        c.rect(lx, y - 8, 12, 12, stroke=0, fill=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 10)
        c.drawString(lx + 18, y - 5, bin_name)
    draw_pdf_y_label(c, "Mean human scalar inference rate", 22, y0 + plot_h / 2)
    c.showPage()
    c.save()

def latex_escape_text(text: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def write_latex_summary_table(df: pd.DataFrame) -> None:
    rows = []
    for label in ["Overall", *DATASET_ORDER]:
        g = df if label == "Overall" else df[df["dataset_label"] == label]
        rows.append(
            {
                "Dataset": label,
                "N": len(g),
                "Pearson $r$": pearson(g["target_logprob"], g["human_si_rate"]),
                "Spearman $\\rho$": spearman(g["target_logprob"], g["human_si_rate"]),
                "Mean SI": g["human_si_rate"].mean(),
                "Mean log $P$": g["target_logprob"].mean(),
            }
        )
    table = pd.DataFrame(rows)
    latex_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Dataset & $N$ & Pearson $r$ & Spearman $\rho$ & Mean SI & Mean log $P$ \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        latex_lines.append(
            f"{latex_escape_text(row['Dataset'])} & {int(row['N'])} & {row['Pearson $r$']:.2f} & {row['Spearman $\\rho$']:.2f} & {row['Mean SI']:.2f} & {row['Mean log $P$']:.2f} \\\\"  # noqa: E501
        )
    latex_lines.extend([r"\hline", r"\end{tabular}"])
    (TABLE_DIR / "hu_qwen_correlation_summary.tex").write_text("\n".join(latex_lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_scores()
    df.to_csv(ANALYSIS_PATH, index=False)
    summarize_correlations(df)
    quintiles, tertiles = make_bins(df)
    write_scatter_pdf(df)
    write_quintile_plot_pdf(quintiles)
    write_tertile_plot_pdf(tertiles)
    write_latex_summary_table(df)

    print(f"Wrote analysis table: {ANALYSIS_PATH}")
    print("Wrote tables:")
    for path in sorted(TABLE_DIR.glob("hu_qwen_*")):
        print(f"  {path}")
    print("Wrote figures:")
    for path in sorted(FIGURE_DIR.glob("hu_qwen_*.pdf")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
