#!/usr/bin/env python3
"""Normalize Hu et al. cross-scale human datasets.

The source datasets use different column names and SI-rate scales. This script
standardizes them into one analysis table with human SI rates on a 0-1 scale.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data_raw" / "hu_et_al_2023" / "cross-scale"
HUMAN_ROOT = RAW_ROOT / "human_data"
TEST_SUITE_ROOT = RAW_ROOT / "test_suites"
OUTPUT_PATH = PROJECT_ROOT / "data_processed" / "hu_cross_scale_items.csv"


DATASET_LABELS = {
    "g18": "Gotzner et al. (2018)",
    "pvt21": "Pankratz & van Tiel (2021)",
    "rx22": "Ronai & Xiang (2022)",
    "vt16": "van Tiel et al. (2016)",
}


def scale_id_to_filename(scale_id: str) -> str:
    return scale_id.replace("/", "_") + ".json"


def has_test_suite(dataset: str, scale_id: str, template_id: str = "default") -> bool:
    filename = scale_id_to_filename(scale_id)
    if dataset == "vt16":
        path = TEST_SUITE_ROOT / dataset / f"template{template_id}" / filename
    else:
        path = TEST_SUITE_ROOT / dataset / filename
    return path.exists()


def row_base(
    *,
    dataset: str,
    item_id: str,
    scale_id: str,
    weak: str,
    strong: str,
    weak_surface: str,
    strong_surface: str,
    pos: str,
    template_id: str,
    scalar_construction: str,
    human_si_rate: float,
    human_si_rate_original: float,
    human_si_rate_source_column: str,
    source_file: str,
    notes: str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "item_id": item_id,
        "scale_id": scale_id,
        "weak": weak,
        "strong": strong,
        "weak_surface": weak_surface,
        "strong_surface": strong_surface,
        "pos": pos,
        "template_id": template_id,
        "scalar_construction": scalar_construction,
        "human_si_rate": human_si_rate,
        "human_si_rate_original": human_si_rate_original,
        "human_si_rate_source_column": human_si_rate_source_column,
        "source_file": source_file,
        "has_hu_test_suite": has_test_suite(dataset, scale_id, template_id),
        "notes": notes,
    }


def normalize_g18() -> list[dict[str, object]]:
    source_file = "cross-scale/human_data/g18.csv"
    df = pd.read_csv(HUMAN_ROOT / "g18.csv")
    rows = []
    for i, row in df.iterrows():
        rows.append(
            row_base(
                dataset="g18",
                item_id=f"g18_{i + 1:03d}",
                scale_id=row["scale_id"],
                weak=row["weak_adj"],
                strong=row["strong_adj"],
                weak_surface=row["weak_adj"],
                strong_surface=row["strong_adj"],
                pos=row["pos"],
                template_id="default",
                scalar_construction=row["scalar_construction"],
                human_si_rate=float(row["si_rate"]),
                human_si_rate_original=float(row["si_rate"]),
                human_si_rate_source_column="si_rate",
                source_file=source_file,
            )
        )
    return rows


def normalize_pvt21() -> list[dict[str, object]]:
    source_file = "cross-scale/human_data/pvt21.csv"
    df = pd.read_csv(HUMAN_ROOT / "pvt21.csv")
    rows = []
    for i, row in df.iterrows():
        scale_id = f"{row['weak_adj']}/{row['strong_adj']}"
        rows.append(
            row_base(
                dataset="pvt21",
                item_id=f"pvt21_{i + 1:03d}",
                scale_id=scale_id,
                weak=row["weak_adj"],
                strong=row["strong_adj"],
                weak_surface=row["weak_adj"],
                strong_surface=row["strong_adj"],
                pos="adj",
                template_id="default",
                scalar_construction=row["combo_sentence"],
                human_si_rate=float(row["SI_rate"]),
                human_si_rate_original=float(row["SI_rate"]),
                human_si_rate_source_column="SI_rate",
                source_file=source_file,
            )
        )
    return rows


def normalize_rx22() -> list[dict[str, object]]:
    source_file = "cross-scale/human_data/rx22.csv"
    df = pd.read_csv(HUMAN_ROOT / "rx22.csv")
    exp1 = pd.read_csv(HUMAN_ROOT / "rx22_processed_exp1.csv")
    constructions = (
        exp1[["Item", "scalar_construction"]]
        .drop_duplicates(subset=["Item"])
        .set_index("Item")["scalar_construction"]
    )
    rows = []
    for _, row in df.iterrows():
        item = int(row["Item"])
        scale_id = f"{row['Weaker']}/{row['Stronger']}"
        scalar_construction = constructions.get(item, "")
        notes = ""
        if not scalar_construction:
            notes = "No scalar construction found in rx22_processed_exp1.csv"
        rows.append(
            row_base(
                dataset="rx22",
                item_id=f"rx22_{item:03d}",
                scale_id=scale_id,
                weak=row["Weaker"],
                strong=row["Stronger"],
                weak_surface=row["Weaker"],
                strong_surface=row["Stronger"],
                pos=row["pos"],
                template_id="default",
                scalar_construction=scalar_construction,
                human_si_rate=float(row["SI percent (Exp 1)"]) / 100.0,
                human_si_rate_original=float(row["SI percent (Exp 1)"]),
                human_si_rate_source_column="SI percent (Exp 1)",
                source_file=source_file,
                notes=notes,
            )
        )
    return rows


def normalize_vt16() -> list[dict[str, object]]:
    source_file = "cross-scale/human_data/vt16.csv"
    df = pd.read_csv(HUMAN_ROOT / "vt16.csv")
    rows = []
    for i, row in df.iterrows():
        for template_id in ["1", "2", "3"]:
            construction_col = f"nonneutral{template_id}_scalar_construction"
            rows.append(
                row_base(
                    dataset="vt16",
                    item_id=f"vt16_{i + 1:03d}_template{template_id}",
                    scale_id=row["scale_id"],
                    weak=row["weak_scalemate"],
                    strong=row["strong_scalemate"],
                    weak_surface=row["weak_inflected"],
                    strong_surface=row["strong_inflected"],
                    pos=row["pos"],
                    template_id=template_id,
                    scalar_construction=row[construction_col],
                    human_si_rate=float(row["si_nonneutral"]) / 100.0,
                    human_si_rate_original=float(row["si_nonneutral"]),
                    human_si_rate_source_column="si_nonneutral",
                    source_file=source_file,
                    notes="van Tiel non-neutral template used by Hu et al. cross-scale analysis",
                )
            )
    return rows


def main() -> None:
    rows = []
    rows.extend(normalize_g18())
    rows.extend(normalize_pvt21())
    rows.extend(normalize_rx22())
    rows.extend(normalize_vt16())

    out = pd.DataFrame(rows)
    out = out.sort_values(["dataset", "scale_id", "template_id"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(out)} rows to {OUTPUT_PATH}")
    print(out.groupby("dataset").size().to_string())
    print()
    print("Rows with Hu test suites:")
    print(out.groupby("dataset")["has_hu_test_suite"].sum().astype(int).to_string())


if __name__ == "__main__":
    main()
