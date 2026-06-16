#!/usr/bin/env python3
"""Build exact-strong Qwen prompts for Hu et al. cross-scale items."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data_processed" / "hu_cross_scale_items.csv"
TEST_SUITE_ROOT = (
    PROJECT_ROOT
    / "data_raw"
    / "hu_et_al_2023"
    / "cross-scale"
    / "test_suites"
)
OUTPUT_PATH = PROJECT_ROOT / "stimuli" / "qwen_exact_strong_prompts.csv"


def scale_id_to_filename(scale_id: str) -> str:
    return scale_id.replace("/", "_") + ".json"


def test_suite_path(row: pd.Series) -> Path:
    filename = scale_id_to_filename(row["scale_id"])
    if row["dataset"] == "vt16":
        return TEST_SUITE_ROOT / row["dataset"] / f"template{row['template_id']}" / filename
    return TEST_SUITE_ROOT / row["dataset"] / filename


def normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    return text


def join_region_contents(regions: list[dict[str, object]], region_numbers: list[int]) -> str:
    by_number = {int(r["region_number"]): str(r["content"]) for r in regions}
    parts = [by_number.get(n, "") for n in region_numbers]
    return normalize_spaces(" ".join(p for p in parts if p != ""))


def find_strong_test_suite_item(row: pd.Series) -> dict[str, object] | None:
    path = test_suite_path(row)
    if not path.exists():
        return None

    suite = json.loads(path.read_text(encoding="utf-8"))
    candidates = {
        str(row["strong"]).strip().lower(),
        str(row["strong_surface"]).strip().lower(),
    }

    for item in suite["items"]:
        condition = item["conditions"][0]
        regions = condition["regions"]
        by_number = {int(r["region_number"]): str(r["content"]) for r in regions}
        alternative = by_number.get(5, "").strip()
        if alternative.lower() in candidates:
            return {
                "hu_test_suite_path": str(path.relative_to(PROJECT_ROOT)),
                "hu_test_suite_item_number": item["item_number"],
                "regions": regions,
                "target_text": alternative,
            }
    return None


def split_from_scalar_construction(row: pd.Series) -> dict[str, object]:
    construction = str(row["scalar_construction"])
    candidates = [
        str(row["strong_surface"]).strip(),
        str(row["strong"]).strip(),
    ]
    lowered = construction.lower()

    for candidate in candidates:
        if not candidate:
            continue
        idx = lowered.rfind(candidate.lower())
        if idx >= 0:
            target_text = construction[idx : idx + len(candidate)]
            return {
                "context_text": normalize_spaces(construction[:idx]),
                "target_text": target_text,
                "continuation_text": " " + target_text,
                "suffix_text": construction[idx + len(candidate) :],
                "full_text": normalize_spaces(construction),
                "prompt_source": "scalar_construction_fallback",
                "hu_test_suite_path": "",
                "hu_test_suite_item_number": "",
                "prompt_builder_note": (
                    "No Hu test suite item was available; split scalar_construction "
                    "at the final strong scalemate occurrence."
                ),
            }

    raise ValueError(
        f"Could not find strong scalemate for {row['dataset']} {row['scale_id']} "
        f"in scalar construction: {construction}"
    )


def build_prompt_row(row: pd.Series) -> dict[str, object]:
    suite_item = find_strong_test_suite_item(row)
    if suite_item is not None:
        regions = suite_item["regions"]
        context_text = join_region_contents(regions, [1, 2, 3, 4])
        target_text = str(suite_item["target_text"])
        suffix_text = join_region_contents(regions, [6])
        full_text = join_region_contents(regions, [1, 2, 3, 4, 5, 6])
        prompt_bits = {
            "context_text": context_text,
            "target_text": target_text,
            "continuation_text": " " + target_text,
            "suffix_text": suffix_text,
            "full_text": full_text,
            "prompt_source": "hu_test_suite_region_5",
            "hu_test_suite_path": suite_item["hu_test_suite_path"],
            "hu_test_suite_item_number": suite_item["hu_test_suite_item_number"],
            "prompt_builder_note": "Target is region 5 (Alternative) from Hu SyntaxGym suite.",
        }
    else:
        prompt_bits = split_from_scalar_construction(row)

    source_cols = {
        "dataset": row["dataset"],
        "dataset_label": row["dataset_label"],
        "item_id": row["item_id"],
        "scale_id": row["scale_id"],
        "weak": row["weak"],
        "strong": row["strong"],
        "weak_surface": row["weak_surface"],
        "strong_surface": row["strong_surface"],
        "pos": row["pos"],
        "template_id": row["template_id"],
        "scalar_construction": row["scalar_construction"],
        "human_si_rate": row["human_si_rate"],
        "has_hu_test_suite": row["has_hu_test_suite"],
        "source_file": row["source_file"],
    }
    prompt_id = (
        f"{row['dataset']}__{row['item_id']}__template_{row['template_id']}__"
        f"{str(row['scale_id']).replace('/', '_')}"
    )
    return {
        "prompt_id": prompt_id,
        **source_cols,
        **prompt_bits,
        "score_target": "exact_strong_scalemate",
        "score_family": "target_word_or_phrase_logprob",
        "context_has_trailing_space": False,
        "continuation_has_leading_space": True,
    }


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    rows = [build_prompt_row(row) for _, row in df.iterrows()]
    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(out)} rows to {OUTPUT_PATH}")
    print(out.groupby(["dataset", "prompt_source"]).size().to_string())


if __name__ == "__main__":
    main()
