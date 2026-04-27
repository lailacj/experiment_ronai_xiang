#!/usr/bin/env python3
"""Build a scoring table for Ronai & Xiang Experiment 3 stimuli.

Experiment 3 contains a single condition:

- Eonly: the Experiment 1 sentence is modified to include the focus particle
  "only"

Because the shared files do not contain the exact original Experiment 3
materials, this script reconstructs a best-effort stimulus table from the
Experiment 1 items.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_experiment_1_stimuli import PROJECT_ROOT, PROMPTS_ROOT, build_rows, write_rows


DEFAULT_INPUT = PROJECT_ROOT / "ronai_xiang_data" / "items.csv"
DEFAULT_OUTPUT = PROMPTS_ROOT / "experiment_3_scoring_stimuli.csv"

# Most items are reconstructed by placing "only" immediately before the weaker
# scalar term. A small number benefit from custom prefix/suffix overrides to
# avoid especially awkward output.
ONLY_PREFIX_OVERRIDES = {
    19: "There is only water ",
    29: "Jimmy writes only books ",
    59: "The rehearsal only went ",
}

ONLY_SUFFIX_OVERRIDES = {
}


def build_condition_rows(input_path: Path) -> list[dict[str, str]]:
    exp1_rows = build_rows(input_path)
    output_rows: list[dict[str, str]] = []

    for base_row in exp1_rows:
        item_id = int(base_row["item_id"])
        weaker_surface = base_row["weaker_surface"]
        stronger_surface = base_row["stronger_surface_guess"]
        weaker_lemma = base_row["weaker_lemma"]
        stronger_lemma = base_row["stronger_lemma"]
        suffix = ONLY_SUFFIX_OVERRIDES.get(item_id, base_row["suffix_after_target"])
        exp1_prefix = base_row["prompt_prefix"]

        prompt_prefix = ONLY_PREFIX_OVERRIDES.get(item_id, f"{exp1_prefix}only ")
        insertion_method = "insert_only_before_target"
        if item_id in ONLY_PREFIX_OVERRIDES and item_id in ONLY_SUFFIX_OVERRIDES:
            insertion_method = "prefix_and_suffix_override"
        elif item_id in ONLY_PREFIX_OVERRIDES:
            insertion_method = "prefix_override"
        elif item_id in ONLY_SUFFIX_OVERRIDES:
            insertion_method = "suffix_override"
        needs_manual_review = "yes" if item_id in ONLY_PREFIX_OVERRIDES or item_id in ONLY_SUFFIX_OVERRIDES else "no"

        weaker_sentence = f"{prompt_prefix}{weaker_surface}{suffix}"
        stronger_sentence = f"{prompt_prefix}{stronger_surface}{suffix}"

        output_rows.append(
            {
                "item_id": str(item_id),
                "condition": "Eonly",
                "sub_experiment": "Only",
                "weaker_lemma": weaker_lemma,
                "weaker_surface": weaker_surface,
                "stronger_lemma": stronger_lemma,
                "stronger_surface_guess": stronger_surface,
                "sentence_frame_experiment_1": base_row["sentence_frame_experiment_1"],
                "sentence_frame_experiment_3_guess": weaker_sentence,
                "prompt_prefix_for_qwen": prompt_prefix,
                "target_word_seen_by_participants": weaker_surface,
                "suffix_after_target": suffix,
                "prompt_with_blank": f"{prompt_prefix}___{suffix}",
                "stronger_sentence_guess": stronger_sentence,
                "only_insertion_method": insertion_method,
                "reconstruction_note": (
                    "Experiment 3 reconstructed from Experiment 1 by adding "
                    "the focus particle only."
                ),
                "needs_manual_review": needs_manual_review,
            }
        )

    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to items.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the reconstructed Experiment 3 scoring table",
    )
    args = parser.parse_args()

    rows = build_condition_rows(args.input)
    write_rows(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
