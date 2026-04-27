#!/usr/bin/env python3
"""Build a scoring table for Ronai & Xiang Experiment 4 stimuli.

Experiment 4 combines:

- the strong-QUD question from Experiment 2
- the "only" answer form from Experiment 3

Because the shared files do not contain the exact original Experiment 4
materials, this script reconstructs a best-effort stimulus table by combining
the previously reconstructed Experiment 2 and Experiment 3 logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_experiment_1_stimuli import PROJECT_ROOT, PROMPTS_ROOT, build_rows, write_rows
from build_experiment_2_stimuli import build_question
from build_experiment_3_stimuli import build_condition_rows as build_experiment_3_rows


DEFAULT_INPUT = PROJECT_ROOT / "ronai_xiang_data" / "items.csv"
DEFAULT_OUTPUT = PROMPTS_ROOT / "experiment_4_scoring_stimuli.csv"


def build_condition_rows(input_path: Path) -> list[dict[str, str]]:
    exp1_rows = build_rows(input_path)
    exp3_rows = {
        int(row["item_id"]): row for row in build_experiment_3_rows(input_path)
    }
    output_rows: list[dict[str, str]] = []

    for base_row in exp1_rows:
        item_id = int(base_row["item_id"])
        exp3_row = exp3_rows[item_id]

        weaker_lemma = base_row["weaker_lemma"]
        weaker_surface = base_row["weaker_surface"]
        stronger_lemma = base_row["stronger_lemma"]
        stronger_surface = base_row["stronger_surface_guess"]
        strong_statement = base_row["stronger_sentence_guess"]
        surface_morphology = base_row["surface_morphology"]

        question_sentence, question_auxiliary, question_method = build_question(
            item_id,
            strong_statement,
            weaker_lemma=weaker_lemma,
            weaker_surface=weaker_surface,
            stronger_lemma=stronger_lemma,
            stronger_surface=stronger_surface,
            question_target_type="stronger",
            surface_morphology=surface_morphology,
        )

        answer_prefix = exp3_row["prompt_prefix_for_qwen"]
        answer_suffix = exp3_row["suffix_after_target"]
        answer_sentence = exp3_row["sentence_frame_experiment_3_guess"]
        stronger_answer_sentence = exp3_row["stronger_sentence_guess"]
        only_method = exp3_row["only_insertion_method"]

        prompt_prefix_for_qwen = f"Sue: {question_sentence}\nMary: {answer_prefix}"
        prompt_with_blank = f"Sue: {question_sentence}\nMary: {answer_prefix}___{answer_suffix}"
        dialogue_with_answer = f"Sue: {question_sentence}\nMary: {answer_sentence}"
        dialogue_with_stronger_answer = f"Sue: {question_sentence}\nMary: {stronger_answer_sentence}"

        needs_manual_review = "yes" if question_method != "aux_inversion" or only_method != "insert_only_before_target" else "no"

        output_rows.append(
            {
                "item_id": str(item_id),
                "condition": "Eonlystrong",
                "sub_experiment": "QUD+only",
                "weaker_lemma": weaker_lemma,
                "weaker_surface": weaker_surface,
                "stronger_lemma": stronger_lemma,
                "stronger_surface_guess": stronger_surface,
                "question_sentence_guess": question_sentence,
                "question_auxiliary": question_auxiliary,
                "question_generation_method": question_method,
                "answer_sentence_guess": answer_sentence,
                "answer_only_insertion_method": only_method,
                "prompt_prefix_for_qwen": prompt_prefix_for_qwen,
                "target_word_seen_by_participants": weaker_surface,
                "answer_suffix_after_target": answer_suffix,
                "prompt_with_blank": prompt_with_blank,
                "dialogue_with_answer_guess": dialogue_with_answer,
                "dialogue_with_stronger_answer_guess": dialogue_with_stronger_answer,
                "reconstruction_note": (
                    "Experiment 4 reconstructed by combining the strong-QUD "
                    "question from Experiment 2 with the only-marked answer "
                    "from Experiment 3."
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
        help="Where to write the reconstructed Experiment 4 scoring table",
    )
    args = parser.parse_args()

    rows = build_condition_rows(args.input)
    write_rows(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
