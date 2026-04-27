#!/usr/bin/env python3
"""Build a scoring table for Ronai & Xiang Experiment 2 stimuli.

Experiment 2 contains two QUD conditions:

- Eweak: the question contains the weaker scalar term
- Estrong: the question contains the stronger scalar term

This script reconstructs a best-effort prompt table with one row per
item-condition pair. The answer side is kept conservative: it reuses the
Experiment 1 sentence context around the target word, because the shared files
do not contain the exact dialogue-coherence edits used in the original study.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from build_experiment_1_stimuli import PROJECT_ROOT, PROMPTS_ROOT, build_rows, write_rows


DEFAULT_INPUT = PROJECT_ROOT / "ronai_xiang_data" / "items.csv"
DEFAULT_OUTPUT = PROMPTS_ROOT / "experiment_2_scoring_stimuli.csv"

AUXILIARIES = {
    "is",
    "are",
    "was",
    "were",
    "might",
    "will",
}

SUBJECT_INITIALS_KEEP_CAPS = {
    "Ann's",
    "Bill's",
    "Cecilia",
    "Chris's",
    "John",
    "Jimmy",
    "Joey's",
    "Kaye's",
    "Peter's",
    "Phoebe",
    "Stu's",
    "Tim's",
    "Tom's",
    "Zack's",
}

# Sentences without an invertible auxiliary need do-support. The shared files do
# not contain part-of-speech annotation, so we anchor those items with a small
# override table.
DO_SUPPORT_ITEMS = {
    3: {"main_verb_source": "target"},
    4: {"main_verb_source": "target"},
    11: {"main_verb_source": "target"},
    24: {"main_verb_source": "target"},
    25: {"main_verb_source": "target"},
    28: {"main_verb_source": "fixed", "surface": "happened", "lemma": "happen", "aux": "Did"},
    29: {"main_verb_source": "fixed", "surface": "writes", "lemma": "write", "aux": "Does"},
    34: {"main_verb_source": "target"},
    40: {"main_verb_source": "target"},
    44: {"main_verb_source": "target"},
    47: {"main_verb_source": "fixed", "surface": "trusts", "lemma": "trust", "aux": "Does"},
    48: {"main_verb_source": "target"},
    49: {"main_verb_source": "target"},
    51: {"main_verb_source": "target", "aux": "Do"},
    52: {"main_verb_source": "target"},
    57: {"main_verb_source": "target"},
    59: {"main_verb_source": "fixed", "surface": "went", "lemma": "go", "aux": "Did"},
}

QUESTION_OVERRIDES = {
    (44, "weaker"): "Did the train slow down?",
}


def lowercase_subject_initial(token: str) -> str:
    if token in SUBJECT_INITIALS_KEEP_CAPS:
        return token
    return token[:1].lower() + token[1:]


def strip_final_period(sentence: str) -> str:
    return sentence[:-1] if sentence.endswith(".") else sentence


def invert_auxiliary(statement: str) -> tuple[str, str, str]:
    words = strip_final_period(statement).split()
    aux_index = next(
        (idx for idx, word in enumerate(words) if word.lower() in AUXILIARIES),
        None,
    )
    if aux_index is None:
        raise ValueError(f"No invertible auxiliary found in: {statement}")

    aux = words[aux_index]
    remainder = words[:aux_index] + words[aux_index + 1 :]
    remainder[0] = lowercase_subject_initial(remainder[0])
    question = f"{aux.capitalize()} {' '.join(remainder)}?"
    return question, aux.lower(), "aux_inversion"


def auto_do_aux(surface_morphology: str) -> str:
    if surface_morphology == "past_tense":
        return "Did"
    if surface_morphology == "third_person_singular":
        return "Does"
    if surface_morphology == "base":
        return "Do"
    return "Do"


def replace_first_word(sentence: str, old: str, new: str) -> str:
    pattern = re.compile(rf"\b{re.escape(old)}\b")
    replaced, count = pattern.subn(new, sentence, count=1)
    if count != 1:
        raise ValueError(f"Could not replace {old!r} in {sentence!r}")
    return replaced


def do_support_question(
    statement: str,
    *,
    main_verb_surface: str,
    main_verb_lemma: str,
    do_aux: str,
) -> tuple[str, str, str]:
    clause = strip_final_period(statement)
    clause = replace_first_word(clause, main_verb_surface, main_verb_lemma)
    words = clause.split()
    words[0] = lowercase_subject_initial(words[0])
    question = f"{do_aux} {' '.join(words)}?"
    return question, do_aux.lower(), "do_support"


def build_question(
    item_id: int,
    statement: str,
    *,
    weaker_lemma: str,
    weaker_surface: str,
    stronger_lemma: str,
    stronger_surface: str,
    question_target_type: str,
    surface_morphology: str,
) -> tuple[str, str, str]:
    override = QUESTION_OVERRIDES.get((item_id, question_target_type))
    if override is not None:
        first_word = override.split()[0].rstrip("?").lower()
        return override, first_word, "question_override"

    config = DO_SUPPORT_ITEMS.get(item_id)
    if config is not None:
        if config["main_verb_source"] == "target":
            if question_target_type == "weaker":
                main_verb_surface = weaker_surface
                main_verb_lemma = weaker_lemma
            else:
                main_verb_surface = stronger_surface
                main_verb_lemma = stronger_lemma
            do_aux = config.get("aux", auto_do_aux(surface_morphology))
        else:
            main_verb_surface = config["surface"]
            main_verb_lemma = config["lemma"]
            do_aux = config["aux"]

        return do_support_question(
            statement,
            main_verb_surface=main_verb_surface,
            main_verb_lemma=main_verb_lemma,
            do_aux=do_aux,
        )

    if any(f" {aux} " in f" {strip_final_period(statement)} " for aux in AUXILIARIES):
        return invert_auxiliary(statement)

    raise ValueError(f"No question-generation strategy found for item {item_id}: {statement}")


def build_condition_rows(input_path: Path) -> list[dict[str, str]]:
    exp1_rows = build_rows(input_path)
    output_rows: list[dict[str, str]] = []

    for base_row in exp1_rows:
        item_id = int(base_row["item_id"])
        weaker_lemma = base_row["weaker_lemma"]
        weaker_surface = base_row["weaker_surface"]
        stronger_lemma = base_row["stronger_lemma"]
        stronger_surface = base_row["stronger_surface_guess"]
        answer_prefix = base_row["prompt_prefix"]
        answer_suffix = base_row["suffix_after_target"]
        weaker_answer_sentence = base_row["sentence_frame_experiment_1"]
        stronger_answer_sentence = base_row["stronger_sentence_guess"]
        surface_morphology = base_row["surface_morphology"]

        for condition, sub_experiment, question_target_type, question_target_surface, statement in [
            ("Eweak", "Weak QUD", "weaker", weaker_surface, weaker_answer_sentence),
            ("Estrong", "Strong QUD", "stronger", stronger_surface, stronger_answer_sentence),
        ]:
            question_sentence, question_auxiliary, question_method = build_question(
                item_id,
                statement,
                weaker_lemma=weaker_lemma,
                weaker_surface=weaker_surface,
                stronger_lemma=stronger_lemma,
                stronger_surface=stronger_surface,
                question_target_type=question_target_type,
                surface_morphology=surface_morphology,
            )

            prompt_prefix_for_qwen = f"Sue: {question_sentence}\nMary: {answer_prefix}"
            prompt_with_blank = f"Sue: {question_sentence}\nMary: {answer_prefix}___{answer_suffix}"
            stronger_answer_prompt = f"Sue: {question_sentence}\nMary: {stronger_answer_sentence}"

            output_rows.append(
                {
                    "item_id": str(item_id),
                    "condition": condition,
                    "sub_experiment": sub_experiment,
                    "weaker_lemma": weaker_lemma,
                    "weaker_surface": weaker_surface,
                    "stronger_lemma": stronger_lemma,
                    "stronger_surface_guess": stronger_surface,
                    "question_target_type": question_target_type,
                    "question_target_surface_guess": question_target_surface,
                    "answer_sentence_guess": weaker_answer_sentence,
                    "answer_prompt_prefix_guess": answer_prefix,
                    "answer_target_word_seen_by_participants": weaker_surface,
                    "answer_suffix_after_target": answer_suffix,
                    "question_source_statement_guess": statement,
                    "question_sentence_guess": question_sentence,
                    "question_auxiliary": question_auxiliary,
                    "question_generation_method": question_method,
                    "prompt_prefix_for_qwen": prompt_prefix_for_qwen,
                    "prompt_with_blank": prompt_with_blank,
                    "dialogue_with_stronger_answer_guess": stronger_answer_prompt,
                    "reconstruction_note": (
                        "Question reconstructed from Experiment 1 materials; "
                        "answer side conservatively reuses the Experiment 1 sentence context."
                    ),
                    "needs_manual_review": "yes",
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
        help="Where to write the reconstructed Experiment 2 scoring table",
    )
    args = parser.parse_args()

    rows = build_condition_rows(args.input)
    write_rows(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
