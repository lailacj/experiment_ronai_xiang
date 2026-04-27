#!/usr/bin/env python3
"""Build a scoring table for Ronai & Xiang Experiment 1 stimuli.

The script reads the Experiment 1 item list, extracts the exact weaker-word
surface form that appears in the sentence, and writes a CSV with:

- the original sentence participants saw
- the prompt prefix before the weaker word
- the weaker word participants saw
- a best-effort stronger-word surface form matched to the same morphology
- the suffix after the target word

This gives us a clean table for language-model scoring experiments.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROMPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PROMPTS_ROOT.parent
DEFAULT_INPUT = PROJECT_ROOT / "ronai_xiang_data" / "items.csv"
DEFAULT_OUTPUT = PROMPTS_ROOT / "experiment_1_scoring_stimuli.csv"

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?(?:-[A-Za-z]+)?")

# A small number of items need explicit help because the sentence uses an
# irregular surface form that is not directly recoverable from the lemma.
ITEM_OVERRIDES = {
    3: {"weaker_surface": "began", "stronger_surface": "completed"},
}

IRREGULAR_PAST = {
    "begin": "began",
}


def is_consonant(char: str) -> bool:
    return char.isalpha() and char.lower() not in "aeiou"


def third_person_singular(lemma: str) -> str:
    if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        return lemma + "es"
    if len(lemma) >= 2 and lemma.endswith("y") and is_consonant(lemma[-2]):
        return lemma[:-1] + "ies"
    return lemma + "s"


def past_tense(lemma: str) -> str:
    if lemma in IRREGULAR_PAST:
        return IRREGULAR_PAST[lemma]
    if lemma.endswith("e"):
        return lemma + "d"
    if len(lemma) >= 2 and lemma.endswith("y") and is_consonant(lemma[-2]):
        return lemma[:-1] + "ied"
    if (
        len(lemma) >= 3
        and is_consonant(lemma[-1])
        and lemma[-1].lower() not in "wxy"
        and lemma[-2].lower() in "aeiou"
        and is_consonant(lemma[-3])
    ):
        return lemma + lemma[-1] + "ed"
    return lemma + "ed"


def morphology_of(lemma: str, surface: str) -> str:
    if surface == lemma:
        return "base"
    if surface == third_person_singular(lemma):
        return "third_person_singular"
    if surface == past_tense(lemma):
        return "past_tense"
    return "unknown"


def apply_morphology(lemma: str, morphology: str) -> str:
    if morphology == "base":
        return lemma
    if morphology == "third_person_singular":
        return third_person_singular(lemma)
    if morphology == "past_tense":
        return past_tense(lemma)
    return lemma


def candidate_surfaces(lemma: str) -> set[str]:
    return {
        lemma,
        third_person_singular(lemma),
        past_tense(lemma),
    }


def find_target_span(item_id: int, weaker_lemma: str, sentence: str) -> tuple[str, int, int, str]:
    override = ITEM_OVERRIDES.get(item_id, {})
    override_surface = override.get("weaker_surface")
    if override_surface:
        for match in TOKEN_RE.finditer(sentence):
            if match.group(0).lower() == override_surface.lower():
                return match.group(0), match.start(), match.end(), "item_override"
        raise ValueError(f"Override surface {override_surface!r} not found in item {item_id}: {sentence}")

    candidates = candidate_surfaces(weaker_lemma)
    matches = [
        match
        for match in TOKEN_RE.finditer(sentence)
        if match.group(0).lower() in candidates
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely identify weaker word for item {item_id}: "
            f"lemma={weaker_lemma!r}, sentence={sentence!r}, matches={[m.group(0) for m in matches]!r}"
        )

    match = matches[0]
    return match.group(0), match.start(), match.end(), "automatic"


def build_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    output_rows: list[dict[str, str]] = []
    for item_id, row in enumerate(rows, start=1):
        weaker_lemma = row["Weaker"].strip()
        stronger_lemma = row["Stronger"].strip()
        sentence = row["Sentence frame (Experiment 1)"].strip()

        weaker_surface, start, end, extraction_method = find_target_span(item_id, weaker_lemma, sentence)
        prompt = sentence[:start]
        suffix = sentence[end:]
        morphology = morphology_of(weaker_lemma, weaker_surface.lower())

        stronger_surface = ITEM_OVERRIDES.get(item_id, {}).get("stronger_surface")
        stronger_guess_method = "item_override" if stronger_surface else "morphology_match"
        if stronger_surface is None:
            stronger_surface = apply_morphology(stronger_lemma, morphology)
            if morphology == "unknown":
                stronger_guess_method = "fallback_to_lemma"

        output_rows.append(
            {
                "item_id": str(item_id),
                "weaker_lemma": weaker_lemma,
                "weaker_surface": weaker_surface,
                "stronger_lemma": stronger_lemma,
                "stronger_surface_guess": stronger_surface,
                "sentence_frame_experiment_1": sentence,
                "prompt_prefix": prompt,
                "target_word_seen_by_participants": weaker_surface,
                "suffix_after_target": suffix,
                "prompt_with_blank": f"{prompt}___{suffix}",
                "stronger_sentence_guess": f"{prompt}{stronger_surface}{suffix}",
                "surface_morphology": morphology,
                "target_extraction_method": extraction_method,
                "stronger_guess_method": stronger_guess_method,
            }
        )

    return output_rows


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to items.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the reconstructed Experiment 1 scoring table",
    )
    args = parser.parse_args()

    rows = build_rows(args.input)
    write_rows(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
