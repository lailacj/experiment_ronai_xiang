#!/usr/bin/env python3
"""Build and score no-frame strong-alternative prompts for Hu cross-scale items.

The existing Hu/Qwen exact-strong analysis scores the stronger scalemate in the
explicit scalar construction frame:

    context:      The elephant is big, but not
    continuation:  enormous

This script scores the same 309 Hu cross-scale items without the ``X but not Y``
frame by placing the stronger alternative in the weaker item's original slot:

    context:      The elephant is
    continuation:  enormous

The scoring code is shared with ``score_qwen_exact_strong.py`` so the two Qwen
results differ only in prompt construction.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from score_qwen_exact_strong import (
    SCORE_COLUMNS,
    float_cell,
    json_cell,
    load_model_and_tokenizer,
    score_continuation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS_INPUT = PROJECT_ROOT / "data_processed" / "hu_cross_scale_items.csv"
DEFAULT_PROMPT_OUTPUT = PROJECT_ROOT / "stimuli" / "qwen_no_frame_strong_prompts.csv"
DEFAULT_SCORE_OUTPUT = PROJECT_ROOT / "model_scores" / "qwen2_7b_no_frame_strong_scores.csv"

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?(?:-[A-Za-z]+)?")

# The Ronai/Xiang subset stores lemmas rather than fully inflected surfaces.
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
    surface = surface.lower()
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


def candidate_surfaces(lemma: str, surface: str) -> set[str]:
    candidates = {lemma.strip().lower(), surface.strip().lower()}
    candidates.add(third_person_singular(lemma.strip().lower()))
    candidates.add(past_tense(lemma.strip().lower()))
    return {candidate for candidate in candidates if candidate}


def weak_sentence_from_scalar_construction(scalar_construction: str) -> str:
    """Return the first-clause sentence that contains the weak scalemate."""

    construction = scalar_construction.strip()
    if not construction:
        raise ValueError("Empty scalar_construction")

    if ", but " not in construction:
        raise ValueError(f"Expected ', but ' in scalar_construction: {construction!r}")

    first_clause = construction.split(", but ", 1)[0].strip()
    if first_clause.endswith((".", "!", "?")):
        return first_clause
    return first_clause + "."


def find_weak_span(row: dict[str, str], weak_sentence: str) -> tuple[str, int, int, str]:
    """Find the weak expression in the reconstructed weak sentence."""

    weak = row["weak"].strip().lower()
    weak_surface = row["weak_surface"].strip().lower()
    candidates = candidate_surfaces(weak, weak_surface)

    matches = [
        match
        for match in TOKEN_RE.finditer(weak_sentence)
        if match.group(0).lower() in candidates
    ]
    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely identify weak expression: "
            f"item_id={row['item_id']!r}, scale_id={row['scale_id']!r}, "
            f"weak={row['weak']!r}, weak_surface={row['weak_surface']!r}, "
            f"weak_sentence={weak_sentence!r}, matches={[m.group(0) for m in matches]!r}"
        )

    match = matches[0]
    method = "exact_surface" if match.group(0).lower() == weak_surface else "morphology_match"
    return match.group(0), match.start(), match.end(), method


def no_frame_stronger_surface(row: dict[str, str], weak_surface_in_sentence: str) -> tuple[str, str]:
    """Choose the stronger surface for the weak slot.

    Most source datasets already store the right stronger surface. The
    Ronai/Xiang subset stores lemmas, so verbs need the weak surface morphology
    transferred to the stronger lemma.
    """

    strong_surface = row["strong_surface"].strip()
    if row["dataset"] != "rx22" or row["pos"] != "verb":
        return strong_surface, "source_strong_surface"

    morphology = morphology_of(row["weak"].strip().lower(), weak_surface_in_sentence.lower())
    if morphology == "unknown":
        return strong_surface, "source_strong_surface_unknown_morphology"
    return apply_morphology(row["strong"].strip().lower(), morphology), f"rx22_{morphology}"


def move_boundary_whitespace(context: str, target_text: str) -> tuple[str, str, str]:
    trailing = context[len(context.rstrip()) :]
    clean_context = context.rstrip()
    continuation = trailing + target_text
    if not continuation.startswith((" ", "\t", "\n")):
        continuation = " " + continuation
    return clean_context, continuation, trailing


def build_prompt_row(row: dict[str, str]) -> dict[str, str]:
    weak_sentence = weak_sentence_from_scalar_construction(row["scalar_construction"])
    weak_surface, start, end, extraction_method = find_weak_span(row, weak_sentence)
    target_text, target_method = no_frame_stronger_surface(row, weak_surface)

    raw_context = weak_sentence[:start]
    suffix_text = weak_sentence[end:]
    context_text, continuation_text, moved_boundary = move_boundary_whitespace(
        raw_context, target_text
    )
    full_text = f"{raw_context}{target_text}{suffix_text}"
    prompt_with_blank = f"{raw_context}___{suffix_text}"

    prompt_id = (
        f"{row['dataset']}__{row['item_id']}__template_{row['template_id']}__"
        f"{row['scale_id'].replace('/', '_')}__no_frame"
    )
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

    return {
        "prompt_id": prompt_id,
        **source_cols,
        "weak_sentence_frame": weak_sentence,
        "weak_surface_in_sentence": weak_surface,
        "target_text": target_text,
        "context_text": context_text,
        "continuation_text": continuation_text,
        "moved_prompt_boundary_text": moved_boundary,
        "suffix_text": suffix_text,
        "prompt_with_blank": prompt_with_blank,
        "full_text": full_text,
        "prompt_source": "weak_sentence_slot_from_first_clause",
        "weak_extraction_method": extraction_method,
        "stronger_surface_method": target_method,
        "prompt_builder_note": (
            "No-frame score: stronger alternative placed in the weaker item's "
            "first-clause slot, without the X-but-not-Y continuation frame."
        ),
        "score_target": "no_frame_strong_scalemate",
        "score_family": "target_word_or_phrase_logprob",
        "context_has_trailing_space": "False",
        "continuation_has_leading_space": str(continuation_text[:1].isspace()),
    }


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_dict_rows(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_prompts(items_input: Path, prompt_output: Path) -> list[dict[str, str]]:
    item_rows = read_dict_rows(items_input)
    prompt_rows = [build_prompt_row(row) for row in item_rows]
    write_dict_rows(prompt_rows, prompt_output)
    return prompt_rows


def score_prompt_rows(
    args: argparse.Namespace,
    prompt_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    model, tokenizer, resolved_model_path = load_model_and_tokenizer(args)
    rows_to_score = prompt_rows[: args.limit] if args.limit is not None else prompt_rows
    output_rows: list[dict[str, str]] = []
    input_columns = list(prompt_rows[0].keys()) if prompt_rows else []
    total = len(rows_to_score)

    for index, row in enumerate(rows_to_score, start=1):
        score = score_continuation(
            model=model,
            tokenizer=tokenizer,
            context=row["context_text"],
            continuation=row["continuation_text"],
        )
        output_rows.append(
            {
                **row,
                "target_token_ids": json_cell(score["token_ids"]),
                "target_tokens": json_cell(score["tokens"]),
                "target_token_logprobs": json_cell(score["token_logprobs"]),
                "target_n_tokens": str(score["n_tokens"]),
                "target_logprob": float_cell(score["logprob"]),
                "model_name": args.model_name,
                "model_path": str(args.model_path),
                "resolved_model_path": str(resolved_model_path),
                "torch_dtype": args.torch_dtype,
                "device_map": args.device_map,
            }
        )

        if args.progress_every and (
            index == 1 or index % args.progress_every == 0 or index == total
        ):
            print(f"Scored {index}/{total} no-frame strong continuations")

    return output_rows, input_columns + SCORE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items-input",
        type=Path,
        default=DEFAULT_ITEMS_INPUT,
        help="Normalized Hu cross-scale item CSV.",
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=DEFAULT_PROMPT_OUTPUT,
        help="Where to write the generated no-frame prompt CSV.",
    )
    parser.add_argument(
        "--score-output",
        type=Path,
        default=DEFAULT_SCORE_OUTPUT,
        help="Where to write Qwen no-frame scores.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip prompt generation and score --prompt-output as an existing CSV.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the no-frame prompt CSV without loading Qwen.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Local Qwen path. May be either the snapshot directory or the "
            "Hugging Face cache root containing snapshots/."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2-7B",
        help="Model label to store in the output CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit on prompt rows for scoring.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device_map. Use "none" with --device for manual placement.',
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Manual torch device used only when --device-map is "none".',
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face downloads. By default, only local files are used.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress after this many rows. Use 0 to disable.",
    )
    args = parser.parse_args()

    if args.build_only and args.score_only:
        parser.error("--build-only and --score-only cannot be used together.")
    if not args.build_only and args.model_path is None:
        parser.error("--model-path is required unless --build-only is set.")

    return args


def main() -> None:
    args = parse_args()

    if args.score_only:
        prompt_rows = read_dict_rows(args.prompt_output)
        print(f"Read {len(prompt_rows)} no-frame prompts from {args.prompt_output}")
    else:
        prompt_rows = build_prompts(args.items_input, args.prompt_output)
        print(f"Wrote {len(prompt_rows)} no-frame prompts to {args.prompt_output}")

    if args.build_only:
        return

    score_rows, columns = score_prompt_rows(args, prompt_rows)
    args.score_output.parent.mkdir(parents=True, exist_ok=True)
    with args.score_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(score_rows)
    print(f"Wrote {len(score_rows)} Qwen no-frame scores to {args.score_output}")


if __name__ == "__main__":
    main()
