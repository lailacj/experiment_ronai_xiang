#!/usr/bin/env python3
"""Score Ronai & Xiang weak/strong alternatives with Qwen.

This script reads the reconstructed stimulus tables in ``stimuli_prompts/`` and
scores both the weak and strong alternative for each prompt:

    log P(" excellent" | "The movie is")

For multi-token continuations, the score is the summed log probability of the
continuation tokens:

    log P(token_1 | prompt) + log P(token_2 | prompt + token_1) + ...

The script also scores the candidate plus the reconstructed suffix, e.g.:

    log P(" excellent." | "The movie is")

No normalization is applied. All output columns are raw model measurements or
metadata needed to audit those measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_SCORES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_SCORES_ROOT.parent
PROMPTS_ROOT = PROJECT_ROOT / "stimuli_prompts"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT.parent
    / "hf-cache"
    / "models--Qwen--Qwen2-7B"
    / "snapshots"
    / "453ed1575b739b5b03ce3758b23befdb0967f40e"
)
DEFAULT_OUTPUT = MODEL_SCORES_ROOT / "qwen2_7b_scored_alternatives.csv"


@dataclass(frozen=True)
class StimulusFile:
    experiment: str
    path: Path
    prompt_column: str
    suffix_column: str
    default_condition: str | None = None


STIMULUS_FILES = {
    "1": StimulusFile(
        experiment="experiment_1",
        path=PROMPTS_ROOT / "experiment_1_scoring_stimuli.csv",
        prompt_column="prompt_prefix",
        suffix_column="suffix_after_target",
        default_condition="ESI",
    ),
    "2": StimulusFile(
        experiment="experiment_2",
        path=PROMPTS_ROOT / "experiment_2_scoring_stimuli.csv",
        prompt_column="prompt_prefix_for_qwen",
        suffix_column="answer_suffix_after_target",
    ),
    "3": StimulusFile(
        experiment="experiment_3",
        path=PROMPTS_ROOT / "experiment_3_scoring_stimuli.csv",
        prompt_column="prompt_prefix_for_qwen",
        suffix_column="suffix_after_target",
    ),
    "4": StimulusFile(
        experiment="experiment_4",
        path=PROMPTS_ROOT / "experiment_4_scoring_stimuli.csv",
        prompt_column="prompt_prefix_for_qwen",
        suffix_column="answer_suffix_after_target",
    ),
}


OUTPUT_COLUMNS = [
    "experiment",
    "source_file",
    "source_row_index",
    "item_id",
    "condition",
    "sub_experiment",
    "weaker_lemma",
    "weaker_surface",
    "stronger_lemma",
    "stronger_surface_guess",
    "target_type",
    "target_surface",
    "prompt_text",
    "context_text",
    "moved_prompt_boundary_text",
    "suffix_text",
    "word_continuation_text",
    "word_token_ids",
    "word_tokens",
    "word_token_logprobs",
    "word_n_tokens",
    "word_logprob",
    "candidate_plus_suffix_text",
    "candidate_plus_suffix_token_ids",
    "candidate_plus_suffix_tokens",
    "candidate_plus_suffix_token_logprobs",
    "candidate_plus_suffix_n_tokens",
    "candidate_plus_suffix_logprob",
    "model_name",
    "model_path",
    "torch_dtype",
    "device_map",
]


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def float_cell(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.10f}"


def split_prompt_boundary(prompt: str, continuation: str) -> tuple[str, str, str]:
    """Move trailing prompt whitespace into the continuation.

    Qwen's tokenizer folds word-initial whitespace into continuation tokens. If
    the CSV prompt is ``"The movie is "`` and the target is ``"excellent"``,
    the probability object we want is therefore scored as:

        context = "The movie is"
        continuation = " excellent"
    """

    context = prompt.rstrip()
    boundary = prompt[len(context) :]
    return context, boundary, boundary + continuation


def resolve_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"

    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_files_only = not args.allow_downloads
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if args.device_map == "none":
        device = torch.device(args.device)
        model.to(device)

    model.eval()
    return model, tokenizer


def model_input_device(model: Any) -> Any:
    import torch

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for device_name in device_map.values():
            if device_name not in {"cpu", "disk"}:
                return torch.device(device_name)
    return next(model.parameters()).device


def continuation_token_ids(tokenizer: Any, context: str, continuation: str) -> tuple[list[int], list[int]]:
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)

    if not context_ids:
        raise ValueError("Cannot score a continuation from an empty tokenized context.")

    if full_ids[: len(context_ids)] != context_ids:
        raise ValueError(
            "Tokenized context is not a prefix of tokenized context+continuation. "
            f"context={context!r}, continuation={continuation!r}, "
            f"context_ids={context_ids!r}, full_ids={full_ids!r}"
        )

    target_ids = full_ids[len(context_ids) :]
    if not target_ids:
        raise ValueError(
            f"Continuation produced no target tokens: context={context!r}, "
            f"continuation={continuation!r}"
        )

    return full_ids, target_ids


def score_continuation(
    *,
    model: Any,
    tokenizer: Any,
    context: str,
    continuation: str,
) -> dict[str, Any]:
    import torch

    full_ids, target_ids = continuation_token_ids(tokenizer, context, continuation)
    context_length = len(full_ids) - len(target_ids)
    input_device = model_input_device(model)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=input_device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[0]
        prediction_logits = logits[context_length - 1 : -1]
        log_probs = torch.nn.functional.log_softmax(prediction_logits.float(), dim=-1)
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=log_probs.device)
        token_logprobs = log_probs.gather(1, target_tensor[:, None]).squeeze(1)

    token_logprob_values = [float(value) for value in token_logprobs.detach().cpu().tolist()]
    return {
        "token_ids": target_ids,
        "tokens": tokenizer.convert_ids_to_tokens(target_ids),
        "token_logprobs": token_logprob_values,
        "logprob": sum(token_logprob_values),
        "n_tokens": len(target_ids),
    }


def read_stimulus_rows(
    stimulus_file: StimulusFile,
    *,
    limit_per_file: int | None,
) -> list[dict[str, str]]:
    with stimulus_file.path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if limit_per_file is not None:
        rows = rows[:limit_per_file]
    return rows


def iter_scoring_inputs(
    stimulus_keys: list[str],
    *,
    limit_per_file: int | None,
) -> Any:
    for key in stimulus_keys:
        stimulus_file = STIMULUS_FILES[key]
        rows = read_stimulus_rows(stimulus_file, limit_per_file=limit_per_file)

        for row_index, row in enumerate(rows, start=1):
            prompt = row[stimulus_file.prompt_column]
            suffix = row[stimulus_file.suffix_column]
            condition = row.get("condition") or stimulus_file.default_condition or ""

            for target_type, target_surface in [
                ("weaker", row["weaker_surface"]),
                ("stronger", row["stronger_surface_guess"]),
            ]:
                yield {
                    "stimulus_file": stimulus_file,
                    "source_row_index": row_index,
                    "row": row,
                    "condition": condition,
                    "prompt": prompt,
                    "suffix": suffix,
                    "target_type": target_type,
                    "target_surface": target_surface,
                }


def build_output_row(
    *,
    scoring_input: dict[str, Any],
    word_score: dict[str, Any],
    candidate_plus_suffix_score: dict[str, Any],
    context_text: str,
    boundary_text: str,
    word_continuation_text: str,
    candidate_plus_suffix_text: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    stimulus_file = scoring_input["stimulus_file"]
    row = scoring_input["row"]

    return {
        "experiment": stimulus_file.experiment,
        "source_file": str(stimulus_file.path.relative_to(PROJECT_ROOT)),
        "source_row_index": str(scoring_input["source_row_index"]),
        "item_id": row["item_id"],
        "condition": scoring_input["condition"],
        "sub_experiment": row.get("sub_experiment", ""),
        "weaker_lemma": row.get("weaker_lemma", ""),
        "weaker_surface": row.get("weaker_surface", ""),
        "stronger_lemma": row.get("stronger_lemma", ""),
        "stronger_surface_guess": row.get("stronger_surface_guess", ""),
        "target_type": scoring_input["target_type"],
        "target_surface": scoring_input["target_surface"],
        "prompt_text": scoring_input["prompt"],
        "context_text": context_text,
        "moved_prompt_boundary_text": boundary_text,
        "suffix_text": scoring_input["suffix"],
        "word_continuation_text": word_continuation_text,
        "word_token_ids": json_cell(word_score["token_ids"]),
        "word_tokens": json_cell(word_score["tokens"]),
        "word_token_logprobs": json_cell(word_score["token_logprobs"]),
        "word_n_tokens": str(word_score["n_tokens"]),
        "word_logprob": float_cell(word_score["logprob"]),
        "candidate_plus_suffix_text": candidate_plus_suffix_text,
        "candidate_plus_suffix_token_ids": json_cell(candidate_plus_suffix_score["token_ids"]),
        "candidate_plus_suffix_tokens": json_cell(candidate_plus_suffix_score["tokens"]),
        "candidate_plus_suffix_token_logprobs": json_cell(
            candidate_plus_suffix_score["token_logprobs"]
        ),
        "candidate_plus_suffix_n_tokens": str(candidate_plus_suffix_score["n_tokens"]),
        "candidate_plus_suffix_logprob": float_cell(candidate_plus_suffix_score["logprob"]),
        "model_name": args.model_name,
        "model_path": str(args.model_path),
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
    }


def score_all(args: argparse.Namespace) -> list[dict[str, str]]:
    model, tokenizer = load_model_and_tokenizer(args)
    output_rows: list[dict[str, str]] = []

    scoring_inputs = list(
        iter_scoring_inputs(args.experiments, limit_per_file=args.limit_per_file)
    )
    total = len(scoring_inputs)

    for index, scoring_input in enumerate(scoring_inputs, start=1):
        target = scoring_input["target_surface"]
        suffix = scoring_input["suffix"]
        context, boundary, word_continuation = split_prompt_boundary(
            scoring_input["prompt"],
            target,
        )
        _, _, candidate_plus_suffix = split_prompt_boundary(
            scoring_input["prompt"],
            target + suffix,
        )

        word_score = score_continuation(
            model=model,
            tokenizer=tokenizer,
            context=context,
            continuation=word_continuation,
        )
        candidate_plus_suffix_score = score_continuation(
            model=model,
            tokenizer=tokenizer,
            context=context,
            continuation=candidate_plus_suffix,
        )

        output_rows.append(
            build_output_row(
                scoring_input=scoring_input,
                word_score=word_score,
                candidate_plus_suffix_score=candidate_plus_suffix_score,
                context_text=context,
                boundary_text=boundary,
                word_continuation_text=word_continuation,
                candidate_plus_suffix_text=candidate_plus_suffix,
                args=args,
            )
        )

        if args.progress_every and (index == 1 or index % args.progress_every == 0 or index == total):
            print(f"Scored {index}/{total} alternatives")

    return output_rows


def write_output(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local Qwen model snapshot path.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2-7B",
        help="Model label to store in the output CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the scored alternative table.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(STIMULUS_FILES),
        default=sorted(STIMULUS_FILES),
        help="Experiment stimulus tables to score.",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Optional smoke-test limit on stimulus rows read from each CSV.",
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
        help="Print progress after this many scored alternatives. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = score_all(args)
    write_output(rows, args.output)
    print(f"Wrote {len(rows)} scored alternatives to {args.output}")


if __name__ == "__main__":
    main()
