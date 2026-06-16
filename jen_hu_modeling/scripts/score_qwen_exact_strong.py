#!/usr/bin/env python3
"""Score exact strong scalemates in Hu et al. cross-scale prompts with Qwen.

The input prompt table is produced by:

    python3 scripts/build_exact_strong_prompts.py

Each row is scored as:

    log P(continuation_text | context_text)

where ``continuation_text`` already includes the leading whitespace needed for
Qwen tokenization, e.g. ``" enormous"``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "stimuli" / "qwen_exact_strong_prompts.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "model_scores" / "qwen2_7b_exact_strong_scores.csv"


SCORE_COLUMNS = [
    "target_token_ids",
    "target_tokens",
    "target_token_logprobs",
    "target_n_tokens",
    "target_logprob",
    "model_name",
    "model_path",
    "resolved_model_path",
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


def resolve_model_path(path: Path) -> Path:
    """Resolve Hugging Face cache roots to a concrete snapshot directory."""

    path = path.expanduser().resolve()
    if (path / "config.json").exists():
        return path

    snapshots = path / "snapshots"
    if snapshots.is_dir():
        snapshot_dirs = [p for p in snapshots.iterdir() if p.is_dir()]
        if not snapshot_dirs:
            raise FileNotFoundError(f"No snapshot directories found under {snapshots}")
        if len(snapshot_dirs) == 1:
            return snapshot_dirs[0]
        return max(snapshot_dirs, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"Could not resolve model path {path}. Pass either a Hugging Face "
        "snapshot directory containing config.json or a cache root containing snapshots/."
    )


def resolve_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"

    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, Path]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model_path = resolve_model_path(args.model_path)
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
        resolved_model_path,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(resolved_model_path, **model_kwargs)

    if args.device_map == "none":
        import torch

        device = torch.device(args.device)
        model.to(device)

    model.eval()
    return model, tokenizer, resolved_model_path


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


def read_prompt_rows(input_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        return rows[:limit]
    return rows


def score_all(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[str]]:
    model, tokenizer, resolved_model_path = load_model_and_tokenizer(args)
    prompt_rows = read_prompt_rows(args.input, args.limit)
    output_rows: list[dict[str, str]] = []
    input_columns = list(prompt_rows[0].keys()) if prompt_rows else []
    total = len(prompt_rows)

    for index, row in enumerate(prompt_rows, start=1):
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
            print(f"Scored {index}/{total} exact strong continuations")

    return output_rows, input_columns + SCORE_COLUMNS


def write_output(rows: list[dict[str, str]], columns: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Prompt CSV produced by build_exact_strong_prompts.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write Qwen scores.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
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
        help="Optional smoke-test limit on prompt rows.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, columns = score_all(args)
    write_output(rows, columns, args.output)
    print(f"Wrote {len(rows)} Qwen scores to {args.output}")


if __name__ == "__main__":
    main()
