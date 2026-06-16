# Jen Hu Cross-Scale Modeling

This folder is for a Qwen-based replication of the cross-scale scalar-diversity
analysis from Hu et al. (2023), "Expectations over unspoken alternatives predict
pragmatic inferences."

The immediate goal is to build the scalar benchmark section for the focus
alternatives paper:

1. Normalize the Hu et al. cross-scale human datasets.
2. Build prompts for the true strong scalemate in each scalar construction.
3. Score those continuations with Qwen.
4. Test whether Qwen expectedness predicts human scalar implicature rates.

## Folder Layout

- `data_raw/`: source materials copied or fetched from the Hu et al. repository.
- `data_processed/`: tidy item-level tables created from the raw source files.
- `stimuli/`: Qwen scoring prompts and continuations.
- `model_scores/`: raw Qwen scoring outputs.
- `analysis/`: joined human/model tables and analysis-ready CSVs.
- `figures/`: generated plots for paper inspection.
- `tables/`: generated summary/correlation tables.
- `scripts/`: data preparation, scoring, and analysis scripts.

## Planned First Pass

The first pass will use only the exact strong scalemate score:

```text
context: The elephant is big, but not
continuation: enormous
```

This keeps the implementation focused on the result needed for the paper's
scalar benchmark. The broader concept-weighted alternative-set analysis can be
added later if needed.

## Regeneration

Normalize the Hu et al. cross-scale human datasets:

```bash
python3 scripts/normalize_hu_cross_scale.py
```

This writes:

```text
data_processed/hu_cross_scale_items.csv
```

Build exact-strong Qwen prompts:

```bash
python3 scripts/build_exact_strong_prompts.py
```

This writes:

```text
stimuli/qwen_exact_strong_prompts.csv
```

The Qwen prompt table uses `context_text` plus `continuation_text`. The
continuation includes a leading space, matching the continuation-token scoring
setup used in the Ronai/Xiang model scoring code.

Score exact strong alternatives with Qwen:

```bash
python3 scripts/score_qwen_exact_strong.py \
  --model-path /users/ljohnst7/data/ljohnst7/hf-cache/models--Qwen--Qwen2-7B
```

This writes:

```text
model_scores/qwen2_7b_exact_strong_scores.csv
```

For a quick cluster smoke test, run:

```bash
python3 scripts/score_qwen_exact_strong.py \
  --model-path /users/ljohnst7/data/ljohnst7/hf-cache/models--Qwen--Qwen2-7B \
  --limit 5 \
  --output model_scores/qwen2_7b_exact_strong_scores_smoke_test.csv
```
