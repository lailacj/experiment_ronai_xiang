# Ronai & Xiang Qwen Baseline

This repository is for a first-pass model-based analysis of the Ronai & Xiang scalar-diversity datasets.

The immediate goal is:

- reconstruct the experimental stimuli as clean prompt tables
- use Qwen to score the relevant scalar alternative in each prompt
- test a simple baseline idea:
  the probability that a stronger alternative is negated or excluded is related to Qwen's probability for that stronger alternative in context

In the motivating example from the paper:

- sentence: `The movie is good.`
- stronger alternative: `excellent`
- simple baseline: higher model expectation for `excellent` should predict a stronger tendency toward the inference `not excellent`

This is intentionally the simplest version of the broader project. It is a baseline for later work on different alternative types, including scalar, ad hoc, and intermediate cases.

## Source Material

The original materials in this repo are:

- [What could have been said_ Alternatives and variability in pragmatic inferences.pdf](./What%20could%20have%20been%20said_%20Alternatives%20and%20variability%20in%20pragmatic%20inferences.pdf)
- [ronai_xiang_data/items.csv](./ronai_xiang_data/items.csv)
- [ronai_xiang_data/Experiment_1_data.csv](./ronai_xiang_data/Experiment_1_data.csv)
- [ronai_xiang_data/Experiment_2_data.csv](./ronai_xiang_data/Experiment_2_data.csv)
- [ronai_xiang_data/Experiment_3_data.csv](./ronai_xiang_data/Experiment_3_data.csv)
- [ronai_xiang_data/Experiment_4_data.csv](./ronai_xiang_data/Experiment_4_data.csv)
- [ronai_xiang_data/analysis_codes.R](./ronai_xiang_data/analysis_codes.R)

Important limitation:

- the shared data include the Experiment 1 carrier sentences and the response files
- they do **not** include a fully explicit item-by-item stimulus file for Experiments 2 to 4
- because of that, the prompt tables in `stimuli_prompts/` are reconstructions, not guaranteed verbatim originals

## What Has Been Done

We created a separate folder, [stimuli_prompts](./stimuli_prompts), so the original `ronai_xiang_data/` stays untouched.

Inside `stimuli_prompts/` there are four generator scripts:

- [build_experiment_1_stimuli.py](./stimuli_prompts/build_experiment_1_stimuli.py)
- [build_experiment_2_stimuli.py](./stimuli_prompts/build_experiment_2_stimuli.py)
- [build_experiment_3_stimuli.py](./stimuli_prompts/build_experiment_3_stimuli.py)
- [build_experiment_4_stimuli.py](./stimuli_prompts/build_experiment_4_stimuli.py)

And four generated prompt tables:

- [experiment_1_scoring_stimuli.csv](./stimuli_prompts/experiment_1_scoring_stimuli.csv): 60 rows
- [experiment_2_scoring_stimuli.csv](./stimuli_prompts/experiment_2_scoring_stimuli.csv): 120 rows
- [experiment_3_scoring_stimuli.csv](./stimuli_prompts/experiment_3_scoring_stimuli.csv): 60 rows
- [experiment_4_scoring_stimuli.csv](./stimuli_prompts/experiment_4_scoring_stimuli.csv): 60 rows

We also added a separate derived-output folder, [model_scores](./model_scores), for Qwen scoring work:

- [score_qwen_alternatives.py](./model_scores/score_qwen_alternatives.py)
- [qwen2_7b_scored_alternatives.csv](./model_scores/qwen2_7b_scored_alternatives.csv)

The score table has 600 records: one row for each reconstructed prompt by alternative type, with both the weaker and stronger alternative scored for every prompt.

### Experiment Coverage

Experiment 1:

- directly reconstructed from `items.csv`
- extracts the exact weaker surface form from the sentence
- also creates a best-effort stronger-form sentence for later scoring

Experiment 2:

- includes the two subconditions:
  - `Eweak`
  - `Estrong`
- reconstructs strong-QUD and weak-QUD question forms
- uses question-generation heuristics plus a few item-level fixes for do-support and wording

Experiment 3:

- includes the `Eonly` condition
- reconstructs `only`-marked answer sentences
- mostly inserts `only` before the weaker scalar term, with a few item-specific overrides where needed

Experiment 4:

- includes the `Eonlystrong` condition
- combines the strong-QUD question logic from Experiment 2 with the `only` answer logic from Experiment 3

## Current Interpretation of the CSVs

These CSVs are meant to support Qwen scoring.

The important columns are:

- a prompt prefix to feed to the model
- the word participants saw in the reconstructed stimulus
- the suffix after the target, when relevant
- the reconstructed sentence or dialogue for inspection

For Experiment 1, the basic use case is:

- prompt: `The movie is `
- target shown to participants: `good`
- stronger alternative to score: `excellent`

For Experiment 2 and 4, prompts include a dialogue like:

- `Sue: Is the movie excellent?`
- `Mary: The movie is ___`

or

- `Sue: Is the movie excellent?`
- `Mary: The movie is only ___`

## Qwen Scoring

The current Qwen scoring script is:

```bash
python3 model_scores/score_qwen_alternatives.py
```

By default, it loads the local cached model:

```text
../hf-cache/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e
```

and writes:

```text
model_scores/qwen2_7b_scored_alternatives.csv
```

The scoring script applies no normalization. It records raw continuation log probabilities.

For word-only scoring, it computes:

```text
log P(" excellent" | "The movie is")
```

For multi-token alternatives, it sums the continuation-token log probabilities:

```text
log P(" life-threatening" | prompt)
= log P(token_1 | prompt) + log P(token_2 | prompt + token_1) + ...
```

It also computes a candidate-plus-suffix score, for example:

```text
log P(" excellent." | "The movie is")
```

or:

```text
log P(" completed the race." | "John")
```

The output keeps both `word_logprob` and `candidate_plus_suffix_logprob`, along with token IDs, tokenizer strings, token-level log probabilities, token counts, and the exact context/continuation strings used for scoring.

Important implementation detail:

- Prompt-final whitespace is moved into the continuation before scoring.
- For example, the CSV prompt `The movie is ` plus target `excellent` is scored as context `The movie is` and continuation ` excellent`.
- This matches Qwen tokenization, where word-initial whitespace is part of the continuation tokenization.

## Important Caveats

1. The Experiment 2 to 4 materials are reconstructed.

- The paper and data files give enough information to build plausible prompts.
- They do not provide a guaranteed exact stimulus file for every item in every later experiment.

2. Some rows include manual wording decisions.

- These were made to keep prompts grammatical and useful for model scoring.
- They should be treated as best-effort reconstructions, not archival ground truth.

3. The `analysis_codes.R` file is an analysis script, not a stimulus-construction script.

- It confirms condition labels and experiment structure.
- It does not tell us the exact original wording for all reconstructed stimuli.

## How to Regenerate the Prompt Tables

Run from the project root:

```bash
python3 stimuli_prompts/build_experiment_1_stimuli.py
python3 stimuli_prompts/build_experiment_2_stimuli.py
python3 stimuli_prompts/build_experiment_3_stimuli.py
python3 stimuli_prompts/build_experiment_4_stimuli.py
```

These scripts:

- read from `ronai_xiang_data/items.csv`
- write to `stimuli_prompts/`

## Project Goal for the Next Agent

The project has now moved past prompt reconstruction and first-pass Qwen scoring.

The next major step is to aggregate human responses and compare them to the raw Qwen log probabilities.

That agent should:

1. Aggregate human data by item and condition.

- Experiment 1: `ESI`
- Experiment 2: `Eweak`, `Estrong`
- Experiment 3: `Eonly`
- Experiment 4: `Eonlystrong`
- The original R script excludes two Experiment 1 participants and one Experiment 3 participant; any Python aggregation should either preserve those exclusions or explicitly document a different choice.

2. Join human summaries to model scores.

- Use `item_id` and `condition`.
- The model table is long by `target_type`, so stronger-only analyses should filter to `target_type == "stronger"`.
- Keep both `word_logprob` and `candidate_plus_suffix_logprob` available during comparison.

3. Decide the comparison-layer normalization or transformation.

- The Qwen table intentionally stores raw log probabilities.
- Any normalization, centering, ranking, condition-wise scaling, or weak-vs-strong contrast should happen in the human/model comparison script, not in the Qwen scoring script.

4. Compare model scores to human responses.

- item-level correlations by condition
- condition-level summaries
- potentially mixed models or simpler regressions for a first pass

5. Keep the original data separate.

- do not edit files in `ronai_xiang_data/`
- put all derived artifacts in a separate folder

## Recommended Next Deliverables

The next agent on the compute cluster should probably produce:

- a script that aggregates human response rates by item and condition
- a first analysis notebook or script comparing Qwen scores to human rates
- optional diagnostic plots for item-level human rates against Qwen log probabilities

## Open Questions

These are still unresolved and should be decided explicitly:

- For the first comparison, should the primary model predictor be `word_logprob` or `candidate_plus_suffix_logprob`?
- Should analyses use raw log probabilities, ranks, item-wise contrasts, condition-wise centering, or another comparison-layer transformation?
- Should the first pass compare only stronger-alternative probabilities to human inference rates, or also include weak-vs-strong differences?
- Should the first pass use the reconstructed prompts exactly as they are, or do a final manual audit of odd cases before scoring?

## Bottom Line

This repo is now at the “stimuli reconstruction complete, first-pass Qwen scoring complete” stage.

The original Ronai & Xiang materials are preserved in `ronai_xiang_data/`. The working prompt tables and generators live in `stimuli_prompts/`. The Qwen scoring script and derived score table live in `model_scores/`. The next step is to aggregate the human response data and evaluate whether Qwen's raw alternative log probabilities predict the Ronai & Xiang human results across experiments.
