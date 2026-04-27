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

The next major step is to move from prompt reconstruction to actual Qwen scoring.

That agent should:

1. Decide the exact scoring rule.

- Minimum baseline:
  score the stronger alternative from the prompt prefix
- Most likely useful normalization:
  compare the stronger alternative to the weaker one, for example with a normalized score like
  `p(stronger) / (p(stronger) + p(weaker))`
- For multi-token continuations, use full continuation probability rather than a naive single-token shortcut

2. Build a scoring script.

- likely one script that reads the four reconstructed CSVs
- runs Qwen on each row
- stores scores in new output files, probably also under `stimuli_prompts/` or a separate `model_scores/` folder

3. Aggregate human data by item and condition.

- Experiment 1: `ESI`
- Experiment 2: `Eweak`, `Estrong`
- Experiment 3: `Eonly`
- Experiment 4: `Eonlystrong`

4. Compare model scores to human responses.

- item-level correlations by condition
- condition-level summaries
- potentially mixed models or simpler regressions for a first pass

5. Keep the original data separate.

- do not edit files in `ronai_xiang_data/`
- put all derived artifacts in a separate folder

## Recommended Next Deliverables

The next agent on the compute cluster should probably produce:

- a Qwen scoring script
- a table of model scores for all reconstructed prompts
- a script that aggregates human response rates by item and condition
- a first analysis notebook or script comparing Qwen scores to human rates

## Open Questions

These are still unresolved and should be decided explicitly:

- Which Qwen model variant should be used on the cluster?
- What exact probability object should be used:
  raw next-token probability, full continuation probability, or a normalized alternative score?
- How should multi-token stronger alternatives be handled consistently?
- Should the first pass use the reconstructed prompts exactly as they are, or do a final manual audit of odd cases before scoring?

## Bottom Line

This repo is now at the “stimuli reconstruction complete, model scoring not yet started” stage.

The original Ronai & Xiang materials are preserved in `ronai_xiang_data/`. The working prompt tables and generators live in `stimuli_prompts/`. The next step is to run Qwen on those prompt tables and evaluate whether the simple alternative-probability baseline predicts the Ronai & Xiang human results across all experiments.
