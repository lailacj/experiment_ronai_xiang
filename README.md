# Ronai & Xiang Qwen Models

This repository is for a first-pass model-based analysis of the Ronai & Xiang scalar-diversity datasets.

The working goal is:

- reconstruct the experimental stimuli as clean prompt tables
- use Qwen to score the relevant scalar alternative in each prompt
- compare different simple model implementations against human response rates

The project currently has two model implementations:

- Baseline model: the probability that a stronger alternative is negated or excluded is related to Qwen's probability for that stronger alternative in context.
- Ordering model: when the query (the stronger alternative) is more likely to be predicted in context than the trigger (the weaker alternative), the query is more likely to be negated.

In the motivating baseline example from the paper:

- sentence: `The movie is good.`
- stronger alternative: `excellent`
- simple baseline: higher model expectation for `excellent` should predict a stronger tendency toward the inference `not excellent`

These are intentionally simple first-pass models for the broader project. They are baselines for later work on different alternative types, including scalar, ad hoc, and intermediate cases.

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

A third derived-output folder, [human_model_analysis](./human_model_analysis), now contains the human/model comparison layer.

Shared human/model data and preparation scripts:

- [aggregate_human_responses.py](./human_model_analysis/aggregate_human_responses.py)
- [human_response_rates_by_item_condition.csv](./human_model_analysis/human_response_rates_by_item_condition.csv)
- [join_human_qwen_scores.py](./human_model_analysis/join_human_qwen_scores.py)
- [human_qwen_item_condition_joined.csv](./human_model_analysis/human_qwen_item_condition_joined.csv)

Baseline-model code and outputs:

- [baseline_model/plot_human_model_scatter.py](./human_model_analysis/baseline_model/plot_human_model_scatter.py)
- [baseline_model/plot_qwen_figure9_style.py](./human_model_analysis/baseline_model/plot_qwen_figure9_style.py)
- [baseline_model/plots](./human_model_analysis/baseline_model/plots)

Ordering-model code and outputs:

- [ordering_model/analyze_ordering_model.py](./human_model_analysis/ordering_model/analyze_ordering_model.py)
- [ordering_model/ordering_model_item_condition.csv](./human_model_analysis/ordering_model/ordering_model_item_condition.csv)
- [ordering_model/ordering_model_binary_summary.csv](./human_model_analysis/ordering_model/ordering_model_binary_summary.csv)
- [ordering_model/plots](./human_model_analysis/ordering_model/plots)

The joined human/model table has one row per experiment/condition/item. It keeps both weaker and stronger model scores, plus differences between stronger and weaker log probabilities.

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

## Human Response Aggregation

The human aggregation script is:

```bash
python3 human_model_analysis/aggregate_human_responses.py
```

It reads the original trial-level CSVs in `ronai_xiang_data/`, applies the participant exclusions used in `analysis_codes.R`, and writes:

```text
human_model_analysis/human_response_rates_by_item_condition.csv
```

The included conditions are:

- Experiment 1: `ESI`
- Experiment 2: `Eweak`, `Estrong`
- Experiment 3: `Eonly`
- Experiment 4: `Eonlystrong`

The human/model join script is:

```bash
python3 human_model_analysis/join_human_qwen_scores.py
```

It joins item-level human response rates to the Qwen score table and writes:

```text
human_model_analysis/human_qwen_item_condition_joined.csv
```

## Baseline Model

The baseline model asks whether human response rates are related to Qwen's probability for the stronger alternative in context:

```text
human response probability ~ Qwen P(stronger alternative | context)
```

The baseline scatter-plot script is:

```bash
python3 human_model_analysis/baseline_model/plot_human_model_scatter.py
```

By default, it plots human response probability on the y-axis and Qwen raw stronger-alternative word log probability on the x-axis. Each panel shows both Pearson correlation and Spearman rank correlation.

Current PNG outputs:

- [qwen_human_response_scatter_by_experiment.png](./human_model_analysis/baseline_model/plots/qwen_human_response_scatter_by_experiment.png): human probabilities vs. Qwen log probabilities
- [qwen_probability_human_response_scatter_by_experiment.png](./human_model_analysis/baseline_model/plots/qwen_probability_human_response_scatter_by_experiment.png): human probabilities vs. Qwen probabilities, using `exp(logprob)`
- [qwen_logprob_human_logprob_scatter_by_experiment.png](./human_model_analysis/baseline_model/plots/qwen_logprob_human_logprob_scatter_by_experiment.png): human log probabilities vs. Qwen log probabilities

Regenerate them with:

```bash
python3 human_model_analysis/baseline_model/plot_human_model_scatter.py
python3 human_model_analysis/baseline_model/plot_human_model_scatter.py --x-transform probability
python3 human_model_analysis/baseline_model/plot_human_model_scatter.py --y-transform logprob
```

The default model score column is:

```text
stronger_word_logprob
```

The script also accepts `--score-column`, so the same plotting code can be reused for `stronger_candidate_plus_suffix_logprob` or stronger-minus-weaker contrast columns.

The baseline folder also keeps the optional Figure-9-style Qwen-only diagnostics:

- [qwen_stronger_word_logprob_figure9_style_raw_logprob.png](./human_model_analysis/baseline_model/plots/qwen_stronger_word_logprob_figure9_style_raw_logprob.png)
- [qwen_stronger_word_logprob_figure9_style_minmax_0_100.png](./human_model_analysis/baseline_model/plots/qwen_stronger_word_logprob_figure9_style_minmax_0_100.png)

Regenerate those with:

```bash
python3 human_model_analysis/baseline_model/plot_qwen_figure9_style.py
```

Current correlations for `stronger_word_logprob`:

| Figure | Experiment 1 | Experiment 2 | Experiment 3 | Experiment 4 |
| --- | --- | --- | --- | --- |
| Human probability vs. Qwen logprob | Pearson `0.138`, Spearman `0.160` | Pearson `0.227`, Spearman `0.263` | Pearson `0.236`, Spearman `0.308` | Pearson `-0.117`, Spearman `-0.083` |
| Human probability vs. Qwen probability | Pearson `0.024`, Spearman `0.160` | Pearson `0.229`, Spearman `0.263` | Pearson `-0.102`, Spearman `0.308` | Pearson `-0.115`, Spearman `-0.083` |
| Human logprob vs. Qwen logprob | Pearson `0.189`, Spearman `0.160` | Pearson `0.195`, Spearman `0.263` | Pearson `0.200`, Spearman `0.308` | Pearson `-0.116`, Spearman `-0.083` |

Spearman correlations are unchanged when converting Qwen log probabilities to probabilities because `exp()` preserves rank order. Pearson correlations can change because the transformation changes spacing and leverage among points.

## Query-Trigger Ordering Model

The simple ordering-model script is:

```bash
python3 human_model_analysis/ordering_model/analyze_ordering_model.py
```

It tests the advisor-meeting idea:

```text
when query is more likely than trigger, is the query also more likely to be negated?
```

Definitions used in the script:

- `query`: the stronger alternative that participants judge or negate
- `trigger`: the weaker alternative that Mary says
- `ordering_score`: `log P(query) - log P(trigger)`

For the default word-score analysis:

```text
query_logprob = stronger_word_logprob
trigger_logprob = weaker_word_logprob
ordering_score = stronger_word_logprob - weaker_word_logprob
```

The binary prediction is:

```text
query_more_likely_than_trigger = ordering_score > 0
```

The script reads:

```text
human_model_analysis/human_qwen_item_condition_joined.csv
```

and writes:

- [ordering_model_item_condition.csv](./human_model_analysis/ordering_model/ordering_model_item_condition.csv)
- [ordering_model_binary_summary.csv](./human_model_analysis/ordering_model/ordering_model_binary_summary.csv)
- [qwen_ordering_model_scatter_by_experiment.png](./human_model_analysis/ordering_model/plots/qwen_ordering_model_scatter_by_experiment.png)
- [qwen_ordering_model_binary_split_by_experiment.png](./human_model_analysis/ordering_model/plots/qwen_ordering_model_binary_split_by_experiment.png)

The item-level CSV records the query/trigger labels, their log probabilities, the continuous ordering score, the binary model prediction, and the human response rate.

Current default word-score results:

| Scope | n items | Query > trigger items | Mean response when query > trigger | Mean response when query <= trigger | Difference | Pearson r | Spearman rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 300 | 143 | `0.674` | `0.501` | `0.173` | `0.253` | `0.272` |
| Experiment 1 | 60 | 22 | `0.469` | `0.339` | `0.130` | `0.153` | `0.164` |
| Experiment 2 | 120 | 60 | `0.600` | `0.359` | `0.241` | `0.368` | `0.407` |
| Experiment 3 | 60 | 17 | `0.693` | `0.681` | `0.012` | `0.046` | `0.041` |
| Experiment 4 | 60 | 44 | `0.870` | `0.931` | `-0.061` | `-0.300` | `-0.282` |

Positive differences mean human response rates are higher when Qwen assigns higher log probability to the stronger query than to the weaker trigger. The current overall pattern is positive, but Experiment 4 goes in the opposite direction.

The script also supports candidate-plus-suffix scores:

```bash
python3 human_model_analysis/ordering_model/analyze_ordering_model.py --score-type candidate-plus-suffix
```

That mode uses:

```text
stronger_candidate_plus_suffix_logprob - weaker_candidate_plus_suffix_logprob
```

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

## How to Regenerate the Human/Model Layer

Run from the project root:

```bash
python3 human_model_analysis/aggregate_human_responses.py
python3 human_model_analysis/join_human_qwen_scores.py
python3 human_model_analysis/baseline_model/plot_human_model_scatter.py
python3 human_model_analysis/baseline_model/plot_human_model_scatter.py --x-transform probability
python3 human_model_analysis/baseline_model/plot_human_model_scatter.py --y-transform logprob
python3 human_model_analysis/ordering_model/analyze_ordering_model.py
```

Optional Figure-9-style Qwen-only bar plots can be regenerated with:

```bash
python3 human_model_analysis/baseline_model/plot_qwen_figure9_style.py
```

## Open Questions

These are still unresolved and should be decided explicitly:

- For each model, should the primary score be `word_logprob` or `candidate_plus_suffix_logprob`?
- For future models, should analyses use raw log probabilities, probabilities, ranks, item-wise contrasts, condition-wise centering, or another comparison-layer transformation?
- How should weak-vs-strong condition differences be modeled beyond the current item-level scatter and ordering analyses?
- Should the reconstructed prompts be used exactly as they are, or should odd cases get a final manual audit before more scoring?

Recent advisor-meeting ideas to implement next:

- Clarify whether within-sentence comparisons should use log probability from the shared prefix.
- For Experiment 2, test whether the stronger alternative is more negated in the strong-QUD condition than in the weak-QUD condition.
- Test whether a weak QUD suppresses implicature compared to the SI condition.

## Bottom Line

This repo is now at the “stimuli reconstruction complete, first-pass Qwen scoring complete, baseline model complete, first query-trigger ordering model complete” stage.

The original Ronai & Xiang materials are preserved in `ronai_xiang_data/`. The working prompt tables and generators live in `stimuli_prompts/`. The Qwen scoring script and derived score table live in `model_scores/`. Shared human aggregates and joined tables live in `human_model_analysis/`; model-specific code and plots live in `human_model_analysis/baseline_model/` and `human_model_analysis/ordering_model/`.
