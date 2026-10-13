# Source Code Guide

## 1. Overview

The `src/` package contains the data preparation, inference, training, evaluation, and analysis code for **depolarAIze**. It supports two research stages:
# Source Code Guide

## 1. Overview

The `src/` package contains the data preparation, inference, training, evaluation, and analysis code for **depolarAIze**. It supports two research stages:

- **Stage 1 — Dialogue Act Classification:** comparison of direct Llama prompting (Strategy A), RoBERTa trained on Llama-generated silver labels (Strategy B), and RoBERTa trained on human gold labels (Strategy C).
- **Stage 2 — Sentiment, Emotion, and Intent:** comparison of joint and independent prompting architectures, followed by experiments on whether auxiliary Dialogue Act, Sentiment, and Emotion labels improve Intent classification.

This document explains the code rather than the research background or full result tables. See the root [`README.md`](../README.md) for the project overview, installation instructions, and final results. Commands below are intended to be run from the repository root.

## 2. Code Structure and File Responsibilities

### A. Shared configuration and utilities

| File | Responsibility |
|---|---|
| `__init__.py` | Marks `src` as a Python package so that scripts can be run with `python3 -m src.<module>`. |
| `config.py` | Defines repository paths, label sets, model identifiers, seeds, training defaults, Stage 2 settings, and the locked final configurations for Strategies A, B, and C. |
| `data_loader.py` | Loads and validates the gold data, normalises expected column names, and provides shared input-building helpers. |
| `utils.py` | Provides random-seed control, Dialogue Act output parsing, result-directory creation, and local/ModelScope/Hugging Face model-path resolution. |
| `evaluate.py` | Implements shared metrics and result writers: Macro-F1, per-class F1, Cohen's kappa, confusion matrices, bootstrap confidence intervals, paired bootstrap comparisons, and multi-label Emotion evaluation. |

### B. Data preparation

| File | Responsibility |
|---|---|
| `prepare_silver_data.py` | Builds the 10,000-example silver candidate pool from raw Winning Arguments utterances. It creates Parent–Reply pairs, removes ID duplicates and exact gold pairs, filters Reply length, and samples with a fixed seed. The required raw `utterances.jsonl` is not included. |
| `prepare_gold_test_candidates.py` | Historical held-out collection script. It downloads the Winning Arguments Corpus through ConvoKit, excludes known gold/silver pairs and unsuitable rows, and samples the first candidate batch for manual annotation. It is not required when using the included final held-out CSV. |
| `prepare_test_topup_candidates.py` | Historical follow-up script that samples an additional held-out annotation batch and writes both an internal tracking copy and an annotator-facing copy. Its heuristic label is for tracking, not a gold annotation. |
| `prepare_heldout_balanced.py` | Creates the final balanced 332-row Stage 1 held-out set by sampling 83 annotated examples from each Dialogue Act class. |
| `prepare_stage2_split.py` | Splits the 300 fully annotated examples into the fixed 36-row Stage 2 development set and 264-row evaluation set, stratified by Intent with seed 42. |

### C. Stage 1 — Dialogue Act Classification

| File | Responsibility |
|---|---|
| `llm_annotate.py` | Runs Strategy A with zero-shot or few-shot Llama prompting, performs the locked Strategy A held-out evaluation, and uses the same prompting code to generate silver labels for Strategy B. The LLM is used for inference only; it is not fine-tuned. |
| `train_roberta.py` | Main RoBERTa entry point for Strategies B and C. It supports grouped silver train/validation splitting, separate sampling and training seeds, class weighting, five-fold Strategy C out-of-fold evaluation, locked held-out runs, a TF-IDF baseline, and training diagnostics. |
| `predict_with_saved_strategy_b.py` | Applies one saved Strategy B checkpoint, or an ensemble obtained by averaging logits across saved seed checkpoints, to a CSV containing `Parent` and `Reply`. The required checkpoints are not included in the current checkout. |
| `evaluate_common_296.py` | Optional historical comparison on the 296 gold examples shared by zero-shot and few-shot Strategy A after excluding the four few-shot demonstrations. It is not part of the final 332-example held-out evaluation. |

### D. Stage 2 — Sentiment, Emotion, and Intent

| File | Responsibility |
|---|---|
| `stage2_pipeline.py` | Runs Experiment 1. `single_prompt` predicts Sentiment, Emotion, and Intent in one Llama call; `multi_module` makes three independent task-specific calls. The multi-module prompts do not pass predictions to one another. The script parses outputs and saves raw and structured predictions. |
| `evaluate_stage2.py` | Evaluates saved Experiment 1 predictions, producing single-label Sentiment/Intent metrics and confusion matrices plus multi-label Emotion metrics. |
| `stage2_intent_sweep.py` | Runs Experiment 2 over all eight combinations of Dialogue Act, Sentiment, and Emotion, including the no-auxiliary-input baseline. It supports human `oracle` labels and two sources of model-predicted labels, and computes paired comparisons against the Parent–Reply-only baseline. |

### E. Error analysis, ablations, and visualisation

| File | Responsibility |
|---|---|
| `analyze_disagree_errors.py` | Optional Stage 1 error analysis that collects final Strategy B/C `disagree` errors and measures text features including negation, hedges, contractions, spelling heuristics, length, and question form. |
| `add_text_features.py` | Applies the same text-feature extraction to the complete gold dataset and writes an analysis-ready CSV. It uses the NLTK English word list through `analyze_disagree_errors.py`. |
| `ablation_parent_negation.py` | Optional Strategy B context-sensitivity experiment. It edits Parent negation for selected examples, retrains the locked Strategy B configuration across seeds, and compares predictions before and after the edit. |
| `ablation_parent_negation_strategy_a.py` | Runs the corresponding Parent-negation analysis with Strategy A's Llama prompt. |
| `score_agree_disagree_relabel.py` | Scores a completed blind re-annotation of 27 difficult `agree`/`disagree` examples and can calculate raw agreement between two annotators. |
| `plot_strategy_b_learning_curve.py` | Aggregates standardised Strategy B run metrics and writes the learning-curve data table and plot. |

### F. Experiment result summaries

| File | Responsibility |
|---|---|
| `summarize_strategy_c_grid.py` | Collects Strategy C configuration summaries from the result subdirectories and writes one ranked warmup/epoch grid table. |

## 3. Execution Workflow

The scripts are not one long sequential pipeline. Stage 1 contains three alternative strategies, and Stage 2 contains two separate experiments.

### Stage 1 workflow

```text
                         DEVELOPMENT / TRAINING                         FINAL TEST

Strategy A     300 gold examples for prompt selection             332 held-out examples
               (296 shared examples for the fair                   Llama inference only
               zero-shot/few-shot comparison)                      no fine-tuning
                      |                                                   ^
                      +--> lock few-shot prompt --------------------------+

Strategy B     raw CMV data --> 10k candidates --> Llama labels
                                      |
                              silver training pool
                                      |
               300 gold examples --> select silver size,
               sample seed, and class-weight setting
                                      |
                      +--> lock 2,500-silver configuration
                                      |
                              fine-tune RoBERTa -----------------> 332 held-out examples

Strategy C     300 gold examples
                      |
               five-fold cross-validation for configuration selection
                      |
               lock configuration and train on all 300 ----------> 332 held-out examples
```

The 300-example gold set has a different role in each strategy. Strategy A uses it to choose the prompt; the common zero-shot/few-shot comparison contains 296 rows because the four few-shot demonstrations are excluded. Strategy B uses the gold set to select its configuration but trains RoBERTa on silver data. Its final configuration uses 2,500 few-shot-labelled silver examples. Strategy C uses the gold set both for five-fold development evaluation and for final training. After configuration selection, it retrains on all 300 gold examples.

All three locked strategies are finally evaluated on the same independent file, `data/cmv_test_candidates_heldout_balanced.csv` (332 rows). Strategy A runs inference on it, while Strategies B and C first train their final RoBERTa models. The held-out data are not used for prompt choice, hyperparameter selection, or training.

### Stage 2 workflow

```text
cmv_300_gold_final.csv
        +--> prepare_stage2_split.py
                +--> 36-row dev split (prompt development)
                +--> 264-row eval split (final evaluation)

Experiment 1
        +--> stage2_pipeline.py --mode single_prompt
        |       +--> one call predicts all three tasks
        +--> stage2_pipeline.py --mode multi_module
                +--> three independent calls: Sentiment, Emotion, Intent
        +--> evaluate_stage2.py

Experiment 2
        +--> stage2_intent_sweep.py --feature-source oracle
        +--> stage2_intent_sweep.py --feature-source predicted
        +--> stage2_intent_sweep.py --feature-source predicted_sp
```

Experiment 1 compares prompting architectures. It does **not** pass one module's prediction to another module. Experiment 2 is a separate auxiliary-information experiment: it adds different combinations of Dialogue Act, Sentiment, and Emotion labels to the Intent prompt.

For Experiment 2:

- `oracle` loads human-annotated auxiliary labels from the Stage 2 split;
- `predicted` loads Strategy A Dialogue Act predictions and multi-module Sentiment/Emotion predictions;
- `predicted_sp` loads the same Strategy A Dialogue Act predictions and single-prompt Sentiment/Emotion predictions.

The predicted-label conditions require the corresponding saved Stage 1 and Experiment 1 prediction files. They use the 260 rows common to these files because the four Strategy A few-shot demonstrations are excluded.

## 4. Main Scripts: Inputs, Processing, and Outputs

### `llm_annotate.py`

- **Purpose / used in:** Strategy A inference and Strategy B silver-label generation.
- **Inputs:** by default, `data/cmv_300_gold_final.csv`; with `--input-csv`, any CSV containing `Parent` and `Reply`; with `--heldout`, the fixed Stage 1 held-out CSV.
- **Processing:** builds zero-shot or few-shot Dialogue Act prompts, runs Llama 3.1 8B Instruct, parses one of the four labels, and records any fallback to `statement`. The four demonstrations are excluded from the few-shot development run. `evaluate_common_296.py` also excludes them from the zero-shot predictions when comparing the two prompts on identical rows.
- **Outputs:** Strategy A predictions, metrics, and confusion matrices under `results/strategy_a/` or `results/strategy_a_heldout/`; custom silver predictions at `--output-csv` plus a `.summary.json` file.
- **Dependencies:** real runs require the configured Llama model and a GPU-capable inference environment. `--bootstrap-existing` only rescores saved predictions and does not call the model.

### `train_roberta.py`

- **Purpose / used in:** all Strategy B and C fine-tuning, development evaluation, diagnostics, and final held-out evaluation.
- **Strategy B input:** a labelled silver CSV with `Parent`, `Reply`, `Dialogue_act`, and `source_root`. The final workflow uses `data/task3_silver_labeled_10k_fewshot.csv`.
- **Strategy B processing:** samples rows with `sample_seed`, performs a `source_root`-grouped train/validation split, tokenises Parent and Reply as a text pair, and trains once per `train_seed`. Sampling randomness and model-training randomness are therefore controlled separately.
- **Strategy C input:** `data/cmv_300_gold_final.csv`.
- **Strategy C processing:** runs stratified five-fold out-of-fold evaluation for each training seed. It supports RoBERTa, a TF-IDF/logistic-regression baseline, class-weight ablations, and warmup/epoch overrides.
- **Outputs:** per-run prediction CSVs, metrics JSON files, confusion matrices, run tables, and aggregate summaries under the requested results directory. Locked held-out commands write to `results/strategy_b_heldout/` and `results/strategy_c_heldout/`.
- **Dependencies:** RoBERTa runs require PyTorch, Transformers, ModelScope model access, and normally a GPU. `--dry-run` checks data/split logic without full training.

### `stage2_pipeline.py`

- **Purpose / used in:** Stage 2 Experiment 1.
- **Inputs:** `data/cmv_300_gold_stage2_dev.csv` or `data/cmv_300_gold_stage2_eval.csv`, selected with `--split`.
- **Processing:** runs either one joint prompt (`single_prompt`) or three independent prompts (`multi_module`). It parses Sentiment and Intent as single-label outputs and Emotion as a six-category multi-label output. Raw responses and fallback flags are retained. Existing partial CSVs are resumed rather than restarted when their schema matches.
- **Outputs:** `results/stage2/<mode>_<split>_predictions.csv`.
- **Dependencies:** run `prepare_stage2_split.py` if the split files are unavailable. Real inference requires the configured Llama model.

### `evaluate_stage2.py`

- **Purpose / used in:** evaluation of Stage 2 Experiment 1 predictions.
- **Inputs:** the prediction CSV produced by `stage2_pipeline.py` for the selected mode and split.
- **Processing:** evaluates Sentiment and Intent as single-label tasks and Emotion as a multi-label task.
- **Outputs:** metrics JSON files and confusion matrices under `results/stage2/`, together with a printed summary.
- **Dependencies:** generation must be complete before evaluation. This script does not call the LLM.

### `stage2_intent_sweep.py`

- **Purpose / used in:** Stage 2 Experiment 2.
- **Inputs:** the Stage 2 dev/eval split plus auxiliary labels selected by `--feature-source`. Predicted conditions load Strategy A few-shot predictions and the relevant Experiment 1 predictions.
- **Processing:** runs Intent prompts for all eight auxiliary-input combinations. It saves results incrementally, evaluates each combination, calculates paired bootstrap differences against the baseline, and compares oracle and predicted results on common rows.
- **Outputs:** prediction CSVs, per-combination metrics JSON files, summary CSVs, `*_vs_base_paired.csv`, and oracle-versus-predicted comparison CSVs under `results/stage2/`.
- **Dependencies:** run `oracle` first because the predicted conditions reuse its Parent–Reply-only baseline. Complete Experiment 1 and Strategy A predictions are required for the predicted conditions. `--eval-only` rescores existing complete predictions without LLM generation.

### Essential preparation scripts

- `prepare_stage2_split.py` deterministically regenerates the 36/264 Stage 2 split.
- `prepare_silver_data.py` regenerates silver candidates only when the external raw Winning Arguments `utterances.jsonl` is available.
- `prepare_heldout_balanced.py` recreates the balanced Stage 1 held-out set from the expanded, manually annotated held-out data.

## 5. Running the Code

Install the environment as described in the root README. The following commands match the current command-line interfaces; they were checked against the source code but were not rerun as part of this documentation update.

### Stage 1 — Strategy A

```bash
# Development runs
python3 -m src.llm_annotate --mode zeroshot
python3 -m src.llm_annotate --mode fewshot

# Locked held-out evaluation
python3 -m src.llm_annotate --heldout

# Regenerate the final few-shot silver labels
python3 -m src.llm_annotate \
  --mode fewshot \
  --input-csv data/task3_silver_candidates_10k.csv \
  --output-csv data/task3_silver_labeled_10k_fewshot.csv
```

Useful options are `--limit N` for a partial run, `--mock` for deterministic local test output, and `--bootstrap-existing --n-boot 2000` for rescoring existing Strategy A predictions.

### Stage 1 — Strategy B

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k_fewshot.csv \
  --sizes 2500 \
  --sample-seeds 123 \
  --train-seeds 42 123 2026 \
  --class-weights \
  --results-dir results/strategy_b_fewshot

python3 -m src.train_roberta strategy_b_heldout
```

`--sample-seeds` controls which silver rows are selected. `--train-seeds` controls model initialisation, data order, dropout, and other training randomness. Use `--no-class-weights` for the corresponding ablation. The held-out command uses only the locked settings in `config.py` and retrains the three seed models.

### Stage 1 — Strategy C

```bash
python3 -m src.train_roberta strategy_c \
  --model roberta \
  --seeds 42 123 2026 \
  --split-seed 42 \
  --class-weights \
  --warmup-steps 20 \
  --epochs 8

python3 -m src.train_roberta strategy_c_heldout
```

The development command performs five-fold out-of-fold evaluation. `--model tfidf` selects the optional TF-IDF/logistic-regression baseline. `strategy_c_diagnose` is an analysis mode for checking whether models fit their own training folds; it is not required for the final workflow.

### Stage 2 — Experiment 1

```bash
python3 -m src.prepare_stage2_split

python3 -m src.stage2_pipeline --mode multi_module --split eval
python3 -m src.evaluate_stage2 --mode multi_module --split eval

python3 -m src.stage2_pipeline --mode single_prompt --split eval
python3 -m src.evaluate_stage2 --mode single_prompt --split eval
```

`--mode` selects one joint prompt or three independent task-specific prompts. `--split dev` uses the 36 prompt-development examples; `--split eval` uses the 264-example final evaluation set. The pipeline also supports `--limit` and `--mock` for smoke tests.

### Stage 2 — Experiment 2

```bash
# Run oracle first
python3 -m src.stage2_intent_sweep --split eval --feature-source oracle

# Predicted Dialogue Act + multi-module Sentiment/Emotion
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted

# Predicted Dialogue Act + single-prompt Sentiment/Emotion
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted_sp
```

To recalculate metrics from complete saved predictions without making new LLM calls:

```bash
python3 -m src.stage2_intent_sweep \
  --split eval \
  --feature-source predicted \
  --eval-only
```

### Apply saved Strategy B checkpoints

```bash
python3 -m src.predict_with_saved_strategy_b \
  --input path/to/parent_reply_rows.csv \
  --output path/to/predictions.csv
```

Add `--seed 42` to use one checkpoint instead of averaging logits across all saved final seeds. This command requires checkpoints under `results/strategy_b_heldout/models/`. They are not included in the current checkout and must first be regenerated with `strategy_b_heldout`.

### Tests

```bash
python3 -m unittest discover -s tests -v
```

Real Llama inference and RoBERTa fine-tuning require model downloads on first use and appropriate compute. The Llama 3.1 8B runs are intended for a GPU environment. Evaluation-only, plotting, summarisation, and most data-preparation steps can run on CPU.

## 6. Shared Utilities and Implementation Notes

### Configuration and loading

`config.py` is the central source for data/result paths, label orders, model identifiers, and locked final configurations. Llama and RoBERTa are configured to load from ModelScope by default. `data_loader.py` validates the required gold columns and normalises the historical `Int` column name to `Intent` when necessary.

### Metrics

Macro-F1 is the principal metric. The shared evaluator also records per-class scores, accuracy, Cohen's kappa, and confusion matrices for single-label tasks. Emotion uses one binary decision per category and macro-averages across its six labels. Bootstrap intervals resample evaluation rows; they do not represent variation from retraining a model or regenerating LLM output.

### Reproducibility

`utils.set_seed()` covers Python `random`, NumPy, PyTorch, and CUDA. Strategy B deliberately separates `sample_seed` from `train_seed`. Its inner validation split is grouped by `source_root` so that one Reddit discussion thread does not occur in both training and validation. Strategy C uses a fixed stratified fold split and reports results across training seeds.

### Parent–Reply tokenisation

RoBERTa receives Parent and Reply through the tokenizer's text-pair interface rather than by manually inserting a literal `[SEP]` string. The tokenizer therefore creates model-appropriate special tokens while applying the configured maximum length of 256 tokens.

### Parsing and fallbacks

Strategy A maps unparseable Dialogue Act output to `statement` and records that a fallback occurred. Stage 2 applies task-specific parsers: Sentiment and Intent require one valid label, while Emotion accepts multiple labels. The parser checks the first output line before considering an unambiguous label in the full response. Stage 2 fallbacks are `Neutral` for Sentiment, `Others` for Intent, and no recognised Emotion category; raw model responses and fallback flags are saved for inspection.

### Saved predictions and evaluation

Inference scripts retain raw output, parsed predictions, gold labels where available, and fallback indicators. Evaluation scripts operate on these saved files, allowing metrics to be recalculated without another model call. Stage 2 generation is resumable: a compatible partial output file is continued from the next unfinished row.

### Optional and historical analyses

The error-analysis, feature, negation-ablation, common-296, learning-curve, and Strategy C grid-summary scripts support secondary analyses. They are not required for the locked Stage 1 or Stage 2 evaluations. Historical result directories remain in the repository, so use the locked settings in `config.py` and the root README to identify the final configurations.

### Known reproducibility limits

- The raw Winning Arguments `utterances.jsonl` used to reconstruct the silver pool is not included. The prepared candidate and labelled silver CSVs are included instead.
- Saved final Strategy B model checkpoints are not included. They must be regenerated before `predict_with_saved_strategy_b.py` can be used.
- Package versions are not fully pinned to the original experiment environment. See `requirements.txt` and the root README for the currently documented environment.
- Real-model commands were not rerun while creating this documentation; the commands and paths above were verified against the current source interfaces.

- **Stage 1 — Dialogue Act Classification:** comparison of direct Llama prompting (Strategy A), RoBERTa trained on Llama-generated silver labels (Strategy B), and RoBERTa trained on human gold labels (Strategy C).
- **Stage 2 — Sentiment, Emotion, and Intent:** comparison of joint and independent prompting architectures, followed by experiments on whether auxiliary Dialogue Act, Sentiment, and Emotion labels improve Intent classification.

This document explains the code rather than the research background or full result tables. See the root [`README.md`](../README.md) for the project overview, installation instructions, and final results. Commands below are intended to be run from the repository root.

## 2. Code Structure and File Responsibilities

### A. Shared configuration and utilities

| File | Responsibility |
|---|---|
| `__init__.py` | Marks `src` as a Python package so that scripts can be run with `python3 -m src.<module>`. |
| `config.py` | Defines repository paths, label sets, model identifiers, seeds, training defaults, Stage 2 settings, and the locked final configurations for Strategies A, B, and C. |
| `data_loader.py` | Loads and validates the gold data, normalises expected column names, and provides shared input-building helpers. |
| `utils.py` | Provides random-seed control, Dialogue Act output parsing, result-directory creation, and local/ModelScope/Hugging Face model-path resolution. |
| `evaluate.py` | Implements shared metrics and result writers: Macro-F1, per-class F1, Cohen's kappa, confusion matrices, bootstrap confidence intervals, paired bootstrap comparisons, and multi-label Emotion evaluation. |

### B. Data preparation

| File | Responsibility |
|---|---|
| `prepare_silver_data.py` | Builds the 10,000-example silver candidate pool from raw Winning Arguments utterances. It creates Parent–Reply pairs, removes ID duplicates and exact gold pairs, filters Reply length, and samples with a fixed seed. The required raw `utterances.jsonl` is not included. |
| `prepare_gold_test_candidates.py` | Historical held-out collection script. It downloads the Winning Arguments Corpus through ConvoKit, excludes known gold/silver pairs and unsuitable rows, and samples the first candidate batch for manual annotation. It is not required when using the included final held-out CSV. |
| `prepare_test_topup_candidates.py` | Historical follow-up script that samples an additional held-out annotation batch and writes both an internal tracking copy and an annotator-facing copy. Its heuristic label is for tracking, not a gold annotation. |
| `prepare_heldout_balanced.py` | Creates the final balanced 332-row Stage 1 held-out set by sampling 83 annotated examples from each Dialogue Act class. |
| `prepare_stage2_split.py` | Splits the 300 fully annotated examples into the fixed 36-row Stage 2 development set and 264-row evaluation set, stratified by Intent with seed 42. |

### C. Stage 1 — Dialogue Act Classification

| File | Responsibility |
|---|---|
| `llm_annotate.py` | Runs Strategy A with zero-shot or few-shot Llama prompting, performs the locked Strategy A held-out evaluation, and uses the same prompting code to generate silver labels for Strategy B. The LLM is used for inference only; it is not fine-tuned. |
| `train_roberta.py` | Main RoBERTa entry point for Strategies B and C. It supports grouped silver train/validation splitting, separate sampling and training seeds, class weighting, five-fold Strategy C out-of-fold evaluation, locked held-out runs, a TF-IDF baseline, and training diagnostics. |
| `predict_with_saved_strategy_b.py` | Applies one saved Strategy B checkpoint, or an ensemble obtained by averaging logits across saved seed checkpoints, to a CSV containing `Parent` and `Reply`. The required checkpoints are not included in the current checkout. |
| `evaluate_common_296.py` | Optional historical comparison on the 296 gold examples shared by zero-shot and few-shot Strategy A after excluding the four few-shot demonstrations. It is not part of the final 332-example held-out evaluation. |

### D. Stage 2 — Sentiment, Emotion, and Intent

| File | Responsibility |
|---|---|
| `stage2_pipeline.py` | Runs Experiment 1. `single_prompt` predicts Sentiment, Emotion, and Intent in one Llama call; `multi_module` makes three independent task-specific calls. The multi-module prompts do not pass predictions to one another. The script parses outputs and saves raw and structured predictions. |
| `evaluate_stage2.py` | Evaluates saved Experiment 1 predictions, producing single-label Sentiment/Intent metrics and confusion matrices plus multi-label Emotion metrics. |
| `stage2_intent_sweep.py` | Runs Experiment 2 over all eight combinations of Dialogue Act, Sentiment, and Emotion, including the no-auxiliary-input baseline. It supports human `oracle` labels and two sources of model-predicted labels, and computes paired comparisons against the Parent–Reply-only baseline. |

### E. Error analysis, ablations, and visualisation

| File | Responsibility |
|---|---|
| `analyze_disagree_errors.py` | Optional Stage 1 error analysis that collects final Strategy B/C `disagree` errors and measures text features including negation, hedges, contractions, spelling heuristics, length, and question form. |
| `add_text_features.py` | Applies the same text-feature extraction to the complete gold dataset and writes an analysis-ready CSV. It uses the NLTK English word list through `analyze_disagree_errors.py`. |
| `ablation_parent_negation.py` | Optional Strategy B context-sensitivity experiment. It edits Parent negation for selected examples, retrains the locked Strategy B configuration across seeds, and compares predictions before and after the edit. |
| `ablation_parent_negation_strategy_a.py` | Runs the corresponding Parent-negation analysis with Strategy A's Llama prompt. |
| `score_agree_disagree_relabel.py` | Scores a completed blind re-annotation of 27 difficult `agree`/`disagree` examples and can calculate raw agreement between two annotators. |
| `plot_strategy_b_learning_curve.py` | Aggregates standardised Strategy B run metrics and writes the learning-curve data table and plot. |

### F. Experiment result summaries

| File | Responsibility |
|---|---|
| `summarize_strategy_c_grid.py` | Collects Strategy C configuration summaries from the result subdirectories and writes one ranked warmup/epoch grid table. |

## 3. Execution Workflow

The scripts are not one long sequential pipeline. Stage 1 contains three alternative strategies, and Stage 2 contains two separate experiments.

### Stage 1 workflow

```text
                         DEVELOPMENT / TRAINING                         FINAL TEST

Strategy A     300 gold examples for prompt selection             332 held-out examples
               (296 shared examples for the fair                   Llama inference only
               zero-shot/few-shot comparison)                      no fine-tuning
                      |                                                   ^
                      +--> lock few-shot prompt --------------------------+

Strategy B     raw CMV data --> 10k candidates --> Llama labels
                                      |
                              silver training pool
                                      |
               300 gold examples --> select silver size,
               sample seed, and class-weight setting
                                      |
                      +--> lock 2,500-silver configuration
                                      |
                              fine-tune RoBERTa -----------------> 332 held-out examples

Strategy C     300 gold examples
                      |
               five-fold cross-validation for configuration selection
                      |
               lock configuration and train on all 300 ----------> 332 held-out examples
```

The 300-example gold set has a different role in each strategy. Strategy A uses it to choose the prompt; the common zero-shot/few-shot comparison contains 296 rows because the four few-shot demonstrations are excluded. Strategy B uses the gold set to select its configuration but trains RoBERTa on silver data. Its final configuration uses 2,500 few-shot-labelled silver examples. Strategy C uses the gold set both for five-fold development evaluation and for final training. After configuration selection, it retrains on all 300 gold examples.

All three locked strategies are finally evaluated on the same independent file, `data/cmv_test_candidates_heldout_balanced.csv` (332 rows). Strategy A runs inference on it, while Strategies B and C first train their final RoBERTa models. The held-out data are not used for prompt choice, hyperparameter selection, or training.

### Stage 2 workflow

```text
cmv_300_gold_final.csv
        +--> prepare_stage2_split.py
                +--> 36-row dev split (prompt development)
                +--> 264-row eval split (final evaluation)

Experiment 1
        +--> stage2_pipeline.py --mode single_prompt
        |       +--> one call predicts all three tasks
        +--> stage2_pipeline.py --mode multi_module
                +--> three independent calls: Sentiment, Emotion, Intent
        +--> evaluate_stage2.py

Experiment 2
        +--> stage2_intent_sweep.py --feature-source oracle
        +--> stage2_intent_sweep.py --feature-source predicted
        +--> stage2_intent_sweep.py --feature-source predicted_sp
```

Experiment 1 compares prompting architectures. It does **not** pass one module's prediction to another module. Experiment 2 is a separate auxiliary-information experiment: it adds different combinations of Dialogue Act, Sentiment, and Emotion labels to the Intent prompt.

For Experiment 2:

- `oracle` loads human-annotated auxiliary labels from the Stage 2 split;
- `predicted` loads Strategy A Dialogue Act predictions and multi-module Sentiment/Emotion predictions;
- `predicted_sp` loads the same Strategy A Dialogue Act predictions and single-prompt Sentiment/Emotion predictions.

The predicted-label conditions require the corresponding saved Stage 1 and Experiment 1 prediction files. They use the 260 rows common to these files because the four Strategy A few-shot demonstrations are excluded.

## 4. Main Scripts: Inputs, Processing, and Outputs

### `llm_annotate.py`

- **Purpose / used in:** Strategy A inference and Strategy B silver-label generation.
- **Inputs:** by default, `data/cmv_300_gold_final.csv`; with `--input-csv`, any CSV containing `Parent` and `Reply`; with `--heldout`, the fixed Stage 1 held-out CSV.
- **Processing:** builds zero-shot or few-shot Dialogue Act prompts, runs Llama 3.1 8B Instruct, parses one of the four labels, and records any fallback to `statement`. The four demonstrations are excluded from the few-shot development run. `evaluate_common_296.py` also excludes them from the zero-shot predictions when comparing the two prompts on identical rows.
- **Outputs:** Strategy A predictions, metrics, and confusion matrices under `results/strategy_a/` or `results/strategy_a_heldout/`; custom silver predictions at `--output-csv` plus a `.summary.json` file.
- **Dependencies:** real runs require the configured Llama model and a GPU-capable inference environment. `--bootstrap-existing` only rescores saved predictions and does not call the model.

### `train_roberta.py`

- **Purpose / used in:** all Strategy B and C fine-tuning, development evaluation, diagnostics, and final held-out evaluation.
- **Strategy B input:** a labelled silver CSV with `Parent`, `Reply`, `Dialogue_act`, and `source_root`. The final workflow uses `data/task3_silver_labeled_10k_fewshot.csv`.
- **Strategy B processing:** samples rows with `sample_seed`, performs a `source_root`-grouped train/validation split, tokenises Parent and Reply as a text pair, and trains once per `train_seed`. Sampling randomness and model-training randomness are therefore controlled separately.
- **Strategy C input:** `data/cmv_300_gold_final.csv`.
- **Strategy C processing:** runs stratified five-fold out-of-fold evaluation for each training seed. It supports RoBERTa, a TF-IDF/logistic-regression baseline, class-weight ablations, and warmup/epoch overrides.
- **Outputs:** per-run prediction CSVs, metrics JSON files, confusion matrices, run tables, and aggregate summaries under the requested results directory. Locked held-out commands write to `results/strategy_b_heldout/` and `results/strategy_c_heldout/`.
- **Dependencies:** RoBERTa runs require PyTorch, Transformers, ModelScope model access, and normally a GPU. `--dry-run` checks data/split logic without full training.

### `stage2_pipeline.py`

- **Purpose / used in:** Stage 2 Experiment 1.
- **Inputs:** `data/cmv_300_gold_stage2_dev.csv` or `data/cmv_300_gold_stage2_eval.csv`, selected with `--split`.
- **Processing:** runs either one joint prompt (`single_prompt`) or three independent prompts (`multi_module`). It parses Sentiment and Intent as single-label outputs and Emotion as a six-category multi-label output. Raw responses and fallback flags are retained. Existing partial CSVs are resumed rather than restarted when their schema matches.
- **Outputs:** `results/stage2/<mode>_<split>_predictions.csv`.
- **Dependencies:** run `prepare_stage2_split.py` if the split files are unavailable. Real inference requires the configured Llama model.

### `evaluate_stage2.py`

- **Purpose / used in:** evaluation of Stage 2 Experiment 1 predictions.
- **Inputs:** the prediction CSV produced by `stage2_pipeline.py` for the selected mode and split.
- **Processing:** evaluates Sentiment and Intent as single-label tasks and Emotion as a multi-label task.
- **Outputs:** metrics JSON files and confusion matrices under `results/stage2/`, together with a printed summary.
- **Dependencies:** generation must be complete before evaluation. This script does not call the LLM.

### `stage2_intent_sweep.py`

- **Purpose / used in:** Stage 2 Experiment 2.
- **Inputs:** the Stage 2 dev/eval split plus auxiliary labels selected by `--feature-source`. Predicted conditions load Strategy A few-shot predictions and the relevant Experiment 1 predictions.
- **Processing:** runs Intent prompts for all eight auxiliary-input combinations. It saves results incrementally, evaluates each combination, calculates paired bootstrap differences against the baseline, and compares oracle and predicted results on common rows.
- **Outputs:** prediction CSVs, per-combination metrics JSON files, summary CSVs, `*_vs_base_paired.csv`, and oracle-versus-predicted comparison CSVs under `results/stage2/`.
- **Dependencies:** run `oracle` first because the predicted conditions reuse its Parent–Reply-only baseline. Complete Experiment 1 and Strategy A predictions are required for the predicted conditions. `--eval-only` rescores existing complete predictions without LLM generation.

### Essential preparation scripts

- `prepare_stage2_split.py` deterministically regenerates the 36/264 Stage 2 split.
- `prepare_silver_data.py` regenerates silver candidates only when the external raw Winning Arguments `utterances.jsonl` is available.
- `prepare_heldout_balanced.py` recreates the balanced Stage 1 held-out set from the expanded, manually annotated held-out data.

## 5. Running the Code

Install the environment as described in the root README. The following commands match the current command-line interfaces; they were checked against the source code but were not rerun as part of this documentation update.

### Stage 1 — Strategy A

```bash
# Development runs
python3 -m src.llm_annotate --mode zeroshot
python3 -m src.llm_annotate --mode fewshot

# Locked held-out evaluation
python3 -m src.llm_annotate --heldout

# Regenerate the final few-shot silver labels
python3 -m src.llm_annotate \
  --mode fewshot \
  --input-csv data/task3_silver_candidates_10k.csv \
  --output-csv data/task3_silver_labeled_10k_fewshot.csv
```

Useful options are `--limit N` for a partial run, `--mock` for deterministic local test output, and `--bootstrap-existing --n-boot 2000` for rescoring existing Strategy A predictions.

### Stage 1 — Strategy B

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k_fewshot.csv \
  --sizes 2500 \
  --sample-seeds 123 \
  --train-seeds 42 123 2026 \
  --class-weights \
  --results-dir results/strategy_b_fewshot

python3 -m src.train_roberta strategy_b_heldout
```

`--sample-seeds` controls which silver rows are selected. `--train-seeds` controls model initialisation, data order, dropout, and other training randomness. Use `--no-class-weights` for the corresponding ablation. The held-out command uses only the locked settings in `config.py` and retrains the three seed models.

### Stage 1 — Strategy C

```bash
python3 -m src.train_roberta strategy_c \
  --model roberta \
  --seeds 42 123 2026 \
  --split-seed 42 \
  --class-weights \
  --warmup-steps 20 \
  --epochs 8

python3 -m src.train_roberta strategy_c_heldout
```

The development command performs five-fold out-of-fold evaluation. `--model tfidf` selects the optional TF-IDF/logistic-regression baseline. `strategy_c_diagnose` is an analysis mode for checking whether models fit their own training folds; it is not required for the final workflow.

### Stage 2 — Experiment 1

```bash
python3 -m src.prepare_stage2_split

python3 -m src.stage2_pipeline --mode multi_module --split eval
python3 -m src.evaluate_stage2 --mode multi_module --split eval

python3 -m src.stage2_pipeline --mode single_prompt --split eval
python3 -m src.evaluate_stage2 --mode single_prompt --split eval
```

`--mode` selects one joint prompt or three independent task-specific prompts. `--split dev` uses the 36 prompt-development examples; `--split eval` uses the 264-example final evaluation set. The pipeline also supports `--limit` and `--mock` for smoke tests.

### Stage 2 — Experiment 2

```bash
# Run oracle first
python3 -m src.stage2_intent_sweep --split eval --feature-source oracle

# Predicted Dialogue Act + multi-module Sentiment/Emotion
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted

# Predicted Dialogue Act + single-prompt Sentiment/Emotion
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted_sp
```

To recalculate metrics from complete saved predictions without making new LLM calls:

```bash
python3 -m src.stage2_intent_sweep \
  --split eval \
  --feature-source predicted \
  --eval-only
```

### Apply saved Strategy B checkpoints

```bash
python3 -m src.predict_with_saved_strategy_b \
  --input path/to/parent_reply_rows.csv \
  --output path/to/predictions.csv
```

Add `--seed 42` to use one checkpoint instead of averaging logits across all saved final seeds. This command requires checkpoints under `results/strategy_b_heldout/models/`. They are not included in the current checkout and must first be regenerated with `strategy_b_heldout`.

### Tests

```bash
python3 -m unittest discover -s tests -v
```

Real Llama inference and RoBERTa fine-tuning require model downloads on first use and appropriate compute. The Llama 3.1 8B runs are intended for a GPU environment. Evaluation-only, plotting, summarisation, and most data-preparation steps can run on CPU.

## 6. Shared Utilities and Implementation Notes

### Configuration and loading

`config.py` is the central source for data/result paths, label orders, model identifiers, and locked final configurations. Llama and RoBERTa are configured to load from ModelScope by default. `data_loader.py` validates the required gold columns and normalises the historical `Int` column name to `Intent` when necessary.

### Metrics

Macro-F1 is the principal metric. The shared evaluator also records per-class scores, accuracy, Cohen's kappa, and confusion matrices for single-label tasks. Emotion uses one binary decision per category and macro-averages across its six labels. Bootstrap intervals resample evaluation rows; they do not represent variation from retraining a model or regenerating LLM output.

### Reproducibility

`utils.set_seed()` covers Python `random`, NumPy, PyTorch, and CUDA. Strategy B deliberately separates `sample_seed` from `train_seed`. Its inner validation split is grouped by `source_root` so that one Reddit discussion thread does not occur in both training and validation. Strategy C uses a fixed stratified fold split and reports results across training seeds.

### Parent–Reply tokenisation

RoBERTa receives Parent and Reply through the tokenizer's text-pair interface rather than by manually inserting a literal `[SEP]` string. The tokenizer therefore creates model-appropriate special tokens while applying the configured maximum length of 256 tokens.

### Parsing and fallbacks

Strategy A maps unparseable Dialogue Act output to `statement` and records that a fallback occurred. Stage 2 applies task-specific parsers: Sentiment and Intent require one valid label, while Emotion accepts multiple labels. The parser checks the first output line before considering an unambiguous label in the full response. Stage 2 fallbacks are `Neutral` for Sentiment, `Others` for Intent, and no recognised Emotion category; raw model responses and fallback flags are saved for inspection.

### Saved predictions and evaluation

Inference scripts retain raw output, parsed predictions, gold labels where available, and fallback indicators. Evaluation scripts operate on these saved files, allowing metrics to be recalculated without another model call. Stage 2 generation is resumable: a compatible partial output file is continued from the next unfinished row.

### Optional and historical analyses

The error-analysis, feature, negation-ablation, common-296, learning-curve, and Strategy C grid-summary scripts support secondary analyses. They are not required for the locked Stage 1 or Stage 2 evaluations. Historical result directories remain in the repository, so use the locked settings in `config.py` and the root README to identify the final configurations.

### Known reproducibility limits

- The raw Winning Arguments `utterances.jsonl` used to reconstruct the silver pool is not included. The prepared candidate and labelled silver CSVs are included instead.
- Saved final Strategy B model checkpoints are not included. They must be regenerated before `predict_with_saved_strategy_b.py` can be used.
- Package versions are not fully pinned to the original experiment environment. See `requirements.txt` and the root README for the currently documented environment.
- Real-model commands were not rerun while creating this documentation; the commands and paths above were verified against the current source interfaces.
