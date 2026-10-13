# Source Code Guide

## Overview

The `src/` package contains the code for both project stages:

- **Stage 1:** Dialogue Act classification using direct Llama prompting (Strategy A), RoBERTa trained on Llama-generated silver labels (Strategy B), and RoBERTa trained on human gold labels (Strategy C).
- **Stage 2:** Sentiment, Emotion, and Intent classification, including the prompting-architecture comparison and the Intent auxiliary-information experiment.

Run all commands from the repository root. See the root [`README.md`](../README.md) for installation, datasets, and final results.

## Main Scripts

| Script | Purpose |
|---|---|
| `config.py` | Central paths, labels, model identifiers, seeds, defaults, and locked final configurations. |
| `prepare_silver_data.py` | Builds the 10,000-example silver candidate pool from raw CMV utterances. |
| `prepare_stage2_split.py` | Creates the fixed 36-example development and 264-example evaluation splits for Stage 2. |
| `llm_annotate.py` | Runs Strategy A and generates Llama-labelled silver data for Strategy B. |
| `train_roberta.py` | Trains and evaluates Strategies B and C. |
| `stage2_pipeline.py` | Runs Stage 2 Experiment 1 using single-prompt or independent task-specific prompts. |
| `evaluate_stage2.py` | Evaluates saved Experiment 1 predictions. |
| `stage2_intent_sweep.py` | Runs Stage 2 Experiment 2 with different auxiliary-label combinations. |
| `evaluate.py` | Shared metrics, confusion matrices, and bootstrap comparisons. |

The remaining scripts support data collection, saved-model prediction, learning-curve plotting, error analysis, ablations, diagnostics, and historical comparisons. They are optional and are not required to run the final experiments.

## Stage 1 Workflow

```text
Strategy A
300 gold development examples -> select the Llama prompt
296 shared examples           -> compare zero-shot and few-shot fairly
locked few-shot prompt        -> evaluate on 332 held-out examples

Strategy B
10k CMV candidates -> Llama few-shot labels -> silver pool
300 gold examples  -> select the configuration
2,500 silver examples -> fine-tune RoBERTa -> evaluate on 332 held-out examples

Strategy C
300 gold examples -> five-fold development evaluation
locked configuration -> train on all 300 gold examples
                     -> evaluate on 332 held-out examples
```

The 332-example held-out set is independent of development and training. It is not used for prompt or configuration selection.

## Stage 2 Workflow

The 300 fully annotated examples are split into 36 development examples for prompt design and 264 examples for final evaluation.

- **Experiment 1:** compares one joint `single_prompt` call with three independent `multi_module` calls for Sentiment, Emotion, and Intent. The multi-module predictions are independent, not sequential.
- **Experiment 2:** tests whether Dialogue Act, Sentiment, and Emotion labels improve Intent classification. `oracle` uses human labels, while `predicted` and `predicted_sp` use saved model predictions.

The predicted conditions require Strategy A Dialogue Act predictions and the corresponding Experiment 1 Sentiment and Emotion predictions. They operate on 260 common rows because the four Strategy A few-shot demonstrations are excluded.

## Running the Main Experiments

### Stage 1 — Strategy A

```bash
python3 -m src.llm_annotate --mode zeroshot
python3 -m src.llm_annotate --mode fewshot
python3 -m src.llm_annotate --heldout
```

Generate the final few-shot silver labels:

```bash
python3 -m src.llm_annotate \
  --mode fewshot \
  --input-csv data/task3_silver_candidates_10k.csv \
  --output-csv data/task3_silver_labeled_10k_fewshot.csv
```

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

`sample_seed` selects the silver subset. `train_seed` controls model-training randomness. Strategy B uses a `source_root`-grouped train/validation split.

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

### Stage 2 — Experiment 1

```bash
python3 -m src.prepare_stage2_split

python3 -m src.stage2_pipeline --mode multi_module --split eval
python3 -m src.evaluate_stage2 --mode multi_module --split eval

python3 -m src.stage2_pipeline --mode single_prompt --split eval
python3 -m src.evaluate_stage2 --mode single_prompt --split eval
```

### Stage 2 — Experiment 2

Run `oracle` first because the predicted conditions reuse its Parent–Reply-only baseline.

```bash
python3 -m src.stage2_intent_sweep --split eval --feature-source oracle
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted_sp
```

Add `--eval-only` to recalculate metrics from complete saved predictions without calling the LLM again.

### Tests

```bash
python3 -m unittest discover -s tests -v
```

## Important Implementation Notes

- RoBERTa receives `Parent` and `Reply` through the tokenizer's text-pair API, with a maximum length of 256 tokens.
- Random seeds cover Python, NumPy, PyTorch, and CUDA.
- Predictions, raw LLM responses, parsing fallbacks, metrics, and confusion matrices are saved under `results/`.
- Stage 2 generation can resume from a compatible partial prediction file.
- Real Llama and RoBERTa runs require model downloads and suitable GPU resources.
- The raw `winning-args-corpus/utterances.jsonl` and final Strategy B checkpoints are not included. Prepared CSV files are included, and checkpoints can be regenerated with `strategy_b_heldout`.
- Optional error-analysis and ablation scripts are not part of the main execution path.
