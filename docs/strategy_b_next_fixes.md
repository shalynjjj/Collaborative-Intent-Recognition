# Strategy B — Next Fixes and Server Runs

## What changed

- Silver train/validation splitting is grouped by `source_root`.
- A Reddit thread cannot appear in both training and validation.
- Every run records `source_root_overlap_count`; the required value is `0`.
- `sample_seed` and `train_seed` are separate parameters.
- `sample_seed` selects a deterministic, nested subset from the full silver pool.
- `train_seed` controls model/training randomness.
- Output filenames include size, sample seed, train seed, and class-weight setting.
- Prediction counts are saved for all four classes.
- Runs with a missing predicted class are marked with `class_collapse=true`.
- Strategy B summaries are rebuilt from all existing metrics files instead of only
  the latest command.
- Legacy runs are retained but summarized separately from new group-split runs.
- Dry runs are stored and summarized separately from real training runs.
- Class weights can be toggled with `--class-weights` or `--no-class-weights`.

## Gold benchmark policy

- Keep all 300 gold samples as one fixed evaluation benchmark for now.
- Do not split the gold set further in this stage.
- Document that repeated comparison on this benchmark is a limitation.
- Treat the reported values as fixed-benchmark comparisons, not results from a fully
  untouched held-out test set.

## Before using the server

1. Pull the latest code.
2. Activate the project environment.
3. Confirm that `data/task3_silver_labeled_10k.csv` exists.
4. Run the automated tests:

```bash
python3 -m unittest discover -s tests -v
```

5. Run one dry-run smoke test:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 100 \
  --sample-seeds 42 \
  --train-seeds 42 \
  --dry-run
```

6. Open the generated metrics JSON and confirm:

```text
source_root_overlap_count = 0
sample_seed = 42
train_seed = 42
```

## Experiment 1 — Key silver sizes

Purpose:

- Recheck the unstable 1000-row setting.
- Test whether the 2500-row result is stable across different silver samples.

Run:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 1000 2500 \
  --sample-seeds 42 123 \
  --train-seeds 42 123 2026 \
  --class-weights
```

Expected number of runs:

```text
2 sizes × 2 sample seeds × 3 train seeds = 12 runs
```

Report:

- Macro-F1 mean and standard deviation.
- Results per sample seed.
- Results per train seed.
- Prediction count for each class.
- Every run with `class_collapse=true`.
- Whether `agree` or `question` is never predicted.

## Experiment 2 — Class-weight ablation

Fixed settings:

```text
silver_size = 2000
sample_seed = 42
train_seeds = 42, 123, 2026
```

Config A — current class weights:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 2000 \
  --sample-seeds 42 \
  --train-seeds 42 123 2026 \
  --class-weights
```

Config B — no class weights:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 2000 \
  --sample-seeds 42 \
  --train-seeds 42 123 2026 \
  --no-class-weights
```

Expected number of runs:

```text
2 configurations × 3 train seeds = 6 runs
```

Report:

- Macro-F1 mean and standard deviation for each configuration.
- Cohen's kappa mean and standard deviation.
- Per-class F1.
- Missing predicted classes and class-collapse count.
- Whether class weights help minority classes or amplify noisy labels.

## Decision after these experiments

- Determine whether the 1000-row variance mainly comes from silver sampling or
  training randomness.
- Determine whether the 2500-row performance is stable.
- Decide whether to retain class weights.
- Do not run the full multi-sample 500-10000 sweep until these results have been
  reviewed.

## Final learning-curve completion

The scoped experiments were reviewed and approved. Complete the missing standardized
sizes with two sample seeds:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 500 5000 8000 \
  --sample-seeds 42 123 \
  --train-seeds 42 123 2026 \
  --class-weights
```

For size 10000, both sample seeds select the same full 10k-row set, so run only one:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 10000 \
  --sample-seeds 42 \
  --train-seeds 42 123 2026 \
  --class-weights
```

After downloading the results, generate the final CSV and plot:

```bash
python3 -m src.plot_strategy_b_learning_curve
```
