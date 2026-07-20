# Strategy C — Standardized Gold-Only Evaluation

## Compatibility audit

- The legacy `strategy_c_full_backup` must not be reused for the class-weight
  ablation.
- Its fold 1 and fold 2 evaluation membership differs from the current fixed
  stratified 5-fold split; only folds 3-5 match exactly.
- The legacy inner validation split used the train seed directly. The standardized
  run records a separate, fixed `split_seed=42` for paired comparison.
- Therefore both RoBERTa weights-on and weights-off configurations must be rerun.

## Standardized evaluation

- Outer CV: stratified 5-fold, fixed seed 42.
- Inner RoBERTa validation split: stratified, fixed split seed 42.
- Train seeds: 42, 123, 2026.
- Model: full fine-tuning only; partial-layer experiments are closed.
- For each train seed, concatenate the five held-out folds into 300 OOF predictions.
- Compute one Macro-F1 and kappa per train seed on all 300 OOF rows.
- Report mean and standard deviation across the three seed-level metrics.
- Retain fold-level metrics only as diagnostics.
- Report missing classes and collapse counts at both fold and OOF levels.

## Server runs

Weights on:

```bash
python3 -m src.train_roberta strategy_c --model roberta \
  --seeds 42 123 2026 --split-seed 42 --class-weights
```

Weights off:

```bash
python3 -m src.train_roberta strategy_c --model roberta \
  --seeds 42 123 2026 --split-seed 42 --no-class-weights
```

The two commands produce 30 total RoBERTa training runs. Results are isolated under:

```text
results/strategy_c/roberta_weights/
results/strategy_c/roberta_no_weights/
```

## TF-IDF baseline

```bash
python3 -m src.train_roberta strategy_c --model tfidf \
  --seeds 42 123 2026 --class-weights
```

- TF-IDF is fitted separately inside each outer training fold.
- Current local weighted TF-IDF OOF Macro-F1: `0.2485`.
- No GPU is required.

## Final comparison

Compare standardized OOF results for:

1. Full RoBERTa with class weights.
2. Full RoBERTa without class weights.
3. Weighted TF-IDF + logistic regression.
4. Strategy B with 2500 silver rows (`0.562 ± 0.042` on the fixed benchmark).

Interpret Strategy C as evidence about the limitations of gold-only learning with
300 samples, not as proof that silver labeling is the only possible solution.
