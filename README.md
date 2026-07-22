# depolarAIze

Local, reproducible Stage 1 experiments for dialogue-act classification in Reddit CMV replies.

The dialogue-act label scheme is:

`agree`, `disagree`, `question`, `statement`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Models are downloaded from ModelScope by default:

- LLM: `LLM-Research/Meta-Llama-3.1-8B-Instruct`
- RoBERTa: `AI-ModelScope/roberta-base`

If you switch back to gated Hugging Face models later, add your token to `.env` and authenticate with Hugging Face as needed.

`.env` should contain:

```bash
HF_TOKEN=your_huggingface_token_here
```

Do not commit `.env` or paste real tokens into notebooks, source files, or chat messages.

## Data

The gold data file is:

`data/cmv_300_gold_final.csv`

The loader expects `Parent`, `Reply`, and `Dialogue_act`. If the old `Int` column is present, it is renamed to `Intent`.

## Strategy A: LLM Annotation

Zero-shot:

```bash
python3 -m src.llm_annotate --mode zeroshot
```

Few-shot:

```bash
python3 -m src.llm_annotate --mode fewshot
```

Smoke test without loading Llama:

```bash
python3 -m src.llm_annotate --mode zeroshot --mock --limit 5
```

Outputs are saved under `results/strategy_a/`, including predictions, metrics, confusion matrices, and `fallback_count`.

## Strategy B: RoBERTa on Silver Labels

First build one 10k raw silver-candidate file from the Winning Arguments Corpus. It removes gold overlaps and keeps `source_root`, `parent_id`, `reply_id`, `Parent`, and `Reply`:

```bash
python3 -m src.prepare_silver_data
```

Then label that 10k candidate file with the LLM:

```bash
python3 -m src.llm_annotate \
  --mode zeroshot \
  --input-csv data/task3_silver_candidates_10k.csv \
  --output-csv data/task3_silver_labeled_10k.csv
```

Strategy B expects a silver-label CSV with `Parent`/`Reply` columns where possible. `model_input` is supported only as a fallback.

The silver label column is inspected automatically. Supported names are:

`pred_label`, `Dialogue_act`, `dialogue_act`, `label`, `da_label`

```bash
python3 -m src.train_roberta strategy_b --silver-csv data/task3_silver_labeled_10k.csv
```

Smoke test without fine-tuning:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 100 \
  --sample-seeds 42 \
  --train-seeds 42 \
  --dry-run
```

Strategy B now uses two independent seed controls:

- `--sample-seeds` deterministically selects rows from the full silver pool. For a
  fixed sample seed, larger sizes are nested extensions of smaller sizes.
- `--train-seeds` controls model initialization, batch order, dropout, and other
  PyTorch/Trainer randomness.

The inner silver train/validation split is group-aware: all rows with the same
`source_root` stay in one partition. Every metrics file records the train/validation
row counts, source-root counts, and `source_root_overlap_count` (which must be zero).

Class weighting can be enabled or disabled explicitly with `--class-weights` and
`--no-class-weights`. New output filenames include silver size, sample seed, train
seed, and weight configuration, so independent runs do not overwrite one another.

The default Strategy B learning-curve sizes are:

`500`, `1000`, `1500`, `2000`, `2500`, `3000`, `5000`, `8000`, `10000`

The `500`-`2500` sizes were added to resolve the shape of the curve below 3000, since macro-F1 was already flat across `3000`-`10000`.

Per-run outputs are saved under `results/strategy_b_zeroshot/` (this is the silver data labeled with zero-shot LLM annotation). After every run,
`runs.csv`, `summary.csv`, and `summary_by_sample.csv` are rebuilt from the union of
all existing per-run metrics files. Legacy results are preserved and summarized
separately from the new group-split experiments.

### Strategy B: Scoped Server Runs

Run these only after the dry-run smoke test succeeds.

Key-size experiment (12 real training runs):

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 1000 2500 \
  --sample-seeds 42 123 \
  --train-seeds 42 123 2026 \
  --class-weights
```

Class-weight ablation at 2000 silver rows (six real training runs):

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 2000 \
  --sample-seeds 42 \
  --train-seeds 42 123 2026 \
  --class-weights

python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 2000 \
  --sample-seeds 42 \
  --train-seeds 42 123 2026 \
  --no-class-weights
```

The key-size and class-weight checks should be reviewed before the final
learning-curve completion below.

### Strategy B: Final Learning-Curve Completion

After the scoped checks confirmed the standardized setup, complete the remaining
sizes with:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 500 5000 8000 \
  --sample-seeds 42 123 \
  --train-seeds 42 123 2026 \
  --class-weights

python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 10000 \
  --sample-seeds 42 \
  --train-seeds 42 123 2026 \
  --class-weights
```

At size 10000, sample seeds 42 and 123 contain the same complete 10k-row pool
(different order only), so only sample seed 42 is run. Generate the final plot and
underlying table after all metrics have been downloaded:

```bash
python3 -m src.plot_strategy_b_learning_curve
```

This writes `results/strategy_b_zeroshot/learning_curve_zeroshot.png` and
`results/strategy_b_zeroshot/learning_curve_zeroshot.csv`. The plot includes Strategy A zero-shot
(`0.5148`) and few-shot (`0.5916`) reference lines.

### Fixed Gold Benchmark Limitation

Strategy B trains only on silver labels and evaluates every configuration on the same
300-sample gold benchmark. The gold set is not split further at this stage. Because
the same benchmark is used to compare configurations, results should be interpreted
as comparisons on a fixed evaluation benchmark, not as performance on a fully
untouched held-out test set. This limitation must be stated in the report.

### Common 296-Sample Evaluation

The four gold examples used as few-shot prompt demonstrations are excluded when
comparing Strategy A with the few-shot-silver Strategy B. This is a scoring-only
step and does not require retraining existing models:

```bash
python3 -m src.evaluate_common_296 \
  --strategy-b-dir results/strategy_b_fewshot
```

This writes `strategy_a_common_296.csv`, `common_296_runs.csv`, and
`common_296_summary.csv` under `results/strategy_b_fewshot/`.

The completed common-benchmark comparison is:

| Configuration | Macro-F1 mean | Macro-F1 std | Runs |
| --- | ---: | ---: | ---: |
| Strategy A: Llama few-shot | 0.5916 | n/a | 1 |
| Strategy B: 2500 few-shot silver, class weights | 0.6179 | 0.0282 | 6 |
| Strategy B: 2500 few-shot silver, no class weights | 0.6113 | 0.0252 | 6 |

The Strategy B value is the mean across two silver sample seeds (`42`, `123`)
and three training seeds (`42`, `123`, `2026`). RoBERTa has a higher observed
mean than the Llama teacher on these 296 rows, but this is not treated as proof
of significant student superiority because no paired significance test has been
run and the 2500-row configuration was selected on the same fixed benchmark.

### Few-Shot Silver Class-Weight Ablation (Completed)

The paired no-weight configuration was run at the selected 2500-row setting with
the same sample and training seeds:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k_fewshot.csv \
  --sizes 2500 \
  --sample-seeds 42 123 \
  --train-seeds 42 123 2026 \
  --no-class-weights \
  --results-dir results/strategy_b_fewshot
```

On the common 296-row benchmark, class weights increased mean Macro-F1 from
`0.6113` to `0.6179` (mean paired difference `+0.0066`). Weights-on was better in
four of six paired runs. The main per-class change was higher `disagree` F1
(`+0.0419` on average), while `statement` F1 decreased (`-0.0198`). There were no
class-collapse runs in either configuration. Class weights are retained as the
final configuration because they provide a modest observed improvement, not
because statistical significance has been established.

### Strategy B Interpretation Limits

- The 300-row gold benchmark was repeatedly used for configuration comparison;
  it is an exploratory fixed benchmark rather than a fully untouched final test.
- Exact Parent-Reply overlap between gold and silver was removed. A later audit
  recovered source roots for 276 gold rows and found that 262 of those rows share
  a Reddit thread with the few-shot silver pool. Results therefore represent
  in-domain evaluation and should not be claimed as unseen-thread performance.
- The few-shot learning curve's 2500-row point has the highest observed mean; no
  significance test established that it is better than the neighboring
  1500/2000/5000 points.
- Human spot-checking of few-shot silver-label accuracy remains future work.

## Strategy C: RoBERTa on Gold Labels

Runs stratified 5-fold cross-validation with train seeds `42`, `123`, and `2026`.
The outer folds are fixed with seed `42`; the inner train/validation split is
controlled separately by `--split-seed` (default `42`).

```bash
python3 -m src.train_roberta strategy_c --model roberta --class-weights
```

Smoke test without fine-tuning:

```bash
python3 -m src.train_roberta strategy_c \
  --model roberta \
  --seeds 42 \
  --class-weights \
  --dry-run
```

Each train seed's five test folds are concatenated into one 300-row out-of-fold (OOF)
prediction file. Macro-F1 and kappa are computed once per seed on those 300 OOF rows;
the final mean and standard deviation are calculated across the three seed-level
scores, rather than treating 15 fold runs as independent observations.

Outputs are isolated by model and weight configuration:

```text
results/strategy_c/roberta_weights/
results/strategy_c/roberta_no_weights/
results/strategy_c/tfidf_weights/
```

Each directory contains fold-level files, `fold_runs.csv`, per-seed OOF predictions
and metrics, `summary_by_seed.csv`, and `summary_oof.csv`. Metrics explicitly record
missing predicted classes and class-collapse counts.

### Strategy C: Class-Weight Ablation

Run both full-fine-tuning configurations with identical folds, split seed, and train
seeds. The old `strategy_c_full_backup` cannot be reused because its fold 1-2 test
membership differs from the current fixed split and its inner split was tied directly
to the train seed.

```bash
python3 -m src.train_roberta strategy_c \
  --model roberta \
  --seeds 42 123 2026 \
  --split-seed 42 \
  --class-weights

python3 -m src.train_roberta strategy_c \
  --model roberta \
  --seeds 42 123 2026 \
  --split-seed 42 \
  --no-class-weights
```

### Strategy C: TF-IDF Baseline

The TF-IDF vectorizer is fitted inside each outer training fold through an sklearn
pipeline, so test-fold text never enters vocabulary fitting. This baseline runs on
CPU:

```bash
python3 -m src.train_roberta strategy_c \
  --model tfidf \
  --seeds 42 123 2026 \
  --class-weights
```

The current weighted TF-IDF baseline has OOF macro-F1 `0.2485`. The three seed-level
scores are identical because this solver converged deterministically on these folds.

### Previous Partial-Fine-Tuning Finding

Earlier exploratory runs produced fold-averaged macro-F1 of approximately `0.20` for
full fine-tuning, `0.16` with only the last 2 layers trainable, and `0.17` with only
the last 4 layers trainable. None learned all four classes reliably. Partial
fine-tuning is therefore closed as an experimental direction; it is not exposed in
the current CLI. Full RoBERTa weights-on/off must be rerun under the standardized OOF
pipeline before comparison with TF-IDF and Strategy B.

## Notebook

`notebooks/stage1_experiments.ipynb` is intentionally thin: it imports from `src/`, runs scripts, and displays saved results.

## Server Run Order

For the RTX 4090 server, use this order before launching full experiments:

1. Install dependencies.
2. Run a local plumbing check:

```bash
python3 -m src.llm_annotate --mode zeroshot --mock --limit 5
python3 -m src.train_roberta strategy_c \
  --model roberta --seeds 42 --split-seed 42 --class-weights --dry-run
```

3. Run the real 15-row Strategy A test:

```bash
python3 -m src.llm_annotate --mode zeroshot --limit 15
```

4. After the 10k silver file has been built and labeled, run the Strategy B dry-run
   shown in the Strategy B section above. The group-aware split requires the
   `source_root` column, so Strategy A gold prediction files are not valid smoke-test
   inputs for Strategy B.

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k.csv \
  --sizes 100 \
  --sample-seeds 42 \
  --train-seeds 42 \
  --dry-run
```

5. Run both standardized full-RoBERTa Strategy C configurations:

```bash
python3 -m src.train_roberta strategy_c \
  --model roberta --seeds 42 123 2026 --split-seed 42 --class-weights
python3 -m src.train_roberta strategy_c \
  --model roberta --seeds 42 123 2026 --split-seed 42 --no-class-weights
```

6. Build and label the 10k silver candidate file, then run only the scoped Strategy B
   experiments documented above (1000/2500 key sizes and the 2000 class-weight
   ablation):

```bash
python3 -m src.prepare_silver_data
python3 -m src.llm_annotate \
  --mode zeroshot \
  --input-csv data/task3_silver_candidates_10k.csv \
  --output-csv data/task3_silver_labeled_10k.csv
```

Full RoBERTa experiments use `AI-ModelScope/roberta-base`, learning rate `2e-5`, batch size `8`, `3` epochs, and seeds `42`, `123`, `2026`.
Each training run reserves an inner validation split and loads the checkpoint with the best validation `macro_f1`.
Strategy B retains class-weighted cross-entropy. Strategy C reports a paired
weights-on/off ablation under identical folds and seeds.
