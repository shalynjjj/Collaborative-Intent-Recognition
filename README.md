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

### Held-Out Test Set

To address the fixed-benchmark limitation above, a held-out test set independent of
`GOLD_CSV` was added. `GOLD_CSV` remains the dev set used for all tuning; `TEST_CSV`
is only touched once per strategy, after that strategy's config is locked.

```bash
python3 -m src.prepare_gold_test_candidates
```

builds `data/cmv_test_candidates.csv` by sampling the Winning Arguments Corpus after
excluding every gold and silver pair. The labeled pilot subset actually used for
evaluation is `data/cmv_test_candidates_pilot80.csv` (`TEST_CSV` in `src/config.py`).

Final configs, locked from gold-set (dev) results only, live in `src/config.py`:

- Strategy A: `STRATEGY_A_FINAL_MODE = "fewshot"` (gold macro-F1 `0.5916` vs
  zeroshot `0.5148`)
- Strategy B: `STRATEGY_B_FINAL_CONFIG` — silver_size `2500`, sample_seed `123`,
  class weights on
- Strategy C: `STRATEGY_C_FINAL_CONFIG` — RoBERTa, warmup `20`, epochs `8`,
  class weights on (gold OOF macro-F1 `0.5309 ± 0.0038`, zero class collapse;
  locked after the full warmup x epochs grid, see "Strategy C: Warmup/Epochs
  Grid" below)

Run each strategy's held-out eval once, after its dev-side tuning is finished:

```bash
python3 -m src.llm_annotate --heldout
python3 -m src.train_roberta strategy_b_heldout
python3 -m src.train_roberta strategy_c_heldout
```

Outputs are saved under `results/strategy_a_heldout/`, `results/strategy_b_heldout/`,
and `results/strategy_c_heldout/`.

### IAA

`TEST_CSV` was double-annotated (xin, xiaying), then reconciled by discussion.
Raw pre-discussion agreement on 130 co-labeled rows: **87.7% (114/130), 16
disagreements**. Kappa not computable — the pre-reconciliation file wasn't kept.

### Held-Out Test Results

| | Strategy A (fewshot) | Strategy B (silver 2500) |
|---|---|---|
| Gold (dev) macro-F1 | 0.5916 | 0.6375 |
| **Held-out macro-F1** | **0.6985** | **0.6175 ± 0.0085** |
| Held-out kappa | 0.60 | 0.511 ± 0.0096 |

Per-class held-out F1 — agree/disagree/question/statement:
A: 0.647 / 0.722 / 0.773 / 0.652. B (mean of 3 seeds): 0.689 / 0.449 / 0.803 / 0.529.

Dev ranks B above A; held-out ranks A above B — the ranking flips depending on
which set is used. B's `disagree` F1 is its weakest class in all 3 seeds.

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

Outputs are isolated by model, weight configuration, and (if overridden)
warmup/epoch config, e.g. `<model>_<weights|no_weights>[_warmup<N>_epochs<N>]`. The
locked final directories (see `STRATEGY_C_FINAL_CONFIG` in `src/config.py`) are:

```text
results/strategy_c/roberta_weights_warmup20_epochs8/    # locked final config
results/strategy_c/roberta_no_weights_warmup20_epochs8/  # weights-off ablation
results/strategy_c/tfidf_weights/                        # baseline
```

Each directory contains fold-level files, `fold_runs.csv`, per-seed OOF predictions
and metrics, `summary_by_seed.csv`, and `summary_oof.csv`. Metrics explicitly record
missing predicted classes and class-collapse counts.

All superseded/closed Strategy C results live under `results/strategy_c/legacy/`:
`pre_refactor/` (pre-refactor loose files, macro-F1 `0.169`), `pre_warmup_fix_weights/`
and `pre_warmup_fix_no_weights/` (unstable runs before the warmup/epochs fix below,
macro-F1 `0.2208`/`0.1629`), and `full_backup/`, `layers2/`, `layers4/` (closed
partial-fine-tuning experiments, see below). Do not use anything under `legacy/`.

### Strategy C: Class-Weight Ablation

Run both full-fine-tuning configurations with identical folds, split seed, and train
seeds. The old `strategy_c_full_backup` (now `results/strategy_c/legacy/full_backup/`)
cannot be reused because its fold 1-2 test membership differs from the current fixed
split and its inner split was tied directly to the train seed. Same for
`results/strategy_c/legacy/pre_refactor/` -- pre-refactor loose files (macro-F1
`0.169`, even worse than the collapsed baseline). Do not use either.

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

### Train-Fit Diagnostic (Is It Data or Config?)

`python3 -m src.train_roberta strategy_c_diagnose` trains each of the 5
folds x 3 seeds once and additionally predicts on the training rows
themselves, saved to `results/strategy_c_diagnose/summary_weights.csv`
(per-run rows plus one `AGGREGATE` row).

Current result: mean train-fit macro-F1 `0.326`, mean eval macro-F1 `0.239`,
11/15 runs have train-fit below `0.35`, and train-fit correlates strongly
with eval (`r=0.777`). This points to **training instability**, not
insufficient data -- a working config should fit ~204 training rows far
better than this most of the time; instead most runs fail to even fit their
own training data, and eval quality just tracks whether that run happened to
converge. Not yet possible to conclude "300 gold rows is insufficient" until
training is stabilized (warmup, more epochs) and re-diagnosed.

### Weights On/Off, Fixed Config

Fixed the instability above by adding `--warmup-steps 20 --epochs 8` to
`run_strategy_c` (roberta only, does not change TRAINING defaults used by
Strategy B). Then tried this one config with weights on and with weights
off, to see if weights still matter now that training is stable.

- Weights on: OOF macro-F1 `0.5309 ± 0.0038`. No class collapse in any of
  the 15 fold/seed runs. This beats TF-IDF (`0.2485`).
- Weights off: OOF macro-F1 `0.3499 ± 0.0229`. Still collapses on 8/15
  fold/seed runs. `statement` F1 is near zero; one seed predicted
  `statement` zero times.

Weights on is much better, so `use_class_weights=True` is kept for all
further tuning. This comparison was only run at this one (warmup=20,
epochs=8) config, not across a full warmup x epochs x weights grid, so it's
possible a different warmup/epochs combo changes this -- not verified.

### Strategy C: Warmup/Epochs Grid (Dev-Only)

Closes the "not verified" gap above with a small grid around the known-good
(warmup=20, epochs=8) point. Runs only against the gold set (dev); `TEST_CSV`
is not touched. `use_class_weights` stays fixed on, since that ablation is
already settled above.

```bash
for warmup in 10 20 30; do
  for epochs in 6 8 10; do
    python3 -m src.train_roberta strategy_c \
      --model roberta \
      --seeds 42 123 2026 \
      --split-seed 42 \
      --class-weights \
      --warmup-steps "$warmup" \
      --epochs "$epochs"
  done
done
```

Each configuration writes to its own
`results/strategy_c/roberta_weights_warmup<N>_epochs<N>/`, so the existing
(20, 8) run is reused rather than overwritten. After the grid finishes,
summarize every warmup/epochs configuration found under `results/strategy_c/`,
ranked by OOF macro-F1:

```bash
python3 -m src.summarize_strategy_c_grid
```

This writes `results/strategy_c/warmup_epochs_grid_summary.csv`. If a
configuration beats `0.5309` with zero class collapse, update
`STRATEGY_C_FINAL_CONFIG` in `src/config.py` and rerun
`strategy_c_heldout` once, matching the touch-held-out-once discipline
already used for Strategy A and B.

**Result: the grid is closed.** All 9 warmup x epochs combinations (plus the
`no_weights` control at warmup=20/epochs=8) have been run; ranked by OOF
macro-F1:

| warmup | epochs | weights | macro-F1 (mean ± std) | class collapse |
| --- | --- | --- | --- | --- |
| 20 | 8  | on  | **0.5309 ± 0.0038** | 0/15 |
| 10 | 10 | on  | 0.5231 ± 0.0406 | 0/15 |
| 30 | 10 | on  | 0.5189 ± 0.0494 | 0/15 |
| 20 | 10 | on  | 0.5166 ± 0.0516 | 0/15 |
| 30 | 8  | on  | 0.5072 ± 0.0314 | 0/15 |
| 10 | 8  | on  | 0.4904 ± 0.0602 | 0/15 |
| 30 | 6  | on  | 0.4371 ± 0.0292 | 0/15 |
| 20 | 6  | on  | 0.4278 ± 0.0511 | 0/15 |
| 10 | 6  | on  | 0.4216 ± 0.0587 | 2/15 |
| 20 | 8  | off | 0.3499 ± 0.0229 | 8/15 |

No configuration beat `0.5309`, so `STRATEGY_C_FINAL_CONFIG` (warmup=20,
epochs=8, weights on) stays locked as-is and `strategy_c_heldout` does not
need to be rerun. Two secondary observations: `epochs=6` is undertrained
across every warmup value (bottom three rows among weighted runs), and
`epochs=10` is competitive on the mean but 8-13x higher variance across
seeds than the locked (20, 8) config, i.e. less reproducible for the same
or worse macro-F1. This also closes the "single ad hoc value, not yet
grid-searched" caveat in `docs/stage1_conclusions_and_report_plan.md`.

### Previous Partial-Fine-Tuning Finding

Earlier exploratory runs produced fold-averaged macro-F1 of approximately `0.20` for
full fine-tuning, `0.16` with only the last 2 layers trainable, and `0.17` with only
the last 4 layers trainable. None learned all four classes reliably. Partial
fine-tuning is therefore closed as an experimental direction; it is not exposed in
the current CLI. (Results in `results/strategy_c/legacy/{full_backup,layers2,layers4}/`
-- superseded once the warmup/epochs fix below solved the instability. Do not use.)
Full RoBERTa weights-on/off must be rerun under the standardized OOF
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

## Stage 2: Multi-Module Affective Reasoning (TODO)

Stage 2: a modular LLM-based pipeline that predicts sentiment, emotion, and communicative intent for each CMV reply, using the reply, its parent comment, and the Stage 1 dialogue-act label as input. Two research questions: 
(1) does decomposing the task into separate
modules beat a single unified prompt, and 
(2) which information-sharing strategy
between modules gives the best intent-recognition performance.

`data/cmv_300_gold_final.csv` already has full `Sentiment`, `Sarcasm`, `Hostility`,
`Contempt`, `Neutral`, `Curiosity`, `Appreciation`, and `Intent` columns for all 300
rows, so the Stage 2 gold-standard annotation required by the proposal's Task 2 is
already done. No new annotation is required for the experiments below.

### Open decisions (resolved)

- **DA feature source:** gold `Dialogue_act` (oracle), not Strategy A's
  predictions. Strategy A's predicted DA is only ~0.6 macro-F1, so using it
  would confound "does DA help" with "are the predictions too noisy to help."
- **Eval protocol:** `python3 -m src.prepare_stage2_split` splits `GOLD_CSV`
  by `Intent` (seed `42`, 12%) into `data/cmv_300_gold_stage2_dev.csv` (36
  rows, prompt iteration only, never reported) and
  `data/cmv_300_gold_stage2_eval.csv` (264 rows, touched once per locked
  prompt config) — same touch-once discipline as `GOLD_CSV`/`TEST_CSV` in
  Stage 1. Both Experiment 1 and Experiment 2 must use this same split.
- **IAA / kappa:** pre-reconciliation labels survived for `Dialogue_act`,
  `Sentiment`, and `Intent` on the gold-300, so kappa is already computed
  (κ = 0.8967 / 0.75 / 0.8133). Emotion (6-category multi-label:
  `Sarcasm`/`Hostility`/`Contempt`/`Neutral`/`Curiosity`/`Appreciation`) is
  the one gap — its pre-reconciliation file was lost, so kappa isn't
  computable; report it as such, or run a small fresh double-annotation pass
  if a real number is needed.

### Pipeline code

All five items below are implemented in `src/stage2_pipeline.py`, following
the same generator/prompt/parse pattern as Strategy A's `src/llm_annotate.py`
(reuses its `make_transformers_generator` for the actual LLM calls).

- [x] Sentiment module: `build_sentiment_prompt` + `run_sentiment_module`
      (3-class: `Positive` / `Negative` / `Neutral`).
- [x] Emotion module: `build_emotion_prompt` + `run_emotion_module`
      (6-category multi-label binary output: `Sarcasm`, `Hostility`,
      `Contempt`, `Neutral`, `Curiosity`, `Appreciation` — one bool column per
      category, parsed independently since a reply can match zero or more).
- [x] Intent module: `build_intent_prompt` + `run_intent_module` (5-class:
      `Information seeking` / `Challenge` / `Counter-argue` / `Support` /
      `Others`). Accepts optional `dialogue_act`/`sentiment`/`emotion_labels`
      context so it can be reused for Experiment 2's input-combination sweep.
- [x] Single-prompt baseline: `build_single_prompt_baseline` +
      `run_single_prompt_baseline`, one LLM call returning sentiment, emotion,
      and intent as three labeled lines, parsed together.
- [x] Multi-module orchestration: `run_multi_module_pipeline` calls the three
      modules above in sequence (sentiment, then emotion, then intent with
      gold `Dialogue_act` as context, per the DA feature-source decision above).

Label strings match `GOLD_CSV`'s exact casing, so predictions compare directly
against the `Sentiment`/`Sarcasm`/.../`Intent` columns with no re-casing.
Predicted-label fallbacks (`FALLBACK_SENTIMENT = "Neutral"`,
`FALLBACK_INTENT = "Others"`) and a per-field `*_fallback` flag are recorded
the same way Strategy A tracks `fallback_used`.


- [ ] Run both architectures on the gold-300 set, compute macro-F1 per label type.
- [ ] Write multi-label evaluation code for emotion (per-category precision /
      recall / F1, then macro-averaged across the 6 categories). This cannot reuse
      Stage 1's single-label confusion-matrix evaluation code.

`src/evaluate_stage2.py` runs Sentiment/Intent through `compute_metrics()`
(macro-F1 + confusion matrix, feeds "Error analysis" below) and Emotion
through the new `compute_emotion_metrics()` (per-category P/R/F1, macro-averaged
across the 6 categories):

```bash
python3 -m src.stage2_pipeline --mode multi_module --split dev
python3 -m src.stage2_pipeline --mode single_prompt --split dev
python3 -m src.evaluate_stage2 --mode multi_module --split dev
python3 -m src.evaluate_stage2 --mode single_prompt --split dev
```

Outputs are saved under `results/stage2/` as
`<mode>_<split>_{sentiment,intent,emotion}_metrics.json`,
`<mode>_<split>_{sentiment,intent}_confusion_matrix.csv`, and one
`<mode>_<split>_summary.json` with all three macro-F1s. Compare the two
`summary.json` files to answer Experiment 1's RQ1.

### Experiment 2 — Information-Sharing Strategy Between Modules

- [ ] Build the 7 input-combination sweep for intent prediction (reply+parent as
      base, adding dialogue act / sentiment / emotion in the combinations listed
      in the proposal).
- [ ] Run the sweep, report Intent macro-F1 per input combination against the
      text-only baseline, identify the best module execution order.
- [ ] Run `bootstrap_ci()` (`src/evaluate.py`, reused from Strategy A's
      held-out eval) on each combination's predictions to get a macro-F1 95%
      CI — at n=264, this is what tells you whether the top combination is
      actually better than the others or just within noise.

```python
from src.evaluate import bootstrap_ci
from src.config import INTENT_LABELS

ci = bootstrap_ci(df["gold_intent"], df["intent"], labels=INTENT_LABELS)
print(ci["macro_f1_boot_mean"], ci["macro_f1_ci95"])
```

### Error analysis and write-up

- [ ] Sample and manually review misclassified examples, categorized by source as
      the proposal requires: incorrect DA prediction, undetected sarcasm, or
      insufficient conversational context.
- [ ] Extend `docs/full-report-draft.tex` and `docs/stage1_full_report.md` (or new
      Stage 2-specific files) with Stage 2 Methodology, Experiments, and Results
      sections, matching the style already used for Stage 1.
