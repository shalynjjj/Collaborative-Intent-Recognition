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

## Error Analysis: Why `disagree` Underperforms (Strategies B & C)

**Note:** this section uses the expanded 332-row held-out set (see
`docs/stage1_full_report.md`), not the pilot80 set the numbers earlier in
this README still reference — the two aren't directly comparable.

`disagree` F1 on the 332-row held-out set: A 0.623, B 0.590, C 0.565. B and C
each mislabel it for a different reason, and each reason was checked against
every misclassified row in that category, not just a handful of examples:

- **B mostly mistakes `disagree` for `statement`** (25%, 62/249 true
  `disagree` rows, pooled across B's 3 held-out seeds). Of those 62, 56% have
  zero negation or hedge words -- e.g. *"Reoccurrence has to be detected to
  be tracked."* -- content-only disagreement with no lexical marker to key
  on. The other 44% don't have this feature, so it's a partial explanation,
  not the whole cause.
- **C mostly mistakes `disagree` for `agree`** (18%, 45/249) -- this is the
  broader bidirectional agree/disagree confusion documented in
  `docs/stage1_full_report.md` (present in all 8 of C's dev-OOF seeds). An
  "affirmative opener" theory (reply starts with "yes"/"indeed"/etc.) was
  tested and ruled out: only 4% (2/45) fit it.
- **Both strategies also leak `disagree`→`question`** (B 12%, C 17%). This
  is the strongest pattern found: 71% (B, 22/31) and 52% (C, 22/42) of these
  errors literally end in a question mark, versus 1% of correctly-labeled
  `disagree` replies in both strategies -- e.g. *"Where did I say that?"* is
  grammatically a question but functionally a rejection of the parent's
  claim, and the model appears to key on the "?" rather than the
  disagreement itself.

Negation, contractions, reply length, and spelling were also checked as
candidate explanations and showed no reliable pattern; only the
rhetorical-question pattern above held up.

Reproduce with:

```bash
python3 -m src.analyze_disagree_errors
```

which reads `results/strategy_{b,c}_heldout/heldout_seed{42,123,2026}_predictions.csv`
and writes `results/error_analysis/disagree_error_text_features.csv` and
`disagree_error_examples.csv`. Full write-up: `docs/error_analysis_disagree.md`.

### Blind Re-Label: 27 Hard Agree/Disagree Rows

Separately from the `disagree` analysis above, the 27 gold-300 rows where
at least 6 of Strategy C's 8 dev-OOF seeds flip `agree`↔`disagree` (see
`docs/stage1_full_report.md`'s Error Analysis) were pulled out for a blind
re-label, to check whether that confusion reflects genuine annotation
ambiguity or a model limitation:

- `results/strategy_c/agree_disagree_hard27_blind.csv` -- the 27 rows,
  shuffled, with the original label stripped and a blank `your_label`
  column to fill in.
- `results/strategy_c/agree_disagree_hard27_answer_key.csv` -- same rows
  with the original gold label and each seed's prediction; not to be opened
  until the blind file is labeled.
- Score a completed blind label with:

```bash
python3 -m src.score_agree_disagree_relabel \
  --blind-csv results/strategy_c/agree_disagree_hard27_blind.csv
```

Add `--blind-csv-2` with a second annotator's filled-in copy to also get
inter-annotator agreement. Status: labeling in progress, not yet scored.

### Full-Dataset Text-Feature Columns (Delivery Artifact)

The negation/hedge/spelling check above previously ran only on
misclassified rows. Since the dataset itself is a deliverable, it now runs
on **every** row, so downstream users (e.g. training a negation classifier)
don't have to recompute it.

```bash
python3 -m src.add_text_features
```

Unions `data/cmv_300_gold_final.csv` (300 rows) and
`data/cmv_test_candidates_heldout_expanded.csv` (530 rows, no overlap) into
`data/cmv_full_with_text_features.csv` (830 rows), reusing the `_features()`
logic from `analyze_disagree_errors.py` on both `Parent` and `Reply`:

`{reply,parent}_word_count`, `{reply,parent}_hedge_rate/has_hedge`,
`{reply,parent}_negation_rate/has_negation`,
`{reply,parent}_contraction_rate`, `{reply,parent}_spelling_error_rate`,
`{reply,parent}_very_long/very_short`, `{reply,parent}_ends_in_question`

Plus `corpus_source` (`gold_300` / `heldout_batch2` / `heldout_pilot80`).

### Parent-Negation Ablation: Does Strategy B Use the Parent?

Strategy B's top confusion is agree/disagree misclassified as `statement`
(see "Error Analysis" above) — possibly because it pattern-matches the
`Reply` alone and may be insensitive to `Parent`. As a small targeted test,
three rows that Strategy B frequently calls `statement` across the locked
seeds were selected. We flipped the `Parent`'s
negation (add/remove), keep `Reply` unchanged, flip gold label to match
(`agree`↔`disagree`). Retrain the same locked config from scratch and
predict on both versions:

- unchanged after the edit → evidence of insensitivity on that case
- changes to match edited gold → evidence of responsiveness to the changed context

```bash
python3 -m src.ablation_parent_negation
```

Writes `results/error_analysis/parent_negation_ablation_predictions.csv`
(per example × variant × seed), `parent_negation_ablation_summary.csv`
(by example and variant), and `parent_negation_ablation_overall.csv`
(paired change statistics).

**Result (3 examples × 3 seeds = 9 edited rows):**

| example | variant | gold | seed 42 | seed 123 | seed 2026 | matches row gold |
| --- | --- | --- | --- | --- | --- | ---: |
| `babies_headcover` | original | disagree | statement | statement | statement | 0/3 |
| `babies_headcover` | edited | agree | statement | statement | statement | **0/3** |
| `brand_new_tires` | original | disagree | disagree | statement | statement | 1/3 |
| `brand_new_tires` | edited | agree | statement | statement | statement | **0/3** |
| `who_healthcare_ranking` | original | agree | statement | disagree | statement | 0/3 |
| `who_healthcare_ranking` | edited | disagree | statement | disagree | statement | **1/3** |

**1/9 edited predictions matched the edited gold label**, but that prediction
was already `disagree` before the edit and therefore did not respond to the
changed Parent. Across the paired comparisons, **8/9 predictions were
unchanged**, **1/9 changed**, and **0/9 changed to the expected edited label**.
Overall, **15/18 predictions were `statement`**. The changed case was seed 42
on `brand_new_tires`, which moved
`disagree`→`statement` (wrong direction; `agree` expected), and seed 123 on
`who_healthcare_ranking` predicts `disagree` for **both** the original and
its opposite — invariant to the negation, not responsive to it.
This is preliminary evidence that Strategy B was insensitive to the Parent
edits in these selected cases; it does not establish that the model generally
ignores Parent or relies only on Reply text.

Candidates were drawn from agree/disagree→`statement` errors and kept only
when a Parent contained a reasonably editable claim (most CMV parents are
multi-paragraph). Only the babies example was predicted as `statement` by all
three seeds before editing. Because the counterfactual labels were manually
constructed and two cases remain semantically debatable, they should be
manually validated before this result is used as confirmatory evidence.

`train_roberta.py`'s `_train_and_predict`/`_run_once` now take an optional
`save_model_dir` to persist the fine-tuned model+tokenizer instead of
discarding it (`run_strategy_b_heldout_eval` saves each seed to
`results/strategy_b_heldout/models/seed{N}/` by default). Once saved,
`src/predict_with_saved_strategy_b.py` predicts on any new Parent/Reply CSV
without retraining:

```bash
python3 -m src.predict_with_saved_strategy_b --input rows.csv --output preds.csv
```

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

## Stage 2: Multi-Module Affective Reasoning

Stage 2 uses Llama-3.1-8B-Instruct to predict sentiment, emotion, and
communicative intent from each CMV Parent–Reply pair. It asks two questions:
(1) under matched input information, do separate task-specific prompts perform
better than one prompt that predicts all three targets; and (2) which auxiliary
signals help Intent under both ideal gold-label and realistic predicted-label
conditions? Both experiments and the manual error analysis are complete;
folding the results into the formal thesis writeup remains open. Full details,
including two bugs found and fixed mid-project (a parser bug and a
prompt-fairness confound that reversed Experiment 1's headline result), are in
[`docs/stage2_report.md`](docs/stage2_report.md).

`data/cmv_300_gold_final.csv` already has full `Sentiment`, `Sarcasm`, `Hostility`,
`Contempt`, `Neutral`, `Curiosity`, `Appreciation`, and `Intent` columns for all 300
rows, so the Stage 2 gold-standard annotation required by the proposal's Task 2 is
already done. No new annotation is required for the experiments below.

### Open decisions (resolved)

- **Auxiliary feature sources:** Experiment 1 does **not** use Dialogue Act;
  both conditions receive only `Parent + Reply`. Experiment 2 reports two
  branches: an oracle branch using gold DA/Sentiment/Emotion and a realistic
  branch using Strategy A few-shot DA plus Experiment 1's predicted Sentiment
  and Emotion.
- **Eval protocol:** `python3 -m src.prepare_stage2_split` splits `GOLD_CSV`
  by `Intent` (seed `42`, 12%) into `data/cmv_300_gold_stage2_dev.csv` (36
  rows, prompt iteration only, never used for final claims) and
  `data/cmv_300_gold_stage2_eval.csv` (264 rows, final evaluation). Both
  experiments use this fixed split. Experiment 1 required one documented
  corrective eval rerun after removing an unintended gold Dialogue Act input;
  the confounded result is not used.
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
      modules above independently. Each module receives only `Parent + Reply`,
      matching the information available to the single-prompt baseline.

Label strings match `GOLD_CSV`'s exact casing, so predictions compare directly
against the `Sentiment`/`Sarcasm`/.../`Intent` columns with no re-casing.
Predicted-label fallbacks (`FALLBACK_SENTIMENT = "Neutral"`,
`FALLBACK_INTENT = "Others"`) and a per-field `*_fallback` flag are recorded
the same way Strategy A tracks `fallback_used`.

### Experiment 1 — Single-Prompt Baseline vs. Multi-Module Pipeline

- [x] Run both architectures on the gold-300 set, compute macro-F1 per label type.
- [x] Write multi-label evaluation code for emotion (per-category precision /
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

**Superseded result (eval, 264 rows), kept for the record:** the table below
was measured before two more bugs were found and fixed — `single_prompt`'s
3-line answer parser silently losing correct answers to hallucinated
continuations, and (more importantly) `build_single_prompt_baseline`'s prompt
being written more sparsely than the three per-task prompts it was compared
against (no label definitions, no "judge Reply against Parent" framing). Both
conditions receive exactly `Parent + Reply` either way.

| | multi_module | single_prompt |
| --- | ---: | ---: |
| Sentiment macro-F1 | 0.572 | 0.532 |
| Intent macro-F1 | **0.387** | 0.360 |
| Emotion macro-F1 | **0.393** | 0.357 |
| 3-way average | **0.450** | 0.416 |

**Current result, after fixing the parser bug and the prompt-fairness bug and
re-running `single_prompt` against the real LLM (eval, 264 rows):**

| | multi_module | single_prompt |
| --- | ---: | ---: |
| Sentiment macro-F1 | 0.572 | **0.625** |
| Intent macro-F1 | 0.387 | **0.409** |
| Emotion macro-F1 | 0.393 | **0.408** |

**The result reverses**: `single_prompt` now leads on all three tasks, the
opposite of the original conclusion above. Most of that original gap was the
prompt-fairness bug, not the 1-call-vs-3-calls architecture itself. A paired
bootstrap (`n_boot=2000`) on these same 264 rows found the new gap is **not
statistically significant** either way (Sentiment diff −0.053, 95% CI
[−0.108, +0.006]; Intent diff −0.022, 95% CI [−0.083, +0.036]) — so the honest
reading is "point estimate now favors single_prompt, but not confidently
distinguishable from no real difference," not a confirmed win for either
architecture. Full root-cause writeup, the manual verification of the parser
fix, and the discovery of the prompt-fairness bug are in
[`docs/stage2_report.md`](docs/stage2_report.md).

On the 36-row dev split, the two architectures looked much closer (and briefly
appeared to favor `single_prompt` on Emotion) after fixing a real parsing bug
(`parse_single_label` matched labels in fixed list order instead of by
first-occurrence position in the text, so a model that answered correctly in
its first word could still get mis-scored if it rambled into unrelated text
mentioning another label word later — see `_first_line` in
`src/stage2_pipeline.py`). Spot-checking dev errors also surfaced 3 prompt
issues, fixed before the eval run: the `Others` intent category was defined
too vaguely (model never predicted it) and got concrete positive cues
(off-topic / meta-commentary about the thread itself); `Challenge` vs.
`Counter-argue` was blurry (any pushback defaulted to `Counter-argue`) and now
explicitly hinges on whether the reply presents its own reasoning; and the
Emotion prompt's `neutral`/`"none"` ambiguity (two ways to say "no strong
emotion", parser only recognized one) was collapsed to one canonical answer.

### Experiment 2 — Oracle and Predicted Auxiliary Information for Intent

Experiment 2 separates two questions that were previously conflated:

1. **Oracle information value:** if auxiliary labels are perfectly correct,
   which of Dialogue Act, Sentiment, and Emotion helps Intent classification?
2. **Realistic pipeline value:** does that benefit remain when auxiliary labels
   come from the actual upstream models and therefore contain errors?

- [x] Oracle sweep: text-only base plus all 7 non-empty auxiliary-label subsets.
- [x] Individual bootstrap CIs for the original oracle results.
- [x] Code for the matched predicted-label sweep and paired comparisons.
- [x] Run the predicted-label sweep on dev and eval.

The oracle branch uses gold DA/Sentiment/Emotion. The predicted branch uses
Strategy A few-shot Dialogue Act predictions plus the corrected Experiment 1
Sentiment and Emotion predictions. The four few-shot prompt demonstrations are
excluded, producing a leakage-free common eval set of 260 rows. All Base,
Oracle, and Predicted comparisons involving this branch are paired on those
same rows.

```bash
python3 -m src.stage2_intent_sweep --split dev --feature-source oracle --eval-only
python3 -m src.stage2_intent_sweep --split dev --feature-source predicted
```

Outputs are separated by feature source so oracle results cannot be overwritten:

- `intent_sweep_oracle_<split>_summary.csv`
- `intent_sweep_predicted_<split>_predictions.csv`
- `intent_sweep_predicted_<split>_summary.csv`
- `intent_sweep_<source>_<split>_vs_base_paired.csv`
- `intent_sweep_oracle_vs_predicted_<split>_paired.csv`

Each newly generated auxiliary-input prediction records the parsed label,
fallback flag, and raw Llama output. The reused text-only baseline has no new
raw output. Fallback counts are passed into the saved metrics rather than being
incorrectly reported as zero. Reusing the identical baseline means the
predicted run makes 7 rather than 8 new calls per row.

Smoke test without loading the LLM (saved separately and not scored):

```bash
python3 -m src.stage2_intent_sweep --split dev --feature-source predicted --limit 5 --mock
```

**Oracle result (eval, 264 rows):**

| combination | macro-F1 | 95% CI | kappa |
| --- | ---: | --- | ---: |
| `dialogue_act` | **0.464** | [0.401, 0.527] | 0.392 |
| `dialogue_act+sentiment` | 0.411 | [0.347, 0.470] | 0.335 |
| `emotion+sentiment` | 0.408 | [0.340, 0.472] | 0.307 |
| `dialogue_act+emotion` | 0.404 | [0.340, 0.463] | 0.331 |
| `emotion` | 0.403 | [0.341, 0.461] | 0.331 |
| `sentiment` | 0.390 | [0.329, 0.447] | 0.291 |
| `dialogue_act+emotion+sentiment` | 0.388 | [0.325, 0.448] | 0.304 |
| `base` (text-only) | 0.387 | [0.322, 0.444] | 0.282 |

**Predicted-input result (leakage-free common eval, 260 rows):**

| combination | macro-F1 | 95% CI | kappa |
| --- | ---: | --- | ---: |
| `dialogue_act` | **0.435** | [0.368, 0.499] | 0.352 |
| `dialogue_act+sentiment` | 0.397 | [0.331, 0.462] | 0.309 |
| `dialogue_act+emotion+sentiment` | 0.392 | [0.329, 0.456] | 0.304 |
| `dialogue_act+emotion` | 0.392 | [0.328, 0.456] | 0.295 |
| `base` (text-only) | 0.380 | [0.318, 0.442] | 0.273 |
| `emotion` | 0.373 | [0.307, 0.438] | 0.273 |
| `emotion+sentiment` | 0.370 | [0.306, 0.434] | 0.254 |
| `sentiment` | 0.360 | [0.298, 0.422] | 0.237 |

**Re-validated with a fresh, independent LLM run:** both tables above were
originally produced before every generation call saved its raw output, so a
later parsing-bug fix couldn't be verified against them without re-running the
model. Both have since been re-run for real (not just re-parsed); the numbers
moved by ≤0.002 everywhere and `dialogue_act` remains the only combination
whose CI excludes 0 in both oracle and predicted mode — the finding holds up
on a completely independent generation. Details in
[`docs/stage2_report.md`](docs/stage2_report.md).

**Final RQ2 finding:** gold Dialogue Act is the strongest oracle feature
(0.464 vs. 0.387 text-only on 264 rows). More importantly, predicted Dialogue
Act also improves Intent on the common 260 rows (0.435 vs. 0.380; paired
difference +0.055, 95% CI [+0.001, +0.112]). This primary paired comparison
provides modest, borderline evidence that most of the Dialogue Act benefit
survives realistic Stage 1 errors. On the same 260 rows, oracle DA is
only +0.024 above predicted DA, and that difference is uncertain (95% CI
[-0.021, +0.065]). Predicted Sentiment and Emotion do not improve the baseline,
alone or when stacked with DA. The other seven-combination rankings are treated
as exploratory because no multiple-comparison correction was applied. These
are auxiliary-information results, not a general claim about module execution
order.

This is also a worked example of why dev/eval are kept separate: the 36-row
dev sweep showed a much starker pattern (`sentiment` clearly hurting,
all-three-combined clearly worst) that did not fully replicate at n=264 — most
of that apparent effect was dev-sample noise, not a real signal, and only
`dialogue_act`'s advantage held up.

### Error analysis and write-up

- [x] Sample and manually review misclassified examples, categorized by source:
      Intent confusion matrix, Dialogue-Act→Intent cascade analysis, and 5 case
      studies (long/multi-topic text, Parent context underused, DA misprediction
      cascading into Intent error, ambiguous Challenge/Counter-argue boundary,
      surface form vs. true intent) — see `docs/stage2_report.md`'s "Error
      Analysis" section.
- [x] Add paired uncertainty for Experiment 1's multi-module vs. single-prompt
      differences — see the paired-bootstrap result above; not significant
      either direction.
- [ ] Extend `docs/full-report-draft.tex` (or the thesis draft) with Stage 2
      Methodology, Experiments, and Results sections — `docs/stage2_report.md`
      has the full content, still needs folding into the formal writeup.
