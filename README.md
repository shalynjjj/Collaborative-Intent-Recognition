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
python3 -m src.train_roberta strategy_b --silver-csv results/strategy_a/zeroshot_predictions.csv --sizes 5 --seeds 42 --dry-run
```

The default Strategy B learning-curve sizes are:

`500`, `1000`, `1500`, `2000`, `2500`, `3000`, `5000`, `8000`, `10000`

The `500`-`2500` sizes were added to resolve the shape of the curve below 3000, since macro-F1 was already flat across `3000`-`10000`.

Per-seed outputs and `summary.csv` are saved under `results/strategy_b/`.

## Strategy C: RoBERTa on Gold Labels

Runs 5-fold cross-validation with seeds `42`, `123`, and `2026`.

```bash
python3 -m src.train_roberta strategy_c
```

Smoke test without fine-tuning:

```bash
python3 -m src.train_roberta strategy_c --seeds 42 --dry-run
```

Per-fold, per-seed outputs and `summary.csv` are saved under `results/strategy_c/`.

### Strategy C: Partial Fine-Tuning (Optional)

Full-parameter fine-tuning of RoBERTa-base on gold-only folds (~240 rows per fold) can be unstable and collapse to predicting only 1-2 classes, since ~125M parameters are being updated from very little data. `--trainable-layers N` freezes the embeddings and all but the last `N` encoder layers, leaving only those layers plus the classification head trainable:

```bash
python3 -m src.train_roberta strategy_c --trainable-layers 2
```

Smoke test:

```bash
python3 -m src.train_roberta strategy_c --seeds 42 --dry-run --trainable-layers 2
```

### What We Tried and Found

We tested 3 versions of Strategy C:

1. **Full fine-tuning** (all layers trainable): the model often collapsed and only predicted 1-2 of the 4 classes. macro-F1 ≈ 0.20.
2. **`--trainable-layers 2`**: did not fix it. The model collapsed even harder, predicting only 1 class every time, no matter the input. macro-F1 ≈ 0.16.
3. **`--trainable-layers 4`**: same collapse. macro-F1 ≈ 0.17.

Since freezing more or fewer layers gave almost the same broken result, the number of frozen layers is probably not the real cause. A more likely cause: the class-weighted loss ([train_roberta.py:110-114](src/train_roberta.py#L110-L114)) combined with no warmup steps, on very little data (~200 rows per fold), may push the model toward a "shortcut" answer very early in training that it never recovers from. Next step: try turning off or lowering the class weights and adding warmup, instead of changing the frozen layers further.

Omitting `--trainable-layers` keeps the original full-parameter fine-tuning behavior. This flag only applies to Strategy C; Strategy B is unaffected.

## Notebook

`notebooks/stage1_experiments.ipynb` is intentionally thin: it imports from `src/`, runs scripts, and displays saved results.

## Server Run Order

For the RTX 4090 server, use this order before launching full experiments:

1. Install dependencies.
2. Run a local plumbing check:

```bash
python3 -m src.llm_annotate --mode zeroshot --mock --limit 5
python3 -m src.train_roberta strategy_c --seeds 42 --dry-run
```

3. Run the real 15-row Strategy A test:

```bash
python3 -m src.llm_annotate --mode zeroshot --limit 15
```

4. Run one small RoBERTa smoke test only after the real Strategy A test succeeds:

```bash
python3 -m src.train_roberta strategy_b \
  --silver-csv results/strategy_a/zeroshot_predictions.csv \
  --sizes 15 \
  --seeds 42
```

5. Run full Strategy C:

```bash
python3 -m src.train_roberta strategy_c
```

6. Build and label the 10k silver candidate file, then run full Strategy B:

```bash
python3 -m src.prepare_silver_data
python3 -m src.llm_annotate \
  --mode zeroshot \
  --input-csv data/task3_silver_candidates_10k.csv \
  --output-csv data/task3_silver_labeled_10k.csv
python3 -m src.train_roberta strategy_b --silver-csv data/task3_silver_labeled_10k.csv
```

Full RoBERTa experiments use `AI-ModelScope/roberta-base`, learning rate `2e-5`, batch size `8`, `3` epochs, and seeds `42`, `123`, `2026`.
Each training run reserves an inner validation split and loads the checkpoint with the best validation `macro_f1`.
Class-weighted cross-entropy is enabled to reduce majority-class collapse on the imbalanced gold labels.
