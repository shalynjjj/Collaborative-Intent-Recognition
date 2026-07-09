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

`3000`, `5000`, `8000`, `10000`

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
