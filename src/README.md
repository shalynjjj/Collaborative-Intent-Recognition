# Source code

All experiment code for both stages lives here. Run every command from the repository root as `python3 -m src.<script>`. Installation and the final results are in the root [`README.md`](../README.md).

## Main scripts

| Script | What it does |
|---|---|
| `config.py` | Paths, label sets, model names, seeds, default settings, and the final configuration of each strategy |
| `prepare_silver_data.py` | Samples the 10,000 silver candidate pairs from the raw CMV utterances |
| `prepare_stage2_split.py` | Splits the gold set into the 36-pair Stage 2 dev set and the 264-pair eval set |
| `llm_annotate.py` | Runs Strategy A, and labels the silver pool with Llama for Strategy B |
| `train_roberta.py` | Trains and evaluates Strategies B and C |
| `stage2_pipeline.py` | Stage 2 Experiment 1, with either one joint prompt or three separate ones |
| `evaluate_stage2.py` | Scores the saved Experiment 1 predictions |
| `stage2_intent_sweep.py` | Stage 2 Experiment 2, adding different combinations of extra labels to the Intent prompt |
| `evaluate.py` | Shared metrics, confusion matrices and bootstrap tests |

The other scripts were used for collecting the held-out data, plotting the learning curve, error analysis, ablations and diagnostics, or predicting with a saved model. You don't need them to reproduce the main results.

## How Stage 1 fits together

```text
Strategy A
300 gold pairs   -> pick the prompt (zero-shot vs few-shot, compared on the 296 pairs
                    that are not few-shot examples)
few-shot prompt  -> run once on the 332 held-out pairs

Strategy B
10k CMV candidates -> Llama few-shot labels -> silver pool
300 gold pairs     -> choose silver size and settings
2,500 silver pairs -> fine-tune RoBERTa -> run once on the 332 held-out pairs

Strategy C
300 gold pairs     -> five-fold cross-validation to choose settings
chosen settings    -> train on all 300 gold pairs -> run once on the 332 held-out pairs
```

The held-out set was never used to choose a prompt or a setting.

## How Stage 2 fits together

The 300 gold pairs are split into 36 for prompt development and 264 for evaluation.

Experiment 1 compares one joint call (`single_prompt`) with three separate calls (`multi_module`) for Sentiment, Emotion and Intent. The three separate calls don't share their outputs.

Experiment 2 tests whether Dialogue Act, Sentiment or Emotion labels help the Intent prompt. `oracle` uses the human labels; `predicted` and `predicted_sp` use saved model predictions. The predicted runs need Strategy A's Dialogue Act predictions and the matching Experiment 1 predictions. They use 260 of the 264 eval pairs, because the four gold pairs used as Strategy A few-shot examples have no Strategy A prediction.

## Commands

### Strategy A

```bash
python3 -m src.llm_annotate --mode zeroshot
python3 -m src.llm_annotate --mode fewshot
python3 -m src.llm_annotate --heldout
```

To regenerate the few-shot silver labels:

```bash
python3 -m src.llm_annotate \
  --mode fewshot \
  --input-csv data/task3_silver_candidates_10k.csv \
  --output-csv data/task3_silver_labeled_10k_fewshot.csv
```

### Strategy B

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

`sample_seed` decides which silver subset is drawn; `train_seed` controls the randomness in training. The train/validation split is grouped by `source_root` (the thread), so replies from one thread never end up on both sides.

### Strategy C

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

### Stage 2, Experiment 1

```bash
python3 -m src.prepare_stage2_split

python3 -m src.stage2_pipeline --mode multi_module --split eval
python3 -m src.evaluate_stage2 --mode multi_module --split eval

python3 -m src.stage2_pipeline --mode single_prompt --split eval
python3 -m src.evaluate_stage2 --mode single_prompt --split eval
```

### Stage 2, Experiment 2

Run `oracle` first; the predicted runs reuse its Parent–Reply-only baseline.

```bash
python3 -m src.stage2_intent_sweep --split eval --feature-source oracle
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted_sp
```

With `--eval-only`, the metrics are recomputed from saved predictions without calling the LLM.

### Tests

```bash
python3 -m unittest discover -s tests -v
```

## Things worth knowing

- RoBERTa gets the Parent and Reply as a text pair through the tokenizer, truncated to 256 tokens.
- `set_seed` fixes the Python, NumPy, PyTorch and CUDA seeds.
- Every run saves its predictions, the raw LLM output, fallback flags, metrics and a confusion matrix under `results/`.
- If a Stage 2 run stops halfway, running the same command again picks up where it left off.
- The real Llama and RoBERTa runs download the models and need a GPU.
- `winning-args-corpus/utterances.jsonl` and the Strategy B checkpoints are not in the repository. The prepared CSVs are, and `strategy_b_heldout` recreates the checkpoints.
