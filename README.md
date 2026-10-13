# Dialogue Act and Intent Recognition in Online Argumentative Discussions Using Large Language Models

We classify replies in Reddit's Change My View (CMV) discussions. Each example is a Parent–Reply pair: the Parent is the comment being answered and the Reply is the one we classify. The Parent matters because a reply that looks like a neutral statement on its own can be disagreeing with, or questioning, the comment above it.

The project has two stages:

- Stage 1 labels the Reply's dialogue act: `agree`, `disagree`, `question` or `statement`.
- Stage 2 labels Sentiment, Emotion and Intent. Intent is the more specific purpose of the reply, such as asking for information, challenging a claim, counter-arguing or supporting.

## Research questions

1. For dialogue acts, how does prompting an LLM directly compare with fine-tuning RoBERTa on LLM-generated (silver) labels or on human (gold) labels?
2. Can silver labels give a competitive classifier when there is little hand-labelled data?
3. For Sentiment, Emotion and Intent, is one joint prompt better or worse than a separate prompt per task?
4. Does giving the model the Dialogue Act, Sentiment or Emotion label help it predict Intent, both with human labels and with predicted ones?

## Method

### Stage 1: dialogue acts

All three strategies were tuned on a 300-pair gold set and then evaluated once on a separate, balanced test set of 332 pairs.

- Strategy A prompts Llama 3.1 8B Instruct directly. We compared a zero-shot and a few-shot prompt and kept the few-shot one.
- Strategy B uses Llama to label a larger pool of CMV pairs and fine-tunes RoBERTa on those labels. The final model uses 2,500 few-shot silver examples with class weights, trained with three seeds.
- Strategy C fine-tunes RoBERTa on the 300 gold pairs only. We chose its settings with five-fold cross-validation and ended up with class weights, 20 warmup steps and 8 epochs.

Strategy A uses a different model (Llama) from B and C (RoBERTa), so comparisons with A mix the effect of the model with the effect of the training approach.

### Stage 2: Sentiment, Emotion and Intent

The same 300 gold pairs were split into 36 for prompt development and 264 for evaluation.

Experiment 1 compares two ways of prompting. In `single_prompt`, one LLM call returns all three labels. In `multi_module`, three separate prompts each handle one task. The three prompts run independently; none of them sees the others' output.

Experiment 2 asks whether extra labels help Intent. We start from a prompt that sees only the Parent and Reply, then add every combination of Dialogue Act, Sentiment and Emotion. The extra labels come from one of three sources:

- `oracle`: the human labels
- `predicted`: Strategy A's Dialogue Act plus the multi-module Sentiment and Emotion predictions
- `predicted_sp`: the same Dialogue Act plus the single-prompt Sentiment and Emotion predictions

We report Macro-F1 throughout. For single-label tasks we also report Cohen's kappa, per-class scores and confusion matrices. Emotion is multi-label, so we compute F1 for each of its six categories and average them. Where two conditions are compared on the same examples, we use a paired bootstrap confidence interval.

## Repository layout

```text
.
├── data/          datasets (see data/README.md)
├── results/       saved predictions, metrics and confusion matrices (see results/README.md)
├── src/           all experiment code (see src/README.md)
├── tests/         unit tests
├── .env.example
├── requirements.txt
└── 2026_depolarAIze_MScProject_Proposal_XinXiaying.pdf
```

The scripts you are most likely to need:

| Script | What it runs |
|---|---|
| `src/llm_annotate.py` | Strategy A, and silver-label generation |
| `src/train_roberta.py` | Strategies B and C |
| `src/stage2_pipeline.py` | Stage 2, Experiment 1 |
| `src/evaluate_stage2.py` | Scoring for Experiment 1 |
| `src/stage2_intent_sweep.py` | Stage 2, Experiment 2 |

`results/` also holds outputs from earlier, superseded runs. `results/README.md` says which files the reported numbers come from.

## Setup

You need Python 3.10 or newer (the current ModelScope release requires it). Dependency versions are not pinned.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Models are downloaded from ModelScope by default. To use Hugging Face instead, copy the template and add your token:

```bash
cp .env.example .env
```

```dotenv
HF_TOKEN=your_huggingface_token_here
```

The first run downloads the models, so you need network access and enough disk space. Llama 3.1 8B needs a GPU in practice. Run every command from the repository root.

## Running the experiments

### Strategy A

```bash
# compare prompts on the gold set
python3 -m src.llm_annotate --mode zeroshot
python3 -m src.llm_annotate --mode fewshot

# final run on the held-out set with the few-shot prompt
python3 -m src.llm_annotate --heldout
```

To regenerate the few-shot silver labels from the candidate pool:

```bash
python3 -m src.llm_annotate \
  --mode fewshot \
  --input-csv data/task3_silver_candidates_10k.csv \
  --output-csv data/task3_silver_labeled_10k_fewshot.csv
```

### Strategy B

```bash
# final configuration on the gold set
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k_fewshot.csv \
  --sizes 2500 \
  --sample-seeds 123 \
  --train-seeds 42 123 2026 \
  --class-weights \
  --results-dir results/strategy_b_fewshot

# held-out evaluation
python3 -m src.train_roberta strategy_b_heldout
```

The held-out command retrains the three models and saves them to `results/strategy_b_heldout/models/`. The checkpoints are not in the repository, so run this before using `src.predict_with_saved_strategy_b`.

### Strategy C

```bash
# five-fold cross-validation with the chosen settings
python3 -m src.train_roberta strategy_c \
  --model roberta \
  --seeds 42 123 2026 \
  --split-seed 42 \
  --class-weights \
  --warmup-steps 20 \
  --epochs 8

# held-out evaluation
python3 -m src.train_roberta strategy_c_heldout
```

### Stage 2, Experiment 1

```bash
python3 -m src.stage2_pipeline --mode multi_module --split eval
python3 -m src.evaluate_stage2 --mode multi_module --split eval

python3 -m src.stage2_pipeline --mode single_prompt --split eval
python3 -m src.evaluate_stage2 --mode single_prompt --split eval
```

### Stage 2, Experiment 2

Run `oracle` first. The two predicted conditions reuse its Parent–Reply-only baseline, and they also need the saved Strategy A and Experiment 1 predictions.

```bash
python3 -m src.stage2_intent_sweep --split eval --feature-source oracle
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted_sp
```

If the predictions already exist, add `--eval-only` to recompute the metrics without calling the LLM again.

### Tests

```bash
python3 -m unittest discover -s tests -v
```

There are 27 tests. They cover silver sampling, the grouped train/validation split, out-of-fold aggregation, Stage 2 output parsing, prompt inputs, loading predicted features, and resuming partial runs.

## Data

| File | Rows | Used for |
|---|---:|---|
| `data/cmv_300_gold_final.csv` | 300 | Stage 1 development; all Stage 2 labels |
| `data/task3_silver_candidates_10k.csv` | 10,000 | Unlabelled silver candidates |
| `data/task3_silver_labeled_10k_fewshot.csv` | 10,000 | Silver labels used for Strategy B |
| `data/cmv_test_candidates_heldout_balanced.csv` | 332 | Stage 1 test set, 83 per class |
| `data/cmv_300_gold_stage2_dev.csv` | 36 | Stage 2 prompt development |
| `data/cmv_300_gold_stage2_eval.csv` | 264 | Stage 2 evaluation |

The Stage 2 split is stratified by Intent with seed 42. To recreate it:

```bash
python3 -m src.prepare_stage2_split
```

`data/README.md` explains how every file was built and how they relate to each other.

## Results

### Stage 1 (332 held-out pairs)

| Strategy | Method | Macro-F1 | Cohen's kappa |
|---|---|---:|---:|
| A | Llama, few-shot prompt | 0.6734 | 0.5582 |
| B | RoBERTa on 2,500 few-shot silver labels | 0.6686 ± 0.0016 | 0.5649 ± 0.0061 |
| C | RoBERTa on 300 gold labels | 0.5859 ± 0.0858 | 0.4833 ± 0.1125 |

B and C are the mean ± standard deviation over three training seeds. A and B are close, and the paired bootstrap does not show a real difference between them. C scores lower on average and varies a lot from seed to seed.

### Stage 2, Experiment 1 (264 eval pairs)

| Prompting | Sentiment | Emotion | Intent |
|---|---:|---:|---:|
| Multi-module | 0.5718 | 0.3994 | 0.3866 |
| Single-prompt | 0.6248 | 0.4255 | 0.4090 |

The single prompt scores higher on all three tasks, but for Sentiment and Intent the paired bootstrap interval includes zero, so we can't say one setup is better. We did not run a paired test for Emotion because it is multi-label.

### Stage 2, Experiment 2 (Intent Macro-F1)

| Extra labels from | Intent input | Rows | Macro-F1 | Change | Paired 95% CI |
|---|---|---:|---:|---:|---|
| Human (`oracle`) | Parent–Reply only | 264 | 0.3848 | | |
| Human (`oracle`) | + Dialogue Act | 264 | 0.4645 | +0.0797 | [0.0277, 0.1297] |
| Predicted | Parent–Reply only | 260 | 0.3779 | | |
| Predicted | + Dialogue Act | 260 | 0.4326 | +0.0547 | [0.0016, 0.1113] |

Dialogue Act is the only extra label that helped, with human labels and with predicted ones. Sentiment, Emotion and their combinations did not. The `predicted_sp` run gives the same picture. The most common Intent error is predicting `Counter-argue` for a reply that is actually `Challenge`.

## Limitations

- Strategy A uses Llama while B and C use RoBERTa, so differences involving A combine model and training effects.
- Nobody checked the silver labels by hand.
- Strategy C only has 300 training examples and is sensitive to the random seed.
- Stage 2 uses a single 36/264 split, and some Intent and Emotion classes have very few examples.
- Dialogue Act and Intent are closely linked in our annotation scheme. The Dialogue Act gain shows that the label is useful under this scheme, not that the model reasons more deeply.
- `data/winning-args-corpus/utterances.jsonl`, which is needed to rebuild the silver candidates, is too large for git and is not included.
- Model checkpoints are not included and dependencies are not pinned, so a rerun may differ slightly depending on hardware and library versions.

All reported numbers come from the files saved in `results/`. We did not rerun the GPU experiments after tidying up the code.

## References

The data comes from the Winning Arguments Corpus: Tan et al. (2016), *Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions*.

Models: `LLM-Research/Meta-Llama-3.1-8B-Instruct` and `AI-ModelScope/roberta-base`.

There is no licence file in this repository. Any reuse must follow the terms of the dataset and the model providers.
