# Dialogue Act and Intent Recognition in Online Argumentative Discussions Using Large Language Models

This project investigates Dialogue Act and Intent recognition in Reddit's Change My View (CMV) discussions using LLMs.

## 1. Project Overview

CMV is an online forum where users present a view and discuss it with other participants. This project represents each interaction as a **Parent–Reply pair**: the Parent is the comment being answered, and the Reply is the response to classify. Interpreting the Reply together with its Parent is important because a sentence that appears neutral in isolation may agree with, challenge, or question the earlier comment.

The project has two stages. **Stage 1** classifies the Reply's Dialogue Act as `agree`, `disagree`, `question`, or `statement`. **Stage 2** extends the analysis to Sentiment, Emotion, and Intent. Here, Intent describes the more specific purpose of a Reply, such as seeking information, challenging a claim, counter-arguing, or offering support.

## 2. Research Objectives

The project addresses four main questions:

- How does direct LLM prompting compare with RoBERTa trained on LLM-generated silver labels or human-annotated gold labels for Dialogue Act classification?
- Can silver labels support a competitive supervised classifier when manually labelled training data are limited?
- For Sentiment, Emotion, and Intent, how does a single joint prompt compare with independent task-specific prompts?
- Do Dialogue Act, Sentiment, or Emotion labels improve Intent classification when the auxiliary labels are human-annotated or model-predicted?

## 3. Methodology

### Stage 1 — Dialogue Act Classification

The three strategies were developed on a 300-example gold set. After configuration selection, each final strategy was evaluated on the same balanced 332-example held-out set.

- **Strategy A — Direct LLM prompting:** Llama 3.1 8B Instruct predicts Dialogue Act labels directly. Zero-shot and few-shot prompts were compared, and few-shot prompting was selected.
- **Strategy B — RoBERTa on silver labels:** Llama labels a larger pool of CMV examples, which is then used to fine-tune RoBERTa. The final configuration uses 2,500 examples from the few-shot-labelled silver pool, class weights, and three training seeds.
- **Strategy C — RoBERTa on gold labels:** RoBERTa is trained directly on the 300 human-annotated examples. Five-fold out-of-fold evaluation was used for configuration selection. The final setting uses class weights, 20 warmup steps, and eight epochs.

This comparison examines whether direct LLM use, LLM-generated training data, or limited gold training data provides the most effective route for this task. Because Strategy A uses Llama while Strategies B and C use RoBERTa, the comparison does not isolate model architecture from training strategy.

### Stage 2 — Sentiment, Emotion, and Intent

The same 300 annotated Parent–Reply pairs were divided into 36 development examples for prompt design and 264 examples for final evaluation.

#### Experiment 1 — Prompting architectures

- **Single-prompt:** one LLM call predicts Sentiment, Emotion, and Intent jointly.
- **Multi-module:** three independent task-specific prompts predict the three targets separately.

Both architectures receive only Parent and Reply. The independent prompts are not a sequential pipeline and do not pass predictions between tasks.

#### Experiment 2 — Auxiliary information for Intent

A Parent–Reply-only Intent prompt was compared with all combinations of Dialogue Act, Sentiment, and Emotion as additional inputs. The experiment used:

- `oracle` auxiliary labels from human annotations;
- `predicted` labels using Strategy A Dialogue Act predictions and multi-module Sentiment and Emotion predictions;
- `predicted_sp` labels using the same Dialogue Act predictions and single-prompt Sentiment and Emotion predictions.

Macro-F1 is the main evaluation measure. Single-label tasks also report Cohen's kappa, per-class scores, and confusion matrices. Emotion is multi-label, so F1 is calculated separately for its six categories and then macro-averaged. Selected comparisons use paired bootstrap confidence intervals.

## 4. Repository Structure

```text
.
├── data/          # Gold, silver, held-out, and Stage 2 datasets
├── docs/          # Experiment reports and error analyses
├── notebooks/     # Exploratory Stage 1 notebook
├── results/       # Saved predictions, metrics, and confusion matrices
├── src/           # Data preparation, training, inference, and evaluation code
├── tests/         # Unit tests for the experiment logic
├── .env.example
├── requirements.txt
└── 2026_depolarAIze_MScProject_Proposal_XinXiaying.pdf
```

The main entry points are:

- `src/llm_annotate.py` — Strategy A and silver-label generation
- `src/train_roberta.py` — Strategies B and C
- `src/stage2_pipeline.py` — Stage 2 Experiment 1
- `src/stage2_intent_sweep.py` — Stage 2 Experiment 2
- `src/evaluate_stage2.py` — Stage 2 Experiment 1 evaluation

The repository retains historical experiments and diagnostic outputs. The final results are identified in Section 7 below.

## 5. Installation and Quick Start

### Environment setup

The repository does not pin Python or dependency versions. The CLI requires Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Models are downloaded from ModelScope by default. If using Hugging Face model access instead, copy the environment template and add a local token:

```bash
cp .env.example .env
```

```dotenv
HF_TOKEN=your_huggingface_token_here
```

Real Llama and RoBERTa runs require network access for the initial model download, sufficient disk space, and suitable compute. Llama 3.1 8B inference is intended for GPU execution. Commands should be run from the repository root.

### Stage 1 — Strategy A

```bash
# Development comparison
python3 -m src.llm_annotate --mode zeroshot
python3 -m src.llm_annotate --mode fewshot

# Final held-out evaluation with the locked few-shot prompt
python3 -m src.llm_annotate --heldout
```

To regenerate the final few-shot silver labels from the included candidate pool:

```bash
python3 -m src.llm_annotate \
  --mode fewshot \
  --input-csv data/task3_silver_candidates_10k.csv \
  --output-csv data/task3_silver_labeled_10k_fewshot.csv
```

### Stage 1 — Strategy B

```bash
# Final development configuration
python3 -m src.train_roberta strategy_b \
  --silver-csv data/task3_silver_labeled_10k_fewshot.csv \
  --sizes 2500 \
  --sample-seeds 123 \
  --train-seeds 42 123 2026 \
  --class-weights \
  --results-dir results/strategy_b_fewshot

# Locked held-out evaluation
python3 -m src.train_roberta strategy_b_heldout
```

The held-out command retrains the three models and saves checkpoints under `results/strategy_b_heldout/models/`. These checkpoints are not included in the current checkout, so they must be regenerated before using `src.predict_with_saved_strategy_b`.

### Stage 1 — Strategy C

```bash
# Selected five-fold OOF configuration
python3 -m src.train_roberta strategy_c \
  --model roberta \
  --seeds 42 123 2026 \
  --split-seed 42 \
  --class-weights \
  --warmup-steps 20 \
  --epochs 8

# Locked held-out evaluation
python3 -m src.train_roberta strategy_c_heldout
```

### Stage 2 — Experiment 1

```bash
python3 -m src.stage2_pipeline --mode multi_module --split eval
python3 -m src.evaluate_stage2 --mode multi_module --split eval

python3 -m src.stage2_pipeline --mode single_prompt --split eval
python3 -m src.evaluate_stage2 --mode single_prompt --split eval
```

### Stage 2 — Experiment 2

Run the oracle condition first because the predicted conditions reuse its Parent–Reply-only baseline:

```bash
python3 -m src.stage2_intent_sweep --split eval --feature-source oracle
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted
python3 -m src.stage2_intent_sweep --split eval --feature-source predicted_sp
```

The predicted conditions require Strategy A few-shot predictions and the corresponding Experiment 1 predictions. Existing complete predictions can be rescored without new LLM calls by adding `--eval-only` to each command.

### Tests

```bash
python3 -m unittest discover -s tests -v
```

The current suite contains 27 tests covering sampling, grouped splitting, out-of-fold aggregation, Stage 2 parsing, prompt inputs, predicted-feature loading, and resumable output generation.

## 6. Data Overview

| File | Rows | Role |
|---|---:|---|
| `data/cmv_300_gold_final.csv` | 300 | Stage 1 gold development data and complete Stage 2 annotations |
| `data/task3_silver_candidates_10k.csv` | 10,000 | Unlabelled silver candidates |
| `data/task3_silver_labeled_10k_fewshot.csv` | 10,000 | Silver pool used by final Strategy B |
| `data/cmv_test_candidates_heldout_balanced.csv` | 332 | Balanced final Stage 1 test set, 83 examples per class |
| `data/cmv_300_gold_stage2_dev.csv` | 36 | Stage 2 prompt-development split |
| `data/cmv_300_gold_stage2_eval.csv` | 264 | Stage 2 final evaluation split |

The Stage 2 split is stratified by Intent using seed 42 and can be regenerated with:

```bash
python3 -m src.prepare_stage2_split
```

## 7. Results

### Stage 1 final held-out evaluation

All three locked configurations were evaluated on the same balanced set of 332 Parent–Reply pairs.

| Strategy | Final method | Macro-F1 | Cohen's kappa |
|---|---|---:|---:|
| A | Llama few-shot prompting | 0.6734 | 0.5582 |
| B | RoBERTa on 2,500 few-shot silver labels | 0.6686 ± 0.0016 | 0.5649 ± 0.0061 |
| C | RoBERTa on 300 gold labels | 0.5859 ± 0.0858 | 0.4833 ± 0.1125 |

The Strategy B and C values are means ± sample standard deviations across three training seeds. Strategies A and B had similar Macro-F1 scores, and their paired-bootstrap comparison did not support a difference. Strategy C had a lower mean and greater variation across training seeds.

### Stage 2 Experiment 1

| Architecture | Sentiment Macro-F1 | Emotion Macro-F1 | Intent Macro-F1 |
|---|---:|---:|---:|
| Multi-module | 0.5718 | 0.3994 | 0.3866 |
| Single-prompt | 0.6248 | 0.4255 | 0.4090 |

The single prompt produced higher point estimates on all three tasks. However, paired bootstrap intervals for Sentiment and Intent included zero, so the experiment did not establish a supported performance difference between the architectures. No corresponding paired test was conducted for the multi-label Emotion score.

### Stage 2 Experiment 2

| Auxiliary source | Intent input | Rows | Macro-F1 | Difference from base | Paired 95% CI |
|---|---|---:|---:|---:|---|
| Gold (`oracle`) | Parent–Reply only | 264 | 0.3848 | — | — |
| Gold (`oracle`) | + Dialogue Act | 264 | 0.4645 | +0.0797 | [0.0277, 0.1297] |
| Predicted | Parent–Reply only | 260 | 0.3779 | — | — |
| Predicted | + Dialogue Act | 260 | 0.4326 | +0.0547 | [0.0016, 0.1113] |

Dialogue Act was the only auxiliary signal whose paired confidence interval against the baseline excluded zero in both the oracle and predicted sweeps. Sentiment, Emotion, and their combinations did not show a supported improvement. The `predicted_sp` sensitivity check led to the same Dialogue-Act-only conclusion.

The main Intent error was `Challenge` being predicted as `Counter-argue`. The complete analysis and examples are in `docs/stage2_intent_error_analysis.md`.

## 8. Limitations and Reproducibility

- Strategy A uses Llama, whereas Strategies B and C use RoBERTa. Differences involving Strategy A therefore combine model-family and training-strategy effects.
- The silver pool was not subjected to a systematic human accuracy audit.
- Strategy C is trained on only 300 gold examples and remains sensitive to the training seed.
- Stage 2 uses one fixed 36/264 split, with relatively few examples for several Intent and Emotion categories.
- Dialogue Act and Intent are strongly associated in the annotation scheme. The auxiliary-input result shows predictive value under this scheme, but does not by itself demonstrate deeper model reasoning.
- The raw `data/winning-args-corpus/utterances.jsonl` file required to rebuild the 10,000 silver candidates is git-ignored and not included.
- Trained checkpoints are not included, and Python dependencies are not version-pinned. Exact reproduction may therefore vary with model availability, hardware, and library versions.

The saved CSV and JSON artifacts contain the final predictions and metrics. The commands above were checked against the current CLIs, and all 27 unit tests passed during the README update. The GPU-intensive experiments were not rerun as part of this documentation work.

Further experiment records are available in:

- `docs/stage1_full_report.md`
- `docs/stage2_report.md`
- `docs/stage2_intent_error_analysis.md`
- `docs/parent_negation_ablation_report.md`

`docs/full-report-draft.tex` is an earlier draft and still contains outdated Stage 2 placeholders.

## 9. References

The CMV data were derived from the Winning Arguments Corpus introduced by Tan et al. (2016), *Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions*.

Configured model identifiers:

- `LLM-Research/Meta-Llama-3.1-8B-Instruct`
- `AI-ModelScope/roberta-base`

No licence file is included in this repository. Reuse must follow the terms of the source dataset and model providers.
