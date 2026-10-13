# Data Guide

## Overview

The `data/` folder contains all datasets used in the project. Every example is a **Parent–Reply pair** from Reddit's r/ChangeMyView, taken from the Winning Arguments Corpus (Tan et al., 2016). The Parent is the comment being answered, and the Reply is the response to classify.

The data falls into three groups:

- **Gold set:** 300 human-annotated pairs labelled for Dialogue Act, Sentiment, Emotion, and Intent. Used for Stage 1 development, Strategy C training, and all of Stage 2.
- **Held-out test set:** 332 human-annotated pairs labelled for Dialogue Act. Used only for the final Stage 1 evaluation.
- **Silver set:** 10,000 pairs labelled by Llama. Used to train Strategy B.

The gold set, the held-out set, and the silver data used by Strategy B share no Parent–Reply pairs. See the root [`README.md`](../README.md) for how these datasets are used in the experiments.

## Main Files

| File | Rows | Purpose |
|---|---:|---|
| `cmv_300_gold_final.csv` | 300 | Full gold set with all four label types. |
| `cmv_300_gold_stage2_dev.csv` | 36 | Stage 2 development split for prompt design. |
| `cmv_300_gold_stage2_eval.csv` | 264 | Stage 2 evaluation split for final results. |
| `cmv_test_candidates_heldout_balanced.csv` | 332 | Final Stage 1 test set, 83 examples per Dialogue Act class. |
| `cmv_test_candidates_heldout_expanded.csv` | 530 | All annotated held-out pairs; the 332 test examples are sampled from it. |
| `task3_silver_candidates_10k.csv` | 10,000 | Unlabelled candidate pairs for silver labelling. |
| `task3_silver_labeled_10k_fewshot.csv` | 10,000 | Llama few-shot labels for the candidates. Final Strategy B training pool. |
| `task3_silver_labeled_10k.csv` | 10,000 | Llama zero-shot labels for an earlier candidate sample. Exploratory runs only. |

The remaining files record how the held-out set was annotated, or provide extra analysis data. They are not required to run the final experiments:

| File | Rows | Purpose |
|---|---:|---|
| `cmv_test_candidates.csv` | 149 | Held-out round 1: sampled candidates. |
| `cmv_test_candidates_random_topup.csv` | 49 | Held-out round 1: random top-up candidates. |
| `cmv_test_candidates_labeled.csv` | 162 | Held-out round 1: annotated candidates. |
| `cmv_test_candidates_pilot80.csv` | 80 | Held-out round 1: balanced pilot set, 20 per class. |
| `cmv_test_candidates_batch2.csv` | 450 | Held-out round 2: sampled pairs with a heuristic label for tracking. |
| `cmv_test_candidates_batch2_annotate.csv` | 450 | Held-out round 2: annotator copy without the heuristic label. |
| `cmv_test_candidates_batch2_annotated.csv` | 450 | Held-out round 2: both annotators' labels and the final label. |
| `cmv_full_with_text_features.csv` | 830 | Gold and expanded held-out pairs with text features for error analysis. |
| `*.summary.json` | — | Model, prompt type, and label counts for each silver file. |
| `winning-args-corpus/` | — | Corpus metadata. |

## Labels and Columns

| Task | Column | Labels |
|---|---|---|
| Dialogue Act | `Dialogue_act` | `agree`, `disagree`, `question`, `statement` |
| Sentiment | `Sentiment` | `Positive`, `Negative`, `Neutral` |
| Emotion | `Sarcasm`, `Hostility`, `Contempt`, `Neutral`, `Curiosity`, `Appreciation` | `True` / `False` per column; several can be `True` |
| Intent | `Intent` | `Information seeking`, `Challenge`, `Counter-argue`, `Support`, `Others` |

- **Gold and Stage 2 files:** `Parent`, `Reply`, `Dialogue_act`, `Sentiment`, the six Emotion columns, and `Intent`.
- **Held-out files:** `sample_id`, `utt_id` (reply ID in the corpus), `Parent`, `Reply`, `token_count`, `length_group` (`short`, `mid`, `long`), `Dialogue_act`, and `source` (`pilot80` or `batch2`).
- **Silver files:** `row_id`, `Parent`, `Reply`, `pred_label` (Llama's Dialogue Act label), `raw_output` (full Llama response), `fallback_used`, `source_root` (thread ID), `parent_id`, and `reply_id`.

## Gold Set Workflow

```text
CMV corpus -> 300 selected pairs
           -> two annotators label all four tasks
           -> disagreements resolved by discussion -> cmv_300_gold_final.csv (300)

cmv_300_gold_final.csv -> prepare_stage2_split.py (stratified by Intent, seed 42)
                       -> cmv_300_gold_stage2_dev.csv (36)
                       -> cmv_300_gold_stage2_eval.csv (264)
```

## Held-Out Set Workflow

New pairs, none of them in the gold set, were annotated by two annotators in two rounds.

```text
Round 1 (pilot)
prepare_gold_test_candidates.py -> cmv_test_candidates.csv (149)
random top-up                   -> cmv_test_candidates_random_topup.csv (49)
annotation                      -> cmv_test_candidates_labeled.csv (162)
20 pairs per class              -> cmv_test_candidates_pilot80.csv (80)

Round 2 (second batch)
prepare_test_topup_candidates.py (seed 2026) -> cmv_test_candidates_batch2.csv (450)
remove heuristic label for annotators        -> cmv_test_candidates_batch2_annotate.csv (450)
annotation and final labels                  -> cmv_test_candidates_batch2_annotated.csv (450)

Combine and balance
pilot80 (80) + batch2 (450)        -> cmv_test_candidates_heldout_expanded.csv (530)
prepare_heldout_balanced.py        -> cmv_test_candidates_heldout_balanced.csv (332)
(83 per class, the size of agree)
```

The expanded set has the natural class distribution: disagree 188, statement 164, question 95, agree 83. The balanced set does not reflect the natural CMV distribution.

## Silver Set Workflow

```text
CMV corpus -> prepare_silver_data.py (gold pairs excluded) -> task3_silver_candidates_10k.csv (10,000)
           -> llm_annotate.py, few-shot prompt             -> task3_silver_labeled_10k_fewshot.csv (10,000)
           -> 2,500-example subset                         -> final Strategy B training data

earlier candidate sample -> llm_annotate.py, zero-shot prompt -> task3_silver_labeled_10k.csv (10,000)
```

The zero-shot file comes from an earlier version of the candidate sample, made before the sampling was corrected. It shares only 358 pairs with the current candidates. The few-shot prompt scored higher on the gold set, so its labels replaced the zero-shot labels.

`add_text_features.py` combines the gold set (300) and the expanded held-out set (530) and adds text features for both Parent and Reply to produce `cmv_full_with_text_features.csv` (830).

## Loading the Data

```python
import pandas as pd

gold = pd.read_csv("data/cmv_300_gold_final.csv", encoding="utf-8-sig")
test = pd.read_csv("data/cmv_test_candidates_heldout_balanced.csv", encoding="utf-8-sig")
```

## Important Notes

- Do not edit these files. The reported results depend on them; save new or modified data to new files.
- Files are UTF-8, and some include a byte-order mark, so use `encoding="utf-8-sig"`.
- The project code reads the gold and held-out files with `latin1`, which garbles a few non-ASCII characters (2 gold rows, 13 held-out rows). The reported results use this setting.
- The zero-shot silver file shares 8 pairs with the held-out set. It was only evaluated on the gold set, so no held-out result is affected.
- Different replies from the same thread can appear in different sets, so results reflect in-domain performance.
- The raw `winning-args-corpus/utterances.jsonl` is not included. It is only needed to rebuild the silver candidates.
- If you use this data, cite Tan et al. (2016), *Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions*.
