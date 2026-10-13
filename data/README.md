# Data

Every example is a Parent–Reply pair from r/ChangeMyView, taken from the Winning Arguments Corpus (Tan et al., 2016). The Parent is the comment being answered; the Reply is the one we classify.

There are three groups of data:

- the gold set: 300 pairs we labelled by hand for Dialogue Act, Sentiment, Emotion and Intent. We used it to develop Stage 1, to train Strategy C, and for all of Stage 2.
- the held-out test set: 332 hand-labelled pairs (Dialogue Act only), used once per strategy for the final Stage 1 evaluation.
- the silver set: 10,000 pairs labelled by Llama, used to train Strategy B.

No pair appears in more than one of the gold set, the held-out set and the silver file used for Strategy B.

## Files used in the experiments

| File | Rows | Contents |
|---|---:|---|
| `cmv_300_gold_final.csv` | 300 | The gold set, with all four label types |
| `cmv_300_gold_stage2_dev.csv` | 36 | Gold pairs used to develop the Stage 2 prompts |
| `cmv_300_gold_stage2_eval.csv` | 264 | Gold pairs the Stage 2 results are reported on |
| `cmv_test_candidates_heldout_balanced.csv` | 332 | The Stage 1 test set, 83 pairs per class |
| `cmv_test_candidates_heldout_expanded.csv` | 530 | Every held-out pair we annotated; the 332 are drawn from these |
| `task3_silver_candidates_10k.csv` | 10,000 | Unlabelled pairs to be labelled by Llama |
| `task3_silver_labeled_10k_fewshot.csv` | 10,000 | Those pairs with Llama's few-shot labels; Strategy B trains on these |
| `task3_silver_labeled_10k.csv` | 10,000 | Zero-shot Llama labels on an earlier sample; only used in early exploratory runs |

## Other files

None of these are needed to rerun the experiments. Most of them are the paper trail of how we built the held-out set.

| File | Rows | Contents |
|---|---:|---|
| `cmv_test_candidates.csv` | 149 | Round 1 candidates |
| `cmv_test_candidates_random_topup.csv` | 49 | Extra random candidates added to round 1 |
| `cmv_test_candidates_labeled.csv` | 162 | Round 1 after annotation |
| `cmv_test_candidates_pilot80.csv` | 80 | Round 1 pilot set, 20 per class |
| `cmv_test_candidates_batch2.csv` | 450 | Round 2 sample, with a heuristic label we used to track progress |
| `cmv_test_candidates_batch2_annotate.csv` | 450 | The copy we annotated, with the heuristic label removed |
| `cmv_test_candidates_batch2_annotated.csv` | 450 | Round 2 with both annotators' labels and the agreed label |
| `cmv_full_with_text_features.csv` | 830 | Gold and held-out pairs with text features added, for error analysis |
| `*.summary.json` | | Model, prompt type and label counts for each silver file |
| `winning-args-corpus/` | | Corpus metadata |

## Labels

| Task | Column | Values |
|---|---|---|
| Dialogue Act | `Dialogue_act` | `agree`, `disagree`, `question`, `statement` |
| Sentiment | `Sentiment` | `Positive`, `Negative`, `Neutral` |
| Emotion | `Sarcasm`, `Hostility`, `Contempt`, `Neutral`, `Curiosity`, `Appreciation` | `True`/`False` in each column; a reply can have several |
| Intent | `Intent` | `Information seeking`, `Challenge`, `Counter-argue`, `Support`, `Others` |

Columns by file type:

- Gold and Stage 2 files: `Parent`, `Reply`, `Dialogue_act`, `Sentiment`, the six Emotion columns, `Intent`.
- Held-out files: `sample_id`, `utt_id` (the reply's ID in the corpus), `Parent`, `Reply`, `token_count`, `length_group` (`short`, `mid`, `long`), `Dialogue_act`, and `source`, which says whether the pair came from `pilot80` or `batch2`.
- Silver files: `row_id`, `Parent`, `Reply`, `pred_label` (Llama's label), `raw_output` (Llama's full answer), `fallback_used`, `source_root` (the thread), `parent_id`, `reply_id`.

## How the data was built

### Gold set

```text
CMV corpus -> 300 pairs, labelled by two annotators for all four tasks,
              disagreements settled by discussion -> cmv_300_gold_final.csv (300)

cmv_300_gold_final.csv -> prepare_stage2_split.py (stratified by Intent, seed 42)
                       -> cmv_300_gold_stage2_dev.csv (36)
                       -> cmv_300_gold_stage2_eval.csv (264)
```

### Held-out set

We collected new pairs (none of them in the gold set) and two annotators labelled them in two rounds.

```text
Round 1
prepare_gold_test_candidates.py -> cmv_test_candidates.csv (149)
extra random candidates         -> cmv_test_candidates_random_topup.csv (49)
annotation                      -> cmv_test_candidates_labeled.csv (162)
20 pairs per class              -> cmv_test_candidates_pilot80.csv (80)

Round 2
prepare_test_topup_candidates.py (seed 2026) -> cmv_test_candidates_batch2.csv (450)
heuristic label removed                      -> cmv_test_candidates_batch2_annotate.csv (450)
annotation                                   -> cmv_test_candidates_batch2_annotated.csv (450)

Combined
pilot80 (80) + batch2 (450) -> cmv_test_candidates_heldout_expanded.csv (530)
prepare_heldout_balanced.py -> cmv_test_candidates_heldout_balanced.csv (332, 83 per class)
```

The annotators never saw the heuristic label, so it couldn't influence them.

The 530 pairs are unbalanced (disagree 188, statement 164, question 95, agree 83), so we kept 83 of each class, the size of the smallest one. This means the test set does not follow the natural class distribution on CMV.

### Silver set

```text
CMV corpus -> prepare_silver_data.py (gold pairs left out) -> task3_silver_candidates_10k.csv (10,000)
           -> llm_annotate.py with the few-shot prompt     -> task3_silver_labeled_10k_fewshot.csv (10,000)
           -> 2,500 of them                                -> Strategy B training data

earlier sample -> llm_annotate.py with the zero-shot prompt -> task3_silver_labeled_10k.csv (10,000)
```

The zero-shot file was made before we fixed the candidate sampling, so it is based on an older sample and has only 358 pairs in common with the current candidates. We switched to the few-shot labels because the few-shot prompt did better on the gold set.

`add_text_features.py` takes the gold set (300) and the 530 held-out pairs, adds text features for both Parent and Reply, and writes `cmv_full_with_text_features.csv` (830).

## Loading

```python
import pandas as pd

gold = pd.read_csv("data/cmv_300_gold_final.csv", encoding="utf-8-sig")
test = pd.read_csv("data/cmv_test_candidates_heldout_balanced.csv", encoding="utf-8-sig")
```

Use `utf-8-sig`: the files are UTF-8 and some start with a byte-order mark.

## Caveats

- Please don't edit these files. The reported results depend on them, so save any changed data under a new name.
- Our code reads the gold and held-out files as `latin1`, which garbles a few non-ASCII characters (2 gold rows and 13 held-out rows). The reported results were produced this way, and we left it as it is.
- The zero-shot silver file shares 8 pairs with the held-out set. It was only ever evaluated on the gold set, so this doesn't affect any held-out result.
- Different replies from the same thread can end up in different sets, so the results describe in-domain performance.
- `winning-args-corpus/utterances.jsonl` is not included. You only need it to rebuild the silver candidates.
- If you use the data, please cite Tan et al. (2016), *Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions*.
