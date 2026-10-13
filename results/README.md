# Results

Saved outputs from every experiment: predictions, metrics, confusion matrices and summary tables. You can check any number we report against these files without rerunning a model. The commands that write them are in [`src/README.md`](../src/README.md).

## Folders

| Folder | What's in it | Written by |
|---|---|---|
| `strategy_a/` | Strategy A on the gold set, zero-shot and few-shot prompts | `llm_annotate.py` |
| `strategy_a_heldout/` | Strategy A on the 332 held-out pairs | `llm_annotate.py --heldout` |
| `strategy_b_fewshot/` | Strategy B on the gold set: the silver-size learning curve, class weights on/off, the final setting | `train_roberta.py strategy_b` |
| `strategy_b_heldout/` | Strategy B on the held-out set, three training seeds | `train_roberta.py strategy_b_heldout` |
| `strategy_b_zeroshot/` | Early Strategy B runs on the zero-shot silver labels | `train_roberta.py strategy_b` |
| `strategy_c/` | Strategy C five-fold cross-validation: the warmup/epoch grid and the final setting | `train_roberta.py strategy_c` |
| `strategy_c_heldout/` | Strategy C on the held-out set, three training seeds | `train_roberta.py strategy_c_heldout` |
| `strategy_c_diagnose/` | A check of whether RoBERTa could fit its own training folds; this is what led us to add warmup and train for longer | `train_roberta.py strategy_c_diagnose` |
| `stage2/` | Stage 2 Experiments 1 and 2 | `stage2_pipeline.py`, `evaluate_stage2.py`, `stage2_intent_sweep.py` |
| `error_analysis/` | Stage 1 error analysis: text features of misclassified `disagree` replies, and the Parent-negation ablation | `analyze_disagree_errors.py`, `ablation_parent_negation*.py` |

## Where the reported numbers come from

```text
Stage 1, gold set
Strategy A, zero-shot vs few-shot (296 pairs) -> strategy_b_fewshot/strategy_a_common_296.csv
Strategy B learning curve                     -> strategy_b_fewshot/summary.csv
                                                 strategy_b_fewshot/learning_curve_fewshot.png
Strategy C warmup/epoch grid                  -> strategy_c/warmup_epochs_grid_summary.csv
Strategy C final setting                      -> strategy_c/roberta_weights_warmup20_epochs8/summary_oof.csv

Stage 1, held-out set (332 pairs)
Strategy A -> strategy_a_heldout/fewshot_metrics.json
Strategy B -> strategy_b_heldout/summary.csv   (one row per training seed)
Strategy C -> strategy_c_heldout/summary.csv   (one row per training seed)

Stage 2, Experiment 1 (264 pairs)
multi-module     -> stage2/multi_module_eval_summary.json
single-prompt    -> stage2/single_prompt_eval_summary.json
paired bootstrap -> stage2/multi_module_vs_single_prompt_eval_paired.csv

Stage 2, Experiment 2
human labels (264)              -> stage2/intent_sweep_oracle_eval_summary.csv
                                   stage2/intent_sweep_oracle_eval_vs_base_paired.csv
predicted labels (260)          -> stage2/intent_sweep_predicted_eval_summary.csv
                                   stage2/intent_sweep_predicted_eval_vs_base_paired.csv
predicted, single-prompt (260)  -> stage2/intent_sweep_predicted_sp_eval_summary.csv
                                   stage2/intent_sweep_predicted_sp_eval_vs_base_paired.csv
```

## What the files contain

Most runs write three files:

- `*_predictions.csv`: one row per example, with `Parent`, `Reply`, the gold and predicted labels, and for LLM runs the raw output and whether a fallback label was used
- `*_metrics.json`: Macro-F1, Cohen's kappa, per-class precision/recall/F1, the confusion matrix and the run settings
- `*_confusion_matrix.csv`: gold labels in rows, predicted labels in columns

`summary*.csv` and `*_summary.json` combine several runs. `*_paired.csv` files hold paired bootstrap comparisons: the difference in Macro-F1 and its 95% confidence interval.

## Reading the file names

| Name | Meaning |
|---|---|
| `silver_<size>_sample<s>_train<t>_weights<0/1>_*` | Strategy B: silver subset size, sampling seed, training seed, class weights off/on |
| `fold<k>_seed<t>_*` | Strategy C, cross-validation fold *k*, training seed *t* |
| `oof_seed<t>_*` | Strategy C, all five folds combined for seed *t* (300 predictions) |
| `roberta_weights_warmup<w>_epochs<e>/` | Strategy C with *w* warmup steps and *e* epochs |
| `heldout_seed<t>_*` | Held-out run with training seed *t* |
| `<mode>_<split>_*` | Stage 2 Experiment 1; `mode` is `multi_module` or `single_prompt`, `split` is `dev` or `eval` |
| `intent_sweep_<source>_<split>_<inputs>_*` | Stage 2 Experiment 2; `source` is `oracle`, `predicted` or `predicted_sp`, `inputs` is `base` (Parent–Reply only) or a combination such as `dialogue_act+sentiment` |

Anything on the Stage 2 `dev` split was only used while writing the prompts and is not reported.

## Old outputs

These are from earlier versions of the experiments. We kept them for reference, but none of the reported numbers come from them:

- `strategy_c/legacy/`: Strategy C before we fixed warmup and epochs, plus runs that fine-tuned only the top layers (`layers2`, `layers4`)
- `strategy_c/tfidf_weights/`: a TF-IDF baseline
- `strategy_c/agree_disagree_hard27_*.csv`: a blind re-labelling of 27 hard `agree`/`disagree` examples
- `stage2/intent_sweep_dev_*` and `stage2/intent_sweep_eval_*` (without `oracle` or `predicted` in the name): the first version of Experiment 2, replaced by the `intent_sweep_oracle_*` files. The numbers differ from the reported ones.
- `stage2/*smoke*` and `stage2/*.log`: test runs and logs
- `strategy_b_zeroshot/silver_<size>_seed<t>_*`: early Strategy B runs with an older naming scheme

## Notes

- Strategies B and C are reported as the mean and standard deviation over seeds 42, 123 and 2026. Strategy A uses greedy decoding, so it has a single run.
- Strategy A's four few-shot examples are gold pairs. That's why the Stage 1 prompt comparison uses 296 pairs and the predicted conditions in Experiment 2 use 260 of the 264 eval pairs.
- Model checkpoints are not included. `train_roberta.py strategy_b_heldout` recreates them in `strategy_b_heldout/models/`.
