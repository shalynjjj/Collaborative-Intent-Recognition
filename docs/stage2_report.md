# Stage 2 Progress Report — Intent/Sentiment/Emotion Classification

Stage 2 classifies each Reddit CMV `Parent`/`Reply` pair on three axes: **Sentiment** (3-class), **Emotion** (6-class multi-label), **Intent** (5-class, the primary target). Two experiments are running: **Experiment 1** compares two prompting architectures; **Experiment 2** (`intent_sweep`) tests whether auxiliary predicted labels help Intent classification. This report is the current, honest state of both — including two implementation bugs found and fixed this session that are not yet reflected in the result files below.

## Dataset & split

Same 300-row gold set used in Stage 1 ([`cmv_300_gold_final.csv`](../data/cmv_300_gold_final.csv)), with 7 additional annotation columns: `Sentiment`, `Sarcasm`/`Hostility`/`Contempt`/`Neutral`/`Curiosity`/`Appreciation` (booleans, the Emotion multi-label set), `Intent`.

Split into **dev (36 rows, 12%)** — prompt iteration only, never reported — and **eval (264 rows, 88%)** — touched once per locked prompt config, via `sklearn.train_test_split(stratify=df["Intent"], random_state=42)`. No `Parent` string is shared between dev and eval (verified — no leakage).

**Label distribution** (counts):

| Sentiment | full (300) | dev (36) | eval (264) |
|---|---:|---:|---:|
| Neutral | 124 | 14 | 110 |
| Negative | 110 | 15 | 95 |
| Positive | 66 | 7 | 59 |

| Emotion (multi-label) | full (300) | dev (36) | eval (264) |
|---|---:|---:|---:|
| Neutral | 130 | 18 | 112 |
| Appreciation | 58 | 5 | 53 |
| Contempt | 51 | 4 | 47 |
| Sarcasm | 48 | 5 | 43 |
| Curiosity | 34 | 3 | 31 |
| Hostility | 26 | 5 | 21 |

| Intent | full (300) | dev (36) | eval (264) |
|---|---:|---:|---:|
| Counter-argue | 99 | 12 | 87 |
| Challenge | 89 | 11 | 78 |
| Support | 63 | 7 | 56 |
| Others | 33 | 4 | 29 |
| Information seeking | 16 | 2 | 14 |

Sentiment is fairly balanced; Intent and especially Emotion are not. Because only Intent was stratified, dev's Sentiment/Emotion proportions drift somewhat from eval's (e.g. Hostility is 13.9% of dev vs. 8.0% of eval) — expected sampling noise at n=36, not a bug.

**Notable structural finding — `Dialogue_act` (Stage 1's label) is near-deterministic for several Intent classes:**

| Intent \ Dialogue_act | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| Support | 57 (90.5%) | 0 | 0 | 6 |
| Information seeking | 0 | 0 | 15 (93.8%) | 1 |
| Counter-argue | 3 | 90 (90.9%) | 0 | 6 |
| Challenge | 1 | 53 | 28 | 7 |
| Others | 3 | 2 | 1 | 27 (81.8%) |

This is annotation-scheme correlation, not something a model learned. It matters directly for interpreting Experiment 2 below.

---

## Experiment 1: `multi_module` vs. `single_prompt`

Same inputs (`Parent` + `Reply`, no auxiliary labels) for both modes ([`stage2_pipeline.py`](../src/stage2_pipeline.py)):

- **`multi_module`**: 3 independent LLM calls, one prompt per task.
- **`single_prompt`**: 1 LLM call, one prompt asking for all 3 answers in a fixed 3-line format, parsed after the fact.

### Results (macro-F1)

| split | mode | Sentiment | Intent | Emotion |
|---|---|---:|---:|---:|
| eval (n=264) | multi_module | **0.572** | **0.387** | **0.393** |
| eval (n=264) | single_prompt | 0.532 | 0.360 | 0.357 |
| dev (n=36) | multi_module | 0.647 | 0.567 | 0.272 |
| dev (n=36) | single_prompt | 0.609 | 0.435 | 0.370 |

`multi_module` leads on all 3 tasks on eval (+0.03–0.04). **No significance test has been run on this gap yet** — it is a point estimate only.

### Root cause found (this session)

Parsing-fallback counts on eval, by mode:

| | sentiment_fallback | intent_fallback | emotion_fallback |
|---|---:|---:|---:|
| multi_module | 1/264 | 3/264 | 34/264 |
| single_prompt | 8/264 | 16/264 | 9/264 |

Inspected `single_prompt`'s fallback rows' raw model output directly and found **two distinct, unrelated failure modes** — not the token-budget-squeeze hypothesis originally suspected (fallback-row output length is *not* shorter than non-fallback rows, ruling that out):

1. **Intent/Sentiment fallback — answer overwritten, not lost.** The model answers correctly on the first 3 lines, then keeps generating and hallucinates a new fake `Parent:`/`Reply:` example, re-answering it with a second, truncated `Intent:` line. `_split_labeled_lines` stored the *last* occurrence of each key, so the empty truncated line silently overwrote the correct first answer. Confirmed on multiple rows (e.g. eval row_id 12, 13, 22).
2. **Sentiment fallback — format abandoned entirely.** In other rows the model ignores the 3-line format from the start and writes free-text analysis/reasoning instead (e.g. row_id 56, 59, 64). Unrelated to bug 1 — this is an instruction-following failure, not a parsing bug.

### Fixes applied this session (not yet re-run against the LLM)

- `_split_labeled_lines` now keeps the **first** occurrence of each key ([stage2_pipeline.py](../src/stage2_pipeline.py)), fixing failure mode 1.
- `make_transformers_generator` gained an optional `stop_strings` param ([llm_annotate.py](../src/llm_annotate.py)); `run_stage2` now passes `stop_strings=["\nParent:"]` only for `single_prompt` mode, so generation stops before the hallucinated continuation starts, instead of relying on parsing to recover from it. Other callers (Strategy A, `multi_module`, `intent_sweep`) are unaffected (default `None`).
- 2 new regression tests reproducing the exact row_id 12 pattern, added to [tests/test_stage2.py](../tests/test_stage2.py). Full suite: 15/15 passing.
- **Not yet done: re-running `single_prompt` on dev/eval with the fix.** All numbers in the table above are from the pre-fix predictions and are expected to improve, at least on Sentiment/Intent. Failure mode 2 (format abandonment) is untouched by these fixes and will still cause some fallbacks.
- `multi_module`'s Emotion fallback (34/264, the single worst number in the table) has a different, not-yet-root-caused mechanism (suspected: the model echoes the prompt's category list back per the docstring note in `_emotion_answer_text`) — out of scope for this session's fix.

---

## Experiment 2: `intent_sweep` — do auxiliary labels help Intent classification?

Tests whether feeding `dialogue_act`/`sentiment`/`emotion` as extra context into the Intent prompt improves Intent macro-F1 over a `base` prompt with none, using either **oracle** (gold) or **predicted** (model's own) values for those auxiliary labels.

### Results, eval (n=264), Δ macro-F1 vs. `base` with bootstrap 95% CI (n_boot=2000)

| combination | oracle Δ | oracle CI excludes 0? | predicted Δ | predicted CI excludes 0? |
|---|---:|:---:|---:|:---:|
| **dialogue_act** | **+0.078** | **Yes** | **+0.055** | **Yes** |
| sentiment | +0.004 | No | −0.020 | No |
| emotion | +0.016 | No | −0.007 | No |
| dialogue_act+sentiment | +0.024 | No | +0.018 | No |
| dialogue_act+emotion | +0.018 | No | +0.012 | No |
| emotion+sentiment | +0.021 | No | −0.010 | No |
| dialogue_act+emotion+sentiment | +0.002 | No | +0.013 | No |

Dev shows the same ranking (dialogue_act 0.567 vs. base 0.391). **Oracle vs. predicted**: no combination shows a significant difference — using the model's own predicted auxiliary labels performs statistically the same as using gold labels.

### Interpretation

`dialogue_act` is the only auxiliary feature that reliably helps, and it holds up whether the feature is oracle or self-predicted. **Given the near-deterministic Dialogue_act↔Intent correlation found in the dataset section above** (Support≈agree 90.5%, Information seeking≈question 93.8%, Counter-argue≈disagree 90.9%), the more conservative reading is that this result is largely **annotation-scheme leakage** rather than evidence the model is doing richer contextual reasoning when given more information. This caveat should be stated explicitly whenever this result is cited.

---

## TODO / Next steps

1. **Re-run `single_prompt` on dev + eval with the two bug fixes**, recompute `evaluate_stage2.py` metrics, and compare fallback counts and macro-F1 before/after. This is the most immediate item — the Experiment 1 numbers above are stale relative to the code.
2. **Run a paired bootstrap significance test for `multi_module` vs. `single_prompt`** (reuse the bootstrap code already written for `intent_sweep`'s `*_vs_base_paired.csv`), on the *post-fix* `single_prompt` predictions, to know whether the architecture gap is real or noise.
3. **Root-cause `multi_module`'s Emotion fallback (34/264)** — inspect raw outputs the same way as was done for `single_prompt`, confirm/refute the "echoes category list" hypothesis, and fix at the prompt or stopping-criteria level rather than only in parsing.
4. **Address `single_prompt` failure mode 2 (format abandonment)** — this needs a prompt-level fix (e.g. a one-shot format example, or an explicit "Answer:" marker as already used for the emotion parser), not a parsing or stop-string fix. Separate piece of work from items 1–3.
5. **Methodological question to raise with advisor**: at n=300 (dev=36, eval=264), is a single fixed-seed hold-out split sufficient, or should eval be evaluated with k-fold cross-validation — particularly for rare classes (Information seeking n=16, Hostility n=26) where a single split's estimate may have high variance?
6. **Caveat to raise with advisor**: the `intent_sweep` headline result ("dialogue_act significantly improves Intent classification") is confounded with the Dialogue_act↔Intent annotation correlation documented above. Worth discussing whether this should be reframed, or whether a follow-up test controlling for that correlation (e.g. evaluating only on rows where dialogue_act is *not* the majority-correlated one for that Intent class) is worth running.
