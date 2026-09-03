# Stage 2 Progress Report — Intent/Sentiment/Emotion Classification

Stage 2 classifies each Reddit CMV `Parent`/`Reply` pair on three axes: **Sentiment** (3-class), **Emotion** (6-class multi-label), **Intent** (5-class, the primary target). Two experiments are running: **Experiment 1** compares two prompting architectures; **Experiment 2** (`intent_sweep`) tests whether auxiliary predicted labels help Intent classification.

This report has been updated across two sessions. Session 1 found and fixed two parsing bugs but never re-ran them against the LLM. Session 2 (this update) re-parsed the already-saved raw model outputs against the fixed code (no LLM calls needed), added two more parsing robustness fixes, manually verified the fixes on real data, and found a prompt-design confound in Experiment 1 that has been fixed in code but **not yet re-run against the LLM** — see the TODO list at the bottom for exactly what is current vs. stale.

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

### Results (macro-F1) — after Session 2's parsing fixes, re-parsed from already-saved raw model output (no new LLM calls)

| split | mode | Sentiment | Intent | Emotion |
|---|---|---:|---:|---:|
| eval (n=264) | multi_module | 0.572 (unchanged) | 0.387 (unchanged) | 0.393 (unchanged) |
| eval (n=264) | single_prompt | 0.532 → **0.584** | 0.360 → **0.364** | 0.357 → **0.383** |
| dev (n=36) | multi_module | 0.647 (unchanged) | 0.567 (unchanged) | 0.272 → **0.308**¹ |
| dev (n=36) | single_prompt | 0.609 → **0.521**² | 0.435 → **0.401**² | 0.370 → 0.371 |

¹ `multi_module`'s dev predictions file predated an earlier (already-merged) fix to `_emotion_answer_text` and had never been re-parsed since — this is stale-data cleanup unrelated to Session 2's own fixes; `eval` was already re-run after that fix, hence unchanged.
² `single_prompt` on **dev went down** after the fix. This is not a regression — it means some of the old dev score was an artifact of the overwrite bug (a correct first answer being silently replaced by a hallucinated second one) coincidentally landing on the right label. At n=36 a couple of flipped rows swing the score noticeably; treat dev numbers as illustrative only, per the existing "dev is never reported" policy.

`single_prompt`'s eval numbers move closer to `multi_module` across all three tasks once the parsing bugs are fixed, but a gap remains. **No significance test has been run on this gap.**

### Root cause (Session 1) and parsing fixes (Session 1 + 2)

Parsing-fallback counts on eval, by mode, **before any fix**:

| | sentiment_fallback | intent_fallback | emotion_fallback |
|---|---:|---:|---:|
| multi_module | 1/264 | 3/264 | 34/264 |
| single_prompt | 8/264 | 16/264 | 9/264 |

Three fixes were made to `parse_single_label`/`_split_labeled_lines` in [`stage2_pipeline.py`](../src/stage2_pipeline.py):

1. **First-occurrence-wins** (Session 1): the model sometimes answers correctly, then hallucinates a second fake `Parent:`/`Reply:` example with its own truncated answer lines. `_split_labeled_lines` used to keep the *last* occurrence of each key, letting the truncated hallucinated line silently overwrite the real answer. Now keeps the *first* occurrence.
2. **Single-match full-text recovery** (Session 2): when the model abandons the 3-line format entirely and writes free-text reasoning instead, the real answer is often still present as prose (e.g. "...The reply is information seeking"), just not on the first line. `parse_single_label` now falls back to scanning the *entire* output, but only accepts the recovery when **exactly one** candidate label is mentioned anywhere in it — if the model instead echoes back several/all candidate labels (ambiguous), it still falls back rather than guessing.
3. **Case/punctuation/separator normalization** (Session 2): matches now normalize both the label and the text (lowercase, collapse hyphens/underscores/whitespace/markdown punctuation to a single space) before comparing, so `"Counter-argue"` / `"counter argue"` / `"COUNTER_ARGUE"` / `"**Counter-Argue**"` all match. No live occurrence found in current saved outputs yet — this is defensive, verified to cause zero regressions on existing data.

Effect of fixes 1+2 on `single_prompt` eval intent fallback: **16/264 → 9/264**. Dev: 1/36 → 0/36. Sentiment fallback and `multi_module`'s numbers were unaffected — the cases there were confirmed (see below) to be genuinely ambiguous or absent, not recoverable.

**Manual verification (Session 2):** all 8 rows across dev+eval whose fallback flag flipped to a real answer because of fixes 1–2 were inspected by hand against their raw model output. All 8 extractions faithfully reflect what the model actually said (some still disagree with the gold label — that's the model being wrong, not a parsing error). No incorrect/coincidental extractions were found. Notably, eval row_id 235 is a clean validation of fix 2: the model wrote pure free-text reasoning ending in the literal sentence *"The reply is information seeking"*, with no `Intent:` field at all — previously discarded as an unrecoverable fallback, now correctly extracted.

`multi_module`'s Emotion fallback (34/264) was inspected the same way: nearly every fallback row's raw output contains 3–6 distinct emotion category words at once, confirming the suspected mechanism — **the model echoes the prompt's full category list back** instead of committing to one answer. Not yet fixed (see TODO).

### Prompt-fairness confound found and fixed (Session 2, code only — not yet re-run)

`build_single_prompt_baseline`'s prompt text was much sparser than the three per-task prompts it's being compared against: it never explained that `Reply` should be judged *in relation to* `Parent`, and included none of the per-label definitions that `build_sentiment_prompt`/`build_emotion_prompt`/`build_intent_prompt` give. This is a confound — the two conditions differed not only in "1 call vs. 3 calls" but also in how much guidance the model received, which plausibly explains some of `single_prompt`'s tendency to go off-format (e.g. eval row_id 22 and 235 both read like the model commenting on `Reply` in isolation rather than classifying it against `Parent`).

**Fix**: the three definition blocks were extracted into shared constants (`_SENTIMENT_DEFINITIONS`, `_EMOTION_DEFINITIONS`, `_INTENT_DEFINITIONS`) reused verbatim by both `build_*_prompt` and `build_single_prompt_baseline`, and the combined prompt's opening line now states the reply is judged "toward the parent comment it is responding to" (matching the sentiment/intent prompts' framing). This makes "1 call vs. 3 calls" the only remaining difference between the two conditions.

**This changes the prompt text sent to the model**, so every `single_prompt` number in this report (including the "after Session 2's parsing fixes" table above) is now stale relative to the current code and must be re-run against the LLM, not just re-parsed. See TODO #1.

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

**Session 2 update**: the same 3 parsing fixes were applied and the `predicted`-mode combinations were re-parsed from their saved raw output. Effect: only **1 row out of 264** changed (in `dialogue_act+emotion`, a fallback flipped to a genuine answer without changing the macro-F1). **The numbers above are unchanged and still current.**

**Known limitation — cannot be refreshed by re-parsing:** the `oracle`-mode run and the `predicted`-mode's `base` combination never had their raw model output saved (only the final label), because they predate the raw-output-saving convention used elsewhere. Any future fix to the parsing logic that would affect these cannot be validated without re-running the LLM. Going forward, always persist `*_raw` columns for every generation call, not just some.

### Interpretation

`dialogue_act` is the only auxiliary feature that reliably helps, and it holds up whether the feature is oracle or self-predicted. **Given the near-deterministic Dialogue_act↔Intent correlation found in the dataset section above** (Support≈agree 90.5%, Information seeking≈question 93.8%, Counter-argue≈disagree 90.9%), the more conservative reading is that this result is largely **annotation-scheme leakage** rather than evidence the model is doing richer contextual reasoning when given more information. This caveat should be stated explicitly whenever this result is cited.

---

## TODO / Next steps

1. **Re-run `single_prompt` on dev + eval against the actual LLM** — required now for two independent reasons: (a) the Session 1 `stop_strings=["\nParent:"]` generation-time fix has never been validated against real generation (only the parsing-side fixes have been validated, via re-parsing old raw output), and (b) the prompt-fairness fix (definitions + parent-relationship framing) changes the prompt text itself, so old raw output is no longer representative. Re-parsing cannot substitute for this — it requires actually calling the model. After re-running, recompute `evaluate_stage2.py` metrics and manually spot-check a sample the same way Session 2 did.
2. **Run a paired bootstrap significance test for `multi_module` vs. `single_prompt`** (reuse the bootstrap code already written for `intent_sweep`'s `*_vs_base_paired.csv`), on the post-re-run `single_prompt` predictions, to know whether the architecture gap is real or noise.
3. **Root-cause is now confirmed for `multi_module`'s Emotion fallback (34/264)**: the model echoes the full category list back instead of answering (see Session 2 manual inspection above). Still needs an actual fix — likely an explicit "Answer:" marker in the emotion prompt (mirroring what `_emotion_answer_text`'s parsing side already expects), or a `stop_strings` addition, analogous to what was done for `single_prompt`.
4. **`intent_sweep`'s `oracle` mode and `predicted`-mode `base` combination need a full re-run against the LLM** to benefit from any parsing fix, since their raw output was never saved. Make sure the re-run also starts saving `*_raw` columns for both, to avoid repeating this limitation.
5. **Methodological question to raise with advisor**: at n=300 (dev=36, eval=264), is a single fixed-seed hold-out split sufficient, or should eval be evaluated with k-fold cross-validation — particularly for rare classes (Information seeking n=16, Hostility n=26) where a single split's estimate may have high variance? (Session 2 saw a very visible illustration of this: `single_prompt` dev intent macro-F1 moved by 0.03 from a single row's fallback flag flipping — n=36 is noisy.)
6. **Caveat to raise with advisor**: the `intent_sweep` headline result ("dialogue_act significantly improves Intent classification") is confounded with the Dialogue_act↔Intent annotation correlation documented above. Worth discussing whether this should be reframed, or whether a follow-up test controlling for that correlation (e.g. evaluating only on rows where dialogue_act is *not* the majority-correlated one for that Intent class) is worth running.
7. **After #1 completes**: consider reporting both "old prompt" and "new prompt" `single_prompt` numbers side by side in the eventual writeup, to make explicit how much of the original architecture gap was actually a prompt-fairness confound rather than the 1-call-vs-3-calls difference itself.
