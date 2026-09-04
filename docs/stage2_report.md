# Stage 2 Progress Report — Intent/Sentiment/Emotion Classification

Stage 2 classifies each Reddit CMV `Parent`/`Reply` pair on three axes: **Sentiment** (3-class), **Emotion** (6-class multi-label), **Intent** (5-class, the primary target). Two experiments are running: **Experiment 1** compares two prompting architectures; **Experiment 2** (`intent_sweep`) tests whether auxiliary predicted labels help Intent classification.

This report has been updated across three sessions. Session 1 found and fixed two parsing bugs but never re-ran them against the LLM. Session 2 re-parsed the already-saved raw model outputs against the fixed code (no LLM calls needed), added two more parsing robustness fixes, manually verified the fixes on real data, and found a prompt-design confound in Experiment 1 that was fixed in code but not yet re-run. **Session 3 (this update) re-ran everything that required a real LLM call**: `single_prompt` (dev+eval) with the fixed prompt, and `intent_sweep`'s `oracle` mode (dev+eval, all 8 combinations) plus `predicted` mode's `base` combination. All numbers in this report are now current as of Session 3, except where a TODO item below says otherwise. A full Intent error analysis (confusion matrix, Dialogue-Act→Intent cascade, case studies) was also added.

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

### Results (macro-F1) — Session 3: real LLM re-run with the fixed prompt + fixed parsing

| split | mode | Sentiment | Intent | Emotion |
|---|---|---:|---:|---:|
| eval (n=264) | multi_module | 0.572 | 0.387 | 0.393 |
| eval (n=264) | single_prompt (original prompt, pre-fix) | 0.532 | 0.360 | 0.357 |
| eval (n=264) | single_prompt (original prompt, parsing fixed, re-parsed only) | 0.584 | 0.364 | 0.383 |
| eval (n=264) | **single_prompt (fair prompt, fresh LLM re-run)** | **0.625** | **0.409** | **0.408** |
| dev (n=36) | multi_module | 0.647 | 0.567 | 0.308 |
| dev (n=36) | single_prompt (fair prompt, fresh LLM re-run) | 0.548 | 0.428 | 0.307 |

**The result reverses.** Once the prompt-fairness confound and the three parsing bugs are both fixed, `single_prompt` **outperforms `multi_module` on all three eval tasks** — the opposite of the original headline finding ("multi_module leads on all 3 tasks"). Most of that original gap was the confound and the parsing bugs, not the 3-calls-vs-1-call architecture itself.

**Paired bootstrap significance test** (same 264 eval rows, `n_boot=2000`, `multi_module − single_prompt`; script output saved to `results/stage2/multi_module_vs_single_prompt_eval_paired.csv`):

| task | difference (multi_module − single_prompt) | 95% CI | excludes 0? |
|---|---:|---|:---:|
| Sentiment | −0.053 | [−0.108, +0.006] | No |
| Intent | −0.022 | [−0.083, +0.036] | No |

**Neither difference is statistically significant** — the CI for both tasks includes 0. So the honest conclusion is two-sided: (a) the point estimates now favor `single_prompt`, reversing the original direction, but (b) we cannot confidently claim `single_prompt` is actually better either — the data is also consistent with "no real architecture difference once both confounds are controlled for." (Emotion is multi-label, so this paired single-macro-F1 bootstrap doesn't directly apply; only a point-estimate comparison is reported for it.)

### Root cause and parsing fixes

Parsing-fallback counts on eval, by mode, **before any fix**:

| | sentiment_fallback | intent_fallback | emotion_fallback |
|---|---:|---:|---:|
| multi_module | 1/264 | 3/264 | 34/264 |
| single_prompt (original prompt) | 8/264 | 16/264 | 9/264 |

Three fixes were made to `parse_single_label`/`_split_labeled_lines` in [`stage2_pipeline.py`](../src/stage2_pipeline.py):

1. **First-occurrence-wins**: the model sometimes answers correctly, then hallucinates a second fake `Parent:`/`Reply:` example with its own truncated answer lines. `_split_labeled_lines` used to keep the *last* occurrence of each key, letting the truncated hallucinated line silently overwrite the real answer. Now keeps the *first* occurrence.
2. **Single-match full-text recovery**: when the model abandons the 3-line format entirely and writes free-text reasoning instead, the real answer is often still present as prose (e.g. "...The reply is information seeking"), just not on the first line. `parse_single_label` now falls back to scanning the *entire* output, but only accepts the recovery when **exactly one** candidate label is mentioned anywhere in it — if the model instead echoes back several/all candidate labels (ambiguous), it still falls back rather than guessing.
3. **Case/punctuation/separator normalization**: matches now normalize both the label and the text (lowercase, collapse hyphens/underscores/whitespace/markdown punctuation to a single space) before comparing, so `"Counter-argue"` / `"counter argue"` / `"COUNTER_ARGUE"` / `"**Counter-Argue**"` all match.

**Manual verification (on the original-prompt data):** all 8 rows across dev+eval whose fallback flag flipped to a real answer because of fixes 1–2 were inspected by hand against their raw model output. All 8 extractions faithfully reflect what the model actually said (some still disagree with the gold label — that's the model being wrong, not a parsing error). Eval row_id 235 is a clean validation of fix 2: the model wrote pure free-text reasoning ending in the literal sentence *"The reply is information seeking"*, with no `Intent:` field at all — previously discarded as an unrecoverable fallback, now correctly extracted.

`multi_module`'s Emotion fallback (34/264) was inspected the same way: nearly every fallback row's raw output contains 3–6 distinct emotion category words at once, confirming the suspected mechanism — **the model echoes the prompt's full category list back** instead of committing to one answer. Not yet fixed (see TODO).

### Prompt-fairness confound: found, fixed, and validated with a real re-run

`build_single_prompt_baseline`'s prompt text was much sparser than the three per-task prompts it's being compared against: it never explained that `Reply` should be judged *in relation to* `Parent`, and included none of the per-label definitions that `build_sentiment_prompt`/`build_emotion_prompt`/`build_intent_prompt` give. This is a confound — the two conditions differed not only in "1 call vs. 3 calls" but also in how much guidance the model received.

**Fix**: the three definition blocks were extracted into shared constants (`_SENTIMENT_DEFINITIONS`, `_EMOTION_DEFINITIONS`, `_INTENT_DEFINITIONS`) reused verbatim by both `build_*_prompt` and `build_single_prompt_baseline`, and the combined prompt's opening line now states the reply is judged "toward the parent comment it is responding to" (matching the sentiment/intent prompts' framing). This makes "1 call vs. 3 calls" the only remaining difference between the two conditions.

**Post-fix fallback counts (real re-run, eval n=264):** sentiment 8→**5**, intent 16→**5** — both improved substantially, consistent with the parsing fixes plus the clearer prompt. **Emotion fallback went the other way: 9→18.** Inspecting the new fallback rows found two failure modes not seen before, both apparently triggered by the more detailed prompt inviting a more "structured" answer style than the plain 3 lines asked for:

- **Python-code-style answers.** Several completions are a fenced code block defining a function that returns/prints the three labels as string literals (e.g. `` ```python\ndef analyze_reply(reply):\n    sentiment = "neutral"\n    emotion = "neutral"\n    intent = "others"\n... `` ) instead of plain `Key: value` lines. The real answer is present as a Python string assignment (`emotion = "neutral"`), but `_split_labeled_lines` only recognizes `key:` lines, not `key = value`, so it's never extracted.
- **Morphological mismatch ("sarcastic" vs. "sarcasm").** One free-text row states *"This reply is a sarcastic jab..."* — the adjective form — but `parse_emotion_labels` checks for the exact substring `"sarcasm"` (the label's own spelling), which `"sarcastic"` does not contain. A real, distinct answer is present and still gets discarded.

Neither is fixed yet — both are parsing-layer gaps analogous to the ones already fixed, just newly surfaced by the richer prompt's effect on generation style; see TODO.

---

## Experiment 2: `intent_sweep` — do auxiliary labels help Intent classification?

Tests whether feeding `dialogue_act`/`sentiment`/`emotion` as extra context into the Intent prompt improves Intent macro-F1 over a `base` prompt with none, using either **oracle** (gold) or **predicted** (model's own) values for those auxiliary labels.

### Results, eval, Δ macro-F1 vs. `base` with bootstrap 95% CI (n_boot=2000) — Session 3: real LLM re-run

| combination | oracle Δ (n=264) | oracle CI excludes 0? | predicted Δ (n=260) | predicted CI excludes 0? |
|---|---:|:---:|---:|:---:|
| **dialogue_act** | **+0.080** | **Yes** | **+0.055** | **Yes** |
| sentiment | +0.003 | No | −0.018 | No |
| emotion | +0.016 | No | −0.005 | No |
| dialogue_act+sentiment | +0.040 | No | +0.017 | No |
| dialogue_act+emotion | +0.018 | No | +0.015 | No |
| emotion+sentiment | +0.023 | No | −0.009 | No |
| dialogue_act+emotion+sentiment | +0.001 | No | +0.012 | No |

Dev shows the same ranking (dialogue_act 0.567 vs. base 0.380, n=36). **Oracle vs. predicted**: no combination shows a significant difference — using the model's own predicted auxiliary labels performs statistically the same as using gold labels (full pairwise comparison in `results/stage2/intent_sweep_oracle_vs_predicted_eval_paired.csv`).

**This table is now a fully real, fresh LLM re-run for both `oracle` (all 8 combinations) and `predicted` (7 non-base combinations, plus `base` copied from the fresh oracle run) — not a re-parse of old raw output.** The headline finding is unchanged and now more robustly established than before: `dialogue_act` is the only combination whose CI excludes 0, in both oracle and predicted mode, on a completely independent generation run from the one that originally produced this conclusion. The numbers moved only slightly from the earlier (partially re-parsed, partially stale) version of this table — e.g. oracle `dialogue_act+sentiment` moved from +0.024 to +0.040, still not significant — consistent with ordinary LLM/hardware-level nondeterminism between runs (also visible in `intent_base`: 3/260 rows flipped between the pre- and post-re-run predicted-mode files despite greedy decoding, likely floating-point differences between the original hardware and this session's GPU server).

**Known limitation, now resolved:** the `oracle`-mode run and the `predicted`-mode's `base` combination previously had no raw model output saved. The fresh re-run now saves `*_raw` for every combination in both modes (`results/stage2/intent_sweep_oracle_{dev,eval}_predictions.csv`), so any future parsing fix can be validated by re-parsing alone going forward.

### Interpretation

`dialogue_act` is the only auxiliary feature that reliably helps, and it holds up whether the feature is oracle or self-predicted. **Given the near-deterministic Dialogue_act↔Intent correlation found in the dataset section above** (Support≈agree 90.5%, Information seeking≈question 93.8%, Counter-argue≈disagree 90.9%), the more conservative reading is that this result is largely **annotation-scheme leakage** rather than evidence the model is doing richer contextual reasoning when given more information. This caveat should be stated explicitly whenever this result is cited.

---

## Error Analysis (Intent)

This section merges the Intent confusion breakdown, the Dialogue-Act→Intent cascade check, the specific Intent–Dialogue-Act combination checks, and five representative case studies into one narrative. All numbers here use the `predicted`-mode `intent_sweep` eval set (n=260 — the 260/264 rows with a Strategy A dialogue-act prediction available; see Dataset section) so that "predicted Dialogue Act" is a genuinely realistic (not gold) input throughout. `intent_base` (Parent+Reply only) is used as the no-auxiliary-input reference; it is confirmed near-identical to `multi_module`'s Intent predictions on the same rows (differs in only 2/260 cells), so conclusions here also describe Experiment 1's architecture.

### Confusion Overview

| gold \ pred | Information seeking | Challenge | Counter-argue | Support | Others |
|---|---:|---:|---:|---:|---:|
| Information seeking (n=14) | 7 | 1 | 2 | 0 | 4 |
| Challenge (n=77) | 13 | 12 | 34 | 3 | 15 |
| Counter-argue (n=85) | 3 | 1 | 64 | 3 | 14 |
| Support (n=55) | 10 | 0 | 17 | 21 | 7 |
| Others (n=29) | 3 | 3 | 11 | 2 | 10 |

Per-class accuracy and dominant confusion:

- **Challenge: 0.16 accuracy (12/77) — by far the weakest class.** 34 of its 65 errors (52%) are misclassified as **Counter-argue**. This is the single largest error type in the whole matrix.
- **Counter-argue: 0.75 accuracy (64/85) — the strongest class**, consistent with it also being the "catch-all" destination for Challenge's errors above: the model appears to default to Counter-argue whenever a reply pushes back at all, regardless of whether it actually presents new reasoning (the distinction the Challenge/Counter-argue definitions hinge on).
- **Support: 0.38 accuracy (21/55)**, most often confused with Counter-argue (17/34 of its errors) — the model reads reinforcing detail added to an agreement as a counter-argument.
- **Others: 0.34 accuracy (10/29)** and **Information seeking: 0.50 accuracy (7/14)**, both most often confused with Counter-argue or each other — smaller classes (n=29, n=14) where a handful of errors swing the rate a lot.

**Takeaway:** the dominant error mode is not confusion between semantically distant classes — it's the model collapsing the *"pushes back without reasoning" (Challenge)* vs. *"pushes back with reasoning" (Counter-argue)* distinction into one bucket (Counter-argue), which is the exact boundary the Intent scheme asks the model to draw most finely.

### Dialogue-Act → Intent Cascade

Strategy A's predicted Dialogue Act (from `results/strategy_a/fewshot_predictions.csv`) is only **55.4% accurate** (144/260) against gold `Dialogue_act` on this set:

| gold DA \ pred DA | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| agree (n=56) | 38 | 1 | 0 | 17 |
| disagree (n=126) | 5 | 52 | 15 | **54** |
| question (n=39) | 0 | 5 | 27 | 7 |
| statement (n=39) | 1 | 8 | 3 | 27 |

**`disagree` is more often mispredicted as `statement` (54/126) than correctly identified (52/126).** Since `disagree` is the Dialogue Act most strongly correlated with Intent (Counter-argue≈disagree 90.9%, from the dataset section), this specific weakness in the upstream Dialogue-Act model is the most consequential one for Experiment 2.

**Does a wrong predicted DA actually drag down the benefit of adding it as context?** Splitting rows by whether the predicted DA was correct:

| predicted DA | n | `intent_base` acc | `intent_dialogue_act` acc | Δ |
|---|---:|---:|---:|---:|
| correct | 144 | 0.465 | 0.569 | **+0.104** |
| wrong | 116 | 0.405 | 0.431 | +0.026 |

Yes — when the predicted DA is right, adding it as context roughly quadruples the gain (+0.104 vs. +0.026) compared to when it's wrong. Adding a wrong DA is not actively harmful *on average* (both deltas are positive), but it is far less useful, which is exactly what the leakage-based interpretation above predicts.

**Row-level flips** (base correct/wrong → +dialogue_act correct/wrong), n=260:

| | +DA wrong | +DA correct |
|---|---:|---:|
| **base wrong** | 112 | 34 |
| **base correct** | 16 | 98 |

Net effect is positive (34 rows flip wrong→correct vs. 16 correct→wrong), but the 16 correct→wrong flips are a real, non-trivial cost that the aggregate macro-F1 number hides — see case studies below for what these look like concretely.

**Per-Intent-class effect of adding Dialogue Act:**

| Intent | n | base acc | +DA acc | Δ |
|---|---:|---:|---:|---:|
| Information seeking | 14 | 0.500 | 0.714 | **+0.214** |
| Support | 55 | 0.382 | 0.582 | **+0.200** |
| Counter-argue | 85 | 0.753 | 0.847 | +0.094 |
| Challenge | 77 | 0.156 | 0.143 | **−0.013** |
| Others | 29 | 0.345 | 0.241 | **−0.103** |

Information seeking and Support improve the most — exactly the two classes with the strongest, cleanest Dialogue-Act correlation (question≈93.8%, agree≈90.5%). **Challenge gets essentially zero benefit (slightly negative)**: it shares its dominant Dialogue Act (`disagree`) with Counter-argue, so being told "this is a disagree-type reply" doesn't discriminate between them at all — consistent with Challenge's confusion above already being dominated by Counter-argue. **Others gets actively worse** (−0.103); see case study 3 below for why.

### Specific Intent–Dialogue-Act Combinations

| combination | n | base acc | +DA acc | DA prediction acc on these rows |
|---|---:|---:|---:|---:|
| Support + agree | 50 | 0.420 | 0.620 | 0.720 |
| Support + statement | 5 | 0.000 | 0.200 | 0.600 |
| Challenge + disagree | 47 | 0.170 | 0.234 | 0.426 |
| Counter-argue + disagree | 78 | 0.782 | 0.846 | 0.410 |

Support+agree (the majority, "expected" combination) benefits clearly. Support+statement is the minority combination (only 5 rows) and starts from 0% base accuracy — a tiny sample, but the direction (some improvement) is at least not reversed by adding DA. Challenge+disagree and Counter-argue+disagree share the same Dialogue Act value; both improve only modestly, and the DA model's own accuracy is worst exactly on these two overlapping cells (41–43%, vs. 55.4% overall) — the two Intent classes hardest to tell apart are also sitting on the Dialogue-Act value the upstream model is worst at recognizing.

### Case Studies

Five representative error patterns, one example each (row IDs refer to `intent_sweep_predicted_eval_predictions.csv`):

1. **Long / multi-topic text.** Row 22 — Parent is a long CMV post *followed by the standard CMV moderator footnote* ("please remember to read through our rules... downvotes don't change views... popular topics wiki..."), then the actual quoted claim. Reply: *"You wouldn't have used a phone to call them in 1984?"* Gold Intent: **Challenge**. `intent_base` predicts **Others** — plausibly because the Others definition's own examples ("mentioning mods, bots, deltas, or other posts") pattern-match the moderator boilerplate sitting in the Parent text, even though the actual Reply is a pointed rebuttal. Adding Dialogue Act doesn't fix this (predicts Information seeking instead, still wrong) — the failure is upstream of the auxiliary-label question, in the Parent text itself burying the signal in boilerplate.

2. **Parent context not used enough.** Row 61 — Parent: *"The medical nomenclature is not mentally retarded, it's intellectually disabled"*. Reply: *"Not in America, it isn't."* Read alone, the Reply is uninterpretable; it only means anything against the Parent's specific claim. Gold Intent: **Challenge** (pushback, no new reasoning). `intent_base` gets this right. But adding the (correctly predicted) Dialogue Act `disagree` **flips it to Counter-argue** — the model appears to use "disagree" as a proxy for "this reply must contain an argument," when here the Parent-dependence is exactly what makes it a bare, reasoning-free Challenge rather than a Counter-argue.

3. **Dialogue Act prediction error cascading into an Intent error.** Row 105 — Reply: *"Its not my problem... I don't suffer from that issue..."* Gold Intent: **Others** (dismissive, doesn't engage the argument), gold Dialogue Act: `statement`. `intent_base` correctly predicts Others. Strategy A mispredicts the Dialogue Act as `disagree` (wrong), and feeding that into the Intent prompt **flips the prediction to Counter-argue** — the disagree→Counter-argue correlation firing on a Dialogue-Act value that was itself wrong. This is one of the three rows behind the Others class's −0.103 net decline above (the other two, row 80 and row 181, also flip Others→something-else, though row 80's Dialogue Act was actually predicted *correctly* — the flip isn't purely a DA-error artifact, see next point).

4. **Ambiguous Intent label boundary (Challenge vs. Counter-argue).** Row 38 — Reply: *"Do terrible students opions [sic] not matter? And I remind you they don't choose to be there."* Gold Intent: **Challenge** — it's a rhetorical question plus a bare assertion, not new reasoning or evidence per the scheme's definition. `intent_base` predicts **Counter-argue**, because the reply is multi-sentence and assertive in tone, which is the same surface signal the correct Counter-argue rows share. This is the same Challenge/Counter-argue boundary responsible for the single largest confusion cell in the matrix above, not an isolated one-off.

5. **Surface form doesn't match the true Intent.** Row 78 — Parent is a long, multi-paragraph post; Reply: *"Writing a book, OP?"* Gold Intent: **Challenge** (a sarcastic jab at the Parent's length, not a genuine request for information). `intent_base` predicts **Information seeking**, misled by the surface question form. The predicted Dialogue Act is also `question` (matching the surface form, not the gold `disagree`), so adding it reinforces the wrong reading rather than correcting it — a case where the auxiliary signal and the surface-text signal agree with each other and are both wrong.

### Summary — where the model breaks down

The two most error-prone situations, in order of how much of the total error they account for: **(a) Challenge vs. Counter-argue** — a fine-grained "pushback with vs. without new reasoning" distinction that the model collapses toward Counter-argue by default, responsible for the single largest confusion cell and the reason Challenge gets zero benefit from Dialogue Act; and **(b) short, Parent-dependent replies** (case studies 2 and 3) — a few words that are only interpretable against the Parent, which both the base model and the Dialogue-Act auxiliary signal can push the wrong way once they've overfit to "disagree implies an argument was made." Both are represented in Appendix-worthy volume in the raw predictions CSV for follow-up reading; this section keeps one example each per the case-study list above.

---

## TODO / Next steps

**Done since the last update** (for the record — these were open items and are now resolved): `single_prompt` was re-run against the real LLM on both dev and eval with the fixed prompt (item 1 below, formerly open); the `multi_module` vs. `single_prompt` paired significance test was run (item 2, formerly open); `intent_sweep`'s `oracle` mode and `predicted`-mode `base` combination were re-run against the real LLM with `*_raw` now saved (item 4, formerly open). Results are folded into the sections above.

1. **Root-cause is confirmed but not yet fixed for two Emotion-parsing gaps**: (a) `multi_module`'s Emotion fallback (34/264 originally) — the model echoes the full category list back instead of answering; (b) `single_prompt`'s new Emotion fallback increase (9→18 after the prompt fix) — Python-code-style answers and morphological mismatches like "sarcastic" vs. "sarcasm" (see Prompt-fairness section above). Likely fixes: an explicit "Answer:" marker requirement, a `key = value` pattern added to `_split_labeled_lines`, and/or a small synonym/stem map for emotion words.
2. **Error Analysis section uses `intent_sweep`'s `predicted`-mode eval set (n=260)** — not yet repeated with `oracle`-mode's gold Dialogue Act (now available with the re-run) to check whether the same Challenge/Counter-argue collapse and Others-declines-with-DA pattern hold with gold DA instead of predicted DA — that would isolate whether the patterns found are about the Dialogue-Act *signal itself* or about Strategy A's *prediction errors* in it. Worth doing since the oracle data now exists.
3. **`intent_sweep`'s `predicted`-mode `dev` split was not re-run** (only `eval` was, plus `base`'s dependency on the fresh oracle numbers) — low priority since dev is never officially reported, but flagged for completeness.
4. **Methodological question to raise with advisor**: at n=300 (dev=36, eval=264), is a single fixed-seed hold-out split sufficient, or should eval be evaluated with k-fold cross-validation — particularly for rare classes (Information seeking n=16, Hostility n=26) where a single split's estimate may have high variance? A concrete illustration: re-running the exact same `intent_sweep` pipeline on the same GPU produced 3/260 different `intent_base` predictions purely from run-to-run nondeterminism (floating-point differences across hardware, despite greedy decoding) — a reminder that point estimates at this scale carry real, not just sampling, noise.
5. **Caveat to raise with advisor**: the `intent_sweep` headline result ("dialogue_act significantly improves Intent classification") is confounded with the Dialogue_act↔Intent annotation correlation documented above. Worth discussing whether this should be reframed, or whether a follow-up test controlling for that correlation (e.g. evaluating only on rows where dialogue_act is *not* the majority-correlated one for that Intent class) is worth running.
6. **Judgment call to raise with advisor**: the Experiment 1 prompt-fairness bug was found and fixed mid-project, reversing the headline result. This report states both the pre-fix and post-fix numbers explicitly rather than only reporting the final version — worth confirming this is the presentation the advisor wants for the eventual write-up.
