# Stage 2 — Intent Error Analysis

**Scope:** `intent_sweep`'s `predicted`-mode eval set (n=260 — the 260/264 eval rows with a Strategy A Dialogue Act prediction available). `intent_base` (Parent+Reply only, no auxiliary input) is used as the reference; it is confirmed near-identical to `multi_module`'s Intent predictions on the same rows (differs in only 2/260 cells), so the findings below also describe Experiment 1's architecture.

## 1. Intent Confusion Matrix

| gold \ pred | Info seeking | Challenge | Counter-argue | Support | Others |
|---|---:|---:|---:|---:|---:|
| Information seeking (n=14) | 7 | 1 | 2 | 0 | 4 |
| Challenge (n=77) | 13 | 12 | 34 | 3 | 15 |
| Counter-argue (n=85) | 3 | 1 | 64 | 3 | 14 |
| Support (n=55) | 10 | 0 | 17 | 21 | 7 |
| Others (n=29) | 3 | 3 | 11 | 2 | 10 |

**Per-class accuracy and dominant confusion:**

| Gold Intent | n | Accuracy | Most confused with |
|---|---:|---:|---|
| Challenge | 77 | **0.156** | Counter-argue (34/65 of its errors, 52%) |
| Counter-argue | 85 | 0.753 | Others (14/21, 67%) |
| Support | 55 | 0.382 | Counter-argue (17/34, 50%) |
| Others | 29 | 0.345 | Counter-argue (11/19, 58%) |
| Information seeking | 14 | 0.500 | Others (4/7, 57%) |

**Finding:** the single largest error type in the whole matrix is **Challenge → Counter-argue**. The model collapses the *"pushes back without new reasoning" (Challenge)* vs. *"pushes back with new reasoning" (Counter-argue)* distinction toward Counter-argue by default — this is the most error-prone boundary in the label scheme, not confusion between semantically distant classes.

## 2. Dialogue Act → Intent Cascade

Strategy A's predicted Dialogue Act is only **55.4% accurate** (144/260) against gold on this set. Its worst confusion: **`disagree` is more often mispredicted as `statement` (54/126) than correctly identified (52/126)**.

| gold DA \ pred DA | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| agree (56) | 38 | 1 | 0 | 17 |
| disagree (126) | 5 | 52 | 15 | **54** |
| question (39) | 0 | 5 | 27 | 7 |
| statement (39) | 1 | 8 | 3 | 27 |

**Does a wrong predicted DA drag down the benefit of adding it as context?**

| predicted DA | n | base Intent acc | +DA Intent acc | Δ |
|---|---:|---:|---:|---:|
| correct | 144 | 0.465 | 0.569 | **+0.104** |
| wrong | 116 | 0.405 | 0.431 | +0.026 |

Yes — when the predicted DA is right, adding it as context roughly quadruples the gain compared to when it's wrong. A wrong DA is not actively harmful *on average* (both deltas are positive), but far less useful.

**Row-level flips** (base correct/wrong → +DA correct/wrong), n=260:

| | +DA wrong | +DA correct |
|---|---:|---:|
| base wrong | 112 | 34 |
| base correct | 16 | 98 |

Net effect is positive (34 flip wrong→correct vs. 16 correct→wrong), but the 16 correct→wrong flips are a real cost the aggregate macro-F1 hides.

## 3. Specific Intent–Dialogue Act Combinations

| combination | n | base acc | +DA acc | DA prediction acc on these rows |
|---|---:|---:|---:|---:|
| Support + agree | 50 | 0.420 | 0.620 | 0.720 |
| Support + statement | 5 | 0.000 | 0.200 | 0.600 |
| Challenge + disagree | 47 | 0.170 | 0.234 | 0.426 |
| Counter-argue + disagree | 78 | 0.782 | 0.846 | 0.410 |

Challenge+disagree and Counter-argue+disagree share the same gold Dialogue Act value; both improve only modestly, and the DA model's own accuracy is worst exactly on these two overlapping cells (41–43%, vs. 55.4% overall) — the two Intent classes hardest to tell apart also sit on the Dialogue Act value the upstream model is worst at recognizing.

## 4. Per-Class Effect of Adding Dialogue Act

| Intent | n | base acc | +DA acc | Δ |
|---|---:|---:|---:|---:|
| Information seeking | 14 | 0.500 | 0.714 | **+0.214** |
| Support | 55 | 0.382 | 0.582 | **+0.200** |
| Counter-argue | 85 | 0.753 | 0.847 | +0.094 |
| Challenge | 77 | 0.156 | 0.143 | **−0.013** |
| Others | 29 | 0.345 | 0.241 | **−0.103** |

Information seeking and Support improve the most — exactly the two classes with the strongest, cleanest Dialogue Act correlation (question≈93.8% for Information seeking, agree≈90.5% for Support, in the full dataset). **Challenge gets essentially no benefit** because it shares its dominant Dialogue Act (`disagree`) with Counter-argue — telling the model "this is a disagree-type reply" doesn't discriminate between them. **Others gets actively worse** (see case study 3 below).

## 5. Case Studies

**(a) Long / multi-topic text.** Parent is a long CMV post *followed by the standard moderator footnote* ("please read our rules... downvotes don't change views..."); Reply: *"You wouldn't have used a phone to call them in 1984?"* Gold: **Challenge**. Predicted (base): **Others** — plausibly because the Others definition's own examples ("mentioning mods, bots, deltas") pattern-match the footnote sitting in the Parent, even though the Reply is a pointed rebuttal. Adding Dialogue Act doesn't fix this (predicts Information seeking instead) — the failure is upstream of the auxiliary-label question.

**(b) Parent context not used enough.** Parent: *"...it's intellectually disabled"* (discussing terminology). Reply: *"Not in America, it isn't."* Uninterpretable without the Parent. Gold: **Challenge** (pushback, no new reasoning). Base gets this right. Adding the *correctly predicted* Dialogue Act `disagree` **flips it to Counter-argue** — the model uses "disagree" as a proxy for "this reply must contain an argument," exactly where the Parent-dependence is what makes it reasoning-free.

**(c) Dialogue Act error cascading into an Intent error.** Reply: *"Its not my problem... I don't suffer from that issue..."* Gold: **Others**, gold DA: `statement`. Base correctly predicts Others. Strategy A mispredicts the DA as `disagree`, and feeding that in **flips the prediction to Counter-argue** — the disagree→Counter-argue correlation firing on a DA value that was itself wrong.

**(d) Ambiguous label boundary (Challenge vs. Counter-argue).** Reply: *"Do terrible students opinions not matter? And I remind you they don't choose to be there."* Gold: **Challenge** — a rhetorical question plus a bare assertion, not new reasoning per the scheme's definition. Predicted: **Counter-argue**, because the reply is multi-sentence and assertive in tone — the same surface signal genuine Counter-argue rows share. Same boundary responsible for the largest confusion cell above.

**(e) Surface form doesn't match the true Intent.** Parent is long and multi-paragraph; Reply: *"Writing a book, OP?"* Gold: **Challenge** (a sarcastic jab at the Parent's length, not a genuine question). Predicted: **Information seeking**, misled by the surface question form. Predicted DA is also `question` (matching the surface form, not gold `disagree`), so adding it reinforces the wrong reading instead of correcting it.

## 6. Summary — Where the Model Breaks Down

Two situations account for most of the error, in order of impact:

1. **Challenge vs. Counter-argue** — a fine-grained "pushback with vs. without new reasoning" distinction the model collapses toward Counter-argue by default. This is the single largest confusion cell and the reason Challenge gets zero benefit from Dialogue Act (it shares Dialogue Act's dominant value with the class it's most often confused with).
2. **Short, Parent-dependent replies** — a few words that are only interpretable against the Parent. Both the base model and the Dialogue-Act auxiliary signal can push these the wrong way once they over-rely on "disagree implies an argument was made."
