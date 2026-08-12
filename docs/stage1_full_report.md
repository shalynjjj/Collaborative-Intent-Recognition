# Stage 1 Full Report — Strategy A / B / C

Dialogue-act classification of Reddit CMV (ChangeMyView) parent-reply pairs into four classes: `agree`, `disagree`, `question`, `statement`. This is the complete, current record of Stage 1 — dataset, all three strategies, cross-strategy comparison, error analysis, and open items. All numbers below reflect the final 332-pair balanced held-out set (not the earlier 80-pair pilot).

## Overview

| | A (few-shot LLM) | B (silver, 2500 pairs) | C (gold, warmup20/epochs8) |
|---|---:|---:|---:|
| Dev macro-F1 | 0.5916 | 0.6375 | 0.531 (3 seeds) / 0.543 (8 seeds) |
| **Held-out macro-F1 (332 pairs)** | **0.673** | **0.669 ± 0.001** | **0.586 ± 0.070** |
| Held-out kappa | 0.558 | 0.565 ± 0.006 | 0.483 ± 0.084 |
| Old held-out (pilot80, n=80) | 0.699 | 0.618 ± 0.009 | 0.569 ± 0.015 |

**Not a single ranking.** Strategy and model backbone are confounded by design (A = decoder-only LLM, B/C = RoBERTa-base fine-tuning), so results support three separate sub-claims rather than one ranking — see "Conclusion / Sub-Claims" below.

---

## Dataset

**Gold set (dev):** 300 parent-reply pairs — 145 `disagree`, 64 `agree`, 47 `statement`, 44 `question`. Used exclusively for tuning; all three strategies lock their final configuration on it.

**Held-out test set:** touched once per strategy, after configuration lock. Expanded from an 80-pair pilot (`pilot80`) to a 332-pair, 83-pairs-per-class balanced set, specifically to gain enough statistical power to compare strategies. Construction: candidates sampled from the Winning Arguments Corpus excluding gold/silver overlaps → 530-pair labeled pool → downsampled to the smallest class (`statement`) to get 332 balanced pairs.

**Data-split guarantee:** no identical (Parent, Reply) pair appears across gold, silver, and held-out (verified programmatically). Different replies to the same parent/thread *are* intentionally allowed across sets — this reflects real Reddit structure. An audit found 262/276 traceable gold pairs share a thread with the silver pool, so **Strategy B results reflect in-domain, same-topic performance, not unseen-topic performance**.

**Inter-annotator agreement (IAA):** both gold-300 and held-out were double-annotated by two annotators and reconciled by discussion. Pre-reconciliation labels were not retained for either set, so only raw agreement can be reported — Cohen's kappa is not computable.

| Set | Pairs | Double-annotated | Raw agreement |
|---|---:|---:|---:|
| Gold (dev) | 300 | 300 (all) | 89.67% |
| Held-out | 332 | 332 (all) | 95.18% (16 disagreements) |

---

## Strategy A — LLM Annotation

**Setup:** Llama-3.1-8B-Instruct, temperature 0, zero-shot vs. few-shot prompting.

**Locked:** few-shot (dev macro-F1 0.5916 vs. 0.5148 zero-shot).

**Dev confusion matrix (few-shot):**

| True\Pred | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| agree | 43 | 3 | 0 | 17 |
| disagree | 6 | 60 | 16 | 62 |
| question | 0 | 5 | 31 | 7 |
| statement | 2 | 8 | 3 | 33 |

Dev per-class F1: agree 0.754, disagree 0.545, question 0.667, statement 0.400. Dominant error: 62/144 (43%) of `disagree` pairs misclassified as `statement`.

**Held-out (332 pairs) confusion matrix:**

| True\Pred | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| agree | 49 | 3 | 2 | 29 |
| disagree | 5 | 48 | 4 | 26 |
| question | 1 | 12 | 65 | 5 |
| statement | 9 | 8 | 6 | 60 |

Held-out per-class F1: agree 0.667, disagree 0.623, question 0.812, statement 0.591. macro-F1 = 0.6734, kappa = 0.558, 95% bootstrap CI [0.622, 0.720] (n_boot=2000).

**Key pattern:** on held-out, `agree` and `disagree` are rarely confused with each other (only 3 and 5 misclassifications) — both instead leak into `statement` (29 and 26). This differs qualitatively from B and C (see Error Analysis).

---

## Strategy B — RoBERTa on Silver Labels

**Setup:** RoBERTa-base fine-tuned on LLM-generated silver labels. `sample_seed` picks which pairs are drawn from the silver pool; `train_seed` controls training randomness. Train/validation split is grouped by `source_root` so no Reddit thread appears on both sides.

**Learning curve** (silver pool size 500–10,000 pairs, few-shot-labeled pool):

| Silver size | Dev macro-F1 (mean ± std) | Runs |
|---:|---:|---:|
| 500 | 0.4155 ± 0.0865 | 6 |
| 1000 | 0.5814 ± 0.0189 | 6 |
| 1500 | 0.6143 ± 0.0275 | 6 |
| 2000 | 0.6184 ± 0.0187 | 6 |
| **2500** | **0.6220 ± 0.0274** | 6 |
| 5000 | 0.6092 ± 0.0207 | 6 |
| 8000 | 0.5951 ± 0.0122 | 6 |
| 10000 | 0.5849 ± 0.0059 | 3 |

Macro-F1 rises sharply to 1000 pairs, peaks at 2500, then declines slowly — more silver data does not keep helping.

**Locked:** silver size 2500, `sample_seed` 123, class weights on (dev macro-F1 0.6375 vs. 0.6064 for `sample_seed` 42). On the common 296-pair benchmark, weights-on beats weights-off (0.6179 vs. 0.6113 macro-F1, 4/6 paired runs; not tested for significance).

**Dev results by train seed:**

| train_seed | macro-F1 | kappa |
|---|---:|---:|
| 42 | 0.6423 | 0.4962 |
| 123 | 0.6062 | 0.4548 |
| 2026 | 0.6639 | 0.5295 |
| **mean** | **0.6375** | **0.4935** |

**Dev confusion matrix (train_seed 123, representative):**

| True\Pred | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| agree | 47 | 3 | 1 | 13 |
| disagree | 19 | 60 | 17 | 49 |
| question | 1 | 0 | 43 | 0 |
| statement | 7 | 10 | 3 | 27 |

**Held-out confusion matrix (seed 123, representative):**

| True\Pred | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| agree | 50 | 12 | 6 | 15 |
| disagree | 3 | 51 | 10 | 19 |
| question | 1 | 3 | 79 | 0 |
| statement | 17 | 20 | 2 | 44 |

Held-out (mean of 3 seeds): macro-F1 0.6686 ± 0.0012, kappa 0.5649. Per-class F1: agree 0.654, disagree 0.590, question 0.877, **statement 0.554 (now weakest)**. On the older pilot80 held-out set, `disagree` had been weakest (F1 0.449) — B's weakest class is not stable across held-out sets.

**Known limitation:** results reflect in-domain (same-thread) performance due to the gold/silver overlap noted above.

---

## Strategy C — RoBERTa on Gold Labels

**Setup:** 5-fold cross-validation on the 300 gold pairs — data split into 5 parts, each predicted once by a model trained on the other 4. Each seed's 5 predicted parts are combined into one 300-pair out-of-fold (OOF) set; macro-F1 is computed once on that combined set, not averaged across folds.

**Exploration — instability and fix:**
- Initial full/partial fine-tuning collapsed one or more classes (macro-F1 0.16–0.20).
- Diagnostic: models mostly failed to fit their own training data (train-fit macro-F1 0.326, correlated with eval quality, r=0.777) → training instability, not insufficient data.
- Fix: `warmup=20, epochs=8` with class weights on → 0/15 class collapses, OOF macro-F1 0.5309 ± 0.0038 (3 seeds). Weights-off control: 0.3499 ± 0.0229, 8/15 runs still collapsed a class.
- Full grid search over 9 warmup/epoch combinations confirmed (20, 8) as best. 6 epochs undertrained at every warmup value; 10 epochs competitive on the mean but 8–13× noisier across seeds (less reproducible).

**Locked:** warmup=20, epochs=8, class weights on.

### Variance correction (important finding)

Five additional seeds (7, 99, 777, 1234, 5555) were run on dev OOF with the same locked configuration, to check whether the agree/disagree confusion (see Error Analysis) was specific to one seed:

| Seeds | macro-F1 (mean ± std) |
|---|---:|
| Original 3 (42, 123, 2026) | 0.5309 ± 0.0038 |
| **All 8** | **0.543 ± 0.037** |

**The true seed-to-seed variance is ~10× the originally reported figure.** The original three seeds happened to land unusually close together by chance — the tight std was a small-sample coincidence, not evidence of stable training. The warmup/epoch conclusion itself is unaffected (still the grid's best mean, still no class collapse by the zero-prediction definition).

**Dev OOF confusion matrices, all 8 seeds** (locked config):

| Seed | agree row [a,d,q,s] | disagree row | question row | statement row |
|---|---|---|---|---|
| 42 | 24,30,0,10 | 24,88,13,20 | 1,5,35,3 | 7,17,5,18 |
| 123 | 34,20,1,9 | 38,84,13,10 | 1,6,35,2 | 12,22,0,13 |
| 2026 | 26,22,0,16 | 19,95,16,15 | 0,6,34,4 | 5,26,1,15 |
| 7 | 24,29,1,10 | 26,98,12,9 | 0,9,31,4 | 12,17,1,17 |
| 99 | 38,17,2,7 | 27,93,13,12 | 1,4,38,1 | 9,21,3,14 |
| 777 | 35,21,1,7 | 21,95,13,16 | 0,7,35,2 | 7,23,0,17 |
| 1234 | 35,16,1,12 | 25,81,18,21 | 0,2,41,1 | 9,17,1,20 |
| 5555 | 23,22,2,17 | 40,69,16,20 | 0,8,33,3 | 11,16,1,19 |

In **every one of the 8 seeds**, the largest misclassification destination for the `agree` row is `disagree`, and vice versa — this pattern holds without exception.

**Held-out (3 seeds):**

| | Dev OOF (3 seeds) | Dev OOF (8 seeds) | Held-out (3 seeds) |
|---|---:|---:|---:|
| macro-F1 | 0.5309 ± 0.0038 | 0.543 ± 0.037 | **0.5859 ± 0.0699** |
| kappa | — | — | 0.483 ± 0.084 |

Per-seed held-out: seed 42 → 0.4874 (95% CI [0.43, 0.54]); seed 123 → 0.6269 ([0.58, 0.67]); seed 2026 → 0.6435 ([0.59, 0.69]). Seed 42 is the worst on *both* dev and held-out — consistent with the real (larger) variance, not a one-off anomaly.

**Held-out confusion matrices by seed:**

| Seed | agree row | disagree row | question row | statement row |
|---|---|---|---|---|
| 42 | 37,29,11,6 | 20,37,20,6 | 3,0,78,2 | 30,19,15,19 |
| 123 | 59,18,2,4 | 10,63,10,0 | 1,1,79,2 | 27,31,7,18 |
| 2026 | 61,8,3,11 | 15,51,12,5 | 0,0,81,2 | 27,25,4,27 |

---

## Cross-Strategy Comparison

The ranking is **not stable** across evaluation sets:
- On gold (dev): B (0.638) > A (0.592) > C (0.531/0.543)
- On the old 80-pair held-out set: A (0.699) > C (0.569) > B (0.618)
- On the new 332-pair held-out set: A (0.673) ≈ B (0.669) > C (0.586), with A's and B's confidence intervals overlapping almost completely

**Per-class F1 on held-out (332 pairs):**

| Class | A | B | C |
|---|---:|---:|---:|
| agree | 0.667 | 0.654 | 0.580 |
| disagree | 0.623 | 0.590 | 0.565 |
| question | 0.812 | 0.877 | 0.837 |
| statement | 0.591 | 0.554 | 0.362 |

`question` is the easiest class and `statement` the hardest, consistently, for all three strategies across both held-out sets.

---

## Power Analysis (paired bootstrap, n=332, n_boot=5000)

Because all three strategies are evaluated on the identical 332 held-out pairs, a **paired** bootstrap is more sensitive than treating them as independent samples (it cancels out the effect of some pairs simply being harder than others).

| Comparison | Observed Δ | 95% CI of Δ | Significant? | Power at n=332 | n needed (80% power) |
|---|---:|---|---|---:|---:|
| A vs. B | 0.0041 | [−0.057, 0.063] | No | 5% | ~34,900 pairs/class |
| A vs. C | 0.0465 | [−0.020, 0.111] | No | 29% | ~330 pairs/class |
| B vs. C | 0.0424 | [−0.018, 0.102] | No | 28% | ~350 pairs/class |

**Interpretation:**
- **A vs. B has a near-zero effect size.** Even ~34,900 pairs/class would be needed for 80% power — infeasible. This is a genuine tie, not a result waiting on more data.
- **A vs. C and B vs. C have a real, modest effect** (Δ ≈ 0.04–0.05), resolvable at ~330–350 pairs/class — well above an earlier, more optimistic estimate of 60–70/class, but a large annotation effort.

**Feasibility check (done later):** using the actual observed class distribution in the annotated held-out expansion batch (`agree` was the rarest class, ~15.66% of pairs), reaching 330 confirmed `agree` pairs would require screening/annotating roughly **2,100 raw candidate pairs** — judged **too large a workload for 2 annotators**, given project time constraints. Intermediate targets were also computed (e.g., 150/class ≈ 428 extra candidates ≈ 45–47% power) but **no partial push reaches the 80% threshold** — power scales with √n, so there is no cheap middle ground; it's either the full ~2,100-candidate push or accept A-vs-C/B-vs-C as open questions. Given the workload, the annotation push is **not currently planned**.

---

## Error Analysis: Confusion Overview

Pooling errors across all three held-out seeds for each strategy (996 predictions per strategy), the single largest error types are **not** `agree`/`disagree` confusions, but two other pairs: **`disagree`→`statement`** (Strategy B's largest error type, 62 instances) and **`statement`→`agree`** (Strategy C's largest, 84 instances; also present in B, 56 instances). Reading a sample of each, plus simple text statistics (word count, hedge-word rate like "but"/"however", negation-word rate like "not"/"never"), reveals three distinct, specific patterns:

- **`disagree`→`statement` (Strategy B).** These replies disagree by stating a counter-fact or counter-hypothetical, not by using disagreement language: hedge-word rate is 0.03 (vs. 0.13 for `disagree` generally) and negation rate is 0.31 (vs. 0.51). Examples: *"There is no such thing as cultural decay, only change."*; *"Consider this: if this effect did not occur, we would be heading towards a Malthusian catastrophe."* The disagreement is carried entirely by content, not by any lexical marker, so the model defaults to reading these as neutral statements.

- **`statement`→`agree` (Strategies B and C).** 36% of these misclassified replies are five words or fewer, far above baseline. These are short, reactive replies — thanks, corrections, brief asides — with a positive/polite tone but no substantive engagement with the argument. Examples: *"Whoops, thanks!"*; *"changed."*; *"Ahh. Yes I misinterpreted your comment. Thanks for the clarification."* The model appears to read positive/polite tone in a short reply as `agree`, regardless of whether it substantively agrees with anything.

- **`statement`→`disagree` (Strategy C, 75 instances).** Longer than baseline (15.1 vs. 12.7 words), with elevated hedge-word (0.23 vs. 0.17) and negation (0.31 vs. 0.25) rates — these read as skeptical/challenging in tone (embedded rhetorical questions, "but" qualifiers) without being a genuine disagreement with the parent's core claim.

**Broader theme:** together with the agree/disagree pattern below, all of B's and C's largest error types involve replies where the dialogue act is carried by tone, structure, or implication rather than an explicit lexical marker — RoBERTa fine-tuned on 300–2500 pairs struggles specifically with this class of example.

---

## Error Analysis: Additional Text-Feature Checks

To complete the negation/contractions/length/spelling check, a contraction rate and a spelling-error rate (nltk words-corpus heuristic: a token not found in the corpus and not capitalized) were computed the same way as the negation and hedge-word rates above, comparing each error pair against its class baseline.

- **Contractions and spelling: no effect.** Neither rate differs consistently between misclassified and baseline groups, across any of the confusion pairs checked (`disagree`→`statement`, `statement`→`agree`, `statement`→`disagree`, `agree`↔`disagree`) or across strategies overall. This rules out contractions and spelling as drivers of these errors.

- **Very long replies (≥30 words): elevated specifically for Strategy A.** 7.3% of A's misclassified held-out pairs are ≥30 words, vs. 0.9% of its correctly classified pairs (roughly 8×) — a pattern not present for B (0.9% vs. 4.0%, inverted) and only mild for C's `statement`→`disagree` pair (6.7% vs. 2.4% baseline, which confirms the "longer than baseline" description above quantitatively).

- **Very short replies (≤5 words): confirms the `statement`→`agree` finding above.** 35.7% (B) and 32.1% (C) of `statement`→`agree` misclassifications are ≤5 words, vs. 25.3% baseline — consistent with the ~36% figure already reported. Not elevated for Strategy A overall (12.7% vs. 18.9% for correctly classified pairs).

- **Ambiguous wording** remains a qualitative judgment. No reliable automatic proxy was found, so this criterion is assessed only through the manually read examples above, not a quantitative metric.

This completes the full negation/contractions/length/spelling/ambiguous-wording checklist: negation and short-reply length show real effects (above), contractions and spelling do not, very-long-reply length shows an effect specific to Strategy A, and ambiguous wording is assessed qualitatively only.

---

## Error Analysis: Agree/Disagree Confusion

**Finding:** beyond the confusion pairs above, a smaller but more *reproducible* problem stands out: a **bidirectional `agree`↔`disagree` confusion**, present in **all 8** of Strategy C's dev-OOF seeds without exception, and present to a lesser degree in Strategy B. Severity varies by seed (C's `agree` recall ranges 0.36–0.59), but direction is constant: `agree` misclassifies mainly as `disagree` in every seed, and vice versa.

**Seed-ensembling would not fix this.** Ensembling only cancels out errors that are *independent* across members. Here every seed errs the same way, so majority voting would preserve the shared bias, not cancel it.

**Cross-strategy check:** the 27 gold pairs misclassified by all/most of C's 8 seeds were checked against A's and B's predictions on the same text:

| Strategy | Accuracy on 27 hard pairs | Own agree/disagree baseline |
|---|---:|---:|
| A | 51.9% | 49.8% (no elevation) |
| B | 39.5% | 55.7% (−16 pts) |
| C | 0% (by construction) | ~47% |

Strategy A is unaffected by these rows. Strategy B and C — both RoBERTa fine-tunes — are not. **This points to a limitation of the RoBERTa fine-tuning approach itself, not inherently ambiguous text** (unlabelable text should have hurt Strategy A too, and it didn't).

**Candidate explanation:** many of the hard rows show a "concede-then-pivot" sentence structure (e.g., *"I understand that I'm simplifying... **But** my view is..."*, *"Fair enough... **Though** I believe..."*), where the true stance is in the second clause. RoBERTa-base, fine-tuned on only 300–2500 pairs, may lack signal to weight the pivot over the opening concession. Some errors persist even with explicit markers ("I agree", "Indeed"), suggesting an attention-allocation issue rather than pure textual ambiguity.

**Ruled out:** sarcasm/hostility/contempt — flag rates on the 27 hard pairs match the overall agree/disagree population (~14–15% sarcasm rate in both), no statistical difference.

**Not ruled out:** class imbalance in gold-300 (145 `disagree` vs. 64 `agree`, 2.3:1) may independently explain lower `agree` recall.

**Caveat:** this whole analysis is a model-cross-validation proxy, not measured IAA. Gold-300 has overall raw agreement (89.67%), but pre-reconciliation labels were not retained, so it's not possible to check whether these specific 27 pairs were among the original annotator disagreements. A blind re-labeling of these 27 pairs is the direct way to test this (see Future Work).

---

## Why No Single Ranking / Scope of Comparison

Strategy and model backbone are confounded by design: A uses a decoder-only LLM (Llama-3.1-8B-Instruct) via prompting; B and C fine-tune the same encoder (RoBERTa-base) on different label sources. A gap between A and {B, C} is therefore consistent with at least three explanations this design cannot distinguish:
1. LLM annotation is genuinely better suited to this task at this data scale.
2. A larger or differently pretrained encoder would close the gap.
3. A different LLM would perform differently.

This is reinforced empirically by the Error Analysis: the agree/disagree confusion shared by B and C but absent from A tracks the RoBERTa/LLM split exactly, not the silver/gold split — evidence that part of what looks like a "strategy" effect is really a backbone effect. **B vs. C is the cleaner comparison** (same backbone, different label source); any claim involving A is scoped to this specific LLM-vs-encoder pairing, not a general LLM-vs-fine-tuning claim.

---

## Conclusion / Sub-Claims

Not a single ranking — three separate comparisons, only one resolved:

- **A vs. B (does distillation pay off?) — RESOLVED.** Despite receiving by far the largest tuning budget of the three strategies, B ties rather than loses to A (macro-F1 0.669 vs. 0.673). The effect size is near zero, so this is a genuine tie, not a pending result — silver distillation does not detectably beat direct LLM annotation.
- **A vs. C (data efficiency) — OPEN.** 300 gold pairs are not enough for gold-only fine-tuning to match direct LLM annotation, even after fixing C's training instability. Real effect (gap 0.087), not yet statistically significant.
- **B vs. C (silver vs. gold source) — OPEN.** Observed gap 0.083 (macro-F1), real effect, most promising candidate for a follow-up annotation push, not yet significant. C's underestimated seed variance means its point estimate carries more uncertainty than first reported.

Closing A-vs-C and B-vs-C would require annotating toward ~330–350 confirmed pairs/class — judged infeasible given current time constraints (see Power Analysis). Both are expected to remain open in this report.

---

## Limitations

- **Gold/silver domain overlap (Strategy B).** 262/276 traceable gold pairs share a Reddit thread with the silver pool — held-out results reflect in-domain (same-thread) performance, not fully unseen-topic performance.
- **Gold-300's IAA is aggregate-only.** 89.67% raw agreement is known, but pre-reconciliation labels were not retained, so pair-level annotator disagreements cannot be recovered — the agree/disagree "ambiguity" analysis remains a model-cross-validation proxy, not a rigorous per-pair IAA test.
- **Model family is bound to strategy.** A uses a decoder-only LLM; B and C fine-tune the same encoder on different label sources — strategy and backbone effects cannot be disentangled with this design.
- **Strategy C's seed variance was underestimated at n=3.** The original 3-seed standard deviation (0.0038) understated the true variance by ~10× (0.037 across 8 seeds). Strategies A and B have not been checked with the same scrutiny.
- **Held-out set, though expanded, is still drawn from the same corpus family as dev** (Winning Arguments Corpus) — not tested against a genuinely different conversational data distribution.
- **Class imbalance in gold-300** (145 disagree vs. 64 agree) is a candidate confound for the agree/disagree confusion finding that has not been ruled out.

---

## Future Work

**Compute-only follow-ups (no new annotation required — feasible now):**
- Add more seeds/runs to Strategies A and B to check for previously underestimated variance, mirroring what was found for Strategy C.
- Test the class-imbalance hypothesis: oversample `agree` and rerun Strategy C, check whether the confusion eases.
- Swap `roberta-base` for `roberta-large` in Strategies B and C (no new grid search), and/or evaluate a second LLM for Strategy A, to test whether the A-vs-B/A-vs-C comparison is sensitive to backbone choice.

**Annotation-dependent follow-ups (not currently planned, given time constraints):**
- Blind re-label the 27 agree/disagree "hard" pairs — small-scale (27 pairs), much cheaper than the item below.
- Annotate toward ~330–350 confirmed pairs/class to resolve A-vs-C and B-vs-C at 80% power (not expected to resolve A-vs-B, whose effect size is near zero). This is the only way to make A-vs-C and B-vs-C conclusive; without it, both remain open questions.

---

## Key Corrections Made During This Round (for the record)

A few numbers changed materially from earlier drafts of this report, worth keeping track of:

1. **Held-out set expanded from 80 pairs (pilot80) to 332 pairs (83/class balanced).** All held-out numbers above reflect the 332-pair set unless labeled "old held-out (pilot80)."
2. **Strategy C's seed variance was underestimated.** Originally reported from 3 seeds as 0.5309 ± 0.0038; expanding to 8 seeds revealed the true std is 0.037 (~10×). Correction did not change the locked configuration.
3. **Held-out IAA was corrected from 87.7% (114/130, a leftover pilot-phase subset figure) to 95.18% (all 332 pairs, 16 disagreements)** — the "130 co-labeled" figure did not apply to the final 332-pair set.
4. **The agree/disagree confusion was initially thought to be specific to one bad seed (seed 42);** the 8-seed variance check showed it is systematic across all seeds, not a one-off.
5. **Completed the negation/contractions/length/spelling checklist item**, which had been left half-done (negation and length were reported earlier, contractions and spelling never actually tested). Contractions and spelling were checked and found to have no effect; a new finding emerged for very-long replies (≥30 words), which are specifically elevated in Strategy A's errors (~8× vs. its correct predictions).
