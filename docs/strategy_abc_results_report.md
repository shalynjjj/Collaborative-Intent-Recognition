# Strategy A/B/C Results Report

**Task:** Classify Reddit CMV parent-reply pairs into `agree`, `disagree`,
`question`, `statement`.

**Strategies:** A = direct LLM annotation (Llama-3.1-8B-Instruct). B = RoBERTa-base
fine-tuned on LLM-generated silver labels. C = RoBERTa-base fine-tuned on 300 gold
(human-labeled) rows.

All three are tuned on the same 300-row gold set (`dev`); each locked config is
evaluated once, afterward, on an independent held-out set.

**Note on comparability:** A, B, and C use different underlying models (decoder-only
LLM vs. fine-tuned encoder). Their scores are reported side by side but are **not**
evidence that one strategy is generally "better" — see §9.

---

## 1. Dataset and Setup

**Gold set (dev):** `data/cmv_300_gold_final.csv`, 300 rows, single-annotated.
Class counts: disagree 145, agree 64, statement 47, question 44. No second
annotation pass exists for this set — any claim about label ambiguity on gold-300
(§8) is a model-based proxy, not measured IAA.

**Held-out set:** independent of dev; touched once per strategy, after that
strategy's config is locked. Built from the Winning Arguments Corpus, excluding
all gold/silver pairs. Expanded from the original **pilot80** (80 rows, 20/class)
to a **332-row, 83/class balanced set** (`cmv_test_candidates_heldout_balanced.csv`).
Unless stated otherwise, "held-out" below means the 332-row set.

**Data-split guarantee:** no identical parent-reply pair appears in both
train/silver and test data. Different replies to the same parent post/topic *are*
allowed to appear on both sides — this is intentional, reflecting realistic Reddit
threads where many replies share a parent. An audit found 262/276 traceable gold
rows share a Reddit thread with the few-shot silver pool; results should be read
as in-domain (same-topic), not unseen-topic, performance.

**IAA (held-out only):** double-annotated (2 annotators), reconciled by
discussion. Raw pre-discussion agreement: 87.7% (114/130), 16 disagreements.
Kappa not computable (pre-reconciliation labels not retained).

---

## 2. Strategy A: LLM Annotation

**Setup:** Llama-3.1-8B-Instruct, temperature 0, zero-shot vs. few-shot.

**Locked:** few-shot (dev macro-F1 0.5916 vs. zero-shot 0.5148).

| | Dev (gold-300) | Held-out (332) |
|---|---:|---:|
| macro-F1 | 0.5916 | **0.6734** |
| kappa | 0.420 | 0.558 |
| 95% bootstrap CI | — | [0.622, 0.720] |

Dev confusion matrix:

| True\Pred | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| agree | 43 | 3 | 0 | 17 |
| disagree | 6 | 60 | 16 | 62 |
| question | 0 | 5 | 31 | 7 |
| statement | 2 | 8 | 3 | 33 |

Dev per-class F1: agree 0.754, disagree 0.545, question 0.667, statement 0.400.
62/144 (43%) of `disagree` rows are misclassified as `statement` — the single
largest error on dev.

Held-out confusion matrix:

| True\Pred | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| agree | 49 | 3 | 2 | 29 |
| disagree | 5 | 48 | 4 | 26 |
| question | 1 | 12 | 65 | 5 |
| statement | 9 | 8 | 6 | 60 |

Held-out per-class F1: agree 0.667, disagree 0.623, question 0.812, statement
0.591. Main error: both `agree` and `disagree` leak into `statement` — not into
each other (only 3 and 5 misclassifications between the two). This differs from
B/C (§8).

---

## 3. Strategy B: RoBERTa on Silver Labels

**Setup:** RoBERTa-base on LLM-labeled silver data. `sample_seed` picks the
silver rows; `train_seed` controls training randomness. Group-aware split by
`source_root`.

**Exploration:** learning curve over silver size 500–10000 (flat above 3000,
best point at 2500); size×sample_seed×train_seed grid; class-weight ablation.

**Learning curve (few-shot-labeled silver pool, matches the locked config's data
source; mean ± std across sample_seed × train_seed runs):**

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

Macro-F1 rises sharply from 500→1000 (+0.166), keeps climbing to a peak at
**2500**, then gently declines through 10000 rather than staying flat — more
silver data does not keep helping past this point. Std is largest at the small
sizes (500: ±0.087) and shrinks as size grows, except a mild bump back up at
1500–2500 (±0.02–0.03) before settling low again at 10000 (±0.006, only 3 runs
since sample_seed 42/123 converge to the same full pool at that size). Full
curve and data: `results/strategy_b_fewshot/learning_curve_fewshot.{csv,png}`.

**Locked:** size 2500, sample_seed 123, class weights on (dev macro-F1 0.6375 vs.
0.6064 for sample_seed 42). Weights-on beats weights-off on the common 296-row
benchmark (0.6179 vs. 0.6113, 4/6 paired runs), no significance test run.

| | Dev (3 train seeds, mean) | Held-out (3 seeds, mean) |
|---|---:|---:|
| macro-F1 | 0.6375 | **0.6686 ± 0.0012** |
| kappa | 0.4935 | 0.5649 |

Held-out confusion matrix (seed 123, representative):

| True\Pred | agree | disagree | question | statement |
|---|---:|---:|---:|---:|
| agree | 50 | 12 | 6 | 15 |
| disagree | 3 | 51 | 10 | 19 |
| question | 1 | 3 | 79 | 0 |
| statement | 17 | 20 | 2 | 44 |

Per-class F1 (mean of 3 seeds): agree 0.654, disagree 0.590, question 0.877,
**statement 0.554**. On this larger set, `statement` — not `disagree` — is B's
weakest class; on the old pilot80 set `disagree` was weakest (0.449). This
ranking is not stable across held-out sets.

**Known limitation:** 262/276 gold rows share a Reddit thread with the silver
pool (in-domain, not unseen-topic evaluation).

---

## 4. Strategy C: RoBERTa on Gold Labels

**Setup:** 5-fold stratified CV on the 300 gold rows, outer fold seed 42.
Reported per-seed score = macro-F1 on the 300-row OOF concatenation (not 15
independent fold scores).

**Exploration:**
- Initial full/partial fine-tuning collapsed classes (macro-F1 0.16–0.20).
  Train-fit diagnostic showed most runs failed to fit their own training data
  (train-fit F1 0.326, r=0.777 with eval) → **training instability**, not a
  data-volume problem.
- Fix: `warmup=20, epochs=8` → 0/15 class collapse, OOF macro-F1
  0.5309 ± 0.0038 (3 seeds). Weights-off control: 0.3499 ± 0.0229, 8/15 collapse.
- Full 9-combination warmup×epochs grid confirmed (20, 8, weights-on) as best;
  epochs=6 undertrained everywhere, epochs=10 competitive but 8–13x noisier.

**Locked:** warmup=20, epochs=8, weights on.

**Variance correction:** 5 additional seeds (7, 99, 777, 1234, 5555) run on dev
OOF with the same locked config:

| seeds | macro-F1 (mean ± std) |
|---|---:|
| original 3 (42/123/2026) | 0.5309 ± 0.0038 |
| all 8 | **0.543 ± 0.037** |

The true seed variance is ~10x the originally reported figure. The
warmup/epochs conclusion is unaffected (still the grid's best mean, still no
class collapse by the zero-prediction definition) — but "0.5309 ± 0.0038"
should not be read as evidence of high training stability.

| | Dev OOF (3 seeds) | Dev OOF (8 seeds) | Held-out (3 seeds) |
|---|---:|---:|---:|
| macro-F1 | 0.5309 ± 0.0038 | 0.543 ± 0.037 | **0.5859 ± 0.0699** |
| kappa | — | — | 0.483 ± 0.084 |

Held-out per-seed: 42 → 0.4874 [CI 0.434, 0.538]; 123 → 0.6269 [0.575, 0.672];
2026 → 0.6435 [0.591, 0.690]. Seed 42 is the worst of three, consistent with the
8-seed dev variance — not a one-off anomaly (see §7).

Held-out confusion matrices:

| seed | agree row | disagree row | question row | statement row |
|---|---|---|---|---|
| 42 | 37,29,11,6 | 20,37,20,6 | 3,0,78,2 | 30,19,15,19 |
| 123 | 59,18,2,4 | 10,63,10,0 | 1,1,79,2 | 27,31,7,18 |
| 2026 | 61,8,3,11 | 15,51,12,5 | 0,0,81,2 | 27,25,4,27 |

---

## 5. Cross-Strategy Comparison

| | A (few-shot) | B (silver 2500) | C (gold) |
|---|---:|---:|---:|
| Dev macro-F1 | 0.5916 | 0.6375 | 0.531 (3 seeds) / 0.543 (8 seeds) |
| Held-out macro-F1 (332) | **0.673** | **0.669** | **0.586** |
| Held-out CI | [0.622, 0.720] | ≈[0.615, 0.719] | wide, seed-dependent |
| Old held-out (pilot80, n=80) | 0.699 | 0.618 | 0.569 |

Ranking flips twice: dev → B > A > C; pilot80 → A > C > B; new 332-row set →
**A ≈ B > C**, with A/B CIs overlapping almost completely. Per your item 3: A's
different model family means this is not read as "A beats C" — see §9.

**Per-class F1:**

| Class | A | B | C |
|---|---:|---:|---:|
| agree | 0.667 | 0.654 | 0.580 |
| disagree | 0.623 | 0.590 | 0.565 |
| question | 0.812 | 0.877 | 0.837 |
| statement | 0.591 | 0.554 | 0.362 |

`question` easiest, `statement` hardest — consistent across all three strategies
and both held-out sets.

---

## 6. Error Analysis: Agree/Disagree Confusion

**Not a "disagree is weak" problem — it's a bidirectional agree↔disagree
confusion**, present in **all 8** of C's dev-OOF seeds without exception, and
elevated (though less severe) in B. Severity varies by seed (agree recall
0.36–0.59 in C) but direction is constant: `agree` misclassifies mainly as
`disagree` and vice versa, in every seed.

**Seed-ensembling would not fix this.** Ensembling averages out errors that are
independent across seeds. Here every seed errs in the same direction — voting
preserves a shared bias instead of canceling it.

**B and C share this problem; A does not.** Checked accuracy on the 27 gold rows
that all/most of C's 8 seeds misclassify:

| Strategy | Accuracy on these 27 rows | Own agree/disagree baseline |
|---|---:|---:|
| A | 51.9% | 49.8% (no elevation) |
| B | 39.5% | 55.7% (−16 pts) |
| C | 0% (by construction) | ~47% |

A is unaffected; B and C — both RoBERTa fine-tunes — are not. This points to a
limitation of the **RoBERTa fine-tuning approach**, not inherently ambiguous
text (if the text itself were unlabelable, A should struggle too).

**Candidate explanation:** concede-then-pivot sentences (*"I understand X...
But Y"*, *"Fair enough... Though I believe..."*) where the true stance is in
the second clause. RoBERTa-base, fine-tuned on 300–2500 rows, may lack signal
to weight the pivot over the opening concession; some errors persist even with
explicit markers ("I agree", "Indeed"), suggesting an attention issue rather
than pure ambiguity.

**Ruled out:** sarcasm/hostility/contempt — flag rates on the 27 hard rows
match the overall agree/disagree population (~14–15% sarcasm both), no
statistical difference.

**Not ruled out:** class imbalance (disagree 145 vs. agree 64, 2.3:1) may
independently explain lower agree recall, regardless of sentence structure.

**Caveat:** this is a model-cross-validation proxy, not measured IAA (§1). A
blind re-labeling of these 27 rows is the direct way to test true annotation
disagreement (§8, open item).

---

## 7. Why No Single Ranking

### 7.1 Power analysis (paired bootstrap, n=332, n_boot=5000)

All three strategies are evaluated on the identical 332 held-out rows, so a
**paired** bootstrap is used: resample row indices with replacement, recompute
each strategy's macro-F1 on the same resampled rows, and look at the
distribution of the pairwise difference. This is more powerful than treating
the three strategies as independent samples, since it removes row-to-row
difficulty as a source of noise.

| Comparison | Observed Δ (macro-F1) | 95% CI of Δ | Significant? | Power at n=332 | n needed for 80% power |
|---|---:|---|---|---:|---:|
| A vs. B | 0.0041 | [−0.057, 0.063] | No | 5% | ~34,900 rows/class |
| A vs. C | 0.0465 | [−0.020, 0.111] | No | 29% | ~330 rows/class |
| B vs. C | 0.0424 | [−0.018, 0.102] | No | 28% | ~350 rows/class |

("n needed" extrapolates the bootstrap standard error as ∝ 1/√n from its
value at n=332, holding the observed effect size Δ fixed — a standard
analytical approximation, not a new empirical run.)

None of the three pairwise comparisons is significant at the current sample
size. But the two cases are qualitatively different:

- **A vs. B has a near-zero effect size (Δ=0.004).** This is not "not enough
  data yet" — even ~34,900 confirmed rows/class would be needed for 80% power,
  which is infeasible for this project. The honest reading is that **A and B
  are practically tied**, not that the comparison is merely underpowered.
- **A vs. C and B vs. C have a real, if modest, effect (Δ≈0.04–0.05)** and
  would become resolvable at roughly **330–350 confirmed rows/class** —
  substantially more than the ~60–70/class estimated in an earlier draft of
  this analysis (which used a different, and apparently optimistic, effect-size
  assumption), but within reach of a further annotation push.

### 7.2 Scope of model comparison

Reported macro-F1 numbers for A, B, and C should not be read as a ranking of
"which strategy is best," for one reason that no amount of additional
annotation can fix: **strategy and model backbone are confounded by design.**
Strategy A uses a decoder-only LLM (Llama-3.1-8B-Instruct) via zero-/few-shot
prompting; Strategies B and C both fine-tune the same encoder backbone
(RoBERTa-base) on different label sources (silver vs. gold). Any gap between A
and {B, C} is therefore consistent with at least three different explanations
that this experiment design cannot separate: (1) LLM annotation is genuinely
better suited to this task than encoder fine-tuning at this data scale, (2) a
larger or differently pretrained encoder would close the gap, or (3) a
different LLM would perform differently. This is reinforced empirically by
§6: the agree/disagree confusion shared by B and C but absent in A tracks the
RoBERTa/LLM split exactly, not the silver/gold split — evidence that at least
part of what looks like a "strategy" effect is really a backbone effect. B vs.
C, by contrast, is a cleaner comparison (same backbone, different label
source) and is the pair this report leans on most for strategy-level claims;
any statement involving A is scoped to this specific LLM-vs-encoder pairing,
not a general claim about LLMs vs. fine-tuning.

### 7.3 Three sub-claims, not a ranking

- **A vs. C (data efficiency):** 300 gold rows are not enough for gold-only
  fine-tuning to match direct LLM annotation, even after fixing C's training
  instability. Not yet significant (§7.1), but with a real effect size and a
  feasible path to resolution.
- **A vs. B (does silver distillation pay off?):** despite B's much larger
  tuning budget, A ties B rather than losing to it — and given the near-zero
  effect size and infeasible power requirement (§7.1), this should be read as
  a genuine tie, not a pending result. Silver distillation did not produce a
  detectable accuracy gain over direct LLM annotation.
- **B vs. C (silver vs. gold source):** the observed gap (~0.04–0.08
  depending on held-out set) has a real effect size and unequal-power caveats
  aside, is the strategy pair most worth a follow-up annotation push per
  §7.1 — but is not yet significant, and C's seed variance (§4) means the
  point estimate itself carries more uncertainty than initially reported.

---

## 8. Limitations

- Gold/silver share Reddit threads (B) — in-domain, not unseen-topic, results.
- No second annotation pass on gold-300 — §6's ambiguity claim is a proxy, not
  measured IAA.
- Model family is bound to strategy — see §7.2 for the full scope statement.
- C's seed variance was underestimated at n=3; A/B have not been checked with
  the same scrutiny.
- Held-out set, though expanded, still comes from the same corpus family as
  dev — not tested against a different distribution.

---

## 9. Open Items

- [ ] Blind re-label the 27 agree/disagree "hard" rows to test real annotation
      disagreement vs. model limitation.
- [ ] Tag misclassified agree/disagree examples (B, C) for negation,
      contractions, reply length, spelling errors, ambiguous wording.
- [ ] Test class-imbalance contribution: oversample `agree`, rerun C, check if
      confusion eases.
- [ ] Add more seeds/runs to A and B to check for underestimated variance.
- [ ] `roberta-large` swap for B/C, and/or a second LLM for A (no new grid
      search), to test backbone sensitivity of the A-vs-B/A-vs-C comparison.
