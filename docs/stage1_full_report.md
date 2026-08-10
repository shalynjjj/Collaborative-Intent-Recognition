# Stage 1 Results — Strategy A / B / C

Dialogue-act tagging on Reddit CMV replies (`agree` / `disagree` / `question` / `statement`). Server re-run complete; results below.

## Overview

| | A (few-shot LLM) | B (silver, 2500 few-shot) | C (gold, warmup20/epochs8) |
|---|---|---|---|
| Held-out macro-F1 | **0.6985** | **0.6175** | **0.5693** |
| Dev macro-F1 | 0.5916 | 0.6375 | 0.5309 |
| Direction (held-out − dev) | +0.107 | −0.020 | +0.038 |

Gold set: 300 rows. Held-out (pilot80): 80 rows, 20/class.

*Not a single ranking — three separate sub-claims (data efficiency A-vs-C, distillation payoff A-vs-B, silver-vs-gold B-vs-C), each scoped to Llama-3.1-8B-Instruct (A) + RoBERTa-base (B/C). See "Cross-cutting" and "Next Steps" below.*

---

## Dataset

**Gold** — 300 human-annotated rows. Used as the tuning/dev set for all three strategies.

**Test (held-out, "pilot80")** — 20 rows/tag, 80 total. Touched exactly once per strategy, after locking its config on gold.

**Why we're annotating a second test set:** the 80-row pilot is underpowered. Strategy A's held-out bootstrap 95% CI is **[0.587, 0.797]** — wide enough to already contain both B's (0.618) and C's (0.569) held-out point estimates, so we cannot yet say A actually beats B or C, only that it looks higher. Power analysis: A-vs-B and A-vs-C become resolvable (~80–90% power) once each class reaches ~60–70 confirmed rows; B-vs-C stays unresolvable even at 250 rows/class (~24% power) — a structural dead end, not a data gap. We're annotating a second batch (450 candidate rows, naturally covering all four classes) toward ~70 confirmed rows/class. **Status: 0/450 annotated so far.**

---

## Strategy A — LLM Annotation (Llama 3.1 8B)

Zero-shot vs. few-shot prompting on gold dev; few-shot locked and evaluated once on held-out. Few-shot examples: 4 fixed demonstrations (one per class), hand-picked from the gold set, excluded from every gold-touching evaluation to avoid leakage.

| Config | Gold (dev) macro-F1 | Kappa | Held-out macro-F1 | Held-out kappa |
|---|---|---|---|---|
| Zero-shot | 0.5148 | 0.298 | — | — |
| **Few-shot (locked)** | **0.5916** | **0.420** | **0.6985** | **0.60** |

**Held-out per-class F1:** agree 0.647 / disagree 0.722 / question 0.773 / statement 0.652 — no collapsed classes, the most balanced spread of the three strategies.

**Flag for discussion:** held-out (0.6985) is *higher* than dev (0.5916) — is the 80-row pilot just an easier sample? (Bootstrap 95% CI: [0.587, 0.797], n=2000.) See Cross-cutting — C shows the same direction of shift, B doesn't.

---

## Strategy B — RoBERTa on LLM-Silver Labels

Fine-tuned on Llama-labeled silver data. Because few-shot beat zero-shot in Strategy A, we switched to few-shot-labeled silver data for training — confirmed below as the right call at every silver size tested.

*[Insert screenshot of the learning-curve chart from the artifact here]*

### Zero-shot vs. few-shot silver — learning curve (gold-dev macro-F1, mean ± std)

| Silver size | Zero-shot silver | Few-shot silver | Δ (few − zero) |
|---|---|---|---|
| 500 | 0.2085 ± 0.0772 | 0.4155 ± 0.0865 | +0.2070 |
| 1000 | 0.3972 ± 0.0702 | 0.5814 ± 0.0189 | +0.1842 |
| 1500 | 0.4682 ± 0.1178 | 0.6143 ± 0.0275 | +0.1461 |
| 2000 | 0.4948 ± 0.0646 | 0.6184 ± 0.0187 | +0.1236 |
| **2500 (selected)** | 0.5623 ± 0.0420 | **0.6220 ± 0.0274** | +0.0597 |
| 5000 | 0.5636 ± 0.0463 | 0.6092 ± 0.0207 | +0.0456 |
| 8000 | 0.5676 ± 0.0125 | 0.5951 ± 0.0122 | +0.0275 |
| 10000 | 0.5132 ± 0.0421 | 0.5849 ± 0.0059 | +0.0717 |

Few-shot wins at every size — the gap is largest at 500 rows (+0.207, roughly double) and narrows past 5000. Curve is flat 3000–10000 for few-shot; 2500 is the selected point but not shown significantly better than neighboring 1500/2000/5000.

### Benchmark summary (locked config: size 2500, sample_seed 123, weights on)

| Benchmark | Macro-F1 | Kappa |
|---|---|---|
| Gold (dev), 2500 silver rows | 0.6375 | — |
| **Held-out (mean of 3 train seeds)** | **0.6175 ± 0.0085** | **0.511 ± 0.0096** |
| Common-296, weights on (vs. A few-shot) | 0.6179 ± 0.0282 | 0.468 ± 0.039 |
| Common-296, weights off | 0.6113 ± 0.0252 | 0.449 ± 0.037 |

**Held-out per-class F1** (mean of 3 seeds): agree 0.689 / disagree 0.449 / question 0.803 / statement 0.529

**Flag for discussion:** dev vs. held-out ranking flips — dev ranks B (0.638) above A (0.592); held-out ranks A (0.699) above B (0.618). The model selected on dev isn't the model that wins on the untouched set. `disagree` is B's weakest class in all 3 seeds, and noticeably worse than A's 0.722 on the same class.

**Class-weight ablation:** +0.0066 macro-F1 on common-296 (0.6179 vs 0.6113), weights-on better in 4/6 paired runs. Mainly from `disagree` (+0.042), at some cost to `statement` (−0.020). Modest, not statistically tested.

**Caveat — state this out loud:** a later audit found 262 of 276 recovered gold rows share a Reddit thread with the few-shot silver pool. Results are in-domain, not unseen-thread performance.

---

## Strategy C — RoBERTa Fine-Tuning on Gold-Standard Data (5-fold CV)

Direct fine-tuning on the 300 gold rows — no LLM involved. Stratified 5-fold CV × 3 train seeds (42/123/2026); OOF macro-F1 computed per seed on the concatenated 300-row set, then averaged. Answers: does 300 human-labeled rows let a small fine-tuned model match/beat direct LLM annotation (A)?

**The training-instability story:** initial full- and partial-fine-tuning runs collapsed classes (macro-F1 only 0.17–0.22). A diagnostic — having the model predict its own training rows — showed this was **training instability, not insufficient data**: models failed to even fit their own training folds. Increasing `warmup_steps` (0→20) and `epochs` (3→8) fixed it.

### Diagnostic: train-fit vs. eval, before vs. after the fix (aggregate over 15 fold/seed runs)

| | Train-fit macro-F1 | Eval macro-F1 | Gap | Corr(train-fit, eval) | Runs w/ train-fit < 0.35 |
|---|---|---|---|---|---|
| Before (default warmup/epochs) | 0.326 | 0.239 | 0.088 | r = 0.777 | 11/15 |
| **After (warmup=20, epochs=8)** | **0.902** | **0.535** | 0.367 | **r = 0.084** | **0/15** |

Before the fix, train-fit strongly predicted eval (r=0.777) — eval quality was tracking whether a run happened to converge, not real learning. After the fix, train-fit no longer predicts eval (r=0.084): every run reliably fits its training data, so remaining eval spread reflects genuine generalization difficulty.

### Warmup × epochs grid search (9 combinations + weights-off control), ranked by gold OOF macro-F1

| Rank | Warmup | Epochs | Weights | Macro-F1 (mean ± std) | Kappa (mean ± std) | Class collapse |
|---|---|---|---|---|---|---|
| 1 | 20 | 8 | On | **0.5309 ± 0.0038** | 0.348 ± 0.0081 | 0/15 |
| 2 | 10 | 10 | On | 0.5231 ± 0.0406 | 0.333 ± 0.0634 | 0/15 |
| 3 | 30 | 10 | On | 0.5189 ± 0.0494 | 0.336 ± 0.0710 | 0/15 |
| 4 | 20 | 10 | On | 0.5166 ± 0.0516 | 0.332 ± 0.0700 | 0/15 |
| 5 | 30 | 8 | On | 0.5072 ± 0.0314 | 0.318 ± 0.0414 | 0/15 |
| 6 | 10 | 8 | On | 0.4904 ± 0.0602 | 0.290 ± 0.0838 | 0/15 |
| 7 | 30 | 6 | On | 0.4371 ± 0.0292 | 0.246 ± 0.0400 | 0/15 |
| 8 | 20 | 6 | On | 0.4278 ± 0.0511 | 0.226 ± 0.0613 | 0/15 |
| 9 | 10 | 6 | On | 0.4216 ± 0.0587 | 0.221 ± 0.0553 | 2/15 |
| 10 | 20 | 8 | Off | 0.3499 ± 0.0229 | 0.163 ± 0.0301 | 8/15 |

6 epochs is undertrained at every warmup value. 10 epochs is competitive on the mean but 8–13x noisier across seeds than the locked config. Class weighting is essential — turning it off drops macro-F1 by ~0.18 and collapses classes in over half the runs.

### Locked config: dev vs. held-out

| | Gold (dev) | Held-out |
|---|---|---|
| Macro-F1 | 0.5309 ± 0.0038 | 0.5693 ± 0.0152 |
| Kappa | 0.348 ± 0.0081 | 0.489 ± 0.0192 |

**Held-out per-class F1** (mean of 3 seeds): agree 0.657 / disagree 0.551 / question 0.836 / statement 0.234

### Dev per-class F1 (5-fold OOF, mean of 3 seeds)

| Class | Seed 42 | Seed 123 | Seed 2026 | Mean | Avg. predicted count (/300) |
|---|---|---|---|---|---|
| agree | 0.400 | 0.456 | 0.456 | 0.4375 | 63.7 |
| disagree | 0.618 | 0.606 | 0.646 | 0.6234 | 140.3 |
| question | 0.722 | 0.753 | 0.716 | 0.7300 | 51.0 |
| statement | 0.367 | 0.321 | 0.309 | 0.3325 | 45.0 |

**Weakest class:** `statement` is clearly C's weakest class on both dev (0.33) and held-out (0.23, ranging 0.095–0.375 across seeds). `disagree` is over-predicted on dev (~140/300 predictions, nearly half), suggesting a bias toward defaulting to it on uncertain inputs.

---

## Cross-cutting

*[Insert screenshot of the per-class grouped bar chart from the artifact here]*

### Held-out per-class F1, all three strategies

| Class | A | B | C |
|---|---|---|---|
| agree | 0.647 | 0.689 | 0.657 |
| disagree | 0.722 | 0.449 | 0.551 |
| question | 0.773 | 0.803 | 0.836 |
| statement | 0.652 | 0.529 | 0.234 |

### Dev vs. held-out direction is not consistent across strategies

| Strategy | Dev macro-F1 | Held-out macro-F1 | Direction |
|---|---|---|---|
| A | 0.5916 | 0.6985 | +0.107 |
| B | 0.6375 | 0.6175 | −0.020 |
| C | 0.5309 | 0.5693 | +0.038 |

**Worth raising as one point, not three:** both A and C score *higher* on the 80-row held-out set than on gold dev; B is the opposite. If pilot80 were simply "easier" across the board, all three should shift the same direction — they don't. Possible explanation: B was the most heavily dev-tuned (silver size × sample_seed × train_seed × weights swept), so it may be mildly overfit to dev's specific 300 rows in a way A and C aren't. Speculative — exactly what the batch2 expansion should let us check.

### Why not a single ranking

1. **Power.** pilot80 is too small — A's bootstrap CI already spans B's and C's point estimates.
2. **Tuning-budget asymmetry.** B was swept most on dev; C's grid is now complete and confirms its config is optimal; A has essentially no tuning surface.
3. **Model-family confound.** A is a decoder-only LLM (Llama-3.1-8B-Instruct); B/C are an encoder (RoBERTa-base). Strategy and backbone vary together by design.

Instead: three sub-claims.
- **A vs C (data efficiency):** 300 gold rows are not enough for gold-only fine-tuning to match direct LLM annotation, even after fixing C's training instability.
- **A vs B (distillation payoff):** if A still matches/beats B after the held-out expansion, that's notable specifically because the tuning budget favors B.
- **B vs C (silver vs. gold source):** the observed gap (0.024, held-out) is too small to resolve at any feasible annotation scale — a genuine non-finding.

---

## Questions & Next Steps

**Open questions:**
- Is the annotation timeline for batch2 (currently 0/450) realistic before the next milestone?
- Is the dev/held-out direction split (A & C up, B down) worth a dedicated analysis now, or does it wait for the batch2 re-run?
- Time-permitting: worth swapping in `roberta-large` for B/C's locked configs, or a second LLM for A, to check backbone-sensitivity of the ranking?
- Does the "three sub-claims instead of one ranking" framing match what's expected for Results/Discussion?

**Next steps:**
- [x] Strategy C warmup/epochs grid — complete, locked config confirmed optimal
- [ ] Annotate batch2 (`cmv_test_candidates_batch2_annotate.csv`) to ~70 confirmed rows/class; keep pre-discussion annotator columns this time so Kappa is computable
- [ ] Re-run all three `*_heldout` evaluations on the expanded set
- [ ] Add bootstrap CI to B and C's held-out metrics (only A has one today)
- [ ] Re-run the A-vs-B / A-vs-C power calculation at the larger n
- [ ] Write the "Scope of Model Comparison" paragraph disclosing the model-family confound
- [ ] Write Results/Discussion/Conclusion using the three-sub-claim framing, not a single ranking table
