# Strategy A/B/C Results Report

**Task:** Dialogue-act classification of Reddit CMV reply pairs into four classes
— `agree`, `disagree`, `question`, `statement`.

**Three strategies compared:**
- **Strategy A** — Direct LLM annotation (Llama-3.1-8B-Instruct, zero-/few-shot prompting).
- **Strategy B** — RoBERTa-base fine-tuned on LLM-generated *silver* labels.
- **Strategy C** — RoBERTa-base fine-tuned directly on 300 *gold* (human-labeled) rows.

All three strategies were tuned on the same 300-row gold set (`dev`), then each
locked configuration was evaluated once on an independent 80-row held-out test
set (`pilot80`, 20 rows/class) to avoid dev-set overfitting in the comparison.

---

## 1. Locked configurations

| Strategy | Final configuration | Selected by |
|---|---|---|
| A | Few-shot prompting, temperature 0 | Gold macro-F1: fewshot 0.5916 vs zeroshot 0.5148 |
| B | Silver size 2500, sample_seed 123, class weights on | Gold macro-F1 0.6375 (sample_seed 123) vs 0.6064 (sample_seed 42); size 2500 was the best point on the learning curve (0.6220 avg across both sample seeds) |
| C | RoBERTa, warmup=20, epochs=8, class weights on | Gold 5-fold OOF macro-F1 0.5309 ± 0.0038, zero class collapse across 15 fold/seed runs; confirmed best after a full 9-combination warmup×epochs grid (see §4) |

## 2. Dev vs. held-out results

| | Strategy A (fewshot) | Strategy B (silver 2500) | Strategy C (gold, warmup20/epochs8) |
|---|---|---|---|
| Gold (dev) macro-F1 | 0.5916 | 0.6375 | 0.5309 ± 0.0038 |
| **Held-out macro-F1** | **0.6985** | **0.6175 ± 0.0085** | **0.5693 ± 0.0152** |
| Held-out Cohen's kappa | 0.600 | 0.511 ± 0.0096 | 0.489 ± 0.0192 |
| Held-out 95% bootstrap CI (macro-F1) | [0.587, 0.797] (n_boot=2000) | not yet computed | not yet computed |

Note the dev→held-out ranking flip: on dev, B > A > C; on the 80-row held-out
set, A > C > B. With only 20 rows/class, the held-out set is underpowered —
A's own bootstrap CI already spans both B's and C's held-out point estimates,
so this ranking is not yet statistically resolvable (see §5).

### Per-class held-out F1

| Class | A (fewshot) | B (silver 2500, mean of 3 seeds) | C (gold, mean of 3 seeds) |
|---|---|---|---|
| agree | 0.647 | 0.689 | 0.657 |
| disagree | 0.722 | 0.449 | 0.551 |
| question | 0.773 | 0.803 | 0.836 |
| statement | 0.652 | 0.529 | 0.234 |

`disagree` is B's weakest class in all 3 seeds. `statement` is consistently
the hardest class for C, and the weakest class overall in this table.

## 3. Strategy-specific notes

**Strategy A.** No real tuning surface — 2 prompting modes, temperature fixed
at 0. Few-shot beats zero-shot by a wide margin on dev (0.5916 vs 0.5148) and
is the only strategy with a bootstrap CI computed so far.

**Strategy B.** Most heavily tuned of the three (silver size × sample_seed ×
train_seed × class weights swept on dev). Two caveats on the current numbers:
(1) an audit found 262 of 276 traceable gold rows share a Reddit thread with
the few-shot silver pool, so held-out results reflect in-domain rather than
fully unseen-thread performance; (2) class weights give only a modest,
not-yet-significance-tested improvement on the common 296-row benchmark
(0.6179 vs 0.6113 macro-F1, weights-on better in 4/6 paired runs).

**Strategy C.** Initial full-fine-tuning and partial-fine-tuning (last-2/4
layers only) runs collapsed classes (macro-F1 0.16–0.20) due to training
instability, not insufficient data — a train-fit diagnostic showed most runs
failed to even fit their own 204-row training fold (train-fit macro-F1
0.326 vs eval 0.239, r=0.777 between the two). Adding `warmup_steps=20,
epochs=8` resolved the instability. A full warmup×epochs grid (§4) then
confirmed that configuration as optimal; no further tuning is planned.

## 4. Strategy C: warmup/epochs grid (dev-only, now complete)

| warmup | epochs | class weights | macro-F1 (mean ± std) | class collapse |
|---|---|---|---|---|
| 20 | 8 | on | **0.5309 ± 0.0038** | 0/15 |
| 10 | 10 | on | 0.5231 ± 0.0406 | 0/15 |
| 30 | 10 | on | 0.5189 ± 0.0494 | 0/15 |
| 20 | 10 | on | 0.5166 ± 0.0516 | 0/15 |
| 30 | 8 | on | 0.5072 ± 0.0314 | 0/15 |
| 10 | 8 | on | 0.4904 ± 0.0602 | 0/15 |
| 30 | 6 | on | 0.4371 ± 0.0292 | 0/15 |
| 20 | 6 | on | 0.4278 ± 0.0511 | 0/15 |
| 10 | 6 | on | 0.4216 ± 0.0587 | 2/15 |
| 20 | 8 | **off** | 0.3499 ± 0.0229 | 8/15 |

Takeaways: class weighting is essential (unweighted training collapses the
`statement` class almost entirely); 6 epochs is undertrained at every warmup
value; 10 epochs is competitive on the mean but 8–13x noisier across seeds
than the locked 8-epoch config. No configuration beat the locked one, so
`STRATEGY_C_FINAL_CONFIG` and the held-out run stand as-is.

## 5. Why we are not reporting a single ranking

1. **Power.** The held-out set (`pilot80`, 20/class) is too small. Strategy
   A's held-out bootstrap 95% CI, `[0.587, 0.797]`, already contains both B's
   (0.618) and C's (0.569) point estimates. Power analysis: A-vs-B and
   A-vs-C become resolvable (~80–90% power) once each class reaches ~60–70
   confirmed rows; B-vs-C stays unresolvable even at 250 rows/class (~24%
   power) — a structural dead end, not something more data fixes.
2. **Tuning-budget asymmetry.** B was swept the most on dev; C's grid (§4) is
   now complete and confirms its locked config is optimal; A has essentially
   no tuning surface. The raw asymmetry in configs-tried remains, but C's
   result is no longer "a single untested value."
3. **Model-family confound.** A uses a decoder-only LLM (Llama-3.1-8B-Instruct,
   zero/few-shot); B and C use an encoder (RoBERTa-base, fine-tuned).
   Strategy and backbone vary together by design; a full cross-architecture
   grid is out of scope. Findings are scoped to this specific model pairing,
   not a general LLM-vs-fine-tuning claim.

Instead of a ranking, results support three separate sub-claims:

- **A vs C (data efficiency):** 300 gold rows are not enough for gold-only
  fine-tuning to match direct LLM annotation, even after fixing C's training
  instability.
- **A vs B (does silver distillation pay off?):** if A still matches/beats B
  after the held-out expansion, that is notable specifically because the
  tuning budget favors B — distilling the LLM's own labels into a fine-tuned
  model would not have demonstrated an accuracy gain over using the LLM
  directly.
- **B vs C (silver vs. gold source):** the observed gap (0.024, held-out) is
  too small to resolve at any feasible annotation scale — a genuine
  statistical non-finding, not a data gap.

## 6. Inter-annotator agreement

The held-out set was double-annotated (2 annotators), then reconciled by
discussion. Raw pre-discussion agreement on the original 130 co-labeled rows:
**87.7% (114/130), 16 disagreements.** Kappa was not computable for that batch
because the pre-reconciliation labels weren't retained separately.

## 7. In progress: held-out set expansion

To get enough power to resolve A-vs-B and A-vs-C (see §5), a second held-out
batch (`cmv_test_candidates_batch2_annotate.csv`, 450 rows, naturally covering
all four classes — approx. 216 disagree / 94 agree / 72 statement / 67
question) is being annotated toward ~70 confirmed rows/class. This batch
keeps pre-discussion annotator columns separately this time, so kappa will be
computable. **Status: annotation not yet started (0/450 labeled).**

Once complete:
- Re-run all three `*_heldout` evaluations on the expanded set.
- Add bootstrap CI to B and C's held-out metrics (currently only A has one).
- Re-run the A-vs-B and A-vs-C power calculation with the larger n.

## 8. Open items before Results/Discussion can be finalized

- [ ] Annotate batch2 to ~70 confirmed rows/class.
- [ ] Re-run held-out evals on the expanded set.
- [ ] Add bootstrap CI (ideally paired across A/B/C on the same rows) for B and C.
- [x] Strategy C warmup/epochs grid — complete, locked config confirmed optimal.
- [ ] Write the "Scope of Model Comparison" paragraph disclosing the model-family confound.
- [ ] Optional: `roberta-base` → `roberta-large` swap for B/C's locked configs (no new grid), and/or a second LLM for A, to check whether the A-vs-B/A-vs-C ranking is backbone-sensitive.
