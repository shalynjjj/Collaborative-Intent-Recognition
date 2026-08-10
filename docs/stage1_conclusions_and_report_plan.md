# Stage 1 (Task 3): Conclusions & Report Structure — Working Notes

Working notes for how to write up the Strategy A/B/C comparison in the thesis
report. Not a ranking of "which strategy is best" — see below for why.

## Why not a single ranking

1. **Power.** `pilot80` (20/class) is too small — Strategy A's held-out
   bootstrap 95% CI is `[0.587, 0.797]`, which already contains both B's
   (`0.618`) and C's (`0.569`) point estimates. A-vs-B and A-vs-C become
   resolvable (~80-90% power) once each class reaches ~60-70 confirmed rows;
   B-vs-C stays unresolvable even at n=250/class (~24% power) — a dead end
   regardless of sample size, not a data gap to fix.
2. **Tuning-budget asymmetry.** B was swept the most on dev (silver size ×
   sample_seed × train_seed × class_weights). C's warmup/epochs grid (see
   README's "Strategy C: Warmup/Epochs Grid") is now complete: 9 warmup x
   epochs combinations plus the weights-off control, all class-weighted
   except the control. The locked config (warmup=20, epochs=8, weights on,
   macro-F1 `0.5309 ± 0.0038`) was already the best point tested and held up
   against the full grid, so no relock was needed. A has ~no tuning surface
   (2 prompting modes, temperature fixed at 0). The asymmetry in raw config
   count vs. B still exists, but the "claim about C is only as strong as an
   ad hoc value" caveat no longer applies.
3. **Model-family confound.** A uses an LLM (Llama-3.1-8B-Instruct,
   decoder-only, zero/few-shot); B and C use RoBERTa-base (encoder,
   fine-tuned). Strategy and backbone vary together. A full
   cross-architecture grid is out of scope for this thesis — disclose this,
   don't try to resolve it.

## The three sub-claims to write instead of a ranking

- **A vs C (data efficiency — what Task 3 actually asks for):** under
  Llama-3.1-8B-Instruct + RoBERTa-base, 300 gold rows are not enough for
  gold-only fine-tuning to match direct LLM annotation, even after fixing
  C's training instability. This maps directly onto the proposal's own
  framing of gold fine-tuning as a "data-efficiency upper bound" — the
  strongest, most directly requested claim.
- **A vs B (does silver distillation pay off?):** if A still matches/beats B
  after the held-out expansion, that is a strong result precisely *because*
  the tuning asymmetry favors B (more data, more sweeping), not A. Frame it
  as: distilling the LLM's own labels into a fine-tuned model did not
  demonstrate an accuracy gain over using the LLM directly.
- **B vs C (silver vs gold source):** report as a genuine negative
  finding — the observed gap (`0.024`) is too small to resolve at any
  feasible scale. State plainly that this is not statistically
  distinguishable, not something more annotation would fix.

Every one of the three claims carries the model-family-confound disclaimer:
results are scoped to Llama-3.1-8B-Instruct + RoBERTa-base specifically, not
a general LLM-vs-fine-tuning claim.

## Mapping onto the required report sections

The proposal's required structure is: 1. Introduction, 2. Related Work,
3. Methodology, 4. Experimental Setup, 5. Results, 6. Discussion and
Limitations, 7. Conclusion.

- **Methodology**: describe A/B/C as implemented; add an explicit "Scope of
  Model Comparison" paragraph stating the model-family confound up front
  (Llama-3.1-8B-Instruct for A, RoBERTa-base for B/C; strategy and backbone
  vary together by design; no cross-architecture grid was run).
- **Experimental Setup**: gold/silver/held-out construction; the held-out
  expansion methodology (why 20/class was underpowered, target ~70/class,
  simple random length-stratified sampling — see README's "Held-Out Test
  Set" section); the tuning-budget table (how many configs were tried per
  strategy on dev before locking); the statistical testing plan (bootstrap
  CI, paired significance tests across A/B/C on the same held-out rows).
- **Results**: per-strategy dev + held-out numbers; class-weight and
  warmup/epochs ablations; the paired significance test results for A-vs-B
  and A-vs-C specifically (not a single combined ranking table pretending
  all three are equally resolvable).
- **Discussion and Limitations**: the three issues above (power, tuning
  asymmetry, model-family confound) as named limitations, plus existing
  ones already tracked in the README (fixed-benchmark reuse, gold/silver
  thread overlap, unreconciled IAA on the original 130 rows, few-shot
  silver-label quality never spot-checked).
- **Conclusion**: the three sub-claims above, each with its own evidentiary
  status — not "Strategy X is best."

## Todo before Results/Discussion can be written

- [x] ~~Reconcile the 82 unresolved rows in `cmv_test_candidates_labeled.csv`~~
      Skipped by decision: those rows exist because annotation on that batch
      stopped once `disagree`/`statement` already had enough. Batch2's 450
      rows cover all four classes from natural random sampling (expected
      ~216 disagree / ~94 agree / ~72 statement / ~67 question), so relying
      on batch2 alone is sufficient; the 82 rows are not needed.
- [ ] Annotate `cmv_test_candidates_batch2_annotate.csv` until every class
      reaches ~70 confirmed rows; keep pre-discussion annotator columns this
      time so Kappa is computable (it wasn't for the original 130).
- [ ] Re-run all three `*_heldout` evals on the expanded set.
- [ ] Add bootstrap CI (ideally paired, across A/B/C on the same rows) to B
      and C's held-out metrics — only A has one today.
- [x] Run the Strategy C warmup/epochs grid; relock `STRATEGY_C_FINAL_CONFIG`
      and re-run `strategy_c_heldout` once if a better config turns up.
      Done: full 9-combo grid (+ weights-off control) confirms (warmup=20,
      epochs=8, weights on) is still the best config, so no relock or
      heldout re-run was needed. See README's "Strategy C: Warmup/Epochs
      Grid" for the ranked table.
- [ ] Write the "Scope of Model Comparison" paragraph for Methodology.
- [ ] Optional/time-permitting: swap `roberta-base` → `roberta-large` for
      B/C's already-locked configs only (no new grid), and/or run A's
      held-out inference once with a second LLM, to check whether the
      A-vs-B/A-vs-C ranking is sensitive to backbone choice.
- [ ] Write the Conclusion using the three-sub-claim framing, not one ranking.
