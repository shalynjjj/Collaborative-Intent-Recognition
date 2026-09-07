# Stage 1 Report — Chapter Framework

Framework for `\input{chapters/4-methodology}` through `\input{chapters/8-conclusion}`, based on the work already documented in `docs/full-report-draft.tex`, `docs/stage1_full_report.md`, and `README.md`.

**Structural note:** the requested file order is `4-methodology → 5-experiments → 6-limitations → 7-future-work → 8-conclusion`, but the existing draft's internal chapter order is `Methodology → Experiments → Conclusion → Future Work → Limitations`. Decide which order you actually want before splitting the file — reordering later means re-checking cross-references (`\autoref{...}`) between chapters.

---

## Chapter 4: Methodology

**4.1 Task and Label Scheme**
- Dialogue-act classification: 4 classes (`agree`, `disagree`, `question`, `statement`)
- Three modelling strategies compared:
  - Strategy A — Direct LLM Annotation (Llama-3.1-8B-Instruct, zero-/few-shot)
  - Strategy B — Silver-Standard Fine-Tuning (RoBERTa-base on LLM-generated silver labels)
  - Strategy C — Gold-Standard Fine-Tuning (RoBERTa-base on the 300-pair gold set, cross-validated)
- What each strategy is meant to establish (prompting sufficiency / distillation value / data-efficiency reference)

**4.2 Gold Set (Dev) and Held-Out Test Set**
- Gold-300: 300 pairs, class counts, double-annotated + reconciled, used only for tuning
- Held-out (332 pairs, 83/class balanced): touched once per strategy after config lock
- No leakage between gold/silver/held-out (verified); note the gold/silver same-thread overlap caveat

**4.3 Inter-Annotator Agreement**
- Raw agreement on gold-300 and held-out; note whether pre-reconciliation labels survived (affects whether Cohen's κ is computable)

---

## Chapter 5: Experiments

**5.1 Strategy A — LLM Annotation**
- Zero-shot vs. few-shot comparison, locked config, held-out macro-F1
- Dominant error pattern (disagree→statement), `question` easiest

**5.2 Strategy B — RoBERTa on Silver Labels**
- Silver pool size sweep (500–10k), locked size + sample_seed
- Class-weight ablation
- Held-out result, weakest class (note if it differs from dev)

**5.3 Strategy C — RoBERTa on Gold Labels**
- 5-fold CV / out-of-fold evaluation setup
- Class-collapse problem found + diagnosis (train-fit check) + fix (warmup + longer training)
- Warmup/epoch grid search result
- 8-seed variance check (compare to original 3-seed estimate)
- Held-out result, per-seed spread

**5.4 Cross-Strategy Comparison**
- Ranking instability across eval sets (gold vs. two held-out versions) — explain why no single ranking is reported
- Per-class pattern consistent across strategies (`question` easiest, `statement` hardest)
- Paired bootstrap power analysis: observed Δ, CI, significance, power, pairs-needed-for-80%-power per comparison

**5.5 Error Analysis**
- Confusion overview: largest error types per strategy, with text-feature explanations (hedge words, negation, length)
- Additional text-feature checks (contractions/spelling ruled out; long-reply effect specific to A)
- Agree/disagree confusion: cross-strategy check on "hard pairs," candidate explanation (concede-then-pivot structure), ruled-out/not-ruled-out alternatives
- (If included) Parent-negation ablation — does the model actually use Parent context?

**5.6 Stage 2 Experiments** *(new section to add — this is where the current Stage 2 work goes)*
- Experiment 1: multi_module vs. single_prompt — results, the prompt-fairness bug found/fixed, significance test
- Experiment 2: intent_sweep — oracle/predicted auxiliary-input results, dialogue_act finding
- Intent error analysis: confusion matrix, DA→Intent cascade, case studies

---

## Chapter 6: Limitations

- Gold/silver domain overlap (Strategy B held-out results are in-domain, not unseen-topic)
- Gold-300 IAA is aggregate-only (no pair-level disagreement recoverable)
- Model family confounded with strategy (A=LLM, B/C=RoBERTa — can't isolate backbone effect from label-source effect)
- Strategy C's seed variance was underestimated at n=3
- Held-out set still same corpus family as dev (not a distribution shift test)
- Class imbalance in gold-300 as an unruled-out confound
- *(Stage 2 additions)*: prompt-fairness confound found mid-project and its implications for how early Stage 2 numbers should be read; single-split (no k-fold) variance risk at n=264; Dialogue_act↔Intent annotation-scheme correlation confounding Experiment 2's headline result

---

## Chapter 7: Future Work

**7.1 Stage 1 Follow-Ups**
- Compute-only (no new annotation): more seeds for A/B, oversampling ablation, backbone swap (roberta-large / second LLM)
- Annotation-dependent (not currently planned): blind re-label of the 27 hard agree/disagree pairs; scaling annotation toward ~330–350 pairs/class for A-vs-C and B-vs-C significance

**7.2 Stage 2 Follow-Ups**
- Fix the two newly-found Emotion-parsing gaps (code-block-style answers, "sarcastic"/"sarcasm" mismatch)
- Re-run the Intent error analysis with oracle-mode gold Dialogue Act (isolate DA-signal effect from Strategy A's prediction errors)
- Methodological question for advisor: single split vs. k-fold at this sample size
- Fold this whole report into the final thesis structure

---

## Chapter 8: Conclusion

- Which pairwise comparison is resolved (A vs. B — genuine tie) and which remain open (A vs. C, B vs. C — real but not-yet-significant gaps), with what it would take to close them
- Two qualifying findings: seed-variance was underestimated for C; agree/disagree confusion is backbone-specific (B/C), not present in A
- *(Stage 2 addition)*: headline result reversal after fixing the prompt-fairness confound — the honest takeaway is "no confirmed architecture winner," and why that's still a meaningful contribution (found + fixed a confound, validated with a real re-run, significance-tested)
