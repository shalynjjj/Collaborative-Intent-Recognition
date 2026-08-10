# Meeting Report Draft — Strategy A/B/C Results

## Update
Rented a server, re-ran A, B, and C end-to-end under the standardized
pipeline (fixed folds/seeds, group-aware silver split, warmup/epochs fix for
C). This report covers the results of that re-run.

## Dialogue-Act Tagging
Task: classify each Reddit CMV reply into one of four dialogue acts —
`agree`, `disagree`, `statement`, `question`.

## Dataset

**Gold** — 300 human-annotated rows. Used as the tuning/dev set for all three
strategies; no strategy is allowed to touch the held-out set until its config
is locked on gold.

**Test (held-out, "pilot80")** — 20 rows/tag, 80 total. Touched exactly once
per strategy, after locking.

**Why we're annotating a second dataset (Test batch2):** the 80-row pilot is
underpowered. Strategy A's held-out bootstrap 95% CI is `[0.587, 0.797]` —
wide enough to already contain both B's (0.618) and C's (0.569) held-out
point estimates, so right now we cannot say A actually beats B or C on
held-out, only that it looks higher. Power analysis says A-vs-B and A-vs-C
become resolvable (~80–90% power) once each class reaches ~60–70 confirmed
rows; B-vs-C stays unresolvable even at 250 rows/class (~24% power) — that
one is a structural dead end, not something more annotation fixes. So we're
annotating a second batch (450 candidate rows, naturally covering all four
classes) toward ~70 confirmed rows/class, specifically to make the A-vs-B and
A-vs-C comparisons statistically meaningful.

## A — LLM Annotation (Llama 3.1 8B)

Few-shot vs. zero-shot prompting, evaluated on gold dev; few-shot locked and
evaluated once on held-out.

| Config | Gold (dev) macro-F1 | Kappa | Held-out macro-F1 | Held-out kappa |
|---|---|---|---|---|
| Zero-shot | 0.5148 | 0.298 | — | — |
| Few-shot | 0.5916 | 0.420 | 0.6985 | 0.60 |

Held-out per-class F1 (agree/disagree/question/statement):
`0.647 / 0.722 / 0.773 / 0.652` — no collapsed classes, the most balanced of
the three strategies.

**Flag for discussion:** held-out (0.6985) is *higher* than dev (0.5916).
Worth raising explicitly — is the 80-row pilot just an easier sample, not
evidence the model generalizes better than dev suggests? (See cross-cutting
note at the end — C shows the same direction of shift, B doesn't.)

## B — RoBERTa on LLM-Silver Labels

Fine-tuned on Llama-labeled silver data; learning curve over silver-set size.

**Learning curve (few-shot silver, gold dev, mean over 2 sample seeds × 3 train seeds):**

| silver_size | macro_f1_mean | macro_f1_std |
|---|---|---|
| 500 | 0.4155 | 0.0865 |
| 1000 | 0.5814 | 0.0189 |
| 1500 | 0.6143 | 0.0275 |
| 2000 | 0.6184 | 0.0187 |
| 2500 | 0.6220 | 0.0274 |
| 5000 | 0.6092 | 0.0207 |
| 8000 | 0.5951 | 0.0122 |
| 10000 | 0.5849 | 0.0059 |

Curve is flat from 3000–10000; 2500 is the selected point but not shown
significantly better than neighboring 1500/2000/5000 — selection was by
highest observed mean only, no significance test.

**Benchmark summary:**

| Benchmark | Macro-F1 | Kappa |
|---|---|---|
| Gold (dev), 2500 silver rows | 0.6375 | — |
| Held-out (mean of 3 train seeds) | 0.6175 ± 0.0085 | 0.511 ± 0.0096 |
| Common-296, weights on (vs. A few-shot) | 0.6179 ± 0.0282 | 0.468 ± 0.039 |
| Common-296, weights off | 0.6113 ± 0.0252 | 0.449 ± 0.037 |

Dev vs. held-out ranking flips: dev ranks B (0.638) above A (0.592);
held-out ranks A (0.699) above B (0.618 mean). Good headline tension for the
meeting — the model selected on dev isn't the model that wins on the
untouched set.

Held-out per-class F1 (mean of 3 seeds): `agree 0.689 / disagree 0.449 /
question 0.803 / statement 0.529` — `disagree` is B's weakest class in all 3
seeds, and noticeably worse than A's 0.722 on the same class.

Class weights: `+0.0066` macro-F1 on common-296 (0.618 vs 0.611), mainly from
`disagree` (`+0.042`), at some cost to `statement` (`−0.020`). Modest, not
statistically tested — framed as "retained by observed improvement," not
proven.

**Caveat worth stating out loud:** a later audit found 262 of 276 recovered
gold rows share a Reddit thread with the few-shot silver pool. Results are
in-domain, not unseen-thread performance — don't let this get cited as
generalization evidence without that qualifier.

## C — RoBERTa Fine-Tuning on Gold-Standard Data (5-Fold CV)

Stratified 5-fold CV, 3 train seeds (42/123/2026); OOF macro-F1 computed per
seed on the concatenated 300-row gold set, then averaged.

**Getting to a stable config first:** initial full- and partial-fine-tuning
runs (last-2/4 layers only) collapsed classes, macro-F1 only `0.16–0.20`. A
train-fit diagnostic showed this was **training instability, not
insufficient data** — most runs failed to even fit their own 204-row
training fold (train-fit macro-F1 `0.326` vs. eval `0.239`, correlated at
`r=0.777`). Adding `warmup_steps=20, epochs=8` fixed it.

| Config | Gold (dev) macro-F1 | Kappa | Held-out macro-F1 | Held-out kappa |
|---|---|---|---|---|
| RoBERTa gold, weights on, warmup20/epochs8 (locked) | 0.5309 ± 0.0038 | 0.348 ± 0.0081 | 0.5693 ± 0.0152 | 0.489 ± 0.0192 |
| RoBERTa gold, weights off (same warmup/epochs) | 0.3499 ± 0.0229 | 0.163 ± 0.0301 | not evaluated (never locked) | — |

Held-out per-class F1 (mean of 3 seeds): `agree 0.657 / disagree 0.551 /
question 0.836 / statement 0.234` — `statement` is by far C's weakest class,
consistent with it being the class that collapses first without weighting.

**Warmup/epochs grid (now complete, dev-only)** — confirms the locked config
is optimal:

| warmup | epochs | weights | macro-F1 (mean ± std) | class collapse |
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
| 20 | 8 | off | 0.3499 ± 0.0229 | 8/15 |

Class weighting is essential (unweighted training collapses `statement`
almost entirely — 8/15 fold/seed runs). 6 epochs is undertrained at every
warmup value. 10 epochs is competitive on the mean but 8–13x noisier across
seeds than the locked config.

**Flag for discussion (same shape as A):** held-out (0.5693) is again
*higher* than dev (0.5309) — see cross-cutting note below.

## Cross-cutting observation (worth raising as one point, not three)

Both **A** (dev 0.592 → held-out 0.699) and **C** (dev 0.531 → held-out
0.569) score *higher* on the 80-row held-out set than on the 300-row gold
dev set. **B** is the opposite (dev 0.638 → held-out 0.618). If the pilot80
set were simply "easier" across the board, all three should shift the same
direction — they don't. Possible explanation: B was the most heavily
dev-tuned (silver size × sample_seed × train_seed × weights swept), so it may
be mildly overfit to dev's specific 300 rows in a way A and C aren't. This
is speculative and exactly the kind of thing the batch2 expansion should let
us actually check, rather than argue about on n=80.

## Question
Open items to get input on:

- Annotation bandwidth/timeline: is standing up batch2 to ~70 confirmed
  rows/class realistic before the next milestone?
- Is the dev/held-out direction split (A & C up, B down) worth a dedicated
  analysis now, or does it wait for the batch2 re-run?
- Time-permitting: worth swapping in `roberta-large` for B/C's already-locked
  configs (no new grid), or running A's held-out inference with a second LLM,
  to check whether the A-vs-B/A-vs-C ranking is backbone-sensitive?
- Does the "three sub-claims instead of one ranking" framing (data
  efficiency A-vs-C, distillation payoff A-vs-B, silver-vs-gold B-vs-C as a
  negative finding) match what's expected for the Results/Discussion
  sections?

## Next steps
- Annotate batch2 (`cmv_test_candidates_batch2_annotate.csv`, currently
  0/450) to ~70 confirmed rows/class; keep pre-discussion annotator columns
  this time so Kappa is computable (it wasn't for the original 130).
- Re-run all three `*_heldout` evals on the expanded set.
- Add bootstrap CI to B and C's held-out metrics (only A has one today).
- Re-run the A-vs-B / A-vs-C power calculation at the larger n.
- Write the "Scope of Model Comparison" paragraph (Llama-3.1-8B-Instruct for
  A vs. RoBERTa-base for B/C — strategy and backbone vary together by
  design, no cross-architecture grid run).
- Write Results/Discussion/Conclusion using the three-sub-claim framing
  above, not a single ranking table.
