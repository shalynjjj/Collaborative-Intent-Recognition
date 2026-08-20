# depolarAIze — Meeting Report (18 August 2026)

## 1. What we were asked to do

**Stage 1:** classify CMV replies (`agree`/`disagree`/`question`/`statement`), comparing A (direct LLM), B (RoBERTa on LLM-distilled labels), C (RoBERTa on gold labels) — does the LLM add value, does distillation preserve it more cheaply, is 300 gold rows enough alone.
**Stage 2:** decompose sentiment/emotion/intent into a modular LLM pipeline using Stage 1's dialogue-act as input — does decomposition beat one prompt, which module inputs help.

## 2. What we completed

- Expanded held-out set, re-ran A/B/C on it.
- Full error analysis on `disagree` (weakest class).
- Both Stage 2 experiments run once on eval.

## 3. Expanded held-out setup

pilot80 (80 rows, 20/class) was underpowered. Added batch2 → 530-row union → downsampled to 83/class → **332-row balanced set**. pilot80 is mostly a subset of it, not a separate set.

## 4. Updated A/B/C results

| Strategy | Dev macro-F1 | Held-out (332) | Held-out (pilot80) |
|---|---:|---:|---:|
| A — Llama few-shot | 0.592 | 0.673 | 0.699 |
| B — RoBERTa/silver | 0.638 | 0.669 ± 0.001 | 0.618 ± 0.009 |
| C — RoBERTa/gold | 0.531 | 0.586 ± 0.070 | 0.569 ± 0.015 |

Ranking not consistent: Dev → B>A>C; pilot80 → A>C>B; 332-set → **A≈B>C**. B gains least Dev→held-out (+0.03 vs A's +0.08) — consistent with (not proven as) B being over-tuned on Dev.

**What this tells us:** the 332-row set is the largest, most powered benchmark we have, so it's the one to trust — and on it, A and B are statistically indistinguishable (§8), C is behind both. The Dev-based ranking (B>A>C) is not reliable evidence for a final answer; it just reflects which config Dev happened to favor.

## 5. Per-class and confusion findings

| Class | A | B | C |
|---|---:|---:|---:|
| agree | 0.667 | 0.654 | 0.580 |
| disagree | 0.623 | 0.590 | 0.565 |
| question | 0.812 | 0.877 | 0.837 |
| statement | 0.591 | 0.554 | **0.362** |

`question` easiest everywhere; `statement` hardest, especially C.

| Strategy | Biggest confusion (% of true class) |
|---|---|
| A | agree→statement 34.9%, disagree→statement 31.3% |
| B | disagree→statement 24.9% |
| C | agree↔disagree 22.1%+18.1%, statement→agree 33.7% |

Agree/disagree confusion tracks **RoBERTa vs. LLM** (B, C have it; A doesn't) — a model limitation, not ambiguous text.

## 6. Why `disagree` is difficult

F1: A 0.62, B 0.59, C 0.57. Checked every misclassified row, not one example:
- B → `statement` (25%): 56% have zero negation/hedge words, e.g. *"Reoccurrence has to be detected to be tracked."*
- C → `agree` (18%): not an "affirmative opener" effect (only 4% fit) — overlaps with the agree/disagree confusion.
- Both → `question` (B 12%, C 17%): 71%/52% literally end in "?", vs. 1% baseline — e.g. *"Where did I say that?"*

Negation, contractions, length, spelling checked — no effect. Only the rhetorical-question pattern held up.

## 7. Qualitative examples and annotation ambiguity

27 gold rows where ≥6/8 of C's seeds flip agree↔disagree, pulled for blind re-label — checks genuine ambiguity vs. model limitation.

- `results/strategy_c/agree_disagree_hard27_blind.csv` / `_answer_key.csv`
- `python3 -m src.score_agree_disagree_relabel` to score
- Example: *"Fair enough, you're entitled to your own opinion. Though I believe you're limiting yourself..."* (true: `disagree`) — opens with agreement, pivots.
- **Status: in progress, not yet scored.**

## 8. What comparisons we can and cannot make

| Comparison | Δ macro-F1 | Significant? |
|---|---:|---|
| A vs. B | 0.004 | No — genuine tie |
| A vs. C | 0.047 | No — real gap, open |
| B vs. C | 0.042 | No — real gap, open |

A/B/C use different model families (LLM vs. two RoBERTa fine-tunes) — not a fair ranking. **B vs. C is the cleanest comparison** (same backbone, different label source).

## 9. Data-split checks

- No identical (Parent, Reply) pairs across gold/silver/held-out.
- Different replies to the same parent kept (realistic Reddit) → 262/276 gold rows share a thread with silver, so B's results are in-domain, not unseen-topic.
- IAA: gold 89.67%, held-out 95.18% raw agreement (kappa not computable).

## 10. Updated conclusions

- A vs. B (distillation payoff) — **resolved**: genuine tie.
- A vs. C (data efficiency) — **open**: real gap, not significant.
- B vs. C (silver vs. gold) — **open**: real gap, not significant.

## 11. Remaining Stage 1 work

**Feasible now (compute-only):** more seeds for A/B, class-imbalance test on C, `roberta-large`/second-LLM backbone check.
**Not planned (annotation cost):** full push to 330–350 rows/class (~2,100 candidates) to resolve A-vs-C/B-vs-C — judged infeasible. 27-row blind re-label still in progress.

## 12. Stage 2 preliminary progress

| Label | Multi-module | Single prompt |
|---|---:|---:|
| Sentiment | 0.572 | 0.532 |
| Intent | **0.464** | 0.360 |
| Emotion | 0.372 | 0.357 |

Modules beat one prompt, especially Intent. Best Intent input: `dialogue_act` alone (0.464) — adding sentiment/emotion hurts.

Open: Challenge/Counter-argue confusion in Intent; Emotion parser bug found and fixed, not yet re-validated; manual error review + write-up pending.

## 13. Questions for supervisor

1. OK with three sub-claims instead of one ranking?
2. OK to formally close the infeasible annotation push?
3. Finish the 27-row blind re-label before write-up, or ship in-progress?
4. Worth the backbone-sensitivity check before finalizing Stage 1?
5. Emotion parser fix — worth a second eval-split touch once re-validated on dev?
