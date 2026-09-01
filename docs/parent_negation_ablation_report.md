# Parent-Negation Ablation — Strategy B Progress Report

## Motivation

At the last meeting, the advisor's hunch was that Strategy B's `agree`/`disagree`→`statement` errors happen because the model classifies the `Reply` in isolation and doesn't use the `Parent`. Proposed test: edit the `Parent`'s negation, keep `Reply` fixed, flip the expected label to match, and see whether the model's prediction follows.

## Method

1. Started from the 23 `agree`/`disagree` rows Strategy B calls `statement` in all 3 locked seeds (heldout eval). Most CMV `Parent` comments are long, multi-paragraph, and don't reduce to one cleanly negatable claim — only 3 were usable.
2. For each: flipped the `Parent`'s negation (added "not" if absent, removed it if present), left `Reply` untouched, flipped the gold label to match (`agree`↔`disagree`).
3. Retrained the exact locked Strategy B config (silver_size=2500, sample_seed=123, class weights on) from scratch on the 3 locked seeds, predicted on both the original and edited version of each row.
4. Scored per-pair: did the prediction change at all, and if it changed, did it land on the new expected label (vs. just changing to something else)?

Reproduce: `python3 -m src.ablation_parent_negation`

## The 3 examples

| id | direction tested | Parent (original → edited) | Reply (unchanged) |
|---|---|---|---|
| `babies_headcover` | disagree → agree | "Some babies **won't** eat if their heads are covered." → "Some babies eat if their heads are covered." | "They'll eat once they're hungry enough." |
| `who_healthcare_ranking` | agree → disagree | "The WHO ranks..." → "The WHO **does not** rank..." | "Yeah. Still one of the best in the world..." |
| `brand_new_tires` | disagree → agree | "...**Not** everyone is driving brand new cars." → "...Everyone is driving brand new cars." | "Everyone on the road is required to have safe tires..." |

`babies_headcover` is the advisor's own worked example from the meeting.

## Results

9 edited-row predictions (3 examples × 3 seeds):

| example | variant | gold | seed 42 | seed 123 | seed 2026 |
|---|---|---|---|---|---|
| `babies_headcover` | original | disagree | statement | statement | statement |
| `babies_headcover` | **edited** | **agree** | statement | statement | statement |
| `brand_new_tires` | original | disagree | disagree | statement | statement |
| `brand_new_tires` | **edited** | **agree** | statement | statement | statement |
| `who_healthcare_ranking` | original | agree | statement | disagree | statement |
| `who_healthcare_ranking` | **edited** | **disagree** | statement | disagree | statement |

Paired summary across all 9: **89% of predictions didn't change** after the edit (8/9), **11% changed** (1/9), and **0% changed to the new expected label** (0/9).

## Interpretation

- **The headline number is 0/9**: not one edited row was correctly reclassified in response to the `Parent` edit. That's evidence Strategy B is not using `Parent` for this error type, for these 3 cases.
- **The advisor's specific hunch — that it might work for the agree direction but not disagree — is not what we found.** Both directions failed equally: 0/6 on the two disagree→agree examples, 0/3 on the one agree→disagree example (the one apparent "hit," seed 123 on `who_healthcare_ranking`, predicted `disagree` for *both* the original and its exact opposite — it didn't respond to the edit at all, it just always says `disagree` for that pair regardless of what the Parent says).
- **Caveat: n=3 examples.** This is a targeted case study, not a powered test. It confirms the general pattern-matching explanation but shouldn't be oversold as "Strategy B never uses Parent."

## Status / next steps

- ✅ Strategy B — done (above).
- 🔄 Strategy A — running now (no retraining needed, just re-prompting with the locked few-shot config; results shortly).
- ⬜ Strategy C — not started (needs retraining on gold-300, ~15–25 min estimated).

Full write-up merged into the main report: `docs/stage1_full_report.md` → Error Analysis → "Parent-Negation Ablation (Strategy B)". This document is the standalone version for the advisor meeting.
