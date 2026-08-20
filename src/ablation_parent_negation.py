"""Parent-negation ablation for Strategy B: does it use the Parent, or just
pattern-match the Reply?

For each hand-picked agree/disagree row that Strategy B misclassifies as
`statement` in all 3 locked seeds (see results/strategy_b_heldout), we flip
the negation in the Parent (add it if absent, remove it if present), leave
the Reply untouched, and flip the gold label to match the now-negated
Parent. Re-running the *same* locked Strategy B config (silver_size=2500,
sample_seed=123, class weights on) on both the original and edited Parent
tells us:
  - prediction stays `statement` on the edited row  -> model ignores Parent
  - prediction matches the new gold label            -> model uses Parent

Usage:
    python3 -m src.ablation_parent_negation
"""

from pathlib import Path

import pandas as pd

from .config import STRATEGY_B_FINAL_CONFIG, STRATEGY_B_FINAL_SILVER_CSV, STRATEGY_B_HELDOUT_DIR, SEEDS
from .train_roberta import _infer_label_column, _prepare_labeled_frame, _sample_silver_subset, _train_and_predict
from .utils import set_seed

OUTPUT_DIR = Path("results/error_analysis")
MODEL_DIR = STRATEGY_B_HELDOUT_DIR / "models"

EXAMPLES = [
    {
        "example_id": "babies_headcover",
        "note": "Advisor's worked example. Parent has explicit negation (\"won't\"); removed.",
        "Parent_original": "Some babies won't eat if their heads are covered.",
        "Reply": "They'll eat once they're hungry enough.",
        "gold_original": "disagree",
        "Parent_edited": "Some babies eat if their heads are covered.",
        "gold_edited": "agree",
    },
    {
        "example_id": "who_healthcare_ranking",
        "note": "Parent has no negation; added (\"does not rank\").",
        "Parent_original": "The WHO ranks the American healthcare system two spots above Cuba's.",
        "Reply": "Yeah. Still one of the best in the world and it's free at the point of use. Therefore it's two spots below America, but for every citizen.",
        "gold_original": "agree",
        "Parent_edited": "The WHO does not rank the American healthcare system two spots above Cuba's.",
        "gold_edited": "disagree",
    },
    {
        "example_id": "brand_new_tires",
        "note": "Only the core claim sentence (\"Not everyone is driving brand new cars\") is flipped; the identifying first sentence is left as-is.",
        "Parent_original": (
            "You are talking about brand new tires, not the guy next to you who's driving a car "
            "he bought for 2k five years ago and never changed tires. Not everyone is driving brand new cars."
        ),
        "Reply": "Everyone on the road is required to have safe tires with tread, it's a law.  Whether or not your drive a junker.",
        "gold_original": "disagree",
        "Parent_edited": (
            "You are talking about brand new tires, not the guy next to you who's driving a car "
            "he bought for 2k five years ago and never changed tires. Everyone is driving brand new cars."
        ),
        "gold_edited": "agree",
    },
]


def _build_eval_df() -> pd.DataFrame:
    rows = []
    for ex in EXAMPLES:
        rows.append(
            {
                "example_id": ex["example_id"],
                "variant": "original",
                "Parent": ex["Parent_original"],
                "Reply": ex["Reply"],
                "Dialogue_act": ex["gold_original"],
            }
        )
        rows.append(
            {
                "example_id": ex["example_id"],
                "variant": "edited",
                "Parent": ex["Parent_edited"],
                "Reply": ex["Reply"],
                "Dialogue_act": ex["gold_edited"],
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = STRATEGY_B_FINAL_CONFIG

    silver = pd.read_csv(STRATEGY_B_FINAL_SILVER_CSV, encoding="latin1")
    label_col = _infer_label_column(silver)
    silver = _prepare_labeled_frame(silver, label_col)
    train_base = _sample_silver_subset(silver, config["silver_size"], config["sample_seed"])

    eval_df = _build_eval_df()
    eval_df = _prepare_labeled_frame(eval_df, "Dialogue_act")

    all_preds = []
    for seed in SEEDS:
        set_seed(seed)
        predictions, _, _ = _train_and_predict(
            train_base,
            eval_df,
            seed,
            label_col,
            split_seed=config["sample_seed"],
            group_col="source_root",
            use_class_weights=config["use_class_weights"],
            save_model_dir=MODEL_DIR / f"seed{seed}",
        )
        run_df = eval_df[["example_id", "variant", "Parent", "Reply", "Dialogue_act"]].copy()
        run_df["seed"] = seed
        run_df["pred_label"] = predictions
        all_preds.append(run_df)
        print(f"seed {seed} done")

    result = pd.concat(all_preds, ignore_index=True)
    result_path = OUTPUT_DIR / "parent_negation_ablation_predictions.csv"
    result.to_csv(result_path, index=False)

    result["matches_gold"] = result["pred_label"] == result["Dialogue_act"]
    result["still_statement"] = result["pred_label"] == "statement"
    summary = (
        result.groupby(["example_id", "variant"])
        .agg(
            gold=("Dialogue_act", "first"),
            n_seeds=("seed", "nunique"),
            n_pred_matches_gold=("matches_gold", "sum"),
            n_pred_statement=("still_statement", "sum"),
            preds=("pred_label", lambda s: list(s)),
        )
        .reset_index()
    )
    summary_path = OUTPUT_DIR / "parent_negation_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print(summary.to_string(index=False))
    print(f"\nWrote {result_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    run()
