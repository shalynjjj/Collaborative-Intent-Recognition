"""Same parent-negation ablation as ablation_parent_negation.py, but for
Strategy A (Llama few-shot prompting) instead of Strategy B.

Strategy A needs no retraining -- it's re-prompted directly with the locked
few-shot config (STRATEGY_A_FINAL_MODE), so this only costs one model load
plus 6 short generations (3 examples x original/edited).

Usage:
    python3 -m src.ablation_parent_negation_strategy_a
"""

from pathlib import Path

import pandas as pd

from .ablation_parent_negation import EXAMPLES, _build_eval_df
from .config import STRATEGY_A_FINAL_MODE
from .llm_annotate import build_prompt, make_transformers_generator
from .utils import parse_label

OUTPUT_DIR = Path("results/error_analysis")


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_df = _build_eval_df()

    generator = make_transformers_generator()

    rows = []
    for _, row in eval_df.iterrows():
        prompt = build_prompt(str(row["Parent"]), str(row["Reply"]), STRATEGY_A_FINAL_MODE)
        raw_output = generator(prompt)
        label, fallback_used = parse_label(raw_output)
        rows.append(
            {
                "example_id": row["example_id"],
                "variant": row["variant"],
                "Parent": row["Parent"],
                "Reply": row["Reply"],
                "Dialogue_act": row["Dialogue_act"],
                "pred_label": label,
                "raw_output": raw_output,
                "fallback_used": fallback_used,
            }
        )
        print(f"{row['example_id']} ({row['variant']}): gold={row['Dialogue_act']} pred={label}")

    result = pd.DataFrame(rows)
    result_path = OUTPUT_DIR / "parent_negation_ablation_strategy_a_predictions.csv"
    result.to_csv(result_path, index=False)

    pairs = result.pivot(index="example_id", columns="variant", values=["pred_label", "Dialogue_act"])
    original_pred = pairs[("pred_label", "original")]
    edited_pred = pairs[("pred_label", "edited")]
    edited_gold = pairs[("Dialogue_act", "edited")]
    changed = original_pred != edited_pred
    matches_gold = edited_pred == edited_gold

    summary = pd.DataFrame(
        {
            "original_pred": original_pred,
            "edited_pred": edited_pred,
            "edited_gold": edited_gold,
            "prediction_changed": changed,
            "edited_matches_gold": matches_gold,
            "changed_to_expected": changed & matches_gold,
        }
    )
    summary_path = OUTPUT_DIR / "parent_negation_ablation_strategy_a_summary.csv"
    summary.to_csv(summary_path)

    print()
    print(summary.to_string())
    print(f"\nWrote {result_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    run()
