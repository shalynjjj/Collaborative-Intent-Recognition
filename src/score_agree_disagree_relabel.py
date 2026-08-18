"""Score a completed blind re-label of the 27 hard agree/disagree rows.

Usage:
    python3 -m src.score_agree_disagree_relabel \\
        --blind-csv results/strategy_c/agree_disagree_hard27_blind.csv \\
        [--blind-csv-2 results/strategy_c/agree_disagree_hard27_blind_annotator2.csv]

Compares `your_label` against the original gold label (were the 27 rows
mislabeled in the first place, or genuinely ambiguous) and, if a second
annotator's file is given, computes raw agreement between the two annotators.
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_BLIND = Path("results/strategy_c/agree_disagree_hard27_blind.csv")
DEFAULT_KEY = Path("results/strategy_c/agree_disagree_hard27_answer_key.csv")


def score(blind_csv: Path, key_csv: Path, blind_csv_2: Path = None) -> None:
    blind = pd.read_csv(blind_csv)
    key = pd.read_csv(key_csv)

    missing = blind["your_label"].isna() | (blind["your_label"].astype(str).str.strip() == "")
    if missing.any():
        print(f"WARNING: {missing.sum()} rows still have an empty your_label -- excluding them.")
        blind = blind[~missing]

    merged = blind.merge(key[["item_id", "gold_index", "Dialogue_act", "n_flip_to_other"]], on="item_id")
    merged["your_label"] = merged["your_label"].str.strip().str.lower()
    merged["matches_gold"] = merged["your_label"] == merged["Dialogue_act"].str.lower()

    n = len(merged)
    n_match = int(merged["matches_gold"].sum())
    print(f"Rows scored: {n}")
    print(f"Your label matches the original gold label: {n_match}/{n} ({n_match / n:.1%})")
    print()
    print("Rows where you disagree with the original gold label:")
    disagreements = merged[~merged["matches_gold"]][["item_id", "gold_index", "Dialogue_act", "your_label", "n_flip_to_other"]]
    print(disagreements.to_string(index=False) if len(disagreements) else "  (none)")

    if blind_csv_2 is not None:
        blind2 = pd.read_csv(blind_csv_2)
        both = merged.merge(blind2[["item_id", "your_label"]], on="item_id", suffixes=("_a1", "_a2"))
        both["your_label_a2"] = both["your_label_a2"].str.strip().str.lower()
        both["iaa_agree"] = both["your_label_a1"] == both["your_label_a2"]
        iaa_n = len(both)
        iaa_match = int(both["iaa_agree"].sum())
        print()
        print(f"Inter-annotator raw agreement: {iaa_match}/{iaa_n} ({iaa_match / iaa_n:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the blind agree/disagree re-label.")
    parser.add_argument("--blind-csv", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--key-csv", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--blind-csv-2", type=Path, default=None, help="Second annotator's filled-in blind csv, for IAA.")
    args = parser.parse_args()
    score(args.blind_csv, args.key_csv, args.blind_csv_2)


if __name__ == "__main__":
    main()
