"""Add negation/hedge/question text-feature columns to every row of the full
annotated corpus (not just the misclassified rows checked in
`analyze_disagree_errors.py`), so the delivered dataset can be used
downstream (e.g. to train a negation classifier).

Unions the 300-row gold set and the 530-row expanded held-out set (no
overlap between them; together they're the full labeled corpus), computes
the same text features already validated in the disagree error analysis on
both Reply and Parent, and writes one combined CSV.

Usage:
    python3 -m src.add_text_features
"""

from pathlib import Path

import pandas as pd

from src.analyze_disagree_errors import _features, _word_set

GOLD_PATH = Path("data/cmv_300_gold_final.csv")
HELDOUT_PATH = Path("data/cmv_test_candidates_heldout_expanded.csv")
OUTPUT_PATH = Path("data/cmv_full_with_text_features.csv")


def _add_features(df: pd.DataFrame, column: str, prefix: str, word_set: set) -> pd.DataFrame:
    feat = df[column].apply(lambda t: _features(t, word_set)).apply(pd.Series)
    feat["has_negation"] = feat["negation_rate"] > 0
    feat["has_hedge"] = feat["hedge_rate"] > 0
    feat["ends_in_question"] = df[column].astype(str).str.strip().str.endswith("?")
    feat = feat.add_prefix(f"{prefix}_")
    return pd.concat([df.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)


def run() -> None:
    word_set = _word_set()

    gold = pd.read_csv(GOLD_PATH)
    gold["corpus_source"] = "gold_300"

    heldout = pd.read_csv(HELDOUT_PATH)
    heldout["corpus_source"] = "heldout_" + heldout["source"].astype(str)

    common_cols = ["Parent", "Reply", "Dialogue_act", "corpus_source"]
    combined = pd.concat([gold[common_cols], heldout[common_cols]], ignore_index=True)

    combined = _add_features(combined, "Reply", "reply", word_set)
    combined = _add_features(combined, "Parent", "parent", word_set)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Rows: {len(combined)} (gold_300: {len(gold)}, heldout_expanded: {len(heldout)})")
    print(f"Columns: {list(combined.columns)}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
