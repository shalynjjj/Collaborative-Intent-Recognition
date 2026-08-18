"""Manual-analysis support for why `disagree` underperforms in Strategies B/C.

Loads B and C's held-out (332-row) predictions across their 3 locked seeds,
isolates every row whose gold label is `disagree`, and for each error
destination (statement / agree / question) computes simple text-feature
rates (negation, hedge words, contractions, spelling-error heuristic, very
long/short reply) against the correctly-classified `disagree` baseline.

Usage:
    python3 -m src.analyze_disagree_errors
"""

import re
from pathlib import Path

import pandas as pd

HEDGE = {"but", "however", "although", "though", "yet", "nonetheless", "nevertheless", "still", "except"}
NEGATION = {"not", "no", "never", "none", "nothing", "nobody", "neither", "nor", "cant", "cannot"}
OUTPUT_DIR = Path("results/error_analysis")


def _word_set():
    import nltk

    try:
        from nltk.corpus import words as nltk_words

        return set(w.lower() for w in nltk_words.words())
    except LookupError:
        nltk.download("words")
        from nltk.corpus import words as nltk_words

        return set(w.lower() for w in nltk_words.words())


def _tokenize(text: str):
    return re.findall(r"[A-Za-z']+", str(text).lower())


def _features(text: str, word_set: set) -> dict:
    toks = _tokenize(text)
    n = max(len(toks), 1)
    hedge_hits = sum(1 for t in toks if t in HEDGE)
    neg_hits = sum(1 for t in toks if t in NEGATION or "n't" in t)
    contractions = len(re.findall(r"\b\w+'\w+\b", str(text)))
    orig_toks = re.findall(r"[A-Za-z']+", str(text))
    misspelled = sum(
        1 for t in orig_toks if t.lower() not in word_set and not t[0].isupper() and len(t) > 2 and "'" not in t
    )
    return {
        "word_count": len(toks),
        "hedge_rate": hedge_hits / n,
        "negation_rate": neg_hits / n,
        "contraction_rate": contractions / n,
        "spelling_error_rate": misspelled / n,
        "very_long": len(toks) >= 30,
        "very_short": len(toks) <= 5,
    }


def _load_heldout(strategy: str, seeds) -> pd.DataFrame:
    dfs = []
    for s in seeds:
        d = pd.read_csv(f"results/strategy_{strategy}_heldout/heldout_seed{s}_predictions.csv")
        d["seed"] = s
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    word_set = _word_set()

    stats_rows = []
    example_rows = []

    for strategy in ["b", "c"]:
        df = _load_heldout(strategy, [42, 123, 2026])
        sub = df[df["Dialogue_act"] == "disagree"].copy()
        feat = sub["Reply"].apply(lambda t: _features(t, word_set)).apply(pd.Series)
        sub = pd.concat([sub, feat], axis=1)

        for pred in sub["pred_label"].unique():
            group = sub[sub["pred_label"] == pred]
            row = {
                "strategy": strategy.upper(),
                "predicted_as": pred,
                "is_baseline": pred == "disagree",
                "n": len(group),
                "share_of_disagree": len(group) / len(sub),
            }
            row.update(group[["word_count", "hedge_rate", "negation_rate", "contraction_rate", "spelling_error_rate"]].mean().round(3).to_dict())
            row["very_long_pct"] = round(group["very_long"].mean(), 3)
            row["very_short_pct"] = round(group["very_short"].mean(), 3)
            stats_rows.append(row)

            if pred != "disagree":
                picked = group[group["Reply"].str.len().between(15, 250)].drop_duplicates(subset="Reply").head(5)
                for _, r in picked.iterrows():
                    example_rows.append(
                        {
                            "strategy": strategy.upper(),
                            "predicted_as": pred,
                            "seed": r["seed"],
                            "Parent": r["Parent"],
                            "Reply": r["Reply"],
                            "word_count": r["word_count"],
                            "negation_rate": r["negation_rate"],
                            "hedge_rate": r["hedge_rate"],
                        }
                    )

    stats_df = pd.DataFrame(stats_rows)
    examples_df = pd.DataFrame(example_rows)

    stats_path = OUTPUT_DIR / "disagree_error_text_features.csv"
    examples_path = OUTPUT_DIR / "disagree_error_examples.csv"
    stats_df.to_csv(stats_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    print(stats_df.to_string(index=False))
    print(f"\nWrote {stats_path}")
    print(f"Wrote {examples_path}")


if __name__ == "__main__":
    run()
