import pandas as pd

from .config import DATA_DIR, DIALOGUE_LABELS

SEED = 42

INPUT_CSV = DATA_DIR / "cmv_test_candidates_heldout_expanded.csv"
OUTPUT_CSV = DATA_DIR / "cmv_test_candidates_heldout_balanced.csv"


def main() -> None:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    counts = df["Dialogue_act"].value_counts()
    n_per_class = counts.min()
    print(f"per-class counts in {INPUT_CSV.name}:\n{counts}\n")
    print(f"downsampling every class to n={n_per_class} (smallest class: {counts.idxmin()})")

    parts = [
        df[df["Dialogue_act"] == label].sample(n=n_per_class, random_state=SEED)
        for label in DIALOGUE_LABELS
    ]
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved {len(out)} rows ({n_per_class}/class) to {OUTPUT_CSV}")
    print(out["Dialogue_act"].value_counts())


if __name__ == "__main__":
    main()