import pandas as pd
from sklearn.model_selection import train_test_split

from .config import GOLD_CSV, STAGE2_DEV_CSV, STAGE2_DEV_FRACTION, STAGE2_EVAL_CSV, STAGE2_SPLIT_SEED


def main() -> None:
    df = pd.read_csv(GOLD_CSV, encoding="latin1")

    dev_df, eval_df = train_test_split(
        df,
        test_size=1 - STAGE2_DEV_FRACTION,
        stratify=df["Intent"],
        random_state=STAGE2_SPLIT_SEED,
    )

    dev_df.to_csv(STAGE2_DEV_CSV, index=False, encoding="utf-8-sig")
    eval_df.to_csv(STAGE2_EVAL_CSV, index=False, encoding="utf-8-sig")

    print(f"dev (prompt-iteration only, never reported): {len(dev_df)} rows -> {STAGE2_DEV_CSV}")
    print(dev_df["Intent"].value_counts(), "\n")
    print(f"eval (touch once per locked prompt config): {len(eval_df)} rows -> {STAGE2_EVAL_CSV}")
    print(eval_df["Intent"].value_counts())


if __name__ == "__main__":
    main()
