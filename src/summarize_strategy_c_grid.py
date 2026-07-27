import argparse
import re
from pathlib import Path

import pandas as pd

from .config import STRATEGY_C_DIR

DIR_PATTERN = re.compile(r"^(?P<model_type>[a-z0-9]+)_(?P<weight_tag>weights|no_weights)_warmup(?P<warmup_steps>\d+)_epochs(?P<epochs>\d+)$")


def collect_grid_results(result_dir: Path = STRATEGY_C_DIR) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(result_dir.glob("*_warmup*_epochs*/summary_oof.csv")):
        run_dir = summary_path.parent
        match = DIR_PATTERN.match(run_dir.name)
        if match is None:
            continue
        summary = pd.read_csv(summary_path)
        if len(summary) != 1:
            raise ValueError(f"Expected exactly one summary row in {summary_path}.")
        row = summary.iloc[0].to_dict()
        row["run_dir"] = run_dir.name
        row["warmup_steps"] = int(match["warmup_steps"])
        row["epochs"] = int(match["epochs"])
        rows.append(row)

    if not rows:
        raise ValueError(f"No warmup/epochs grid results found under {result_dir}.")

    table = pd.DataFrame(rows)
    duplicate_key = ["model_type", "use_class_weights", "warmup_steps", "epochs"]
    if table.duplicated(duplicate_key).any():
        raise ValueError("Duplicate warmup/epochs grid configurations were found.")

    ordered_cols = [
        "run_dir",
        "warmup_steps",
        "epochs",
        "use_class_weights",
        "macro_f1_mean",
        "macro_f1_std",
        "cohen_kappa_mean",
        "cohen_kappa_std",
        "oof_class_collapse_seeds",
    ]
    remaining_cols = [col for col in table.columns if col not in ordered_cols]
    table = table[ordered_cols + remaining_cols]
    return table.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Strategy C's warmup/epochs grid results, best config first."
    )
    parser.add_argument("--result-dir", type=Path, default=STRATEGY_C_DIR)
    args = parser.parse_args()

    table = collect_grid_results(args.result_dir)
    display_cols = [
        "run_dir",
        "warmup_steps",
        "epochs",
        "use_class_weights",
        "macro_f1_mean",
        "macro_f1_std",
        "oof_class_collapse_seeds",
    ]
    print(table[display_cols].to_string(index=False))

    csv_path = args.result_dir / "warmup_epochs_grid_summary.csv"
    table.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
