import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from .config import STRATEGY_B_DIR


LEARNING_CURVE_SIZES = [500, 1000, 1500, 2000, 2500, 5000, 8000, 10000]
STRATEGY_A_ZERO_SHOT = 0.5147993742785979
STRATEGY_A_FEW_SHOT = 0.5916267942583732


def _default_label(result_dir: Path) -> str:
    prefix = "strategy_b_"
    name = result_dir.name
    return name[len(prefix):] if name.startswith(prefix) else name


def collect_learning_curve(
    result_dir: Path,
    expected_sizes: Iterable[int] = LEARNING_CURVE_SIZES,
) -> pd.DataFrame:
    rows = []
    for path in sorted(result_dir.glob("silver_*_sample*_train*_weights1_metrics.json")):
        if "dryrun" in path.name:
            continue
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        if (
            metrics.get("strategy") == "B"
            and metrics.get("experiment_version") == "v2_group_split"
            and metrics.get("use_class_weights") is True
            and metrics.get("dry_run") is False
        ):
            rows.append(
                {
                    "silver_size": int(metrics["silver_size"]),
                    "sample_seed": int(metrics["sample_seed"]),
                    "train_seed": int(metrics["train_seed"]),
                    "macro_f1": float(metrics["macro_f1"]),
                }
            )

    runs = pd.DataFrame(rows)
    if runs.empty:
        raise ValueError(f"No standardized Strategy B metrics found under {result_dir}.")
    duplicate_key = ["silver_size", "sample_seed", "train_seed"]
    if runs.duplicated(duplicate_key).any():
        raise ValueError("Duplicate standardized Strategy B run keys were found.")

    expected_sizes = list(expected_sizes)
    available = set(runs["silver_size"])
    missing = [size for size in expected_sizes if size not in available]
    if missing:
        raise ValueError(f"Missing standardized Strategy B sizes: {missing}")

    selected = runs[runs["silver_size"].isin(expected_sizes)]
    table = (
        selected.groupby("silver_size", as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_runs=("macro_f1", "size"),
        )
        .sort_values("silver_size")
        .reset_index(drop=True)
    )
    return table


def save_learning_curve(table: pd.DataFrame, result_dir: Path, label: str) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = result_dir / f"learning_curve_{label}.csv"
    plot_path = result_dir / f"learning_curve_{label}.png"
    table.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        table["silver_size"],
        table["macro_f1_mean"],
        yerr=table["macro_f1_std"].fillna(0),
        marker="o",
        linewidth=2,
        capsize=4,
        label=f"Strategy B: RoBERTa on {label} silver labels",
    )
    axis.axhline(
        STRATEGY_A_ZERO_SHOT,
        color="tab:orange",
        linestyle="--",
        label=f"Strategy A zero-shot ({STRATEGY_A_ZERO_SHOT:.4f})",
    )
    axis.axhline(
        STRATEGY_A_FEW_SHOT,
        color="tab:green",
        linestyle=":",
        label=f"Strategy A few-shot ({STRATEGY_A_FEW_SHOT:.4f})",
    )
    axis.set_xlabel("Number of silver training samples")
    axis.set_ylabel("Macro F1 on the 300-sample gold benchmark")
    axis.set_title(f"Strategy B Learning Curve ({label} silver labels)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the standardized Strategy B learning curve.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=STRATEGY_B_DIR,
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Suffix for output filenames, e.g. 'fewshot'. Defaults to the result-dir name.",
    )
    args = parser.parse_args()
    label = args.label or _default_label(args.result_dir)
    table = collect_learning_curve(args.result_dir)
    csv_path, plot_path = save_learning_curve(table, args.result_dir, label)
    print(table.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
