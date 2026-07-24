import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import DIALOGUE_LABELS, RESULTS_DIR, STRATEGY_A_DIR
from .evaluate import compute_metrics
from .llm_annotate import exclude_fewshot_examples


EXPECTED_COMMON_ROWS = 296


def _common_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="latin1")
    required = {"Parent", "Reply", "pred_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    filtered = exclude_fewshot_examples(frame)
    if len(filtered) != EXPECTED_COMMON_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_COMMON_ROWS} common rows in {path}, got {len(filtered)}."
        )
    return filtered


def _score_predictions(path: Path) -> Dict:
    frame = _common_predictions(path)
    gold_col = "gold_label" if "gold_label" in frame.columns else "Dialogue_act"
    if gold_col not in frame.columns:
        raise ValueError(f"No gold-label column found in {path}.")
    metrics = compute_metrics(frame[gold_col], frame["pred_label"])
    report = metrics["classification_report"]
    return {
        "n_samples": len(frame),
        "macro_f1": metrics["macro_f1"],
        "cohen_kappa": metrics["cohen_kappa"],
        "accuracy": report["accuracy"],
        **{f"{label}_f1": report[label]["f1-score"] for label in DIALOGUE_LABELS},
    }


def evaluate_strategy_a(output_dir: Path) -> pd.DataFrame:
    rows = []
    for mode in ("zeroshot", "fewshot"):
        path = STRATEGY_A_DIR / f"{mode}_predictions.csv"
        row = {"strategy": "A", "configuration": mode, "predictions_file": path.name}
        row.update(_score_predictions(path))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "strategy_a_common_296.csv", index=False)
    return table


def evaluate_strategy_b(result_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for metrics_path in sorted(result_dir.glob("silver_*_sample*_train*_weights*_metrics.json")):
        with metrics_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("strategy") != "B"
            or metadata.get("experiment_version") != "v2_group_split"
            or metadata.get("dry_run") is not False
        ):
            continue

        predictions_path = metrics_path.with_name(
            metrics_path.name.replace("_metrics.json", "_predictions.csv")
        )
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions for {metrics_path}: {predictions_path}")
        row = {
            "strategy": "B",
            "silver_size": int(metadata["silver_size"]),
            "sample_seed": int(metadata["sample_seed"]),
            "train_seed": int(metadata["train_seed"]),
            "use_class_weights": bool(metadata["use_class_weights"]),
            "predictions_file": predictions_path.name,
        }
        row.update(_score_predictions(predictions_path))
        rows.append(row)

    runs = pd.DataFrame(rows)
    if runs.empty:
        raise ValueError(f"No standardized Strategy B prediction files found under {result_dir}.")
    runs = runs.sort_values(
        ["silver_size", "use_class_weights", "sample_seed", "train_seed"]
    ).reset_index(drop=True)
    runs.to_csv(result_dir / "common_296_runs.csv", index=False)

    summary = (
        runs.groupby(["silver_size", "use_class_weights"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            cohen_kappa_mean=("cohen_kappa", "mean"),
            cohen_kappa_std=("cohen_kappa", "std"),
            n_runs=("macro_f1", "size"),
            **{
                f"{label}_f1_mean": (f"{label}_f1", "mean")
                for label in DIALOGUE_LABELS
            },
        )
        .sort_values(["silver_size", "use_class_weights"])
        .reset_index(drop=True)
    )
    summary.to_csv(result_dir / "common_296_summary.csv", index=False)
    return runs, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore Strategy A and Strategy B on the shared 296-row benchmark."
    )
    parser.add_argument(
        "--strategy-b-dir",
        type=Path,
        default=RESULTS_DIR / "strategy_b_fewshot",
    )
    args = parser.parse_args()
    args.strategy_b_dir.mkdir(parents=True, exist_ok=True)

    strategy_a = evaluate_strategy_a(args.strategy_b_dir)
    _, strategy_b = evaluate_strategy_b(args.strategy_b_dir)
    print("Strategy A on common 296 rows:")
    print(strategy_a.to_string(index=False))
    print("\nStrategy B on common 296 rows:")
    print(strategy_b.to_string(index=False))


if __name__ == "__main__":
    main()
