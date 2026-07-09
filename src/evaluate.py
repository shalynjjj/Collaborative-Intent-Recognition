import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from .config import DIALOGUE_LABELS


def compute_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: Optional[List[str]] = None,
    fallback_count: int = 0,
) -> Dict:
    label_names = labels or DIALOGUE_LABELS
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=label_names, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=label_names),
        "fallback_count": int(fallback_count),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_names).tolist(),
        "labels": label_names,
    }


def save_metrics(metrics: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_confusion_matrix(metrics: Dict, output_path: Path) -> None:
    labels = metrics["labels"]
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path)


def summarize_runs(rows: List[Dict], group_cols: List[str], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [col for col in ("macro_f1", "cohen_kappa") if col in df.columns]
    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary

