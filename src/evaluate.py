import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import DIALOGUE_LABELS, EMOTION_LABELS


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


def bootstrap_ci(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: Optional[List[str]] = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict:
    """Percentile bootstrap CI for macro-F1 / kappa over a fixed prediction set.

    Resamples (y_true, y_pred) row pairs with replacement to estimate how much
    the reported score would move under a different draw of the same-size
    eval set. This is evaluation-sample variance, not model-refit variance
    (there is nothing to retrain for a fixed LLM prompt/output pair) -- it
    does not require calling the model or re-reading the source data again.
    """
    label_names = labels or DIALOGUE_LABELS
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    n = len(y_true)
    rng = np.random.default_rng(seed)

    f1_samples = np.empty(n_boot)
    kappa_samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_samples[i] = f1_score(
            y_true[idx], y_pred[idx], labels=label_names, average="macro", zero_division=0
        )
        kappa_samples[i] = cohen_kappa_score(y_true[idx], y_pred[idx], labels=label_names)

    lo, hi = np.percentile(f1_samples, [2.5, 97.5])
    kappa_lo, kappa_hi = np.percentile(kappa_samples, [2.5, 97.5])
    return {
        "n_boot": n_boot,
        "boot_seed": seed,
        "macro_f1_boot_mean": float(f1_samples.mean()),
        "macro_f1_boot_std": float(f1_samples.std(ddof=1)),
        "macro_f1_ci95": [float(lo), float(hi)],
        "cohen_kappa_boot_mean": float(kappa_samples.mean()),
        "cohen_kappa_boot_std": float(kappa_samples.std(ddof=1)),
        "cohen_kappa_ci95": [float(kappa_lo), float(kappa_hi)],
    }


def compute_emotion_metrics(
    df: pd.DataFrame,
    labels: Optional[List[str]] = None,
    gold_prefix: str = "gold_",
    pred_prefix: str = "emotion_",
) -> Dict:
    """Per-category precision/recall/F1 for the 6 emotion columns, then
    macro-averaged. A row can match zero, one, or several categories at
    once, so this can't reuse compute_metrics' single-label confusion matrix.
    """
    label_names = labels or EMOTION_LABELS
    per_category = {}
    for label in label_names:
        y_true = df[f"{gold_prefix}{label.lower()}"].astype(bool)
        y_pred = df[f"{pred_prefix}{label.lower()}"].astype(bool)
        per_category[label] = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "support": int(y_true.sum()),
        }
    macro_f1 = float(np.mean([category["f1"] for category in per_category.values()]))
    return {"per_category": per_category, "macro_f1": macro_f1, "labels": label_names}


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

