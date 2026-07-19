import argparse
import inspect
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import (
    DIALOGUE_LABELS,
    ID2LABEL,
    LABEL2ID,
    SEEDS,
    STRATEGY_B_SIZES,
    STRATEGY_B_DIR,
    STRATEGY_C_DIR,
    TRAINING,
)
from .data_loader import build_model_input, load_gold_data
from .evaluate import compute_metrics, save_confusion_matrix, save_metrics, summarize_runs
from .utils import ensure_results_dirs, resolve_model_path, set_seed


LABEL_COLUMN_CANDIDATES = ("pred_label", "Dialogue_act", "dialogue_act", "label", "da_label")


def _infer_label_column(df: pd.DataFrame) -> str:
    for column in LABEL_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Could not infer silver label column. Expected one of: "
        f"{list(LABEL_COLUMN_CANDIDATES)}"
    )


def _prepare_labeled_frame(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    out = df.copy()
    if {"parent", "reply"}.issubset(out.columns) and not {"Parent", "Reply"}.issubset(out.columns):
        out = out.rename(columns={"parent": "Parent", "reply": "Reply"})
    if not {"Parent", "Reply"}.issubset(out.columns):
        if "model_input" not in out.columns:
            raise ValueError("Input data needs Parent/Reply columns, or model_input as fallback.")
        split_text = out["model_input"].fillna("").astype(str).str.split(" [SEP] ", n=1, expand=True)
        out["Parent"] = split_text[0]
        out["Reply"] = split_text[1] if split_text.shape[1] > 1 else ""
    if "model_input" not in out.columns:
        out["model_input"] = build_model_input(out)
    out[label_col] = out[label_col].astype(str).str.strip().str.lower()
    unknown = sorted(set(out[label_col]) - set(DIALOGUE_LABELS))
    if unknown:
        raise ValueError(f"Unexpected labels in {label_col}: {unknown}")
    out["label_id"] = out[label_col].map(LABEL2ID)
    return out


def _dry_run_predictions(train_df: pd.DataFrame, eval_df: pd.DataFrame, label_col: str) -> List[str]:
    majority = train_df[label_col].mode().iloc[0]
    return [majority] * len(eval_df)


def _make_inner_validation_split(
    train_df: pd.DataFrame,
    label_col: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify = train_df[label_col] if train_df[label_col].value_counts().min() >= 2 else None
    inner_train, inner_val = train_test_split(
        train_df,
        test_size=TRAINING.validation_size,
        random_state=seed,
        stratify=stratify,
    )
    return inner_train.reset_index(drop=True), inner_val.reset_index(drop=True)


def _training_args(output_dir: Path, seed: int, TrainingArguments):
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": TRAINING.learning_rate,
        "per_device_train_batch_size": TRAINING.batch_size,
        "per_device_eval_batch_size": TRAINING.batch_size,
        "num_train_epochs": TRAINING.epochs,
        "weight_decay": TRAINING.weight_decay,
        "optim": TRAINING.optimizer,
        "report_to": [],
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "seed": seed,
        "data_seed": seed,
    }
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs[eval_key] = "epoch"
    if "warmup_steps" in params:
        kwargs["warmup_steps"] = TRAINING.warmup_steps
    if "logging_strategy" in params:
        kwargs["logging_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def _compute_class_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["label_id"].value_counts().reindex(range(len(DIALOGUE_LABELS)), fill_value=0)
    safe_counts = counts.clip(lower=1)
    weights = len(frame) / (len(DIALOGUE_LABELS) * safe_counts)
    return weights.to_numpy(dtype=np.float32)


def _freeze_backbone_layers(model, trainable_layers: int) -> Tuple[int, int]:
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    encoder_layers = model.roberta.encoder.layer
    num_freeze = max(0, len(encoder_layers) - trainable_layers)
    for layer in encoder_layers[:num_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _train_and_predict(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    seed: int,
    train_label_col: str,
    freeze_layers: int | None = None,
) -> List[str]:
    set_seed(seed)
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    model_path = resolve_model_path(TRAINING.model_name, TRAINING.model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inner_train_df, inner_val_df = _make_inner_validation_split(train_df, train_label_col, seed)

    class TextDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, has_labels: bool = True):
            self.parents = frame["Parent"].fillna("").astype(str).tolist()
            self.replies = frame["Reply"].fillna("").astype(str).tolist()
            self.labels = frame["label_id"].tolist() if has_labels else None

        def __len__(self) -> int:
            return len(self.parents)

        def __getitem__(self, idx: int) -> Dict:
            encoded = tokenizer(
                self.parents[idx],
                self.replies[idx],
                truncation=True,
                padding="max_length",
                max_length=TRAINING.max_length,
            )
            if self.labels is not None:
                encoded["labels"] = int(self.labels[idx])
            return encoded

    def trainer_compute_metrics(eval_pred) -> Dict[str, float]:
        logits, label_ids = eval_pred
        pred_ids = np.argmax(logits, axis=1)
        pred_labels = [ID2LABEL[int(idx)] for idx in pred_ids]
        gold_labels = [ID2LABEL[int(idx)] for idx in label_ids]
        return {
            "macro_f1": f1_score(
                gold_labels,
                pred_labels,
                labels=DIALOGUE_LABELS,
                average="macro",
                zero_division=0,
            ),
            "cohen_kappa": cohen_kappa_score(
                gold_labels,
                pred_labels,
                labels=DIALOGUE_LABELS,
            ),
            "accuracy": accuracy_score(gold_labels, pred_labels),
        }

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
            )
            loss = loss_fct(logits.view(-1, len(DIALOGUE_LABELS)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(DIALOGUE_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if freeze_layers is not None:
        trainable, total = _freeze_backbone_layers(model, freeze_layers)
        print(
            f"Partial fine-tuning: {trainable:,}/{total:,} parameters trainable "
            f"({trainable / total:.1%}), last {freeze_layers} encoder layers unfrozen."
        )

    tmp_root = Path("results") / "tmp_trainer"
    tmp_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix=f"roberta_seed{seed}_", dir=tmp_root))
    args = _training_args(output_dir, seed, TrainingArguments)

    class_weights = None
    if TRAINING.use_class_weights:
        class_weights = torch.tensor(_compute_class_weights(inner_train_df), dtype=torch.float32)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=TextDataset(inner_train_df),
        eval_dataset=TextDataset(inner_val_df),
        compute_metrics=trainer_compute_metrics,
    )
    try:
        trainer.train()
        predictions = trainer.predict(TextDataset(eval_df, has_labels=False)).predictions
        pred_ids = np.argmax(predictions, axis=1)
        return [ID2LABEL[int(idx)] for idx in pred_ids]
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(output_dir, ignore_errors=True)


def _run_once(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    seed: int,
    train_label_col: str,
    eval_label_col: str,
    dry_run: bool,
    freeze_layers: int | None = None,
) -> Tuple[List[str], Dict]:
    predictions = (
        _dry_run_predictions(train_df, eval_df, train_label_col)
        if dry_run
        else _train_and_predict(train_df, eval_df, seed, train_label_col, freeze_layers)
    )
    metrics = compute_metrics(eval_df[eval_label_col], predictions)
    return predictions, metrics


def run_strategy_b(
    silver_csv: Path,
    sizes: List[int],
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    ensure_results_dirs()
    seeds = seeds or SEEDS
    gold = _prepare_labeled_frame(load_gold_data(), "Dialogue_act")
    silver = pd.read_csv(silver_csv, encoding="latin1")
    label_col = _infer_label_column(silver)
    silver = _prepare_labeled_frame(silver, label_col)

    rows: List[Dict] = []
    for size in sizes:
        if size > len(silver):
            raise ValueError(
                f"Requested silver size {size}, but {silver_csv} only has {len(silver)} rows."
            )
        train_base = silver.head(size).copy()
        for seed in seeds:
            set_seed(seed)
            predictions, metrics = _run_once(
                train_base,
                gold,
                seed,
                label_col,
                "Dialogue_act",
                dry_run,
            )
            metrics.update({"strategy": "B", "silver_size": size, "seed": seed, "dry_run": dry_run})
            metrics.update({"use_class_weights": TRAINING.use_class_weights})

            pred_df = gold[["Parent", "Reply", "Dialogue_act"]].copy()
            pred_df["pred_label"] = predictions
            pred_df.to_csv(STRATEGY_B_DIR / f"silver_{size}_seed{seed}_predictions.csv", index=False)
            save_metrics(metrics, STRATEGY_B_DIR / f"silver_{size}_seed{seed}_metrics.json")
            save_confusion_matrix(metrics, STRATEGY_B_DIR / f"silver_{size}_seed{seed}_confusion_matrix.csv")
            rows.append(
                {
                    "silver_size": size,
                    "seed": seed,
                    "macro_f1": metrics["macro_f1"],
                    "cohen_kappa": metrics["cohen_kappa"],
                }
            )

    return summarize_runs(rows, ["silver_size"], STRATEGY_B_DIR / "summary.csv")


def _clear_strategy_c_outputs() -> None:
    STRATEGY_C_DIR.mkdir(parents=True, exist_ok=True)
    for path in STRATEGY_C_DIR.glob("fold*_seed*_*"):
        if path.is_file():
            path.unlink()
    for filename in ("summary.csv", "summary_by_fold.csv"):
        path = STRATEGY_C_DIR / filename
        if path.exists():
            path.unlink()


def _summarize_strategy_c(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = ["macro_f1", "cohen_kappa"]
    by_fold = (
        df.groupby("fold", dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    by_fold.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in by_fold.columns
    ]
    STRATEGY_C_DIR.mkdir(parents=True, exist_ok=True)
    by_fold.to_csv(STRATEGY_C_DIR / "summary_by_fold.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "folds": int(df["fold"].nunique()),
                "seeds": ",".join(str(seed) for seed in sorted(df["seed"].unique())),
                "runs": int(len(df)),
                "macro_f1_mean": df["macro_f1"].mean(),
                "macro_f1_std": df["macro_f1"].std(),
                "cohen_kappa_mean": df["cohen_kappa"].mean(),
                "cohen_kappa_std": df["cohen_kappa"].std(),
                "dry_run": bool(df["dry_run"].any()),
                "use_class_weights": bool(TRAINING.use_class_weights),
            }
        ]
    )
    summary.to_csv(STRATEGY_C_DIR / "summary.csv", index=False)
    return summary


def run_strategy_c(
    seeds: List[int] | None = None,
    dry_run: bool = False,
    freeze_layers: int | None = None,
) -> pd.DataFrame:
    ensure_results_dirs()
    seeds = seeds or SEEDS
    if not dry_run:
        _clear_strategy_c_outputs()
    gold = _prepare_labeled_frame(load_gold_data(), "Dialogue_act")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows: List[Dict] = []
    for fold, (train_idx, eval_idx) in enumerate(
        splitter.split(gold["model_input"], gold["Dialogue_act"]),
        start=1,
    ):
        train_df = gold.iloc[train_idx].reset_index(drop=True)
        eval_df = gold.iloc[eval_idx].reset_index(drop=True)
        for seed in seeds:
            set_seed(seed)
            predictions, metrics = _run_once(
                train_df,
                eval_df,
                seed,
                "Dialogue_act",
                "Dialogue_act",
                dry_run,
                freeze_layers,
            )
            metrics.update({"strategy": "C", "fold": fold, "seed": seed, "dry_run": dry_run, "freeze_layers": freeze_layers})
            metrics.update({"use_class_weights": TRAINING.use_class_weights})

            pred_df = eval_df[["Parent", "Reply", "Dialogue_act"]].copy()
            pred_df["pred_label"] = predictions
            pred_df.to_csv(STRATEGY_C_DIR / f"fold{fold}_seed{seed}_predictions.csv", index=False)
            save_metrics(metrics, STRATEGY_C_DIR / f"fold{fold}_seed{seed}_metrics.json")
            save_confusion_matrix(metrics, STRATEGY_C_DIR / f"fold{fold}_seed{seed}_confusion_matrix.csv")
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "macro_f1": metrics["macro_f1"],
                    "cohen_kappa": metrics["cohen_kappa"],
                    "dry_run": dry_run,
                }
            )

    return _summarize_strategy_c(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Strategy B or C RoBERTa experiments.")
    subparsers = parser.add_subparsers(dest="strategy", required=True)

    b_parser = subparsers.add_parser("strategy_b")
    b_parser.add_argument("--silver-csv", type=Path, required=True)
    b_parser.add_argument("--sizes", type=int, nargs="+", default=STRATEGY_B_SIZES)
    b_parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    b_parser.add_argument("--dry-run", action="store_true")

    c_parser = subparsers.add_parser("strategy_c")
    c_parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    c_parser.add_argument("--dry-run", action="store_true")
    c_parser.add_argument(
        "--trainable-layers",
        type=int,
        default=None,
        help="If set, freeze embeddings and all but the last N encoder layers (partial fine-tuning).",
    )

    args = parser.parse_args()
    if args.strategy == "strategy_b":
        summary = run_strategy_b(args.silver_csv, args.sizes, args.seeds, args.dry_run)
    else:
        summary = run_strategy_c(args.seeds, args.dry_run, args.trainable_layers)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
