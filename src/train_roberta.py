import argparse
import inspect
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split

from .config import (
    DIALOGUE_LABELS,
    ID2LABEL,
    LABEL2ID,
    RESULTS_DIR,
    SAMPLE_SEEDS,
    SEEDS,
    STRATEGY_B_FINAL_CONFIG,
    STRATEGY_B_FINAL_SILVER_CSV,
    STRATEGY_B_SIZES,
    STRATEGY_B_DIR,
    STRATEGY_B_HELDOUT_DIR,
    STRATEGY_C_DIR,
    STRATEGY_C_FINAL_CONFIG,
    STRATEGY_C_HELDOUT_DIR,
    TEST_CSV,
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
    split_seed: int,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if group_col is not None:
        if group_col not in train_df.columns:
            raise ValueError(
                f"Strategy B requires '{group_col}' for group-aware validation splitting."
            )
        if train_df[group_col].isna().any():
            raise ValueError(f"Strategy B cannot group-split rows with missing '{group_col}'.")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TRAINING.validation_size,
            random_state=split_seed,
        )
        train_idx, val_idx = next(
            splitter.split(train_df, groups=train_df[group_col].astype(str))
        )
        inner_train = train_df.iloc[train_idx]
        inner_val = train_df.iloc[val_idx]
        overlap = set(inner_train[group_col].astype(str)) & set(inner_val[group_col].astype(str))
        if overlap:
            raise AssertionError(
                f"Group split leaked {len(overlap)} {group_col} values into both partitions."
            )
        return inner_train.reset_index(drop=True), inner_val.reset_index(drop=True)

    stratify = train_df[label_col] if train_df[label_col].value_counts().min() >= 2 else None
    inner_train, inner_val = train_test_split(
        train_df,
        test_size=TRAINING.validation_size,
        random_state=split_seed,
        stratify=stratify,
    )
    return inner_train.reset_index(drop=True), inner_val.reset_index(drop=True)


def _split_metadata(
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
    group_col: Optional[str],
) -> Dict:
    metadata = {
        "train_rows": int(len(inner_train_df)),
        "validation_rows": int(len(inner_val_df)),
    }
    if group_col is not None:
        train_groups = set(inner_train_df[group_col].astype(str))
        val_groups = set(inner_val_df[group_col].astype(str))
        metadata.update(
            {
                "group_column": group_col,
                "train_source_root_count": len(train_groups),
                "validation_source_root_count": len(val_groups),
                "source_root_overlap_count": len(train_groups & val_groups),
            }
        )
    return metadata


def _training_args(output_dir: Path, seed: int, TrainingArguments, overrides: Optional[Dict] = None):
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
    # Diagnostic-only overrides (e.g. warmup_steps, num_train_epochs). Never
    # set by Strategy A/B or the normal Strategy C path -- only
    # diagnose_strategy_c_train_fit passes these, so default behavior for
    # every other caller is unchanged.
    if overrides:
        kwargs.update(overrides)
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
    freeze_layers: Optional[int] = None,
    split_seed: Optional[int] = None,
    group_col: Optional[str] = None,
    use_class_weights: Optional[bool] = None,
    also_predict_train: bool = False,
    training_overrides: Optional[Dict] = None,
) -> Tuple[List[str], Dict, Optional[Dict]]:
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
    inner_train_df, inner_val_df = _make_inner_validation_split(
        train_df,
        train_label_col,
        seed if split_seed is None else split_seed,
        group_col,
    )
    split_metadata = _split_metadata(inner_train_df, inner_val_df, group_col)

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
    args = _training_args(output_dir, seed, TrainingArguments, training_overrides)

    class_weights = None
    weights_enabled = TRAINING.use_class_weights if use_class_weights is None else use_class_weights
    if weights_enabled:
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
        pred_labels = [ID2LABEL[int(idx)] for idx in pred_ids]

        train_metrics = None
        if also_predict_train:
            train_predictions = trainer.predict(TextDataset(inner_train_df, has_labels=False)).predictions
            train_pred_ids = np.argmax(train_predictions, axis=1)
            train_pred_labels = [ID2LABEL[int(idx)] for idx in train_pred_ids]
            train_metrics = compute_metrics(inner_train_df[train_label_col], train_pred_labels)

        return pred_labels, split_metadata, train_metrics
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
    freeze_layers: Optional[int] = None,
    split_seed: Optional[int] = None,
    group_col: Optional[str] = None,
    use_class_weights: Optional[bool] = None,
    also_predict_train: bool = False,
    training_overrides: Optional[Dict] = None,
) -> Tuple[List[str], Dict, Dict]:
    train_metrics = None
    if dry_run:
        inner_train_df, inner_val_df = _make_inner_validation_split(
            train_df,
            train_label_col,
            seed if split_seed is None else split_seed,
            group_col,
        )
        split_metadata = _split_metadata(inner_train_df, inner_val_df, group_col)
        predictions = _dry_run_predictions(train_df, eval_df, train_label_col)
    else:
        predictions, split_metadata, train_metrics = _train_and_predict(
            train_df,
            eval_df,
            seed,
            train_label_col,
            freeze_layers,
            split_seed,
            group_col,
            use_class_weights,
            also_predict_train,
            training_overrides,
        )
    metrics = compute_metrics(eval_df[eval_label_col], predictions)
    if train_metrics is not None:
        metrics["train_fit_macro_f1"] = train_metrics["macro_f1"]
        metrics["train_fit_cohen_kappa"] = train_metrics["cohen_kappa"]
    return predictions, metrics, split_metadata


def _sample_silver_subset(silver: pd.DataFrame, size: int, sample_seed: int) -> pd.DataFrame:
    if size > len(silver):
        raise ValueError(f"Requested silver size {size}, but the pool has only {len(silver)} rows.")
    # One deterministic ordering per sample_seed keeps future learning-curve sizes nested.
    return silver.sample(frac=1, random_state=sample_seed).head(size).reset_index(drop=True)


def rebuild_strategy_b_summaries(results_dir: Optional[Path] = None) -> pd.DataFrame:
    """Rebuild Strategy B summaries from the union of all saved per-run metrics."""
    results_dir = results_dir or STRATEGY_B_DIR
    records: Dict[Tuple, Dict] = {}
    for path in sorted(results_dir.glob("*_metrics.json")):
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        if metrics.get("strategy") != "B" or "silver_size" not in metrics:
            continue
        is_legacy = "sample_seed" not in metrics or "train_seed" not in metrics
        prediction_counts = metrics.get("prediction_counts")
        if prediction_counts is None and metrics.get("confusion_matrix"):
            matrix = np.asarray(metrics["confusion_matrix"])
            prediction_counts = {
                label: int(matrix[:, idx].sum()) for idx, label in enumerate(DIALOGUE_LABELS)
            }
        prediction_counts = prediction_counts or {}
        inferred_collapse = bool(prediction_counts) and any(
            prediction_counts.get(label, 0) == 0 for label in DIALOGUE_LABELS
        )
        record = {
            "experiment_version": "legacy" if is_legacy else "v2_group_split",
            "silver_size": int(metrics["silver_size"]),
            "sample_seed": "legacy" if is_legacy else str(metrics["sample_seed"]),
            "train_seed": int(metrics.get("train_seed", metrics.get("seed", -1))),
            "use_class_weights": bool(metrics.get("use_class_weights", True)),
            "dry_run": bool(metrics.get("dry_run", False)),
            "macro_f1": float(metrics["macro_f1"]),
            "cohen_kappa": float(metrics["cohen_kappa"]),
            "class_collapse": bool(metrics.get("class_collapse", inferred_collapse)),
            "metrics_file": path.name,
        }
        report = metrics.get("classification_report", {})
        for label in DIALOGUE_LABELS:
            record[f"{label}_f1"] = float(report.get(label, {}).get("f1-score", np.nan))
            record[f"{label}_prediction_count"] = prediction_counts.get(label, np.nan)
        key = (
            record["experiment_version"],
            record["silver_size"],
            record["sample_seed"],
            record["train_seed"],
            record["use_class_weights"],
            record["dry_run"],
        )
        records[key] = record

    if not records:
        return pd.DataFrame()

    runs = pd.DataFrame(records.values()).sort_values(
        ["experiment_version", "silver_size", "sample_seed", "train_seed"]
    )
    runs.to_csv(results_dir / "runs.csv", index=False)

    def aggregate(group_cols: List[str], output_path: Path) -> pd.DataFrame:
        aggregations = {
            "runs": ("macro_f1", "size"),
            "macro_f1_mean": ("macro_f1", "mean"),
            "macro_f1_std": ("macro_f1", "std"),
            "cohen_kappa_mean": ("cohen_kappa", "mean"),
            "cohen_kappa_std": ("cohen_kappa", "std"),
            "class_collapse_runs": ("class_collapse", "sum"),
        }
        for label in DIALOGUE_LABELS:
            aggregations[f"{label}_f1_mean"] = (f"{label}_f1", "mean")
            aggregations[f"{label}_prediction_count_mean"] = (
                f"{label}_prediction_count",
                "mean",
            )
        result = runs.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()
        result.to_csv(output_path, index=False)
        return result

    summary = aggregate(
        ["experiment_version", "silver_size", "use_class_weights", "dry_run"],
        results_dir / "summary.csv",
    )
    aggregate(
        ["experiment_version", "silver_size", "sample_seed", "use_class_weights", "dry_run"],
        results_dir / "summary_by_sample.csv",
    )
    return summary


def run_strategy_b(
    silver_csv: Path,
    sizes: List[int],
    sample_seeds: Optional[List[int]] = None,
    train_seeds: Optional[List[int]] = None,
    dry_run: bool = False,
    use_class_weights: Optional[bool] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    ensure_results_dirs()
    results_dir = results_dir or STRATEGY_B_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_seeds = sample_seeds or SAMPLE_SEEDS
    train_seeds = train_seeds or SEEDS
    weights_enabled = TRAINING.use_class_weights if use_class_weights is None else use_class_weights
    gold = _prepare_labeled_frame(load_gold_data(), "Dialogue_act")
    silver = pd.read_csv(silver_csv, encoding="latin1")
    label_col = _infer_label_column(silver)
    silver = _prepare_labeled_frame(silver, label_col)

    for size in sizes:
        for sample_seed in sample_seeds:
            train_base = _sample_silver_subset(silver, size, sample_seed)
            for train_seed in train_seeds:
                set_seed(train_seed)
                predictions, metrics, split_metadata = _run_once(
                    train_base,
                    gold,
                    train_seed,
                    label_col,
                    "Dialogue_act",
                    dry_run,
                    split_seed=sample_seed,
                    group_col="source_root",
                    use_class_weights=weights_enabled,
                )
                prediction_counts = {
                    label: int(predictions.count(label)) for label in DIALOGUE_LABELS
                }
                missing_classes = [
                    label for label, count in prediction_counts.items() if count == 0
                ]
                metrics.update(
                    {
                        "strategy": "B",
                        "experiment_version": "v2_group_split",
                        "silver_size": size,
                        "sample_seed": sample_seed,
                        "train_seed": train_seed,
                        "dry_run": dry_run,
                        "use_class_weights": weights_enabled,
                        "prediction_counts": prediction_counts,
                        "missing_prediction_classes": missing_classes,
                        "class_collapse": bool(missing_classes),
                        **split_metadata,
                    }
                )

                weight_tag = "weights1" if weights_enabled else "weights0"
                run_tag = "_dryrun" if dry_run else ""
                prefix = (
                    f"silver_{size}_sample{sample_seed}_train{train_seed}_{weight_tag}{run_tag}"
                )
                pred_df = gold[["Parent", "Reply", "Dialogue_act"]].copy()
                pred_df["pred_label"] = predictions
                pred_df.to_csv(results_dir / f"{prefix}_predictions.csv", index=False)
                save_metrics(metrics, results_dir / f"{prefix}_metrics.json")
                save_confusion_matrix(
                    metrics,
                    results_dir / f"{prefix}_confusion_matrix.csv",
                )

    return rebuild_strategy_b_summaries(results_dir)


def run_strategy_b_heldout_eval(dry_run: bool = False) -> pd.DataFrame:
    """One-shot confirmatory eval of the locked Strategy B config on TEST_CSV.

    Trains on the single silver subset fixed by STRATEGY_B_FINAL_CONFIG and
    evaluates against TEST_CSV instead of GOLD_CSV. Takes no size/seed-list
    arguments on purpose -- this must not turn into a second sweep. Run once,
    after the silver-size sweep against GOLD_CSV is finalized.
    """
    ensure_results_dirs()
    STRATEGY_B_HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    config = STRATEGY_B_FINAL_CONFIG
    test_df = _prepare_labeled_frame(load_gold_data(TEST_CSV), "Dialogue_act")
    silver = pd.read_csv(STRATEGY_B_FINAL_SILVER_CSV, encoding="latin1")
    label_col = _infer_label_column(silver)
    silver = _prepare_labeled_frame(silver, label_col)

    # Pair-level dedup (prepare_gold_test_candidates.py) doesn't catch a single
    # utterance appearing as TEST_CSV's Reply and a different silver row's
    # Parent (or vice versa). Filter those out of the sampling pool so no
    # future silver_size/sample_seed choice can pull in a contaminated row.
    if "utt_id" in test_df.columns and {"parent_id", "reply_id"}.issubset(silver.columns):
        test_utt_ids = set(test_df["utt_id"])
        before = len(silver)
        silver = silver[
            ~silver["parent_id"].isin(test_utt_ids) & ~silver["reply_id"].isin(test_utt_ids)
        ]
        excluded_count = before - len(silver)
        if excluded_count:
            print(f"excluded {excluded_count} silver rows overlapping TEST_CSV utterance ids")

    train_base = _sample_silver_subset(silver, config["silver_size"], config["sample_seed"])

    rows = []
    for train_seed in SEEDS:
        set_seed(train_seed)
        predictions, metrics, split_metadata = _run_once(
            train_base,
            test_df,
            train_seed,
            label_col,
            "Dialogue_act",
            dry_run,
            split_seed=config["sample_seed"],
            group_col="source_root",
            use_class_weights=config["use_class_weights"],
        )
        prediction_counts = {label: int(predictions.count(label)) for label in DIALOGUE_LABELS}
        missing_classes = [label for label, count in prediction_counts.items() if count == 0]
        metrics.update(
            {
                "strategy": "B",
                "eval_set": "heldout_test",
                "silver_size": config["silver_size"],
                "sample_seed": config["sample_seed"],
                "train_seed": train_seed,
                "dry_run": dry_run,
                "use_class_weights": config["use_class_weights"],
                "prediction_counts": prediction_counts,
                "missing_prediction_classes": missing_classes,
                "class_collapse": bool(missing_classes),
                **split_metadata,
            }
        )

        run_tag = "_dryrun" if dry_run else ""
        prefix = f"heldout_seed{train_seed}{run_tag}"
        pred_df = test_df[["Parent", "Reply", "Dialogue_act"]].copy()
        pred_df["pred_label"] = predictions
        pred_df.to_csv(STRATEGY_B_HELDOUT_DIR / f"{prefix}_predictions.csv", index=False)
        save_metrics(metrics, STRATEGY_B_HELDOUT_DIR / f"{prefix}_metrics.json")
        save_confusion_matrix(metrics, STRATEGY_B_HELDOUT_DIR / f"{prefix}_confusion_matrix.csv")
        rows.append(
            {
                "train_seed": train_seed,
                "macro_f1": metrics["macro_f1"],
                "cohen_kappa": metrics["cohen_kappa"],
            }
        )

    summary = pd.DataFrame(rows)
    summary_name = "summary_dryrun.csv" if dry_run else "summary.csv"
    summary.to_csv(STRATEGY_B_HELDOUT_DIR / summary_name, index=False)
    return summary


def _strategy_c_output_dir(
    model_type: str,
    use_class_weights: bool,
    dry_run: bool,
    warmup_steps: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Path:
    weight_tag = "weights" if use_class_weights else "no_weights"
    dry_tag = "_dryrun" if dry_run else ""
    config_tag = ""
    if warmup_steps is not None:
        config_tag += f"_warmup{warmup_steps}"
    if epochs is not None:
        config_tag += f"_epochs{epochs}"
    return STRATEGY_C_DIR / f"{model_type}_{weight_tag}{config_tag}{dry_tag}"


def _prediction_diagnostics(predictions: List[str]) -> Tuple[Dict[str, int], List[str]]:
    counts = {label: int(predictions.count(label)) for label in DIALOGUE_LABELS}
    missing = [label for label, count in counts.items() if count == 0]
    return counts, missing


def _summarize_strategy_c_oof(
    output_dir: Path,
    fold_rows: List[Dict],
    predictions_by_seed: Dict[int, List[pd.DataFrame]],
    model_type: str,
    use_class_weights: bool,
    split_seed: int,
    dry_run: bool,
) -> pd.DataFrame:
    fold_df = pd.DataFrame(fold_rows).sort_values(["fold", "train_seed"])
    fold_df.to_csv(output_dir / "fold_runs.csv", index=False)

    seed_rows = []
    for train_seed, frames in sorted(predictions_by_seed.items()):
        oof = pd.concat(frames, ignore_index=True).sort_values("gold_index").reset_index(drop=True)
        if len(oof) != 300 or oof["gold_index"].nunique() != 300:
            raise AssertionError(
                f"Expected 300 unique OOF rows for train_seed={train_seed}, "
                f"got {len(oof)} rows and {oof['gold_index'].nunique()} unique indices."
            )
        metrics = compute_metrics(oof["Dialogue_act"], oof["pred_label"])
        counts, missing = _prediction_diagnostics(oof["pred_label"].tolist())
        metrics.update(
            {
                "strategy": "C",
                "aggregation": "seed_level_oof",
                "model_type": model_type,
                "train_seed": train_seed,
                "split_seed": split_seed,
                "outer_fold_seed": 42,
                "use_class_weights": use_class_weights,
                "dry_run": dry_run,
                "prediction_counts": counts,
                "missing_prediction_classes": missing,
                "class_collapse": bool(missing),
            }
        )
        oof.to_csv(output_dir / f"oof_seed{train_seed}_predictions.csv", index=False)
        save_metrics(metrics, output_dir / f"oof_seed{train_seed}_metrics.json")
        seed_row = {
            "train_seed": train_seed,
            "macro_f1": metrics["macro_f1"],
            "cohen_kappa": metrics["cohen_kappa"],
            "class_collapse": bool(missing),
            "fold_collapse_runs": int(
                fold_df.loc[fold_df["train_seed"] == train_seed, "class_collapse"].sum()
            ),
        }
        for label in DIALOGUE_LABELS:
            seed_row[f"{label}_f1"] = metrics["classification_report"][label]["f1-score"]
            seed_row[f"{label}_prediction_count"] = counts[label]
        seed_rows.append(seed_row)

    by_seed = pd.DataFrame(seed_rows).sort_values("train_seed")
    by_seed.to_csv(output_dir / "summary_by_seed.csv", index=False)
    summary_row = {
        "model_type": model_type,
        "use_class_weights": use_class_weights,
        "dry_run": dry_run,
        "split_seed": split_seed,
        "outer_fold_seed": 42,
        "seeds": ",".join(str(seed) for seed in by_seed["train_seed"]),
        "seed_runs": int(len(by_seed)),
        "macro_f1_mean": by_seed["macro_f1"].mean(),
        "macro_f1_std": by_seed["macro_f1"].std(),
        "cohen_kappa_mean": by_seed["cohen_kappa"].mean(),
        "cohen_kappa_std": by_seed["cohen_kappa"].std(),
        "oof_class_collapse_seeds": int(by_seed["class_collapse"].sum()),
        "fold_class_collapse_runs": int(fold_df["class_collapse"].sum()),
        "fold_runs": int(len(fold_df)),
    }
    for label in DIALOGUE_LABELS:
        summary_row[f"{label}_f1_mean"] = by_seed[f"{label}_f1"].mean()
        summary_row[f"{label}_prediction_count_mean"] = by_seed[
            f"{label}_prediction_count"
        ].mean()
        summary_row[f"folds_missing_{label}"] = int(
            fold_df["missing_prediction_classes"].fillna("").str.split(",").apply(
                lambda labels: label in labels
            ).sum()
        )
    summary = pd.DataFrame([summary_row])
    summary.to_csv(output_dir / "summary_oof.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def _tfidf_predictions(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_seed: int,
    use_class_weights: bool,
) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=train_seed,
                    class_weight="balanced" if use_class_weights else None,
                ),
            ),
        ]
    )
    pipeline.fit(train_df["model_input"], train_df["Dialogue_act"])
    return pipeline.predict(eval_df["model_input"]).tolist()


def run_strategy_c(
    seeds: Optional[List[int]] = None,
    dry_run: bool = False,
    model_type: str = "roberta",
    use_class_weights: Optional[bool] = None,
    split_seed: int = 42,
    warmup_steps: Optional[int] = None,
    epochs: Optional[int] = None,
) -> pd.DataFrame:
    """warmup_steps/epochs override TRAINING's roberta defaults for this run
    only (never touches the global TRAINING singleton), so Strategy B and any
    other Strategy C run without these flags are unaffected. Results are
    written to a config-tagged directory so they never mix with runs under
    different hyperparameters (see _strategy_c_output_dir)."""
    ensure_results_dirs()
    seeds = seeds or SEEDS
    weights_enabled = TRAINING.use_class_weights if use_class_weights is None else use_class_weights
    output_dir = _strategy_c_output_dir(model_type, weights_enabled, dry_run, warmup_steps, epochs)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_overrides = {}
    if warmup_steps is not None:
        training_overrides["warmup_steps"] = warmup_steps
    if epochs is not None:
        training_overrides["num_train_epochs"] = epochs
    gold = _prepare_labeled_frame(load_gold_data(), "Dialogue_act").reset_index(drop=True)
    gold["gold_index"] = gold.index
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_rows: List[Dict] = []
    predictions_by_seed: Dict[int, List[pd.DataFrame]] = {seed: [] for seed in seeds}
    for fold, (train_idx, eval_idx) in enumerate(
        splitter.split(gold["model_input"], gold["Dialogue_act"]),
        start=1,
    ):
        train_df = gold.iloc[train_idx].reset_index(drop=True)
        eval_df = gold.iloc[eval_idx].reset_index(drop=True)
        for train_seed in seeds:
            set_seed(train_seed)
            if model_type == "tfidf":
                predictions = _tfidf_predictions(
                    train_df, eval_df, train_seed, weights_enabled
                )
                metrics = compute_metrics(eval_df["Dialogue_act"], predictions)
                split_metadata = {"train_rows": len(train_df), "validation_rows": 0}
            else:
                predictions, metrics, split_metadata = _run_once(
                    train_df,
                    eval_df,
                    train_seed,
                    "Dialogue_act",
                    "Dialogue_act",
                    dry_run,
                    split_seed=split_seed,
                    use_class_weights=weights_enabled,
                    training_overrides=training_overrides or None,
                )
            counts, missing = _prediction_diagnostics(predictions)
            metrics.update(
                {
                    "strategy": "C",
                    "model_type": model_type,
                    "fold": fold,
                    "train_seed": train_seed,
                    "split_seed": split_seed,
                    "outer_fold_seed": 42,
                    "dry_run": dry_run,
                    "use_class_weights": weights_enabled,
                    "warmup_steps": warmup_steps,
                    "epochs": epochs,
                    "prediction_counts": counts,
                    "missing_prediction_classes": missing,
                    "class_collapse": bool(missing),
                    **split_metadata,
                }
            )

            pred_df = eval_df[["gold_index", "Parent", "Reply", "Dialogue_act"]].copy()
            pred_df["pred_label"] = predictions
            predictions_by_seed[train_seed].append(pred_df)
            prefix = f"fold{fold}_seed{train_seed}"
            pred_df.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
            save_metrics(metrics, output_dir / f"{prefix}_metrics.json")
            save_confusion_matrix(metrics, output_dir / f"{prefix}_confusion_matrix.csv")
            fold_rows.append(
                {
                    "fold": fold,
                    "train_seed": train_seed,
                    "macro_f1": metrics["macro_f1"],
                    "cohen_kappa": metrics["cohen_kappa"],
                    "predicted_class_count": len(DIALOGUE_LABELS) - len(missing),
                    "missing_prediction_classes": ",".join(missing),
                    "class_collapse": bool(missing),
                }
            )

    return _summarize_strategy_c_oof(
        output_dir,
        fold_rows,
        predictions_by_seed,
        model_type,
        weights_enabled,
        split_seed,
        dry_run,
    )


def run_strategy_c_heldout_eval(dry_run: bool = False) -> pd.DataFrame:
    """One-shot confirmatory eval of the locked Strategy C config on TEST_CSV.

    Trains on the full 300-row gold set (no outer CV split -- CV was only
    needed to pick the config on GOLD_CSV) and evaluates against TEST_CSV
    instead. Takes no seed-list/model/weight arguments on purpose -- this
    must not turn into a second sweep. Run once, after STRATEGY_C_FINAL_CONFIG
    is locked.
    """
    ensure_results_dirs()
    STRATEGY_C_HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    config = STRATEGY_C_FINAL_CONFIG
    gold = _prepare_labeled_frame(load_gold_data(), "Dialogue_act").reset_index(drop=True)
    test_df = _prepare_labeled_frame(load_gold_data(TEST_CSV), "Dialogue_act")

    training_overrides = {}
    if config.get("warmup_steps") is not None:
        training_overrides["warmup_steps"] = config["warmup_steps"]
    if config.get("epochs") is not None:
        training_overrides["num_train_epochs"] = config["epochs"]

    rows = []
    for train_seed in SEEDS:
        set_seed(train_seed)
        if config["model_type"] == "tfidf":
            predictions = _tfidf_predictions(
                gold, test_df, train_seed, config["use_class_weights"]
            )
            metrics = compute_metrics(test_df["Dialogue_act"], predictions)
            split_metadata = {"train_rows": len(gold), "validation_rows": 0}
        else:
            predictions, metrics, split_metadata = _run_once(
                gold,
                test_df,
                train_seed,
                "Dialogue_act",
                "Dialogue_act",
                dry_run,
                split_seed=config["split_seed"],
                use_class_weights=config["use_class_weights"],
                training_overrides=training_overrides or None,
            )
        counts, missing = _prediction_diagnostics(predictions)
        metrics.update(
            {
                "strategy": "C",
                "eval_set": "heldout_test",
                "model_type": config["model_type"],
                "train_seed": train_seed,
                "dry_run": dry_run,
                "use_class_weights": config["use_class_weights"],
                "warmup_steps": config.get("warmup_steps"),
                "epochs": config.get("epochs"),
                "prediction_counts": counts,
                "missing_prediction_classes": missing,
                "class_collapse": bool(missing),
                **split_metadata,
            }
        )

        run_tag = "_dryrun" if dry_run else ""
        prefix = f"heldout_seed{train_seed}{run_tag}"
        pred_df = test_df[["Parent", "Reply", "Dialogue_act"]].copy()
        pred_df["pred_label"] = predictions
        pred_df.to_csv(STRATEGY_C_HELDOUT_DIR / f"{prefix}_predictions.csv", index=False)
        save_metrics(metrics, STRATEGY_C_HELDOUT_DIR / f"{prefix}_metrics.json")
        save_confusion_matrix(metrics, STRATEGY_C_HELDOUT_DIR / f"{prefix}_confusion_matrix.csv")
        rows.append(
            {
                "train_seed": train_seed,
                "macro_f1": metrics["macro_f1"],
                "cohen_kappa": metrics["cohen_kappa"],
            }
        )

    summary = pd.DataFrame(rows)
    summary_name = "summary_dryrun.csv" if dry_run else "summary.csv"
    summary.to_csv(STRATEGY_C_HELDOUT_DIR / summary_name, index=False)
    return summary


def diagnose_strategy_c_train_fit(
    fold: int = 1,
    train_seed: int = 42,
    use_class_weights: bool = True,
    split_seed: int = 42,
    save: bool = True,
    warmup_steps: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Dict:
    """Strategy C diagnostic: does RoBERTa fit its own training fold?

    Trains on one fold (same 5-fold split as run_strategy_c) and predicts
    both on the held-out fold (normal eval) and on the training rows
    themselves (also_predict_train). Low scores on both -> broken training
    config (bad LR/warmup/epochs), not "not enough data" -- a model this
    size should be able to fit ~240 rows if training is actually working.
    High train-fit but low eval -> genuine overfitting, which does support
    "300 gold rows isn't enough for full RoBERTa fine-tuning."

    warmup_steps/epochs override TRAINING's defaults for this call only --
    they are never touched globally, so Strategy A/B and the normal
    Strategy C path are unaffected regardless of what's passed here.

    Single fold/seed by design: this is a diagnostic, not a sweep. Not
    wired into run_strategy_c's normal loop.
    """
    gold = _prepare_labeled_frame(load_gold_data(), "Dialogue_act").reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(gold["model_input"], gold["Dialogue_act"]))
    train_idx, eval_idx = folds[fold - 1]
    train_df = gold.iloc[train_idx].reset_index(drop=True)
    eval_df = gold.iloc[eval_idx].reset_index(drop=True)

    overrides = {}
    if warmup_steps is not None:
        overrides["warmup_steps"] = warmup_steps
    if epochs is not None:
        overrides["num_train_epochs"] = epochs

    set_seed(train_seed)
    predictions, metrics, split_metadata = _run_once(
        train_df,
        eval_df,
        train_seed,
        "Dialogue_act",
        "Dialogue_act",
        dry_run=False,
        split_seed=split_seed,
        use_class_weights=use_class_weights,
        also_predict_train=True,
        training_overrides=overrides or None,
    )

    result = {
        "fold": fold,
        "train_seed": train_seed,
        "use_class_weights": use_class_weights,
        "warmup_steps": warmup_steps,
        "epochs": epochs,
        "train_rows": split_metadata["train_rows"],
        "validation_rows": split_metadata["validation_rows"],
        "eval_rows": len(eval_df),
        "train_fit_macro_f1": metrics.get("train_fit_macro_f1"),
        "train_fit_cohen_kappa": metrics.get("train_fit_cohen_kappa"),
        "eval_macro_f1": metrics["macro_f1"],
        "eval_cohen_kappa": metrics["cohen_kappa"],
    }
    print(json.dumps(result, indent=2))
    if save:
        output_dir = RESULTS_DIR / "strategy_c_diagnose"
        output_dir.mkdir(parents=True, exist_ok=True)
        weight_tag = "weights" if use_class_weights else "no_weights"
        output_path = output_dir / f"fold{fold}_seed{train_seed}_{weight_tag}.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved to: {output_path}")
    return result


def run_strategy_c_diagnose_sweep(
    folds: Optional[List[int]] = None,
    train_seeds: Optional[List[int]] = None,
    use_class_weights: bool = True,
    split_seed: int = 42,
    warmup_steps: Optional[int] = None,
    epochs: Optional[int] = None,
) -> pd.DataFrame:
    """Run diagnose_strategy_c_train_fit across every fold x seed combination.

    Default is all 5 folds x the project's standard SEEDS (42, 123, 2026) --
    same granularity as run_strategy_c's real sweep, so the "does this
    fold/seed combo overfit vs fail to fit" pattern can be judged from more
    than one anecdotal run before concluding whether Strategy C's problem is
    training instability or genuine data insufficiency.

    warmup_steps/epochs let this diagnostic be re-run with a fixed config
    change (e.g. warmup_steps=20, epochs=8) to check whether that stabilizes
    training, without touching the global TRAINING defaults used elsewhere.
    """
    folds = folds or [1, 2, 3, 4, 5]
    train_seeds = train_seeds or SEEDS

    rows = []
    for fold in folds:
        for train_seed in train_seeds:
            result = diagnose_strategy_c_train_fit(
                fold, train_seed, use_class_weights, split_seed, save=False,
                warmup_steps=warmup_steps, epochs=epochs,
            )
            rows.append(result)

    summary = pd.DataFrame(rows)
    summary["gap"] = summary["train_fit_macro_f1"] - summary["eval_macro_f1"]

    low_fit = int((summary["train_fit_macro_f1"] < 0.35).sum())
    correlation = summary["train_fit_macro_f1"].corr(summary["eval_macro_f1"])
    summary["low_train_fit_count"] = None
    summary["train_fit_eval_correlation"] = None
    stats_row = pd.DataFrame([{
        "fold": "AGGREGATE",
        "train_seed": f"n={len(summary)}",
        "train_fit_macro_f1": summary["train_fit_macro_f1"].mean(),
        "eval_macro_f1": summary["eval_macro_f1"].mean(),
        "gap": summary["gap"].mean(),
        "low_train_fit_count": low_fit,
        "train_fit_eval_correlation": round(correlation, 3),
    }])
    summary_with_stats = pd.concat([summary, stats_row], ignore_index=True)

    output_dir = RESULTS_DIR / "strategy_c_diagnose"
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_tag = "weights" if use_class_weights else "no_weights"
    config_tag = ""
    if warmup_steps is not None:
        config_tag += f"_warmup{warmup_steps}"
    if epochs is not None:
        config_tag += f"_epochs{epochs}"
    summary_path = output_dir / f"summary_{weight_tag}{config_tag}.csv"
    summary_with_stats.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nmean train_fit_macro_f1: {summary['train_fit_macro_f1'].mean():.3f}")
    print(f"mean eval_macro_f1: {summary['eval_macro_f1'].mean():.3f}")
    print(f"mean gap: {summary['gap'].mean():.3f}")
    print(f"train_fit/eval correlation: {correlation:.3f}")
    print(f"runs where train_fit itself is low (<0.35): {low_fit}/{len(summary)}")
    print(f"saved to: {summary_path} (aggregate stats in the last row)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Strategy B or C RoBERTa experiments.")
    subparsers = parser.add_subparsers(dest="strategy", required=True)

    b_parser = subparsers.add_parser("strategy_b")
    b_parser.add_argument("--silver-csv", type=Path, required=True)
    b_parser.add_argument("--sizes", type=int, nargs="+", default=STRATEGY_B_SIZES)
    b_parser.add_argument("--sample-seeds", type=int, nargs="+", default=SAMPLE_SEEDS)
    b_parser.add_argument("--train-seeds", type=int, nargs="+", default=SEEDS)
    b_parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=TRAINING.use_class_weights,
    )
    b_parser.add_argument("--dry-run", action="store_true")
    b_parser.add_argument("--results-dir", type=Path, default=None)

    b_heldout_parser = subparsers.add_parser(
        "strategy_b_heldout",
        help="One-shot eval of the locked STRATEGY_B_FINAL_CONFIG against TEST_CSV.",
    )
    b_heldout_parser.add_argument("--dry-run", action="store_true")

    c_parser = subparsers.add_parser("strategy_c")
    c_parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    c_parser.add_argument("--dry-run", action="store_true")
    c_parser.add_argument(
        "--model",
        choices=("roberta", "tfidf"),
        default="roberta",
    )
    c_parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=TRAINING.use_class_weights,
    )
    c_parser.add_argument("--split-seed", type=int, default=42)
    c_parser.add_argument(
        "--warmup-steps", type=int, default=None,
        help="Override TRAINING.warmup_steps for this Strategy C run only (roberta only).",
    )
    c_parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override TRAINING.epochs for this Strategy C run only (roberta only).",
    )

    c_heldout_parser = subparsers.add_parser(
        "strategy_c_heldout",
        help="One-shot eval of the locked STRATEGY_C_FINAL_CONFIG against TEST_CSV.",
    )
    c_heldout_parser.add_argument("--dry-run", action="store_true")

    c_diagnose_parser = subparsers.add_parser(
        "strategy_c_diagnose",
        help="Does RoBERTa fit its own training data? Sweeps folds x seeds by default.",
    )
    c_diagnose_parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    c_diagnose_parser.add_argument("--train-seeds", type=int, nargs="+", default=SEEDS)
    c_diagnose_parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=TRAINING.use_class_weights,
    )
    c_diagnose_parser.add_argument("--split-seed", type=int, default=42)
    c_diagnose_parser.add_argument(
        "--warmup-steps", type=int, default=None,
        help="Override TRAINING.warmup_steps for this diagnostic run only.",
    )
    c_diagnose_parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override TRAINING.epochs for this diagnostic run only.",
    )

    args = parser.parse_args()
    if args.strategy == "strategy_b":
        summary = run_strategy_b(
            args.silver_csv,
            args.sizes,
            args.sample_seeds,
            args.train_seeds,
            args.dry_run,
            args.class_weights,
            args.results_dir,
        )
    elif args.strategy == "strategy_b_heldout":
        summary = run_strategy_b_heldout_eval(args.dry_run)
    elif args.strategy == "strategy_c_heldout":
        summary = run_strategy_c_heldout_eval(args.dry_run)
    elif args.strategy == "strategy_c_diagnose":
        run_strategy_c_diagnose_sweep(
            args.folds,
            args.train_seeds,
            args.class_weights,
            args.split_seed,
            args.warmup_steps,
            args.epochs,
        )
        return
    else:
        summary = run_strategy_c(
            args.seeds,
            args.dry_run,
            args.model,
            args.class_weights,
            args.split_seed,
            args.warmup_steps,
            args.epochs,
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
