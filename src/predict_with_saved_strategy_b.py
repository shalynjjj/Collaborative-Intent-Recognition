"""Predict with an already-trained, saved Strategy B model (no retraining).

Loads the model+tokenizer saved under
results/strategy_b_heldout/models/seed{N}/ (see save_model_dir in
train_roberta.py / ablation_parent_negation.py) and predicts on any CSV with
Parent/Reply columns. Averages logits across all available seeds unless
--seed is given.

Usage:
    python3 -m src.predict_with_saved_strategy_b --input some_rows.csv --output preds.csv
    python3 -m src.predict_with_saved_strategy_b --input some_rows.csv --output preds.csv --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DIALOGUE_LABELS, ID2LABEL, SEEDS, STRATEGY_B_HELDOUT_DIR, TrainingConfig

MODEL_DIR = STRATEGY_B_HELDOUT_DIR / "models"
MAX_LENGTH = TrainingConfig().max_length


def _available_seeds() -> list:
    return [s for s in SEEDS if (MODEL_DIR / f"seed{s}").exists()]


def predict(input_csv: Path, output_csv: Path, seeds=None) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seeds = seeds or _available_seeds()
    if not seeds:
        raise FileNotFoundError(
            f"No saved models found under {MODEL_DIR}. Run run_strategy_b_heldout_eval "
            "or ablation_parent_negation with save_model_dir set first."
        )

    df = pd.read_csv(input_csv)
    parents = df["Parent"].fillna("").astype(str).tolist()
    replies = df["Reply"].fillna("").astype(str).tolist()

    all_logits = []
    for seed in seeds:
        model_dir = MODEL_DIR / f"seed{seed}"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()

        encoded = tokenizer(
            parents,
            replies,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits.numpy()
        all_logits.append(logits)
        print(f"seed {seed}: predicted {len(df)} rows")

    mean_logits = np.mean(all_logits, axis=0)
    pred_ids = np.argmax(mean_logits, axis=1)
    df["pred_label"] = [ID2LABEL[int(i)] for i in pred_ids]
    for i, label in enumerate(DIALOGUE_LABELS):
        df[f"prob_{label}"] = np.exp(mean_logits[:, i]) / np.exp(mean_logits).sum(axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with saved Strategy B model(s).")
    parser.add_argument("--input", type=Path, required=True, help="CSV with Parent/Reply columns.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Use only this seed's model (default: average all saved seeds).")
    args = parser.parse_args()
    seeds = [args.seed] if args.seed is not None else None
    predict(args.input, args.output, seeds)


if __name__ == "__main__":
    main()
