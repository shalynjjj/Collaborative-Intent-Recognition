import argparse
import json
from typing import Dict

import pandas as pd

from .config import INTENT_LABELS, SENTIMENT_LABELS, STAGE2_DIR
from .evaluate import compute_emotion_metrics, compute_metrics, save_confusion_matrix, save_metrics


def load_predictions(mode: str, split: str) -> pd.DataFrame:
    path = STAGE2_DIR / f"{mode}_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run "
            f"`python3 -m src.stage2_pipeline --mode {mode} --split {split}` first."
        )
    return pd.read_csv(path)


def run_stage2_eval(mode: str, split: str) -> Dict:
    df = load_predictions(mode, split)
    prefix = STAGE2_DIR / f"{mode}_{split}"

    sentiment_metrics = compute_metrics(df["gold_sentiment"], df["sentiment"], labels=SENTIMENT_LABELS)
    save_metrics(sentiment_metrics, prefix.with_name(f"{prefix.name}_sentiment_metrics.json"))
    save_confusion_matrix(
        sentiment_metrics, prefix.with_name(f"{prefix.name}_sentiment_confusion_matrix.csv")
    )

    intent_metrics = compute_metrics(df["gold_intent"], df["intent"], labels=INTENT_LABELS)
    save_metrics(intent_metrics, prefix.with_name(f"{prefix.name}_intent_metrics.json"))
    save_confusion_matrix(intent_metrics, prefix.with_name(f"{prefix.name}_intent_confusion_matrix.csv"))

    emotion_metrics = compute_emotion_metrics(df)
    save_metrics(emotion_metrics, prefix.with_name(f"{prefix.name}_emotion_metrics.json"))

    summary = {
        "mode": mode,
        "split": split,
        "rows": int(len(df)),
        "sentiment_macro_f1": sentiment_metrics["macro_f1"],
        "intent_macro_f1": intent_metrics["macro_f1"],
        "emotion_macro_f1": emotion_metrics["macro_f1"],
    }
    save_metrics(summary, prefix.with_name(f"{prefix.name}_summary.json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 predictions.")
    parser.add_argument("--mode", choices=["multi_module", "single_prompt"], required=True)
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    args = parser.parse_args()

    summary = run_stage2_eval(mode=args.mode, split=args.split)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
