import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .config import EMOTION_LABELS, INTENT_LABELS, STAGE2_DEV_CSV, STAGE2_DIR, STAGE2_EVAL_CSV, STAGE2_MAX_NEW_TOKENS
from .evaluate import bootstrap_ci, compute_metrics, save_metrics
from .llm_annotate import make_transformers_generator
from .stage2_pipeline import make_stage2_mock_generator, run_intent_module
from .utils import ensure_results_dirs, set_seed

# 7 non-empty subsets of {dialogue_act, sentiment, emotion}, added on top of
# the reply+parent text-only baseline (reported separately, not counted among
# the "7"). All three feature sources are GOLD columns (oracle), matching the
# DA feature-source decision in the README: using noisy predicted features
# would confound "does this information help" with "is the prediction noisy".
FEATURES = ["dialogue_act", "sentiment", "emotion"]
COMBINATIONS = [
    frozenset(combo)
    for r in range(1, len(FEATURES) + 1)
    for combo in itertools.combinations(FEATURES, r)
]


def combo_name(combo: frozenset) -> str:
    return "base" if not combo else "+".join(sorted(combo))


def gold_emotion_labels(row: pd.Series) -> List[str]:
    return [label for label in EMOTION_LABELS if bool(row.get(label))]


def run_intent_sweep_dataframe(
    df: pd.DataFrame,
    generator: Callable[[str], str],
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    all_combos = [frozenset()] + COMBINATIONS  # frozenset() = text-only baseline
    rows = []
    writer = None
    file_handle = None
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        file_handle = output_csv.open("w", newline="", encoding="utf-8")

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Intent sweep"):
            parent, reply = str(row["Parent"]), str(row["Reply"])
            gold_da = row.get("Dialogue_act")
            gold_sentiment = row.get("Sentiment")
            gold_emotion = gold_emotion_labels(row)

            out = {"row_id": idx, "Parent": parent, "Reply": reply, "gold_intent": row.get("Intent")}
            for combo in all_combos:
                label, fallback_used, _ = run_intent_module(
                    parent,
                    reply,
                    generator,
                    dialogue_act=gold_da if "dialogue_act" in combo else None,
                    sentiment=gold_sentiment if "sentiment" in combo else None,
                    emotion_labels=gold_emotion if "emotion" in combo else None,
                )
                name = combo_name(combo)
                out[f"intent_{name}"] = label
                out[f"intent_{name}_fallback"] = fallback_used
            rows.append(out)

            if file_handle is not None:
                if writer is None:
                    writer = csv.DictWriter(file_handle, fieldnames=list(out.keys()))
                    writer.writeheader()
                writer.writerow(out)
                file_handle.flush()
    finally:
        if file_handle is not None:
            file_handle.close()

    return pd.DataFrame(rows)


def run_sweep(split: str = "dev", limit: Optional[int] = None, mock: bool = False) -> pd.DataFrame:
    ensure_results_dirs()
    set_seed(42)

    csv_path = STAGE2_DEV_CSV if split == "dev" else STAGE2_EVAL_CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if limit is not None:
        df = df.head(limit).copy()

    generator = (
        make_stage2_mock_generator()
        if mock
        else make_transformers_generator(max_new_tokens=STAGE2_MAX_NEW_TOKENS)
    )
    pred_path = STAGE2_DIR / f"intent_sweep_{split}_predictions.csv"
    return run_intent_sweep_dataframe(df, generator, output_csv=pred_path)


def evaluate_sweep(split: str = "dev") -> pd.DataFrame:
    pred_path = STAGE2_DIR / f"intent_sweep_{split}_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} not found -- run `python3 -m src.stage2_intent_sweep --split {split}` first.")
    df = pd.read_csv(pred_path)

    all_combos = [frozenset()] + COMBINATIONS
    rows = []
    for combo in all_combos:
        name = combo_name(combo)
        metrics = compute_metrics(df["gold_intent"], df[f"intent_{name}"], labels=INTENT_LABELS)
        ci = bootstrap_ci(df["gold_intent"], df[f"intent_{name}"], labels=INTENT_LABELS)
        save_metrics(
            {**metrics, "bootstrap": ci},
            STAGE2_DIR / f"intent_sweep_{split}_{name}_metrics.json",
        )
        rows.append(
            {
                "combination": name,
                "macro_f1": metrics["macro_f1"],
                "macro_f1_ci95_lo": ci["macro_f1_ci95"][0],
                "macro_f1_ci95_hi": ci["macro_f1_ci95"][1],
                "cohen_kappa": metrics["cohen_kappa"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary.to_csv(STAGE2_DIR / f"intent_sweep_{split}_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Experiment 2 intent input-combination sweep.")
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock outputs.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip generation, just (re)score an existing intent_sweep_<split>_predictions.csv.",
    )
    args = parser.parse_args()

    if not args.eval_only:
        run_sweep(split=args.split, limit=args.limit, mock=args.mock)

    summary = evaluate_sweep(split=args.split)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
