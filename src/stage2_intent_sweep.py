import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .config import (
    EMOTION_LABELS,
    INTENT_LABELS,
    STAGE2_DEV_CSV,
    STAGE2_DIR,
    STAGE2_EVAL_CSV,
    STAGE2_MAX_NEW_TOKENS,
    STRATEGY_A_DIR,
)
from .evaluate import bootstrap_ci, compute_metrics, paired_bootstrap_difference, save_metrics
from .llm_annotate import make_transformers_generator
from .stage2_pipeline import make_stage2_mock_generator, run_intent_module
from .utils import ensure_results_dirs, set_seed

# Seven non-empty subsets of {dialogue_act, sentiment, emotion}, plus a
# Parent+Reply baseline. ``oracle`` measures ideal information value using
# gold auxiliary labels; ``predicted`` measures the realistic pipeline using
# outputs from Stage 1 and Experiment 1.
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


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def predicted_emotion_labels(row: pd.Series) -> List[str]:
    return [
        label for label in EMOTION_LABELS
        if _as_bool(row.get(f"predicted_emotion_{label.lower()}", False))
    ]


def _output_stem(feature_source: str, split: str) -> str:
    return f"intent_sweep_{feature_source}_{split}"


def _prediction_path(feature_source: str, split: str) -> Path:
    return STAGE2_DIR / f"{_output_stem(feature_source, split)}_predictions.csv"


def _load_oracle_predictions(split: str) -> pd.DataFrame:
    canonical = _prediction_path("oracle", split)
    legacy = STAGE2_DIR / f"intent_sweep_{split}_predictions.csv"
    path = canonical if canonical.exists() else legacy
    if not path.exists():
        raise FileNotFoundError(f"No oracle predictions found for split={split}")
    return pd.read_csv(path)


def load_feature_dataframe(split: str, feature_source: str) -> pd.DataFrame:
    """Load Stage 2 rows and attach either gold or genuinely predicted inputs."""
    csv_path = STAGE2_DEV_CSV if split == "dev" else STAGE2_EVAL_CSV
    stage2 = pd.read_csv(csv_path, encoding="utf-8-sig")
    if feature_source == "oracle":
        return stage2

    # Strategy A few-shot predictions deliberately exclude its four prompt
    # demonstrations. Inner matching therefore produces the leakage-free
    # common benchmark (260/264 eval rows; all comparisons use these same rows).
    da_path = STRATEGY_A_DIR / "fewshot_predictions.csv"
    if not da_path.exists():
        raise FileNotFoundError(f"Predicted Dialogue Act file not found: {da_path}")
    da = pd.read_csv(da_path)[["Parent", "Reply", "pred_label"]].rename(
        columns={"pred_label": "predicted_dialogue_act"}
    )
    if da.duplicated(["Parent", "Reply"]).any():
        raise ValueError("Strategy A predictions contain duplicate Parent-Reply pairs")
    merged = stage2.merge(da, on=["Parent", "Reply"], how="inner", validate="one_to_one")

    module_path = STAGE2_DIR / f"multi_module_{split}_predictions.csv"
    if not module_path.exists():
        raise FileNotFoundError(f"Experiment 1 predictions not found: {module_path}")
    modules = pd.read_csv(module_path)
    module_cols = ["Parent", "Reply", "sentiment"] + [
        f"emotion_{label.lower()}" for label in EMOTION_LABELS
    ]
    modules = modules[module_cols].rename(
        columns={
            "sentiment": "predicted_sentiment",
            **{
                f"emotion_{label.lower()}": f"predicted_emotion_{label.lower()}"
                for label in EMOTION_LABELS
            },
        }
    )
    if modules.duplicated(["Parent", "Reply"]).any():
        raise ValueError("Experiment 1 predictions contain duplicate Parent-Reply pairs")
    merged = merged.merge(modules, on=["Parent", "Reply"], how="left", validate="one_to_one")
    if merged["predicted_sentiment"].isna().any():
        raise ValueError("Some common rows are missing predicted Sentiment/Emotion outputs")
    return merged


def run_intent_sweep_dataframe(
    df: pd.DataFrame,
    generator: Callable[[str], str],
    feature_source: str = "oracle",
    reused_base: Optional[pd.DataFrame] = None,
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
            if feature_source == "oracle":
                dialogue_act = row.get("Dialogue_act")
                sentiment = row.get("Sentiment")
                emotion = gold_emotion_labels(row)
            elif feature_source == "predicted":
                dialogue_act = row.get("predicted_dialogue_act")
                sentiment = row.get("predicted_sentiment")
                emotion = predicted_emotion_labels(row)
            else:
                raise ValueError(f"Unknown feature_source: {feature_source}")

            out = {"row_id": idx, "Parent": parent, "Reply": reply, "gold_intent": row.get("Intent")}
            for combo in all_combos:
                name = combo_name(combo)
                if not combo and reused_base is not None:
                    key = (parent, reply)
                    base_row = reused_base.loc[key]
                    label = base_row["intent_base"]
                    fallback_used = _as_bool(base_row["intent_base_fallback"])
                    raw = ""
                else:
                    label, fallback_used, raw = run_intent_module(
                        parent,
                        reply,
                        generator,
                        dialogue_act=dialogue_act if "dialogue_act" in combo else None,
                        sentiment=sentiment if "sentiment" in combo else None,
                        emotion_labels=emotion if "emotion" in combo else None,
                    )
                out[f"intent_{name}"] = label
                out[f"intent_{name}_fallback"] = fallback_used
                out[f"intent_{name}_raw"] = raw
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


def run_sweep(
    split: str = "dev",
    feature_source: str = "oracle",
    limit: Optional[int] = None,
    mock: bool = False,
) -> pd.DataFrame:
    ensure_results_dirs()
    set_seed(42)

    df = load_feature_dataframe(split, feature_source)
    if limit is not None:
        df = df.head(limit).copy()

    generator = (
        make_stage2_mock_generator()
        if mock
        else make_transformers_generator(max_new_tokens=STAGE2_MAX_NEW_TOKENS)
    )
    reused_base = None
    if feature_source == "predicted":
        base = _load_oracle_predictions(split).set_index(["Parent", "Reply"])
        reused_base = base[["intent_base", "intent_base_fallback"]]
    pred_path = (
        STAGE2_DIR / f"{_output_stem(feature_source, split)}_smoke{limit}_predictions.csv"
        if limit is not None
        else _prediction_path(feature_source, split)
    )
    return run_intent_sweep_dataframe(
        df,
        generator,
        feature_source=feature_source,
        reused_base=reused_base,
        output_csv=pred_path,
    )


def evaluate_sweep(split: str = "dev", feature_source: str = "oracle") -> pd.DataFrame:
    if feature_source == "oracle":
        df = _load_oracle_predictions(split)
    else:
        pred_path = _prediction_path(feature_source, split)
        if not pred_path.exists():
            raise FileNotFoundError(f"{pred_path} not found -- run the sweep first")
        df = pd.read_csv(pred_path)

    all_combos = [frozenset()] + COMBINATIONS
    rows = []
    for combo in all_combos:
        name = combo_name(combo)
        fallback_count = int(df[f"intent_{name}_fallback"].map(_as_bool).sum())
        metrics = compute_metrics(
            df["gold_intent"],
            df[f"intent_{name}"],
            labels=INTENT_LABELS,
            fallback_count=fallback_count,
        )
        ci = bootstrap_ci(df["gold_intent"], df[f"intent_{name}"], labels=INTENT_LABELS)
        save_metrics(
            {**metrics, "bootstrap": ci},
            STAGE2_DIR / f"{_output_stem(feature_source, split)}_{name}_metrics.json",
        )
        rows.append(
            {
                "combination": name,
                "macro_f1": metrics["macro_f1"],
                "macro_f1_ci95_lo": ci["macro_f1_ci95"][0],
                "macro_f1_ci95_hi": ci["macro_f1_ci95"][1],
                "cohen_kappa": metrics["cohen_kappa"],
                "fallback_count": fallback_count,
            }
        )

    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary.to_csv(STAGE2_DIR / f"{_output_stem(feature_source, split)}_summary.csv", index=False)

    comparisons = []
    for combo in COMBINATIONS:
        name = combo_name(combo)
        paired = paired_bootstrap_difference(
            df["gold_intent"], df[f"intent_{name}"], df["intent_base"], labels=INTENT_LABELS
        )
        comparisons.append({"combination": name, "reference": "base", **paired})
    pd.DataFrame(comparisons).to_csv(
        STAGE2_DIR / f"{_output_stem(feature_source, split)}_vs_base_paired.csv", index=False
    )
    return summary


def compare_oracle_and_predicted(split: str) -> pd.DataFrame:
    """Compare oracle and predicted auxiliary inputs on their exact common rows."""
    oracle = _load_oracle_predictions(split)
    predicted_path = _prediction_path("predicted", split)
    if not predicted_path.exists():
        raise FileNotFoundError(predicted_path)
    predicted = pd.read_csv(predicted_path)
    keep = ["Parent", "Reply", "gold_intent"] + [
        f"intent_{combo_name(combo)}" for combo in COMBINATIONS
    ]
    merged = oracle[keep].merge(
        predicted[keep],
        on=["Parent", "Reply"],
        suffixes=("_oracle", "_predicted"),
        validate="one_to_one",
    )
    rows = []
    for combo in COMBINATIONS:
        name = combo_name(combo)
        gold_a = merged["gold_intent_oracle"]
        if not gold_a.equals(merged["gold_intent_predicted"]):
            raise ValueError("Oracle/predicted gold Intent labels do not align")
        paired = paired_bootstrap_difference(
            gold_a,
            merged[f"intent_{name}_oracle"],
            merged[f"intent_{name}_predicted"],
            labels=INTENT_LABELS,
        )
        rows.append({"combination": name, "a": "oracle", "b": "predicted", "n_rows": len(merged), **paired})
    result = pd.DataFrame(rows)
    result.to_csv(STAGE2_DIR / f"intent_sweep_oracle_vs_predicted_{split}_paired.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Experiment 2 intent input-combination sweep.")
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--feature-source", choices=["oracle", "predicted"], default="oracle")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock outputs.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip generation, just (re)score an existing intent_sweep_<split>_predictions.csv.",
    )
    args = parser.parse_args()

    if not args.eval_only:
        predictions = run_sweep(
            split=args.split,
            feature_source=args.feature_source,
            limit=args.limit,
            mock=args.mock,
        )
        if args.limit is not None:
            print(f"Smoke run saved {len(predictions)} rows; scoring skipped for a partial split.")
            return

    summary = evaluate_sweep(split=args.split, feature_source=args.feature_source)
    print(summary.to_string(index=False))
    if args.feature_source == "predicted" and args.limit is None:
        print("\nOracle vs predicted auxiliary inputs (paired common rows):")
        print(compare_oracle_and_predicted(args.split).to_string(index=False))


if __name__ == "__main__":
    main()
