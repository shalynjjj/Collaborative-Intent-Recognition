import argparse
import csv
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .config import (
    DIALOGUE_LABELS,
    FALLBACK_LABEL,
    LLM,
    SILVER_CANDIDATES_CSV,
    SILVER_LABELED_CSV,
    STRATEGY_A_DIR,
    STRATEGY_A_FINAL_MODE,
    STRATEGY_A_HELDOUT_DIR,
    TEST_CSV,
)
from .data_loader import load_gold_data
from .evaluate import bootstrap_ci, compute_metrics, save_confusion_matrix, save_metrics
from .utils import ensure_results_dirs, parse_label, resolve_model_path, set_seed


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            "python-dotenv is required for local API/model secrets. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    load_dotenv()


FEWSHOT_EXAMPLES = [
    {
        "parent": "There are more than 2 colours",
        "reply": "Exactly. Not everything is black and white.",
        "label": "agree",
    },
    {
        "parent": "Complexity != worth. Dungeons and Dragons is complex - I don't "
        "expect a college department dedicated to it.",
        "reply": "No, it doesn't equal worth, but most of your complaints about "
        "\"worthless\" academic pursuits seem to grossly oversimplifying or "
        "completely misunderstanding those fields.",
        "label": "disagree",
    },
    {
        "parent": "They have nukes",
        "reply": "Are they insane enough to use them though?",
        "label": "question",
    },
    {
        "parent": "Sounds horrible huh? But seriously, the only rule is that they "
        "can't make more than 2 million, so yeah. They'd have to give it away.",
        "reply": "Some would burn it first!",
        "label": "statement",
    },
]


def exclude_fewshot_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Return the common benchmark after removing prompt demonstrations."""
    fewshot_pairs = {(example["parent"], example["reply"]) for example in FEWSHOT_EXAMPLES}
    keep = [
        (parent, reply) not in fewshot_pairs
        for parent, reply in zip(df["Parent"], df["Reply"])
    ]
    return df.loc[keep].copy()


def build_prompt(parent: str, reply: str, mode: str) -> str:
    examples = ""
    if mode == "fewshot":
        blocks = [
            f"Parent: {ex['parent']}\nReply: {ex['reply']}\nLabel: {ex['label']}"
            for ex in FEWSHOT_EXAMPLES
        ]
        examples = "\nExamples:\n" + "\n\n".join(blocks) + "\n"

    return f"""Classify the dialogue act of the Reddit CMV reply.
Allowed labels: {", ".join(DIALOGUE_LABELS)}.

Definitions:
- agree: the reply explicitly or implicitly supports the parent's view.
- disagree: the reply challenges, rejects, or argues against the parent's view.
- question: the reply primarily asks for clarification, evidence, or elaboration.
- statement: the reply provides information, explanation, elaboration, or commentary without clearly agreeing, disagreeing, or asking a question.

Return only one label.
{examples}
Parent: {parent}
Reply: {reply}
Label:"""


def make_mock_generator() -> Callable[[str], str]:
    def generate(prompt: str) -> str:
        lowered = prompt.lower()
        if "?" in lowered.split("reply:", maxsplit=1)[-1]:
            return "question"
        if "agree" in lowered:
            return "agree"
        if "disagree" in lowered or "not" in lowered:
            return "disagree"
        return FALLBACK_LABEL

    return generate


def make_transformers_generator(
    max_new_tokens: Optional[int] = None, stop_strings: Optional[List[str]] = None
) -> Callable[[str], str]:
    load_env()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(LLM.model_id, LLM.model_source)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    auth_kwargs = {"token": hf_token} if LLM.model_source == "huggingface" and hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_path, **auth_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **auth_kwargs,
    )
    resolved_max_new_tokens = max_new_tokens if max_new_tokens is not None else LLM.max_new_tokens

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generation_kwargs = {
            "max_new_tokens": resolved_max_new_tokens,
            "do_sample": LLM.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if LLM.temperature > 0:
            generation_kwargs.update({"temperature": LLM.temperature, "top_p": LLM.top_p})
        if stop_strings:
            # Stops generation once the model starts hallucinating a new
            # Parent/Reply example past its actual answer, instead of relying
            # on max_new_tokens to cut it off mid-line.
            generation_kwargs.update({"stop_strings": stop_strings, "tokenizer": tokenizer})
        output_ids = model.generate(**inputs, **generation_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def annotate_dataframe(
    df: pd.DataFrame,
    mode: str,
    generator: Callable[[str], str],
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    writer = None
    file_handle = None
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        file_handle = output_csv.open("w", newline="", encoding="utf-8")

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
            prompt = build_prompt(str(row["Parent"]), str(row["Reply"]), mode)
            raw_output = generator(prompt)
            label, fallback_used = parse_label(raw_output)
            out = {
                "row_id": idx,
                "Parent": row["Parent"],
                "Reply": row["Reply"],
                "pred_label": label,
                "raw_output": raw_output,
                "fallback_used": fallback_used,
            }
            if "Dialogue_act" in row:
                out["gold_label"] = row["Dialogue_act"]
            for column in ("source_root", "parent_id", "reply_id"):
                if column in row:
                    out[column] = row[column]
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


def run_strategy_a(mode: str, limit: Optional[int] = None, mock: bool = False) -> Dict:
    ensure_results_dirs()
    set_seed(42)
    df = load_gold_data()

    if mode == "fewshot":
        df = exclude_fewshot_examples(df)

    if limit is not None:
        df = df.head(limit).copy()

    pred_path = STRATEGY_A_DIR / f"{mode}_predictions.csv"
    generator = make_mock_generator() if mock else make_transformers_generator()
    predictions = annotate_dataframe(df, mode, generator, output_csv=pred_path)
    fallback_count = int(predictions["fallback_used"].sum())

    metrics = compute_metrics(
        predictions["gold_label"],
        predictions["pred_label"],
        fallback_count=fallback_count,
    )
    metrics.update(
        {
            "strategy": "A",
            "prompting": mode,
            "model": "mock" if mock else LLM.model_id,
            "model_source": "mock" if mock else LLM.model_source,
        }
    )

    metrics_path = STRATEGY_A_DIR / f"{mode}_metrics.json"
    cm_path = STRATEGY_A_DIR / f"{mode}_confusion_matrix.csv"
    save_metrics(metrics, metrics_path)
    save_confusion_matrix(metrics, cm_path)
    return metrics


def run_strategy_a_heldout(mock: bool = False) -> Dict:
    """One-shot confirmatory run of the locked Strategy A config on TEST_CSV.

    Always uses STRATEGY_A_FINAL_MODE -- this does not take a `mode` argument
    on purpose, so it can't be pointed at a config that hasn't been locked in
    config.py. Run this once, after prompt iteration is finished.
    """
    ensure_results_dirs()
    set_seed(42)
    mode = STRATEGY_A_FINAL_MODE
    df = load_gold_data(TEST_CSV)

    fewshot_pairs = {(ex["parent"], ex["reply"]) for ex in FEWSHOT_EXAMPLES}
    df = df[~df.apply(lambda row: (row["Parent"], row["Reply"]) in fewshot_pairs, axis=1)]

    pred_path = STRATEGY_A_HELDOUT_DIR / f"{mode}_predictions.csv"
    generator = make_mock_generator() if mock else make_transformers_generator()
    predictions = annotate_dataframe(df, mode, generator, output_csv=pred_path)
    fallback_count = int(predictions["fallback_used"].sum())

    metrics = compute_metrics(
        predictions["gold_label"],
        predictions["pred_label"],
        fallback_count=fallback_count,
    )
    metrics.update(
        {
            "strategy": "A",
            "eval_set": "heldout_test",
            "prompting": mode,
            "model": "mock" if mock else LLM.model_id,
            "model_source": "mock" if mock else LLM.model_source,
        }
    )

    save_metrics(metrics, STRATEGY_A_HELDOUT_DIR / f"{mode}_metrics.json")
    save_confusion_matrix(metrics, STRATEGY_A_HELDOUT_DIR / f"{mode}_confusion_matrix.csv")
    return metrics


def add_bootstrap_ci(
    predictions_csv: Path, metrics_json: Path, n_boot: int = 2000, seed: int = 42
) -> Dict:
    """Attach a bootstrap macro-F1/kappa CI to an already-saved metrics file.

    Reads the (gold_label, pred_label) pairs already written by a prior run
    and resamples them -- the LLM is temperature=0 (LLM.temperature in
    config.py), so its output is deterministic and re-running it would only
    reproduce the same predictions. This also means it never re-reads
    TEST_CSV, so it does not count against the "TEST_CSV touched once"
    rule for held-out runs.
    """
    predictions = pd.read_csv(predictions_csv)
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    metrics["bootstrap"] = bootstrap_ci(
        predictions["gold_label"], predictions["pred_label"], n_boot=n_boot, seed=seed
    )
    save_metrics(metrics, metrics_json)
    return metrics


def run_silver_annotation(
    input_csv: Path,
    output_csv: Path,
    mode: str,
    limit: Optional[int] = None,
    mock: bool = False,
) -> Dict:
    ensure_results_dirs()
    set_seed(42)
    df = pd.read_csv(input_csv, encoding="latin1")
    missing = {"Parent", "Reply"}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing)}")
    if limit is not None:
        df = df.head(limit).copy()

    generator = make_mock_generator() if mock else make_transformers_generator()
    predictions = annotate_dataframe(df, mode, generator, output_csv=output_csv)
    fallback_count = int(predictions["fallback_used"].sum())

    summary = {
        "strategy": "B_silver_annotation",
        "prompting": mode,
        "model": "mock" if mock else LLM.model_id,
        "model_source": "mock" if mock else LLM.model_source,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": int(len(predictions)),
        "fallback_count": fallback_count,
        "label_counts": predictions["pred_label"].value_counts().to_dict(),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Strategy A LLM annotation.")
    parser.add_argument("--mode", choices=["zeroshot", "fewshot"], default="zeroshot")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock labels.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="Run the locked STRATEGY_A_FINAL_MODE config once against TEST_CSV. "
        "Ignores --mode/--input-csv/--output-csv.",
    )
    parser.add_argument(
        "--bootstrap-existing",
        action="store_true",
        help="Attach a bootstrap macro-F1/kappa CI to an already-saved metrics "
        "file by resampling its predictions.csv, instead of calling the LLM. "
        "Combine with --heldout for the held-out metrics, or --mode for the "
        "dev-set metrics.",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    if args.bootstrap_existing:
        if args.heldout:
            mode = STRATEGY_A_FINAL_MODE
            pred_path = STRATEGY_A_HELDOUT_DIR / f"{mode}_predictions.csv"
            metrics_path = STRATEGY_A_HELDOUT_DIR / f"{mode}_metrics.json"
        else:
            pred_path = STRATEGY_A_DIR / f"{args.mode}_predictions.csv"
            metrics_path = STRATEGY_A_DIR / f"{args.mode}_metrics.json"
        metrics = add_bootstrap_ci(pred_path, metrics_path, n_boot=args.n_boot)
        print(json.dumps(metrics["bootstrap"], indent=2))
    elif args.heldout:
        metrics = run_strategy_a_heldout(mock=args.mock)
        print(json.dumps(metrics, indent=2))
    elif args.input_csv is not None:
        output_csv = args.output_csv or SILVER_LABELED_CSV
        summary = run_silver_annotation(
            input_csv=args.input_csv or SILVER_CANDIDATES_CSV,
            output_csv=output_csv,
            mode=args.mode,
            limit=args.limit,
            mock=args.mock,
        )
        print(json.dumps(summary, indent=2))
    else:
        metrics = run_strategy_a(mode=args.mode, limit=args.limit, mock=args.mock)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
