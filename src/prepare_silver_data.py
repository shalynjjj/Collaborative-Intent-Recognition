import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import RAW_UTTERANCES_JSONL, SILVER_CANDIDATES_CSV, SILVER_CANDIDATE_SIZE
from .data_loader import load_gold_data
from .utils import set_seed


def _norm_text(value: object) -> str:
    return " ".join(str(value or "").split())


def load_utterance_pairs(jsonl_path: Path) -> pd.DataFrame:
    utterances: Dict[str, Dict] = {}
    with Path(jsonl_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            utterances[row["id"]] = row

    pairs: List[Dict] = []
    for row in utterances.values():
        parent_id = row.get("reply-to")
        if not parent_id or parent_id not in utterances:
            continue
        parent = _norm_text(utterances[parent_id].get("text"))
        reply = _norm_text(row.get("text"))
        if not parent or not reply:
            continue
        pairs.append(
            {
                "source_root": row.get("root"),
                "parent_id": parent_id,
                "reply_id": row["id"],
                "Parent": parent,
                "Reply": reply,
            }
        )
    return pd.DataFrame(pairs).drop_duplicates(subset=["parent_id", "reply_id"])


def remove_gold_overlap(candidates: pd.DataFrame) -> pd.DataFrame:
    gold = load_gold_data()
    gold_pairs = set(zip(gold["Parent"].map(_norm_text), gold["Reply"].map(_norm_text)))
    mask = [
        (_norm_text(parent), _norm_text(reply)) not in gold_pairs
        for parent, reply in zip(candidates["Parent"], candidates["Reply"])
    ]
    return candidates.loc[mask].reset_index(drop=True)


def sample_silver_candidates(
    input_jsonl: Path,
    output_csv: Path,
    size: int,
    seed: int,
    min_reply_tokens: int,
    max_reply_tokens: int,
) -> pd.DataFrame:
    set_seed(seed)
    candidates = load_utterance_pairs(input_jsonl)
    candidates = remove_gold_overlap(candidates)
    reply_token_counts = candidates["Reply"].str.split().str.len()
    candidates = candidates[reply_token_counts.between(min_reply_tokens, max_reply_tokens)].reset_index(drop=True)
    if size > len(candidates):
        raise ValueError(f"Requested {size} rows, but only {len(candidates)} candidates remain.")
    sampled = candidates.sample(n=size, random_state=seed).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_csv, index=False)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 10k Strategy B silver candidate Parent/Reply pairs.")
    parser.add_argument("--input-jsonl", type=Path, default=RAW_UTTERANCES_JSONL)
    parser.add_argument("--output-csv", type=Path, default=SILVER_CANDIDATES_CSV)
    parser.add_argument("--size", type=int, default=SILVER_CANDIDATE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-reply-tokens", type=int, default=1)
    parser.add_argument("--max-reply-tokens", type=int, default=30)
    args = parser.parse_args()

    sampled = sample_silver_candidates(
        input_jsonl=args.input_jsonl,
        output_csv=args.output_csv,
        size=args.size,
        seed=args.seed,
        min_reply_tokens=args.min_reply_tokens,
        max_reply_tokens=args.max_reply_tokens,
    )
    print(f"Wrote {len(sampled)} rows to {args.output_csv}")
    print(sampled[["parent_id", "reply_id", "Parent", "Reply"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
