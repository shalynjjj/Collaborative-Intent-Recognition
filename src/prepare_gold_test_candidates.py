import re
from pathlib import Path

import pandas as pd
from convokit import Corpus, download

from .config import DATA_DIR, GOLD_CSV, SILVER_CANDIDATES_CSV

SEED = 42
N = 150

OUTPUT_CSV = DATA_DIR / "cmv_test_candidates.csv"

DELTA_THANKS = {
    "thank you", "thank you!", "thanks!", "thanks",
    "yahoo!", "yahoo", "cheers!", "cheers",
    "awesome", "great", "nice", "cool",
    "lol", "haha", "hah", "lmao",
}


def clean(text):
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if t.lower() in {"[deleted]", "[removed]", ""}:
        return ""
    t = re.sub(r"[!?.]{4,}", lambda m: m.group()[0], t)
    return t


def token_len(text):
    return len(text.split())


def is_low_quality(reply):
    if reply.lower().strip("! .,") in DELTA_THANKS:
        return True
    tokens = reply.split()
    has_url = any(t.startswith("http") or t.startswith("[http") for t in tokens)
    if has_url and len(tokens) <= 3:
        return True
    return False


def _norm(text):
    return " ".join(str(text or "").split())


def load_excluded_pairs() -> set:
    """Pairs already used as gold (300) or already trained on as silver (10k)."""
    excluded = set()

    gold = pd.read_csv(GOLD_CSV, encoding="latin1")
    excluded |= set(zip(gold["Parent"].map(_norm), gold["Reply"].map(_norm)))

    if Path(SILVER_CANDIDATES_CSV).exists():
        silver = pd.read_csv(SILVER_CANDIDATES_CSV, encoding="latin1")
        excluded |= set(zip(silver["Parent"].map(_norm), silver["Reply"].map(_norm)))

    return excluded


def load_excluded_utt_ids() -> set:
    """Utterance ids already used anywhere (as parent or reply) in silver (10k).

    Pair-level exclusion only blocks exact (Parent, Reply) duplicates. A single
    utterance can appear as the Parent of one pair and the Reply of a different
    pair, so it needs its own id-level check -- otherwise a test candidate's
    Reply text can silently match a silver row's Parent text (or vice versa),
    which is a milder but real train/test overlap that pair-level dedup misses.
    """
    excluded_ids = set()
    if Path(SILVER_CANDIDATES_CSV).exists():
        silver = pd.read_csv(SILVER_CANDIDATES_CSV, encoding="latin1")
        for column in ("parent_id", "reply_id"):
            if column in silver.columns:
                excluded_ids |= set(silver[column].dropna())
    return excluded_ids


def collect_data(excluded_pairs: set, excluded_utt_ids: set) -> pd.DataFrame:
    corpus = Corpus(filename=download("winning-args-corpus"))
    rows = []

    for utt in corpus.iter_utterances():
        if utt.reply_to is None:
            continue

        if utt.speaker.id and "deltabot" in utt.speaker.id.lower():
            continue

        try:
            parent = corpus.get_utterance(utt.reply_to)
        except KeyError:
            continue

        if parent.speaker.id and "deltabot" in parent.speaker.id.lower():
            continue

        reply = clean(utt.text)
        if not reply:
            continue

        parent_text = clean(parent.text)
        if not parent_text:
            continue

        length = token_len(reply)
        if not (1 <= length <= 30):
            continue

        if is_low_quality(reply):
            continue

        if (_norm(parent_text), _norm(reply)) in excluded_pairs:
            continue

        if utt.id in excluded_utt_ids or parent.id in excluded_utt_ids:
            continue

        rows.append({
            "utt_id": utt.id,
            "parent": parent_text,
            "reply": reply,
            "token_count": length,
        })

    df = pd.DataFrame(rows)
    print(f"total candidates after filtering + dedup against gold/silver: {len(df)}")
    return df


def sample_data(df: pd.DataFrame, n: int = N, seed: int = SEED) -> pd.DataFrame:
    df = df.copy()
    df["length_group"] = pd.cut(
        df["token_count"],
        bins=[0, 10, 20, 30],
        labels=["short", "mid", "long"],
    )

    targets = {
        "short": int(n * 0.40),
        "mid": int(n * 0.35),
        "long": int(n * 0.25),
    }

    parts = []
    for group, target in targets.items():
        sub = df[df["length_group"] == group]
        if len(sub) == 0:
            print(f"warning: no samples in group '{group}'")
            continue
        take = min(target, len(sub))
        if take < target:
            print(f"warning: '{group}' only has {len(sub)}, wanted {target}, taking {take}")
        parts.append(sub.sample(n=take, random_state=seed))

    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    out["sample_id"] = [f"CMV_TEST_{i + 1:04d}" for i in range(len(out))]
    print(f"final sample size: {len(out)}")
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": df["sample_id"],
        "utt_id": df["utt_id"],
        "Parent": df["parent"],
        "Reply": df["reply"],
        "token_count": df["token_count"],
        "length_group": df["length_group"],
        "Dialogue_act": "",
        "notes": "",
    })


def main():
    print("loading exclusion set (gold 300 + silver 10k)...")
    excluded = load_excluded_pairs()
    excluded_utt_ids = load_excluded_utt_ids()
    print(f"excluding {len(excluded)} known parent/reply pairs")
    print(f"excluding {len(excluded_utt_ids)} known silver utterance ids (parent or reply role)")

    print("collecting...")
    df = collect_data(excluded, excluded_utt_ids)

    print("sampling...")
    sampled = sample_data(df, n=N, seed=SEED)

    out = build(sampled)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nsaved to: {OUTPUT_CSV}")
    print(out["length_group"].value_counts().sort_index())


if __name__ == "__main__":
    main()
