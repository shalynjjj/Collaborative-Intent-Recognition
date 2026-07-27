import re
from pathlib import Path

import pandas as pd
from convokit import Corpus, download

from .config import DATA_DIR, GOLD_CSV, SILVER_CANDIDATES_CSV
from .prepare_gold_test_candidates import _norm, clean, is_low_quality, token_len

SEED = 2026
N = 450

OUTPUT_CSV = DATA_DIR / "cmv_test_candidates_batch2.csv"

# Annotator-facing copy: same rows, same shuffled order, but with
# heuristic_label stripped out. Annotators must never see which class a row
# was guessed to be -- knowing "this one looks like a question" biases their
# independent judgment before they even read the text (see the proposal's
# own "Sure, great logic." example: it reads agree, but is annotated
# disagree). Use OUTPUT_CSV only as the internal tracking copy for counting
# progress toward each class's target.
ANNOTATE_CSV = DATA_DIR / "cmv_test_candidates_batch2_annotate.csv"

# Every file that already carries candidates drawn from the corpus for the
# held-out test set, whether or not annotation on it is finished. All of
# these must be excluded so batch 2 never re-samples something already
# collected.
EXISTING_TEST_CSVS = [
    DATA_DIR / "cmv_test_candidates.csv",
    DATA_DIR / "cmv_test_candidates_labeled.csv",
    DATA_DIR / "cmv_test_candidates_random_topup.csv",
]

# Purely descriptive, post-hoc tag for the internal tracking copy -- it plays
# no role in which rows get sampled. Sampling below is simple random
# (stratified only by length_group, same as the original
# prepare_gold_test_candidates.py), so this heuristic can't bias which
# rows enter the batch or what their real label turns out to be.
QUESTION_PATTERN = re.compile(
    r"\?|\b(what|why|how|who|when|where|which|could you|can you|would you|"
    r"do you|did you|isn'?t it|aren'?t you)\b",
    re.IGNORECASE,
)
AGREE_PATTERN = re.compile(
    r"\b(i agree|you'?re right|you are right|that'?s true|exactly|"
    r"well said|good point|agreed|fair point)\b",
    re.IGNORECASE,
)
DISAGREE_PATTERN = re.compile(
    r"\b(disagree|that'?s wrong|you'?re wrong|not true|i don'?t think|"
    r"makes no sense|incorrect|false)\b",
    re.IGNORECASE,
)

LENGTH_TARGETS = {"short": 0.40, "mid": 0.35, "long": 0.25}


def heuristic_label(reply: str) -> str:
    if QUESTION_PATTERN.search(reply):
        return "question"
    if AGREE_PATTERN.search(reply):
        return "agree"
    if DISAGREE_PATTERN.search(reply):
        return "disagree"
    return "statement"


def load_excluded_pairs() -> set:
    excluded = set()

    gold = pd.read_csv(GOLD_CSV, encoding="latin1")
    excluded |= set(zip(gold["Parent"].map(_norm), gold["Reply"].map(_norm)))

    if Path(SILVER_CANDIDATES_CSV).exists():
        silver = pd.read_csv(SILVER_CANDIDATES_CSV, encoding="latin1")
        excluded |= set(zip(silver["Parent"].map(_norm), silver["Reply"].map(_norm)))

    for path in EXISTING_TEST_CSVS:
        if path.exists():
            df = pd.read_csv(path)
            excluded |= set(zip(df["Parent"].map(_norm), df["Reply"].map(_norm)))

    return excluded


def load_excluded_utt_ids() -> set:
    excluded_ids = set()

    if Path(SILVER_CANDIDATES_CSV).exists():
        silver = pd.read_csv(SILVER_CANDIDATES_CSV, encoding="latin1")
        for column in ("parent_id", "reply_id"):
            if column in silver.columns:
                excluded_ids |= set(silver[column].dropna())

    for path in EXISTING_TEST_CSVS:
        if path.exists():
            df = pd.read_csv(path)
            if "utt_id" in df.columns:
                excluded_ids |= set(df["utt_id"].dropna())

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

        rows.append(
            {
                "utt_id": utt.id,
                "parent": parent_text,
                "reply": reply,
                "token_count": length,
                "heuristic_label": heuristic_label(reply),
            }
        )

    df = pd.DataFrame(rows)
    print(f"total fresh candidates after filtering + dedup against gold/silver/existing test files: {len(df)}")
    return df


def sample_data(df: pd.DataFrame, n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Simple random sampling stratified only by length_group, matching
    prepare_gold_test_candidates.py. Not stratified by heuristic_label: the
    natural gold-set class distribution (disagree ~48%, agree ~21%,
    statement ~16%, question ~15%) means n=450 clears 50+ per class with
    >99% probability even in the rarest class, without risking the
    compositional bias a keyword-based class heuristic would introduce.
    """
    df = df.copy()
    df["length_group"] = pd.cut(
        df["token_count"],
        bins=[0, 10, 20, 30],
        labels=["short", "mid", "long"],
    )

    parts = []
    for group, frac in LENGTH_TARGETS.items():
        sub = df[df["length_group"] == group]
        want = int(round(n * frac))
        take = min(want, len(sub))
        if take < want:
            print(f"warning: length='{group}' only has {len(sub)}, wanted {want}")
        if take > 0:
            parts.append(sub.sample(n=take, random_state=seed))

    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    out["sample_id"] = [f"CMV_TEST2_{i + 1:04d}" for i in range(len(out))]
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "utt_id": df["utt_id"],
            "Parent": df["parent"],
            "Reply": df["reply"],
            "token_count": df["token_count"],
            "length_group": df["length_group"],
            "heuristic_label": df["heuristic_label"],
            "Dialogue_act_xin": "",
            "Dialogue_act_xiaying": "",
            "Dialogue_act_final": "",
            "notes": "",
        }
    )


def main() -> None:
    print("loading exclusion set (gold + silver + all existing test-candidate files)...")
    excluded_pairs = load_excluded_pairs()
    excluded_utt_ids = load_excluded_utt_ids()
    print(f"excluding {len(excluded_pairs)} known parent/reply pairs")
    print(f"excluding {len(excluded_utt_ids)} known utterance ids")

    print("collecting...")
    df = collect_data(excluded_pairs, excluded_utt_ids)

    print("sampling (simple random, length-stratified only)...")
    sampled = sample_data(df)

    out = build(sampled)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    annotate_df = out.drop(columns=["heuristic_label"])
    annotate_df.to_csv(ANNOTATE_CSV, index=False, encoding="utf-8-sig")

    print(f"\nsaved tracking copy (internal use only) to: {OUTPUT_CSV}")
    print(f"saved annotator-facing copy to: {ANNOTATE_CSV}")
    print("\nheuristic_label distribution (descriptive only, not used for sampling):")
    print(out["heuristic_label"].value_counts())
    print("\nlength_group distribution:")
    print(out["length_group"].value_counts().sort_index())


if __name__ == "__main__":
    main()
