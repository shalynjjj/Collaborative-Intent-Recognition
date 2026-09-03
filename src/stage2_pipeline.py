import argparse
import csv
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .config import (
    EMOTION_LABELS,
    FALLBACK_INTENT,
    FALLBACK_SENTIMENT,
    INTENT_LABELS,
    SENTIMENT_LABELS,
    STAGE2_DEV_CSV,
    STAGE2_DIR,
    STAGE2_EVAL_CSV,
    STAGE2_MAX_NEW_TOKENS,
)
from .llm_annotate import make_transformers_generator
from .utils import ensure_results_dirs, set_seed

GOLD_COLUMNS = [
    "Sentiment",
    "Sarcasm",
    "Hostility",
    "Contempt",
    "Neutral",
    "Curiosity",
    "Appreciation",
    "Intent",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _first_line(raw_output: str) -> str:
    """The model's actual answer is always the first line -- anything after
    that is unconstrained rambling (observed: hallucinated Python docstrings
    that restate all candidate labels, e.g. "Returns: positive, negative, or
    neutral"). Scoping to the first line avoids matching label words that
    only appear in that rambling continuation.
    """
    stripped = (raw_output or "").strip()
    return stripped.splitlines()[0] if stripped else ""


_NON_ALNUM_RUN = re.compile(r"[^a-z0-9]+")


def _normalize_for_label_match(text: str) -> str:
    """Case/whitespace/punctuation-insensitive form for matching a label
    against model output. Lowercases, then collapses every run of
    non-alphanumeric characters (spaces, hyphens, underscores, punctuation,
    markdown markup like `**`/`_`) into a single space, so a multi-word
    label matches regardless of how the model spelled the separator --
    "Counter-argue" == "counter argue" == "COUNTER_ARGUE" == "**counter -
    argue**".
    """
    return _NON_ALNUM_RUN.sub(" ", text.lower()).strip()


def _find_label(text: str, labels: List[str]) -> Optional[str]:
    normalized = _normalize_for_label_match(text)
    best_label, best_pos = None, None
    for label in labels:
        pos = normalized.find(_normalize_for_label_match(label))
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_label, best_pos = label, pos
    return best_label


def parse_single_label(raw_output: str, labels: List[str], fallback: str) -> Tuple[str, bool]:
    match = _find_label(_first_line(raw_output), labels)
    if match is not None:
        return match, False

    # The model sometimes abandons the requested format entirely (free-text
    # reasoning instead of a bare label) but still commits to exactly one
    # candidate label somewhere later in its answer -- recover that instead
    # of discarding a real answer as a fallback. Require an *exact one*
    # match across the full output, not just the first line: when the model
    # instead echoes back several/all candidate labels (e.g. restating the
    # option list, or rambling into a second hallucinated example with a
    # different answer), there is no way to tell which one is the real
    # answer, so that case still falls back rather than guessing.
    normalized_full = _normalize_for_label_match(raw_output or "")
    found = {label for label in labels if _normalize_for_label_match(label) in normalized_full}
    if len(found) == 1:
        return next(iter(found)), False

    return fallback, True


def _emotion_answer_text(raw_output: str) -> str:
    """The emotion prompt ends with "Categories:", and the model's completion
    often continues that sentence by echoing the full category list back
    (observed on ~39% of dev rows) before giving its real answer later,
    introduced by "answer" / "the final answer is". Scoping to an explicit
    answer marker -- when present -- avoids matching every category word in
    that echoed list; `_first_line` alone was catching all of them as
    positives, which is why Hostility/Contempt/Curiosity precision was so low.
    """
    stripped = (raw_output or "").strip()
    if not stripped:
        return ""
    for line in stripped.splitlines():
        idx = line.lower().rfind("answer")
        if idx != -1:
            after = line[idx:]
            colon = after.find(":")
            return (after[colon + 1 :] if colon != -1 else after).strip()
    return stripped.splitlines()[0]


def parse_emotion_labels(raw_output: str) -> Tuple[Dict[str, bool], bool]:
    text = _emotion_answer_text(raw_output).lower()
    result = {label: (label.lower() in text) for label in EMOTION_LABELS}
    if "none" in text and not any(result.values()):
        # Defensive synonym: the prompt asks for "neutral", but the model may
        # still occasionally answer "none" -- treat them as equivalent rather
        # than falling back.
        result["Neutral"] = True
        return result, False
    fallback_used = not any(result.values())
    return result, fallback_used


def _split_labeled_lines(raw_output: str) -> Dict[str, str]:
    """First occurrence of each key wins. The model sometimes keeps
    generating past its answer and hallucinates a new Parent/Reply example
    with its own (often truncated) Sentiment/Emotion/Intent lines; taking the
    last occurrence would let that hallucinated continuation silently
    overwrite the real answer.
    """
    fields: Dict[str, str] = {}
    for line in (raw_output or "").splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            if key not in fields:
                fields[key] = value.strip()
    return fields


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


# Shared verbatim across the per-task prompts (multi_module) and the combined
# prompt (single_prompt) so that the two Experiment 1 conditions differ only
# in call structure (3 calls vs. 1), not in how much guidance the model gets.
_SENTIMENT_DEFINITIONS = """- positive: the reply expresses approval, agreement in tone, or a favorable attitude.
- negative: the reply expresses disapproval, frustration, or an unfavorable attitude.
- neutral: the reply is matter-of-fact, with no clear positive or negative tone."""

_EMOTION_DEFINITIONS = """- sarcasm: the reply says the opposite of what it means, or mocks the parent through exaggeration or irony.
- hostility: the reply is openly aggressive, angry, or attacking.
- contempt: the reply is dismissive or belittling toward the parent or its argument.
- curiosity: the reply asks for clarification, evidence, or more information out of genuine interest.
- appreciation: the reply thanks, praises, or acknowledges the parent's point positively.
- neutral: none of the other categories apply; the reply is emotionally flat. Always include this
  category by itself if no other category applies -- do not answer "none"."""

_INTENT_DEFINITIONS = """- information seeking: the reply primarily asks a question to get clarification or evidence.
- challenge: the reply pushes back, dismisses, or mocks the parent's claim WITHOUT presenting its
  own reasoning or alternative claim -- short rebuttals, sarcastic jabs, or "prove it" pushback with
  no new argument all count here, even if they read as confident or dismissive.
- counter-argue: the reply pushes back AND presents its own reasoning, evidence, or alternative
  claim to support that pushback -- there must be actual argument content beyond disagreement itself.
- support: the reply agrees with and reinforces the parent's position.
- others: the reply does not engage with the parent's argument at all -- off-topic remarks, jokes
  with no argumentative point, or meta-commentary about the thread/forum itself (e.g. mentioning
  mods, bots, deltas, or other posts) all count here, even if they are on-topic in subject matter."""


def build_sentiment_prompt(parent: str, reply: str) -> str:
    return f"""Classify the sentiment of the Reddit CMV reply toward the parent comment.
Allowed labels: positive, negative, neutral.

Definitions:
{_SENTIMENT_DEFINITIONS}

Return only one label.

Parent: {parent}
Reply: {reply}
Sentiment:"""


def build_emotion_prompt(parent: str, reply: str) -> str:
    return f"""Identify which of the following categories apply to the Reddit CMV reply. One or more categories may apply.
Categories: sarcasm, hostility, contempt, curiosity, appreciation, neutral.

Definitions:
{_EMOTION_DEFINITIONS}

Return a comma-separated list of every category that applies.

Parent: {parent}
Reply: {reply}
Categories:"""


def build_intent_prompt(
    parent: str,
    reply: str,
    dialogue_act: Optional[str] = None,
    sentiment: Optional[str] = None,
    emotion_labels: Optional[List[str]] = None,
) -> str:
    context_lines = []
    if dialogue_act:
        context_lines.append(f"Dialogue act: {dialogue_act}")
    if sentiment:
        context_lines.append(f"Sentiment: {sentiment}")
    if emotion_labels:
        context_lines.append(f"Emotion: {', '.join(emotion_labels)}")
    context = ("\n".join(context_lines) + "\n") if context_lines else ""

    return f"""Classify the communicative intent of the Reddit CMV reply.
Allowed labels: information seeking, challenge, counter-argue, support, others.

Definitions:
{_INTENT_DEFINITIONS}

Return only one label.
{context}
Parent: {parent}
Reply: {reply}
Intent:"""


def build_single_prompt_baseline(parent: str, reply: str) -> str:
    return f"""Analyze the Reddit CMV reply's sentiment, emotion, and communicative intent, all toward
the parent comment it is responding to, and report all three using exactly this format (one per line):
Sentiment: <positive|negative|neutral>
Emotion: <comma-separated subset of sarcasm, hostility, contempt, curiosity, appreciation, neutral, or "none">
Intent: <information seeking|challenge|counter-argue|support|others>

Sentiment definitions:
{_SENTIMENT_DEFINITIONS}

Emotion definitions:
{_EMOTION_DEFINITIONS}

Intent definitions:
{_INTENT_DEFINITIONS}

Parent: {parent}
Reply: {reply}
"""


# ---------------------------------------------------------------------------
# Module calls (each independently callable -- "multi-module orchestration"
# below is just these three run in sequence)
# ---------------------------------------------------------------------------


def run_sentiment_module(parent: str, reply: str, generator: Callable[[str], str]) -> Tuple[str, bool, str]:
    raw = generator(build_sentiment_prompt(parent, reply))
    label, fallback_used = parse_single_label(raw, SENTIMENT_LABELS, FALLBACK_SENTIMENT)
    return label, fallback_used, raw


def run_emotion_module(
    parent: str, reply: str, generator: Callable[[str], str]
) -> Tuple[Dict[str, bool], bool, str]:
    raw = generator(build_emotion_prompt(parent, reply))
    labels, fallback_used = parse_emotion_labels(raw)
    return labels, fallback_used, raw


def run_intent_module(
    parent: str,
    reply: str,
    generator: Callable[[str], str],
    dialogue_act: Optional[str] = None,
    sentiment: Optional[str] = None,
    emotion_labels: Optional[List[str]] = None,
) -> Tuple[str, bool, str]:
    prompt = build_intent_prompt(
        parent, reply, dialogue_act=dialogue_act, sentiment=sentiment, emotion_labels=emotion_labels
    )
    raw = generator(prompt)
    label, fallback_used = parse_single_label(raw, INTENT_LABELS, FALLBACK_INTENT)
    return label, fallback_used, raw


def run_single_prompt_baseline(parent: str, reply: str, generator: Callable[[str], str]) -> Dict:
    raw = generator(build_single_prompt_baseline(parent, reply))
    fields = _split_labeled_lines(raw)

    sentiment, sentiment_fallback = parse_single_label(
        fields.get("sentiment", "") or raw, SENTIMENT_LABELS, FALLBACK_SENTIMENT
    )
    emotion_labels, emotion_fallback = parse_emotion_labels(fields.get("emotion", "") or raw)
    intent, intent_fallback = parse_single_label(fields.get("intent", "") or raw, INTENT_LABELS, FALLBACK_INTENT)

    return {
        "sentiment": sentiment,
        "sentiment_fallback": sentiment_fallback,
        **{f"emotion_{label.lower()}": value for label, value in emotion_labels.items()},
        "emotion_fallback": emotion_fallback,
        "intent": intent,
        "intent_fallback": intent_fallback,
        "raw_output": raw,
    }


def run_multi_module_pipeline(
    parent: str,
    reply: str,
    generator: Callable[[str], str],
) -> Dict:
    """Run Experiment 1's three independent task-specific prompts.

    All three modules receive the same source information (Parent + Reply).
    Auxiliary labels such as dialogue act are intentionally excluded here so
    that the comparison with the single-prompt baseline isolates prompt/task
    decomposition.  Auxiliary-input combinations belong to Experiment 2.
    """
    sentiment, sentiment_fallback, sentiment_raw = run_sentiment_module(parent, reply, generator)
    emotion_labels, emotion_fallback, emotion_raw = run_emotion_module(parent, reply, generator)
    intent, intent_fallback, intent_raw = run_intent_module(parent, reply, generator)
    return {
        "sentiment": sentiment,
        "sentiment_fallback": sentiment_fallback,
        **{f"emotion_{label.lower()}": value for label, value in emotion_labels.items()},
        "emotion_fallback": emotion_fallback,
        "intent": intent,
        "intent_fallback": intent_fallback,
        "sentiment_raw": sentiment_raw,
        "emotion_raw": emotion_raw,
        "intent_raw": intent_raw,
    }


# ---------------------------------------------------------------------------
# Mock generator for plumbing smoke tests (no LLM load). Dispatches on the
# fixed instruction phrasing in each prompt above, mirroring the mock
# generator already used for Strategy A in llm_annotate.py.
# ---------------------------------------------------------------------------


def _mock_sentiment(reply_text: str) -> str:
    if any(w in reply_text for w in ("thank", "agree", "great", "good point")):
        return "positive"
    if any(w in reply_text for w in ("wrong", "stupid", "hate", "disagree")):
        return "negative"
    return "neutral"


def _mock_emotion(reply_text: str) -> str:
    hits = []
    if "?" in reply_text:
        hits.append("curiosity")
    if any(w in reply_text for w in ("thank", "good point", "well said")):
        hits.append("appreciation")
    if any(w in reply_text for w in ("stupid", "idiot", "pathetic")):
        hits.append("contempt")
    if any(w in reply_text for w in ("hate", "furious", "!!")):
        hits.append("hostility")
    if "/s" in reply_text or "yeah right" in reply_text:
        hits.append("sarcasm")
    return ", ".join(hits) if hits else "none"


def _mock_intent(reply_text: str) -> str:
    if "?" in reply_text:
        return "information seeking"
    if any(w in reply_text for w in ("agree", "well said", "exactly")):
        return "support"
    if any(w in reply_text for w in ("but", "however", "actually")):
        return "counter-argue"
    if any(w in reply_text for w in ("why", "how is that", "prove")):
        return "challenge"
    return "others"


def make_stage2_mock_generator() -> Callable[[str], str]:
    def generate(prompt: str) -> str:
        lowered = prompt.lower()
        reply_text = lowered.split("reply:", maxsplit=1)[-1]

        if "classify the sentiment" in lowered:
            return _mock_sentiment(reply_text)
        if "identify which of the following categories" in lowered:
            return _mock_emotion(reply_text)
        if "classify the communicative intent" in lowered:
            return _mock_intent(reply_text)
        if "report all three" in lowered:
            return (
                f"Sentiment: {_mock_sentiment(reply_text)}\n"
                f"Emotion: {_mock_emotion(reply_text)}\n"
                f"Intent: {_mock_intent(reply_text)}"
            )
        raise ValueError("Unrecognized Stage 2 prompt for mock generator")

    return generate


# ---------------------------------------------------------------------------
# Dataframe-level driver + CLI
# ---------------------------------------------------------------------------


def annotate_stage2_dataframe(
    df: pd.DataFrame,
    mode: str,
    generator: Callable[[str], str],
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    done_row_ids: set = set()
    writer = None
    file_handle = None

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        # Resume support: a long LLM run over a remote/SSH connection can get
        # cut off partway through. If a previous partial run's output is
        # already on disk, keep its rows and append only the ones still
        # missing, instead of overwriting and re-generating from row 0.
        if output_csv.exists() and output_csv.stat().st_size > 0:
            existing = pd.read_csv(output_csv)
            rows.extend(existing.to_dict("records"))
            done_row_ids = set(existing["row_id"].tolist())
            file_handle = output_csv.open("a", newline="", encoding="utf-8")
            writer = csv.DictWriter(file_handle, fieldnames=list(existing.columns))
        else:
            file_handle = output_csv.open("w", newline="", encoding="utf-8")

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Stage 2 ({mode})"):
            if idx in done_row_ids:
                continue
            parent, reply = str(row["Parent"]), str(row["Reply"])

            if mode == "multi_module":
                out = run_multi_module_pipeline(parent, reply, generator)
            elif mode == "single_prompt":
                out = run_single_prompt_baseline(parent, reply, generator)
            else:
                raise ValueError(f"Unknown Stage 2 mode: {mode!r} (expected multi_module or single_prompt)")

            out = {"row_id": idx, "Parent": parent, "Reply": reply, **out}
            for gold_col in GOLD_COLUMNS:
                if gold_col in row:
                    out[f"gold_{gold_col.lower()}"] = row[gold_col]
            rows.append(out)

            if file_handle is not None:
                if writer is None:
                    writer = csv.DictWriter(file_handle, fieldnames=list(out.keys()))
                    writer.writeheader()
                elif list(out.keys()) != writer.fieldnames:
                    raise ValueError(
                        f"{output_csv} has different columns than this run would produce -- "
                        "it looks like a resume from an incompatible earlier run (different mode "
                        "or code version). Delete it to start fresh."
                    )
                writer.writerow(out)
                file_handle.flush()
    finally:
        if file_handle is not None:
            file_handle.close()

    return pd.DataFrame(rows)


def run_stage2(mode: str, split: str = "dev", limit: Optional[int] = None, mock: bool = False) -> pd.DataFrame:
    ensure_results_dirs()
    set_seed(42)

    csv_path = STAGE2_DEV_CSV if split == "dev" else STAGE2_EVAL_CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if limit is not None:
        df = df.head(limit).copy()

    # single_prompt's completion sometimes runs past its answer and
    # hallucinates a new "Parent: ..." example; cut generation off there so
    # the real answer can't be overwritten by a truncated duplicate field
    # (see _split_labeled_lines). 
    stop_strings = ["\nParent:"] if mode == "single_prompt" else None
    generator = (
        make_stage2_mock_generator()
        if mock
        else make_transformers_generator(max_new_tokens=STAGE2_MAX_NEW_TOKENS, stop_strings=stop_strings)
    )
    pred_path = STAGE2_DIR / f"{mode}_{split}_predictions.csv"
    return annotate_stage2_dataframe(df, mode, generator, output_csv=pred_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 sentiment/emotion/intent modules.")
    parser.add_argument("--mode", choices=["multi_module", "single_prompt"], default="multi_module")
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock outputs.")
    args = parser.parse_args()

    predictions = run_stage2(mode=args.mode, split=args.split, limit=args.limit, mock=args.mock)
    print(f"saved {len(predictions)} rows to {STAGE2_DIR / f'{args.mode}_{args.split}_predictions.csv'}")


if __name__ == "__main__":
    main()
