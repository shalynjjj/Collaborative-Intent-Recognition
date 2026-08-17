import argparse
import csv
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


def parse_single_label(raw_output: str, labels: List[str], fallback: str) -> Tuple[str, bool]:
    text = _first_line(raw_output).lower()
    best_label, best_pos = None, None
    for label in labels:
        pos = text.find(label.lower())
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_label, best_pos = label, pos
    if best_label is not None:
        return best_label, False
    return fallback, True


def parse_emotion_labels(raw_output: str) -> Tuple[Dict[str, bool], bool]:
    text = _first_line(raw_output).lower()
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
    fields: Dict[str, str] = {}
    for line in (raw_output or "").splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip().lower()] = value.strip()
    return fields


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def build_sentiment_prompt(parent: str, reply: str) -> str:
    return f"""Classify the sentiment of the Reddit CMV reply toward the parent comment.
Allowed labels: positive, negative, neutral.

Definitions:
- positive: the reply expresses approval, agreement in tone, or a favorable attitude.
- negative: the reply expresses disapproval, frustration, or an unfavorable attitude.
- neutral: the reply is matter-of-fact, with no clear positive or negative tone.

Return only one label.

Parent: {parent}
Reply: {reply}
Sentiment:"""


def build_emotion_prompt(parent: str, reply: str) -> str:
    return f"""Identify which of the following categories apply to the Reddit CMV reply. One or more categories may apply.
Categories: sarcasm, hostility, contempt, curiosity, appreciation, neutral.

Definitions:
- sarcasm: the reply says the opposite of what it means, or mocks the parent through exaggeration or irony.
- hostility: the reply is openly aggressive, angry, or attacking.
- contempt: the reply is dismissive or belittling toward the parent or its argument.
- curiosity: the reply asks for clarification, evidence, or more information out of genuine interest.
- appreciation: the reply thanks, praises, or acknowledges the parent's point positively.
- neutral: none of the other categories apply; the reply is emotionally flat. Always include this
  category by itself if no other category applies -- do not answer "none".

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
- information seeking: the reply primarily asks a question to get clarification or evidence.
- challenge: the reply pushes back, dismisses, or mocks the parent's claim WITHOUT presenting its
  own reasoning or alternative claim -- short rebuttals, sarcastic jabs, or "prove it" pushback with
  no new argument all count here, even if they read as confident or dismissive.
- counter-argue: the reply pushes back AND presents its own reasoning, evidence, or alternative
  claim to support that pushback -- there must be actual argument content beyond disagreement itself.
- support: the reply agrees with and reinforces the parent's position.
- others: the reply does not engage with the parent's argument at all -- off-topic remarks, jokes
  with no argumentative point, or meta-commentary about the thread/forum itself (e.g. mentioning
  mods, bots, deltas, or other posts) all count here, even if they are on-topic in subject matter.

Return only one label.
{context}
Parent: {parent}
Reply: {reply}
Intent:"""


def build_single_prompt_baseline(parent: str, reply: str) -> str:
    return f"""Analyze the Reddit CMV reply and report all three of the following, using exactly this format (one per line):
Sentiment: <positive|negative|neutral>
Emotion: <comma-separated subset of sarcasm, hostility, contempt, curiosity, appreciation, neutral, or "none">
Intent: <information seeking|challenge|counter-argue|support|others>

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
    dialogue_act: Optional[str] = None,
) -> Dict:
    sentiment, sentiment_fallback, sentiment_raw = run_sentiment_module(parent, reply, generator)
    emotion_labels, emotion_fallback, emotion_raw = run_emotion_module(parent, reply, generator)
    intent, intent_fallback, intent_raw = run_intent_module(
        parent, reply, generator, dialogue_act=dialogue_act
    )
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
    writer = None
    file_handle = None
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        file_handle = output_csv.open("w", newline="", encoding="utf-8")

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Stage 2 ({mode})"):
            parent, reply = str(row["Parent"]), str(row["Reply"])

            if mode == "multi_module":
                out = run_multi_module_pipeline(parent, reply, generator, dialogue_act=row.get("Dialogue_act"))
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

    generator = (
        make_stage2_mock_generator()
        if mock
        else make_transformers_generator(max_new_tokens=STAGE2_MAX_NEW_TOKENS)
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
