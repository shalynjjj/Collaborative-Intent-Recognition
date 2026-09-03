import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import INTENT_LABELS
from src.stage2_pipeline import (
    _split_labeled_lines,
    annotate_stage2_dataframe,
    parse_single_label,
    run_multi_module_pipeline,
    run_single_prompt_baseline,
)


class Stage2ExperimentOneTests(unittest.TestCase):
    def test_multi_module_intent_receives_no_auxiliary_gold_labels(self):
        prompts = []

        def generator(prompt):
            prompts.append(prompt)
            if "Classify the sentiment" in prompt:
                return "neutral"
            if "Identify which of the following categories" in prompt:
                return "neutral"
            if "Classify the communicative intent" in prompt:
                return "others"
            raise AssertionError("unexpected prompt")

        run_multi_module_pipeline("parent text", "reply text", generator)

        intent_prompt = next(p for p in prompts if "Classify the communicative intent" in p)
        self.assertIn("Parent: parent text", intent_prompt)
        self.assertIn("Reply: reply text", intent_prompt)
        self.assertNotIn("Dialogue act:", intent_prompt)
        self.assertNotIn("Sentiment:", intent_prompt)
        self.assertNotIn("Emotion:", intent_prompt)


class SplitLabeledLinesTests(unittest.TestCase):
    def test_keeps_first_occurrence_of_a_repeated_key(self):
        # Reproduces results/stage2/single_prompt_eval_predictions.csv row_id
        # 12: the model answers correctly, then hallucinates a new
        # Parent/Reply example whose truncated Intent: line must not clobber
        # the real answer.
        raw = (
            "Sentiment: neutral\nEmotion: none\nIntent: challenge\n"
            "Parent: Your link indicates...\nReply: That's not the point...\n"
            "Sentiment: negative\nEmotion: contempt\nIntent:"
        )
        fields = _split_labeled_lines(raw)
        self.assertEqual(fields["sentiment"], "neutral")
        self.assertEqual(fields["emotion"], "none")
        self.assertEqual(fields["intent"], "challenge")


class ParseSingleLabelRecoveryTests(unittest.TestCase):
    def test_recovers_the_one_label_mentioned_in_free_text_reasoning(self):
        # Reproduces results/stage2/single_prompt_eval_predictions.csv row_id
        # 13: the model abandons the 3-line format and writes free-text
        # reasoning instead, but still commits to exactly one label ("support")
        # -- that must be used instead of discarding it as a fallback.
        raw = (
            "The reply agrees with and reinforces the parent's argument, so "
            "this is clearly a case of support rather than any kind of pushback."
        )
        label, fallback_used = parse_single_label(raw, INTENT_LABELS, "others")
        self.assertEqual(label, "Support")
        self.assertFalse(fallback_used)

    def test_still_falls_back_when_multiple_labels_are_echoed(self):
        # The model restates several/all candidate labels instead of
        # committing to one (e.g. rambling into a second hallucinated
        # example with a different answer) -- there is no way to tell which
        # one is the real answer, so this must still fall back rather than
        # guessing the earliest-mentioned one.
        raw = (
            "Sentiment: positive\nEmotion: none\nIntent: support\n\n"
            "Parent: another example\nReply: another reply\n"
            "Sentiment: negative\nEmotion: contempt\nIntent: challenge"
        )
        label, fallback_used = parse_single_label(raw, INTENT_LABELS, "others")
        self.assertEqual(label, "others")
        self.assertTrue(fallback_used)

    def test_still_falls_back_when_no_label_is_present_anywhere(self):
        raw = "If you have a Y chromosome, you're male. That's it."
        label, fallback_used = parse_single_label(raw, INTENT_LABELS, "others")
        self.assertEqual(label, "others")
        self.assertTrue(fallback_used)


class ParseSingleLabelNormalizationTests(unittest.TestCase):
    def test_matches_label_regardless_of_separator_spelling(self):
        for raw in [
            "Counter argue",  # space instead of hyphen
            "COUNTER_ARGUE",  # underscore, upper case
            "**Counter-Argue**",  # markdown emphasis around it
            "counter   -   argue",  # stray extra whitespace around the hyphen
        ]:
            with self.subTest(raw=raw):
                label, fallback_used = parse_single_label(raw, INTENT_LABELS, "others")
                self.assertEqual(label, "Counter-argue")
                self.assertFalse(fallback_used)

    def test_matches_multiword_label_across_a_line_break(self):
        raw = "Information\nseeking"
        label, fallback_used = parse_single_label(raw, INTENT_LABELS, "others")
        self.assertEqual(label, "Information seeking")
        self.assertFalse(fallback_used)

    def test_trailing_punctuation_does_not_block_a_match(self):
        raw = "Support."
        label, fallback_used = parse_single_label(raw, INTENT_LABELS, "others")
        self.assertEqual(label, "Support")
        self.assertFalse(fallback_used)


class AnnotateStage2DataframeResumeTests(unittest.TestCase):
    def test_resumes_from_a_partial_output_file_without_regenerating_done_rows(self):
        df = pd.DataFrame(
            [
                {"Parent": "p0", "Reply": "great point, I agree"},
                {"Parent": "p1", "Reply": "actually, here is why you're wrong"},
                {"Parent": "p2", "Reply": "why is that true?"},
            ]
        )
        calls = []

        def generator(prompt):
            calls.append(prompt)
            if "Classify the sentiment" in prompt:
                return "neutral"
            if "Identify which of the following categories" in prompt:
                return "neutral"
            return "others"

        with tempfile.TemporaryDirectory() as tmp:
            output_csv = Path(tmp) / "predictions.csv"

            first = annotate_stage2_dataframe(df.iloc[:2], "multi_module", generator, output_csv=output_csv)
            self.assertEqual(len(first), 2)
            calls_after_first_run = len(calls)

            second = annotate_stage2_dataframe(df, "multi_module", generator, output_csv=output_csv)

            self.assertEqual(len(second), 3)
            self.assertEqual(sorted(second["row_id"].tolist()), [0, 1, 2])
            # Only row 2 (the new one) should have triggered fresh LLM calls.
            self.assertEqual(len(calls) - calls_after_first_run, 3)

            on_disk = pd.read_csv(output_csv)
            self.assertEqual(len(on_disk), 3)

    def test_rejects_resuming_from_an_incompatible_output_file(self):
        df = pd.DataFrame([{"Parent": "p0", "Reply": "r0"}, {"Parent": "p1", "Reply": "r1"}])

        def generator(prompt):
            return "others" if "Classify the communicative intent" in prompt else "neutral"

        with tempfile.TemporaryDirectory() as tmp:
            output_csv = Path(tmp) / "predictions.csv"
            annotate_stage2_dataframe(df.iloc[:1], "multi_module", generator, output_csv=output_csv)
            with self.assertRaises(ValueError):
                annotate_stage2_dataframe(df, "single_prompt", lambda p: "Sentiment: neutral\nEmotion: none\nIntent: others", output_csv=output_csv)


class SinglePromptBaselineTests(unittest.TestCase):
    def test_hallucinated_continuation_does_not_overwrite_real_answer(self):
        raw = (
            "Sentiment: neutral\nEmotion: none\nIntent: challenge\n"
            "Parent: Your link indicates...\nReply: That's not the point...\n"
            "Sentiment: negative\nEmotion: contempt\nIntent:"
        )
        result = run_single_prompt_baseline("parent text", "reply text", lambda prompt: raw)
        self.assertEqual(result["sentiment"], "Neutral")
        self.assertEqual(result["intent"], "Challenge")
        self.assertFalse(result["sentiment_fallback"])
        self.assertFalse(result["intent_fallback"])


if __name__ == "__main__":
    unittest.main()
