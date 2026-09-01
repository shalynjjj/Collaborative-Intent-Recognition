import unittest

from src.stage2_pipeline import _split_labeled_lines, run_multi_module_pipeline, run_single_prompt_baseline


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
