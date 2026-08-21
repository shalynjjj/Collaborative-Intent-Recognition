import unittest

from src.stage2_pipeline import run_multi_module_pipeline


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


if __name__ == "__main__":
    unittest.main()
