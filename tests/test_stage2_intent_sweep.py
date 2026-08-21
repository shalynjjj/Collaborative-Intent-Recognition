import unittest

import pandas as pd

from src.stage2_intent_sweep import run_intent_sweep_dataframe


class Stage2PredictedSweepTests(unittest.TestCase):
    def test_predicted_mode_uses_predicted_not_gold_auxiliary_labels(self):
        prompts = []

        def generator(prompt):
            prompts.append(prompt)
            return "counter-argue"

        df = pd.DataFrame(
            [
                {
                    "Parent": "p",
                    "Reply": "r",
                    "Intent": "Counter-argue",
                    "Dialogue_act": "agree",
                    "Sentiment": "Positive",
                    "predicted_dialogue_act": "disagree",
                    "predicted_sentiment": "Negative",
                    "predicted_emotion_sarcasm": True,
                }
            ]
        )
        run_intent_sweep_dataframe(df, generator, feature_source="predicted")
        combined = "\n".join(prompts)
        self.assertIn("Dialogue act: disagree", combined)
        self.assertIn("Sentiment: Negative", combined)
        self.assertIn("Emotion: Sarcasm", combined)
        self.assertNotIn("Dialogue act: agree", combined)
        self.assertNotIn("Sentiment: Positive", combined)


if __name__ == "__main__":
    unittest.main()
