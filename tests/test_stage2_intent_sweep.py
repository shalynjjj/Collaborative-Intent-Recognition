import tempfile
import unittest
from pathlib import Path

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


class RunIntentSweepDataframeResumeTests(unittest.TestCase):
    def test_resumes_from_a_partial_output_file_without_regenerating_done_rows(self):
        df = pd.DataFrame(
            [
                {"Parent": "p0", "Reply": "r0", "Intent": "Support", "Dialogue_act": "agree", "Sentiment": "Positive"},
                {"Parent": "p1", "Reply": "r1", "Intent": "Challenge", "Dialogue_act": "disagree", "Sentiment": "Negative"},
            ]
        )
        calls = []

        def generator(prompt):
            calls.append(prompt)
            return "support"

        with tempfile.TemporaryDirectory() as tmp:
            output_csv = Path(tmp) / "sweep.csv"

            first = run_intent_sweep_dataframe(df.iloc[:1], generator, feature_source="oracle", output_csv=output_csv)
            self.assertEqual(len(first), 1)
            calls_after_first_run = len(calls)

            second = run_intent_sweep_dataframe(df, generator, feature_source="oracle", output_csv=output_csv)

            self.assertEqual(len(second), 2)
            self.assertEqual(sorted(second["row_id"].tolist()), [0, 1])
            # 8 combinations (base + 7) worth of fresh calls for the one new row only.
            self.assertEqual(len(calls) - calls_after_first_run, 8)


if __name__ == "__main__":
    unittest.main()
