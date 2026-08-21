import unittest

import pandas as pd

from src.ablation_parent_negation import summarize_paired_results


class TestParentNegationAblation(unittest.TestCase):
    def test_paired_summary_distinguishes_match_from_response(self):
        rows = [
            {"example_id": "a", "variant": "original", "seed": 1, "Dialogue_act": "agree", "pred_label": "disagree"},
            {"example_id": "a", "variant": "edited", "seed": 1, "Dialogue_act": "disagree", "pred_label": "disagree"},
            {"example_id": "a", "variant": "original", "seed": 2, "Dialogue_act": "agree", "pred_label": "statement"},
            {"example_id": "a", "variant": "edited", "seed": 2, "Dialogue_act": "disagree", "pred_label": "disagree"},
        ]

        summary = summarize_paired_results(pd.DataFrame(rows)).iloc[0]

        self.assertEqual(summary["n_pairs"], 2)
        self.assertEqual(summary["n_prediction_unchanged"], 1)
        self.assertEqual(summary["n_prediction_changed"], 1)
        self.assertEqual(summary["n_edited_matches_gold"], 2)
        self.assertEqual(summary["n_changed_to_expected"], 1)


if __name__ == "__main__":
    unittest.main()
