import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import DIALOGUE_LABELS
from src.train_roberta import (
    _strategy_c_output_dir,
    _summarize_strategy_c_oof,
)


class StrategyCTests(unittest.TestCase):
    def test_output_directories_separate_weight_configs(self):
        weighted = _strategy_c_output_dir("roberta", True, False)
        unweighted = _strategy_c_output_dir("roberta", False, False)
        dry_run = _strategy_c_output_dir("roberta", True, True)
        self.assertNotEqual(weighted, unweighted)
        self.assertNotEqual(weighted, dry_run)

    def test_oof_summary_requires_and_writes_300_unique_rows_per_seed(self):
        frames = []
        fold_rows = []
        for fold in range(1, 6):
            start = (fold - 1) * 60
            gold_indices = list(range(start, start + 60))
            labels = [DIALOGUE_LABELS[index % 4] for index in gold_indices]
            frames.append(
                pd.DataFrame(
                    {
                        "gold_index": gold_indices,
                        "Parent": ["parent"] * 60,
                        "Reply": ["reply"] * 60,
                        "Dialogue_act": labels,
                        "pred_label": labels,
                    }
                )
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "train_seed": 42,
                    "macro_f1": 1.0,
                    "cohen_kappa": 1.0,
                    "predicted_class_count": 4,
                    "missing_prediction_classes": "",
                    "class_collapse": False,
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = _summarize_strategy_c_oof(
                output_dir=output_dir,
                fold_rows=fold_rows,
                predictions_by_seed={42: frames},
                model_type="test",
                use_class_weights=True,
                split_seed=42,
                dry_run=False,
            )
            oof = pd.read_csv(output_dir / "oof_seed42_predictions.csv")
            self.assertEqual(len(oof), 300)
            self.assertEqual(oof["gold_index"].nunique(), 300)
            self.assertEqual(summary.loc[0, "macro_f1_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
