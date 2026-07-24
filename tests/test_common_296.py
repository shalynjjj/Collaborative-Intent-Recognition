import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluate_common_296 import evaluate_strategy_b
from src.llm_annotate import FEWSHOT_EXAMPLES, exclude_fewshot_examples


class CommonBenchmarkTests(unittest.TestCase):
    def test_excludes_only_the_four_fewshot_demonstrations(self):
        ordinary = pd.DataFrame(
            [{"Parent": f"parent-{index}", "Reply": f"reply-{index}"} for index in range(296)]
        )
        demonstrations = pd.DataFrame(
            [
                {"Parent": example["parent"], "Reply": example["reply"]}
                for example in FEWSHOT_EXAMPLES
            ]
        )
        frame = pd.concat([ordinary, demonstrations], ignore_index=True)
        filtered = exclude_fewshot_examples(frame)
        self.assertEqual(len(filtered), 296)
        self.assertEqual(filtered["Parent"].tolist(), ordinary["Parent"].tolist())

    def test_strategy_b_common_summary_separates_weight_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            rows = [
                {
                    "Parent": f"parent-{index}",
                    "Reply": f"reply-{index}",
                    "Dialogue_act": "agree" if index % 2 else "disagree",
                    "pred_label": "agree" if index % 2 else "disagree",
                }
                for index in range(296)
            ]
            for use_weights in (False, True):
                prefix = f"silver_2500_sample42_train42_weights{int(use_weights)}"
                pd.DataFrame(rows).to_csv(result_dir / f"{prefix}_predictions.csv", index=False)
                metadata = {
                    "strategy": "B",
                    "experiment_version": "v2_group_split",
                    "silver_size": 2500,
                    "sample_seed": 42,
                    "train_seed": 42,
                    "use_class_weights": use_weights,
                    "dry_run": False,
                }
                (result_dir / f"{prefix}_metrics.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )

            runs, summary = evaluate_strategy_b(result_dir)
            self.assertEqual(len(runs), 2)
            self.assertEqual(set(summary["use_class_weights"]), {False, True})
            self.assertTrue((result_dir / "common_296_runs.csv").exists())
            self.assertTrue((result_dir / "common_296_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
