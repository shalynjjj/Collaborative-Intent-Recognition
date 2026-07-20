import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import train_roberta


class StrategyBTests(unittest.TestCase):
    def _silver_frame(self) -> pd.DataFrame:
        rows = []
        labels = ["agree", "disagree", "question", "statement"]
        for root in range(20):
            for reply in range(3):
                rows.append(
                    {
                        "source_root": f"root-{root}",
                        "reply_id": f"reply-{root}-{reply}",
                        "Parent": f"parent {root}",
                        "Reply": f"reply {reply}",
                        "pred_label": labels[(root + reply) % len(labels)],
                    }
                )
        return pd.DataFrame(rows)

    def test_group_split_has_no_source_root_overlap(self):
        frame = self._silver_frame()
        train, validation = train_roberta._make_inner_validation_split(
            frame, "pred_label", split_seed=42, group_col="source_root"
        )
        self.assertTrue(
            set(train["source_root"]).isdisjoint(set(validation["source_root"]))
        )

    def test_sample_seed_is_reproducible_and_nested(self):
        frame = self._silver_frame()
        first = train_roberta._sample_silver_subset(frame, 10, 42)
        repeated = train_roberta._sample_silver_subset(frame, 10, 42)
        larger = train_roberta._sample_silver_subset(frame, 20, 42)
        different = train_roberta._sample_silver_subset(frame, 10, 123)
        self.assertEqual(first["reply_id"].tolist(), repeated["reply_id"].tolist())
        self.assertEqual(first["reply_id"].tolist(), larger["reply_id"].head(10).tolist())
        self.assertNotEqual(first["reply_id"].tolist(), different["reply_id"].tolist())

    def test_summary_rebuild_unions_legacy_and_new_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            legacy = {
                "strategy": "B",
                "silver_size": 500,
                "seed": 42,
                "use_class_weights": True,
                "macro_f1": 0.25,
                "cohen_kappa": 0.1,
            }
            new = {
                "strategy": "B",
                "silver_size": 1000,
                "sample_seed": 42,
                "train_seed": 123,
                "use_class_weights": True,
                "macro_f1": 0.4,
                "cohen_kappa": 0.2,
            }
            (result_dir / "legacy_metrics.json").write_text(json.dumps(legacy))
            (result_dir / "new_metrics.json").write_text(json.dumps(new))
            with patch.object(train_roberta, "STRATEGY_B_DIR", result_dir):
                train_roberta.rebuild_strategy_b_summaries()
            runs = pd.read_csv(result_dir / "runs.csv")
            self.assertEqual(set(runs["silver_size"]), {500, 1000})
            self.assertTrue((result_dir / "summary_by_sample.csv").exists())


if __name__ == "__main__":
    unittest.main()
