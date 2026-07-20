import json
import tempfile
import unittest
from pathlib import Path

from src.plot_strategy_b_learning_curve import collect_learning_curve


class StrategyBLearningCurveTests(unittest.TestCase):
    def test_collects_only_standardized_weighted_real_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            for size in (500, 1000):
                for sample_seed in (42, 123):
                    metrics = {
                        "strategy": "B",
                        "experiment_version": "v2_group_split",
                        "silver_size": size,
                        "sample_seed": sample_seed,
                        "train_seed": 42,
                        "macro_f1": size / 10000 + sample_seed / 100000,
                        "use_class_weights": True,
                        "dry_run": False,
                    }
                    path = result_dir / (
                        f"silver_{size}_sample{sample_seed}_train42_weights1_metrics.json"
                    )
                    path.write_text(json.dumps(metrics), encoding="utf-8")
            table = collect_learning_curve(result_dir, expected_sizes=[500, 1000])
            self.assertEqual(table["silver_size"].tolist(), [500, 1000])
            self.assertEqual(table["n_runs"].tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()
