import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.summarize_strategy_c_grid import collect_grid_results


class StrategyCGridTests(unittest.TestCase):
    def _write_summary(self, result_dir: Path, run_dir_name: str, macro_f1_mean: float) -> None:
        run_dir = result_dir / run_dir_name
        run_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "model_type": "roberta",
                    "use_class_weights": True,
                    "macro_f1_mean": macro_f1_mean,
                    "macro_f1_std": 0.01,
                    "cohen_kappa_mean": 0.5,
                    "cohen_kappa_std": 0.01,
                    "oof_class_collapse_seeds": 0,
                }
            ]
        ).to_csv(run_dir / "summary_oof.csv", index=False)

    def test_ranks_configs_by_macro_f1_best_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            self._write_summary(result_dir, "roberta_weights_warmup20_epochs8", 0.5309)
            self._write_summary(result_dir, "roberta_weights_warmup10_epochs6", 0.55)
            self._write_summary(result_dir, "roberta_weights_warmup30_epochs10", 0.48)

            table = collect_grid_results(result_dir)

            self.assertEqual(
                table["run_dir"].tolist(),
                [
                    "roberta_weights_warmup10_epochs6",
                    "roberta_weights_warmup20_epochs8",
                    "roberta_weights_warmup30_epochs10",
                ],
            )
            self.assertEqual(table.loc[0, "warmup_steps"], 10)
            self.assertEqual(table.loc[0, "epochs"], 6)

    def test_ignores_non_grid_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            self._write_summary(result_dir, "roberta_weights_warmup20_epochs8", 0.5309)
            (result_dir / "roberta_weights").mkdir()
            pd.DataFrame([{"macro_f1_mean": 0.99}]).to_csv(
                result_dir / "roberta_weights" / "summary_oof.csv", index=False
            )

            table = collect_grid_results(result_dir)

            self.assertEqual(len(table), 1)
            self.assertEqual(table.loc[0, "run_dir"], "roberta_weights_warmup20_epochs8")


if __name__ == "__main__":
    unittest.main()
