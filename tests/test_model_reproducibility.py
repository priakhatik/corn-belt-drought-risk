"""End-to-end checks for reproducible model artifacts and split validation."""

import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


FEATURES = (
    "ndvi_mean", "evi_mean", "ndvi_lag1", "ndvi_change", "ndvi_roll3",
    "precip_deficit_4wk", "tmax_roll4wk", "tmax_anomaly_roll4wk", "precip_roll4wk",
)
SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "model.py"


class ModelReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "project"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "processed").mkdir()
        shutil.copy2(SOURCE, self.root / "scripts" / "model.py")

    def write_data(self, single_class_test=False):
        rows = []
        for i, year in enumerate((2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022)):
            label = i % 2
            if single_class_test and year >= 2021:
                label = 0
            row = {"date": f"{year}-07-01", "stressed": label}
            row.update({name: float(i + 1 + j / 10) for j, name in enumerate(FEATURES)})
            rows.append(row)
        with (self.root / "processed" / "corn_belt_features.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["date", "stressed", *FEATURES])
            writer.writeheader()
            writer.writerows(rows)

    def run_model(self):
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        return subprocess.run(
            [sys.executable, str(self.root / "scripts" / "model.py")],
            cwd=self.tmp.name, env=env, capture_output=True, text=True, timeout=120,
        )

    def test_model_runs_outside_repo_and_creates_artifacts(self):
        self.write_data()
        result = self.run_model()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for filename in ("confusion_matrix.png", "feature_importance.png",
                         "roc_curve.png", "drought_risk_rf_model.pkl"):
            self.assertTrue((self.root / "processed" / "models" / filename).is_file(), filename)

    def test_single_class_test_period_fails_before_fitting(self):
        self.write_data(single_class_test=True)
        result = self.run_model()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("test labels must contain both binary classes", result.stderr)
        self.assertFalse((self.root / "processed" / "models").exists())


if __name__ == "__main__":
    unittest.main()
