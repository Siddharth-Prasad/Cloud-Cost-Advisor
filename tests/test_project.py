"""Unit tests. Run:  python -m unittest discover -s tests -v   (or: pytest)"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_data import generate_synthetic_data  # noqa: E402
from src import config as C  # noqa: E402
from src.anomaly import detect_anomalies  # noqa: E402
from src.data_preprocessing import (DataValidationError, clean_data, daily_totals,  # noqa: E402
                                    load_and_clean_data, time_based_split)
from src.forecasting import forecast_total_cost  # noqa: E402
from src.optimizer import (analyze_resource_efficiency, efficiency_score,  # noqa: E402
                           find_right_size, what_if)


def make_df(rid, cpu, mem, env="prod", itype="m5.xlarge", hours=24, days=40, storage=50):
    rows = []
    for d in pd.date_range("2026-01-01", periods=days):
        rows.append(dict(date=d, resource_id=rid, instance_type=itype, region="ap-south-1",
                         environment=env, cpu_utilization=cpu, memory_utilization=mem,
                         running_hours=hours, storage_gb=storage, network_gb=1.0, cost_usd=5.0))
    return pd.DataFrame(rows)


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "d.csv"
        generate_synthetic_data(days=60, n_resources=10, out_path=cls.path)

    def test_generation_and_cleaning(self):
        df, rep = load_and_clean_data(self.path, return_report=True)
        self.assertEqual(rep["duplicates_removed"], 3)
        self.assertFalse(df.isna().any().any())
        self.assertIn("instance_type_code", df.columns)

    def test_missing_column_raises(self):
        with self.assertRaises(DataValidationError):
            clean_data(pd.DataFrame({"date": ["2026-01-01"], "cost_usd": [1]}))

    def test_minimal_csv_gets_defaults(self):
        raw = make_df("a", 50, 50).drop(columns=["region", "environment", "network_gb"])
        df, _ = clean_data(raw)
        self.assertTrue((df["environment"] == "prod").all())

    def test_time_split_has_no_leakage(self):
        df = load_and_clean_data(self.path)
        train, test = time_based_split(df)
        self.assertLess(train["date"].max(), test["date"].min())


class TestOptimizer(unittest.TestCase):
    def test_idle_detected(self):
        r = analyze_resource_efficiency(make_df("idle", 2, 8, env="dev"))
        self.assertEqual(r.iloc[0]["Status"], "Idle")
        self.assertGreater(r.iloc[0]["Est. Monthly Savings ($)"], 0)

    def test_underutilized_rightsized(self):
        r = analyze_resource_efficiency(make_df("u", 10, 20, env="prod", itype="m5.2xlarge")).iloc[0]
        self.assertEqual(r["Status"], "Underutilized")
        self.assertNotEqual(r["Recommended Type"], "m5.2xlarge")
        self.assertLess(C.HOURLY_RATE[r["Recommended Type"]], C.HOURLY_RATE["m5.2xlarge"])

    def test_healthy_untouched(self):
        r = analyze_resource_efficiency(make_df("ok", 50, 60, env="staging", hours=12)).iloc[0]
        self.assertEqual(r["Status"], "Optimal")
        self.assertEqual(r["Est. Monthly Savings ($)"], 0)

    def test_offhours_schedule(self):
        r = analyze_resource_efficiency(make_df("dev1", 40, 50, env="dev", itype="m5.large")).iloc[0]
        self.assertIn("SCHEDULE", r["Rules Triggered"])

    def test_overloaded_flagged_as_risk(self):
        r = analyze_resource_efficiency(make_df("hot", 95, 90, itype="t3.micro")).iloc[0]
        self.assertEqual(r["Status"], "Overloaded")
        self.assertEqual(r["Est. Monthly Savings ($)"], 0)

    def test_no_double_counting(self):
        df = pd.concat([make_df("a", 2, 8, env="dev"), make_df("b", 10, 20, itype="m5.2xlarge")])
        r = analyze_resource_efficiency(df)
        self.assertLessEqual(r["Est. Monthly Savings ($)"].sum(), r["Monthly Cost ($)"].sum())
        self.assertTrue(0 <= efficiency_score(r) <= 100)

    def test_right_size_never_picks_costlier(self):
        self.assertIsNone(find_right_size("t3.micro", 90, 90))

    def test_what_if(self):
        r = analyze_resource_efficiency(make_df("idle", 2, 8, env="dev"))
        w = what_if(r, ["idle"])
        self.assertAlmostEqual(w["current_monthly"] - w["savings"], w["new_monthly"])


class TestForecastAnomaly(unittest.TestCase):
    def test_forecast_shape(self):
        d = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=60),
                          "total_cost": np.linspace(100, 130, 60)})
        fc = forecast_total_cost(d, 14)
        self.assertEqual((fc["kind"] == "forecast").sum(), 14)
        self.assertGreater(fc[fc.kind == "forecast"]["forecast"].iloc[-1], 125)

    def test_injected_spike_found(self):
        df = pd.concat([make_df(f"r{i}", 40, 50, days=60) for i in range(5)], ignore_index=True)
        df["cost_usd"] = df["cost_usd"] + np.random.default_rng(0).normal(0, 0.05, len(df))
        df.loc[10, "cost_usd"] = 60.0
        an = detect_anomalies(df)
        self.assertIn(60.0, an["cost_usd"].round(1).tolist())


class TestModel(unittest.TestCase):
    def test_training_pipeline(self):
        from src.train_model import train_and_save_model
        tmp = tempfile.TemporaryDirectory()
        p = Path(tmp.name) / "d.csv"
        generate_synthetic_data(days=60, n_resources=10, out_path=p)
        old = (C.MODEL_PATH, C.METRICS_PATH)
        C.MODEL_PATH, C.METRICS_PATH = Path(tmp.name) / "m.pkl", Path(tmp.name) / "m.json"
        try:
            s = train_and_save_model(p, verbose=False)
            self.assertTrue(C.MODEL_PATH.exists())
            self.assertIn(s["best_model"], s["results"])
            self.assertGreater(s["results"][s["best_model"]]["R2"], 0.3)
        finally:
            C.MODEL_PATH, C.METRICS_PATH = old


if __name__ == "__main__":
    unittest.main()
