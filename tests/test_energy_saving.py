import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from energy_saving.data import (
    CONTROL_COLUMNS, POWER_COLUMNS, POWER_GROUPS, RESPONSE_COLUMNS, STATE_COLUMNS,
    build_samples, cooling_kw, thermal_check,
)
from energy_saving.optimization import choose_candidate, energy_summary, optimize_day
from energy_saving.response import ResponseModel
from simulate_energy_saving_a import default_config, run


class EnergySavingTests(unittest.TestCase):
    def hourly(self):
        dates = pd.date_range("2025-07-01", periods=24 * 25, freq="h")
        h = pd.DataFrame(index=dates)
        for i, c in enumerate(POWER_COLUMNS):
            h[c] = 10.0 + i
        for group, columns in POWER_GROUPS.items():
            h[group] = h[columns].sum(axis=1)
        for c in CONTROL_COLUMNS:
            h[c] = 9.0 + np.sin(np.arange(len(h)) / 6) * 0.2
        h["OutdoorTdbin"], h["OutdoorWetTemp"] = 30.0, 25.0
        h["PriChWFlow"], h["CWFlow01"] = 100.0, 110.0
        h["PriChWTempSupply"], h["PriChWTempReturn"] = 9.0, 14.0
        h["CWTempSupply"], h["CWTempReturn"] = 30.0, 35.0
        h["TotalRealTimeLoad"] = cooling_kw(100, 9, 14)
        return h

    def test_model_excludes_evaluation_and_next_hour_history(self):
        h = self.hourly()
        h.loc["2025-07-16", CONTROL_COLUMNS] = 99
        samples = build_samples(h, "2025-07-16")
        self.assertFalse((samples["x"].index.normalize() == pd.Timestamp("2025-07-16")).any())
        self.assertNotIn(pd.Timestamp("2025-07-17"), samples["x"].index)
        self.assertEqual(len(samples["eval_x"]), 24)
        self.assertLess(samples["x"][CONTROL_COLUMNS].max().max(), 10)
        h = h.drop(pd.Timestamp("2025-07-03 05:00"))
        self.assertNotIn(pd.Timestamp("2025-07-03 06:00"), build_samples(h, "2025-07-16")["x"].index)

    def test_cooling_equation_and_infeasible_candidate_rejected(self):
        self.assertAlmostEqual(float(cooling_kw(3600, 10, 11)), 4186)
        pred = pd.DataFrame([
            [100, 20, 10, 100, 9, 14],   # baseline supplies581kW
            [1, 1, 1, 1, 10, 11],       # cheap but insufficient cooling
            [80, 20, 10, 100, 9, 14],
        ], columns=RESPONSE_COLUMNS)
        best, reason, reliable, _, _ = choose_candidate(pred, 500)
        self.assertEqual(best, 2)
        self.assertEqual(reason, "optimized")
        self.assertTrue(reliable)
        self.assertEqual(choose_candidate(pred, 600)[:3], (0, "baseline_cooling_infeasible", False))
        self.assertEqual(choose_candidate(pred, 500, error_margin_kw=30)[0], 0)

    def test_inconsistent_cooling_units_fail_balance_check(self):
        samples = build_samples(self.hourly(), "2025-07-16")
        self.assertTrue(thermal_check(samples["x"], samples["y"])["passed"])
        wrong = samples["y"].copy()
        wrong["PriChWFlow"] *= 1000
        self.assertFalse(thermal_check(samples["x"], wrong)["passed"])

    def test_supported_controls_keep_other_features_and_inactive_settings(self):
        samples = build_samples(self.hourly(), "2025-07-16")
        model = ResponseModel(n_estimators=5).fit(samples["x"], samples["y"])
        baseline = samples["eval_x"].iloc[0]
        candidates, reason, _ = model.supported_candidates(baseline, context_radius=5, joint_radius=5)
        self.assertEqual(reason, "supported")
        for c in candidates.columns:
            if c not in CONTROL_COLUMNS:
                self.assertTrue((candidates[c] == baseline[c]).all())
        self.assertTrue((candidates[CONTROL_COLUMNS].to_numpy() >= model.control_min).all())
        self.assertTrue((candidates[CONTROL_COLUMNS].to_numpy() <= model.control_max).all())

    def test_unchanging_setpoints_or_missing_state_support_return_baseline(self):
        hourly = self.hourly()
        hourly[CONTROL_COLUMNS] = 9.0
        samples = build_samples(hourly, "2025-07-16")
        model = ResponseModel(n_estimators=5).fit(samples["x"], samples["y"])
        baseline = samples["eval_x"].iloc[0]
        candidates, reason, _ = model.supported_candidates(baseline, context_radius=5, joint_radius=5)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(reason, "insufficient_control_variation")
        baseline = baseline.copy()
        baseline["on_ChPower01"] = 0
        self.assertEqual(model.supported_candidates(baseline)[1], "no_same_equipment_support")

    def test_energy_integrates_only_eligible_hours(self):
        detail = pd.DataFrame({
            "eligible": [True, False], "duration_hours": [1, 1],
            "baseline_power_kw": [100, 1000], "optimized_power_kw": [80, 1],
            "observed_power_kw": [110, 900],
        })
        summary = energy_summary(detail)
        self.assertEqual(summary["simulation_baseline_kwh"], 100)
        self.assertEqual(summary["simulation_optimized_kwh"], 80)
        self.assertEqual(summary["saving_percent"], 20)
        self.assertEqual(summary["coverage_percent"], 50)
        self.assertFalse(summary["full_day_reportable"])
        detail["eligible"] = False
        self.assertIsNone(energy_summary(detail)["saving_percent"])

    def test_controlled_model_optimizes_only_when_checks_pass(self):
        samples = build_samples(self.hourly(), "2025-07-16")
        class ControlledModel:
            def supported_candidates(self, baseline, **kwargs):
                candidate = baseline.copy()
                candidate[CONTROL_COLUMNS[0]] += 0.1
                return pd.DataFrame([baseline, candidate]), "supported", 10

            def predict(self, x):
                return pd.DataFrame([[100, 20, 10, 100, 9, 15], [80, 20, 10, 100, 9, 15]],
                                    columns=RESPONSE_COLUMNS)
        cfg = default_config()
        enabled = optimize_day(ControlledModel(), samples, cfg, True, 0)
        self.assertTrue(enabled.optimized.all())
        self.assertAlmostEqual(energy_summary(enabled)["saving_kwh"], 24 * 20)
        disabled = optimize_day(ControlledModel(), samples, cfg, False, 0)
        self.assertFalse(disabled.eligible.any())
        np.testing.assert_array_equal(disabled.baseline_power_kw, disabled.optimized_power_kw)

    def test_exports_and_failed_unit_check_withhold_savings(self):
        cfg = default_config()
        cfg.n_estimators = 5
        cfg.flow_unit = "unknown"
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = tmp
            _, summary, metadata = run(cfg, hourly=self.hourly())
            self.assertIsNone(summary["saving_percent"])
            self.assertFalse(metadata["global_checks_passed"])
            self.assertTrue(metadata["thermal_balance"]["passed"])
            for name in ("response_model.joblib", "model_validation.csv", "hourly_optimization.csv",
                         "energy_summary.csv", "analysis_metadata.json"):
                self.assertTrue((Path(tmp) / name).is_file())
            loaded = joblib.load(Path(tmp) / "response_model.joblib")
            self.assertEqual(loaded["response_names"], RESPONSE_COLUMNS)
            self.assertEqual(len(pd.read_csv(Path(tmp) / "hourly_optimization.csv")), 24)
            json.loads((Path(tmp) / "analysis_metadata.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
