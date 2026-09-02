import unittest

import numpy as np
import pandas as pd
import torch

from equipment_forecasting.data import build_equipment_windows
from equipment_forecasting.metrics import grouped_metrics
from equipment_forecasting.model import (
    EquipmentResponseTransformer,
    concat_outputs,
    grouped_masked_huber_loss,
)
from equipment_forecasting.targets import (
    EQUIPMENT_TARGET_NAMES,
    FUTURE_PLAN_NAMES,
    build_equipment_masks,
    build_future_control_plan,
    derive_equipment_targets,
)
from load_forecasting.data import WAVELET_COLUMNS


class EquipmentForecastingTests(unittest.TestCase):
    def _frame(self, periods=80):
        time = pd.date_range("2026-01-01", periods=periods, freq="h")
        data = {"timeStamp": time}
        required = set(WAVELET_COLUMNS)
        required.update(EQUIPMENT_TARGET_NAMES)
        for i in range(1, 4):
            required.update({
                f"ChAMPS0{i}", f"ChChWTempSupplySetPoint0{i}",
                f"ChEnterEvapTemp0{i}", f"ChLeaveEvapTemp0{i}",
                f"ChLeaveCondTemp0{i}", f"ChEnterCondTemp0{i}",
                f"ChRealtimeEfficiencyKW0{i}", f"ChPower0{i}",
                f"PriChWPPower0{i}", f"CWPPower0{i}", f"CTPower0{i}",
                f"CTOutletTemp0{i}", f"PriChWPVSDFreq0{i}",
                f"CWPVSDFreq0{i}", f"CTVSDFreq0{i}",
            })
        required.update({"CWTempReturn01", "CWTempSupply01"})
        for index, name in enumerate(required):
            data[name] = np.full(periods, float(index + 1))
        data["TotalRealTimeLoad"] = np.arange(periods, dtype=float) + 100
        data["OutdoorTdbin"] = np.arange(periods, dtype=float) + 20
        data["OutdoorWetTemp"] = np.arange(periods, dtype=float) + 15
        for i in range(1, 4):
            data[f"ChAMPS0{i}"] = np.ones(periods)
            data[f"ChEnterEvapTemp0{i}"] = np.full(periods, 10.0)
            data[f"ChLeaveEvapTemp0{i}"] = np.full(periods, 7.0)
            data[f"ChLeaveCondTemp0{i}"] = np.full(periods, 35.0)
            data[f"ChEnterCondTemp0{i}"] = np.full(periods, 30.0)
            data[f"PriChWPVSDFreq0{i}"] = np.full(periods, 40.0)
            data[f"CWPVSDFreq0{i}"] = np.full(periods, 40.0)
            data[f"CTVSDFreq0{i}"] = np.full(periods, 40.0)
        data["PriChWTempReturn01"] = np.full(periods, 12.0)
        data["PriChWTempSupply01"] = np.full(periods, 7.0)
        data["CWTempReturn01"] = np.full(periods, 35.0)
        data["CWTempSupply01"] = np.full(periods, 30.0)
        return pd.DataFrame(data)

    def test_target_and_plan_schemas_have_expected_dimensions(self):
        frame = self._frame(24)
        targets = derive_equipment_targets(frame)
        plan = build_future_control_plan(frame)

        self.assertEqual(len(EQUIPMENT_TARGET_NAMES), 28)
        self.assertEqual(len(FUTURE_PLAN_NAMES), 24)
        self.assertEqual(targets.shape, (24, 28))
        self.assertEqual(plan.shape, (24, 24))
        self.assertEqual(targets["EvapDeltaT01"].iloc[0], 3.0)
        self.assertEqual(targets["CondDeltaT01"].iloc[0], 5.0)
        self.assertEqual(targets["PriChWDeltaT"].iloc[0], 5.0)

    def test_masks_disable_efficiency_when_chiller_is_off(self):
        frame = self._frame(2)
        frame.loc[0, "ChAMPS01"] = 0
        masks = build_equipment_masks(frame)

        self.assertEqual(masks.shape, (2, 28))
        self.assertEqual(masks[0, EQUIPMENT_TARGET_NAMES.index("ChPower01")], 1)
        self.assertEqual(
            masks[0, EQUIPMENT_TARGET_NAMES.index("ChRealtimeEfficiencyKW01")], 0
        )
        self.assertEqual(masks[0, EQUIPMENT_TARGET_NAMES.index("EvapDeltaT01")], 0)

    def test_build_equipment_windows_returns_expected_shapes(self):
        history, plan, target, mask, history_names = build_equipment_windows(
            self._frame(), 56, 24,
            history_columns=["TotalRealTimeLoad", "OutdoorTdbin"],
        )

        self.assertEqual(history.shape, (1, 56, 16))
        self.assertEqual(plan.shape, (1, 24, 24))
        self.assertEqual(target.shape, (1, 24, 28))
        self.assertEqual(mask.shape, (1, 24, 28))
        self.assertEqual(len(history_names), 16)

    def test_multitask_model_outputs_four_groups(self):
        model = EquipmentResponseTransformer(
            history_dim=16, plan_dim=24, d_model=32, nhead=4, num_layers=1
        )
        output = model(torch.randn(2, 56, 16), torch.randn(2, 24, 24))

        self.assertEqual(tuple(output["chiller"].shape), (2, 24, 12))
        self.assertEqual(tuple(output["pump"].shape), (2, 24, 8))
        self.assertEqual(tuple(output["tower"].shape), (2, 24, 6))
        self.assertEqual(tuple(output["system"].shape), (2, 24, 2))
        self.assertEqual(tuple(concat_outputs(output).shape), (2, 24, 28))

    def test_grouped_loss_ignores_masked_targets(self):
        prediction = torch.zeros(1, 2, 28)
        target = torch.zeros_like(prediction)
        mask = torch.ones_like(prediction)
        target[..., 3] = 1000
        mask[..., 3] = 0

        loss = grouped_masked_huber_loss(prediction, target, mask)

        self.assertEqual(float(loss), 0.0)

    def test_metrics_are_reported_per_group(self):
        true = np.ones((2, 3, 28)) * 10
        pred = true + 1
        mask = np.ones_like(true)

        result = grouped_metrics(pred, true, mask)

        self.assertEqual(set(result["groups"]), {"chiller", "pump", "tower", "system"})
        self.assertAlmostEqual(result["groups"]["chiller"]["mape"], 10.0)
        self.assertEqual(len(result["targets"]), 28)


if __name__ == "__main__":
    unittest.main()
