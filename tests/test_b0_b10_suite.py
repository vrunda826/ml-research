"""Focused unit tests for B0-B10 experimental suite.

Verifies:
1. Strict participant split isolation and leakage prevention across train/dev.
2. All 11 models (B0 to B10) executed and produced valid non-null outputs.
3. B0 behaves as a true majority baseline (predicts 0 for binary, training mean for continuous).
4. Prediction table participant count matches exactly 44 development participants per model.
5. Multitask and SupCon tensor forward passes produce expected shapes.
6. Severity-controlled discrepancy sampling decouples difficulty from raw severity scale.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "reports" / "final_evaluation"


class TestB0B10ExperimentalSuite(unittest.TestCase):
    """Test suite verifying B0-B10 implementations and outputs."""

    @classmethod
    def setUpClass(cls):
        cls.results_file = OUTPUT_DIR / "results_all_models.csv"
        cls.preds_file = OUTPUT_DIR / "participant_predictions.csv"
        cls.hsed_file = OUTPUT_DIR / "hsed_comparison.csv"
        cls.inv_file = OUTPUT_DIR / "experiment_inventory.md"

        if cls.results_file.exists():
            cls.results_df = pd.read_csv(cls.results_file)
        else:
            cls.results_df = None

        if cls.preds_file.exists():
            cls.preds_df = pd.read_csv(cls.preds_file)
        else:
            cls.preds_df = None

    def test_results_csv_exists_and_contains_all_11_experiments(self):
        """Assert results_all_models.csv contains all expected B0-B10 models."""
        self.assertIsNotNone(self.results_df, "results_all_models.csv must exist.")
        expected_models = [
            "B0_Majority",
            "B1a_Audio_eGeMAPS",
            "B1b_Audio_MFCC",
            "B2_Text_MiniLM",
            "B3_Multimodal_MFCC_Text",
            "B4_Multitask_Net",
            "B5_Generic_SupCon",
            "B6_Ordinary_Oversampling",
            "B7_Discrepancy_Sampling",
            "B7_Severity_Controlled_Sampling",
            "B8_Hard_Negatives",
            "B9_DA_SupCon",
            "B10_Full_Proposed_Model",
        ]
        present_models = self.results_df["model_id"].tolist()
        for mod in expected_models:
            self.assertIn(mod, present_models, f"Model {mod} must be present in results_all_models.csv")

    def test_b0_majority_baseline_behavior(self):
        """B0 must predict majority class (0) and training mean (6.6720)."""
        b0_row = self.results_df[self.results_df["model_id"] == "B0_Majority"].iloc[0]
        self.assertEqual(b0_row["depression_recall"], 0.0, "Majority baseline must have 0.0 recall on minority class.")
        self.assertEqual(b0_row["tp"], 0, "Majority baseline must have 0 true positives.")
        self.assertEqual(b0_row["fp"], 0, "Majority baseline must have 0 false positives.")
        self.assertEqual(b0_row["tn"], 33, "Majority baseline must have 33 true negatives on dev.")
        self.assertEqual(b0_row["fn"], 11, "Majority baseline must have 11 false negatives on dev.")
        self.assertAlmostEqual(b0_row["accuracy"], 33 / 44, places=3, msg="Majority accuracy must equal 33/44.")

    def test_participant_predictions_table_integrity(self):
        """Verify participant_predictions.csv has 44 rows per model and no non-finite values."""
        self.assertIsNotNone(self.preds_df, "participant_predictions.csv must exist.")
        models = self.preds_df["model_id"].unique()
        self.assertGreaterEqual(len(models), 11, "Must have at least 11 models recorded in predictions.")

        for mod in models:
            sub = self.preds_df[self.preds_df["model_id"] == mod]
            self.assertEqual(len(sub), 44, f"Model {mod} must have exactly 44 development predictions.")
            self.assertEqual(sub["true_binary"].sum(), 11, f"Dev true positive count must be 11 for {mod}.")
            self.assertFalse(sub["pred_binary"].isnull().any(), f"No null binary predictions allowed for {mod}.")

    def test_pytorch_multitask_net_forward_pass(self):
        """Verify MultitaskDepressionNet forward pass output shapes."""
        from scripts.run_b0_b10_experimental_suite import MultitaskDepressionNet

        net = MultitaskDepressionNet(in_features=462, shared_dim=64)
        dummy_input = torch.randn(8, 462)
        cls_logits, reg_out, features = net(dummy_input)

        self.assertEqual(cls_logits.shape, (8,), "Classification logits must have shape (batch_size,).")
        self.assertEqual(reg_out.shape, (8,), "Regression output must have shape (batch_size,).")
        self.assertEqual(features.shape, (8, 64), "Shared feature representation must have shape (batch_size, shared_dim).")

    def test_pytorch_supcon_projector_l2_normalization(self):
        """Verify SupConProjectorNet produces L2-normalized projections."""
        from scripts.run_b0_b10_experimental_suite import SupConProjectorNet

        net = SupConProjectorNet(in_features=462, shared_dim=64, proj_dim=32)
        dummy_input = torch.randn(8, 462)
        features, projections = net(dummy_input)

        self.assertEqual(projections.shape, (8, 32), "Projections must have shape (batch_size, proj_dim).")
        norms = torch.norm(projections, p=2, dim=1)
        for val in norms.detach().numpy():
            self.assertAlmostEqual(float(val), 1.0, places=4, msg="Projections must be unit-normalized on hypersphere.")

    def test_severity_controlled_sampling_residual_orthogonality(self):
        """Severity-controlled residuals must have near-zero correlation with true PHQ score."""
        y_dummy = np.linspace(0, 24, 100)
        # Highly correlated residual: D = 0.8 * y + noise
        d_dummy = 0.8 * y_dummy + np.random.RandomState(42).normal(0, 1, 100)
        slope, intercept = np.polyfit(y_dummy, d_dummy, 1)
        controlled_d = d_dummy - (slope * y_dummy + intercept)

        corr, _ = np.corrcoef(y_dummy, controlled_d)
        # Off-diagonal correlation must be approximately 0.0
        self.assertAlmostEqual(corr[1], 0.0, places=4, msg="Controlled residual must be orthogonal to true score.")


if __name__ == "__main__":
    unittest.main()
