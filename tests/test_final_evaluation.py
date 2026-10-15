"""Focused automated unit tests for Final Evaluation deliverables and Proposed Method."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
FINAL_EVAL_DIR = ROOT_DIR / "reports" / "final_evaluation"
PLOTS_DIR = FINAL_EVAL_DIR / "plots"


class TestFinalEvaluation(unittest.TestCase):
    """Unit tests verifying final evaluation tables, proposed method, and claims."""

    def setUp(self):
        self.comp_path = FINAL_EVAL_DIR / "final_model_comparison.csv"
        self.claims_path = FINAL_EVAL_DIR / "final_hsed_analysis.csv"
        self.summary_path = FINAL_EVAL_DIR / "final_results_summary.json"
        self.log_path = FINAL_EVAL_DIR / "experiment_log.md"

    # 1. Final comparison table integrity
    def test_01_final_model_comparison_integrity(self):
        self.assertTrue(self.comp_path.exists())
        df = pd.read_csv(self.comp_path)

        # Must contain all 6 baselines and proposed method variants
        self.assertGreaterEqual(len(df), 20)
        self.assertIn("Baseline_none", df["model_id"].values)
        self.assertIn("Baseline_mm_mfcc_text", df["model_id"].values)
        self.assertIn("Proposed_mm_mfcc_text_continuous_g1.0", df["model_id"].values)

        # Participant counts must strictly match cohort
        self.assertTrue((df["n_train_participants"] == 125).all())
        self.assertTrue((df["n_dev_participants"] == 44).all())

        # Baseline Dummy must have zero delta vs itself
        dummy_row = df[df["model_id"] == "Baseline_none"].iloc[0]
        self.assertAlmostEqual(dummy_row["delta_mae_vs_dummy_oof"], 0.0)
        self.assertAlmostEqual(dummy_row["delta_mae_vs_dummy_dev"], 0.0)

    # 2. Proposed Method: DAWR systematically reduces underprediction on high-PHQ cases
    def test_02_dawr_reduces_high_phq_underprediction(self):
        df = pd.read_csv(self.comp_path)

        base_disc = df[df["model_id"] == "Baseline_mm_mfcc_text"]["dev_dep_mean_discrepancy"].iloc[0]
        dawr_g05_disc = df[df["model_id"] == "Proposed_mm_mfcc_text_continuous_g0.5"]["dev_dep_mean_discrepancy"].iloc[0]
        dawr_g10_disc = df[df["model_id"] == "Proposed_mm_mfcc_text_continuous_g1.0"]["dev_dep_mean_discrepancy"].iloc[0]
        dawr_g20_disc = df[df["model_id"] == "Proposed_mm_mfcc_text_continuous_g2.0"]["dev_dep_mean_discrepancy"].iloc[0]

        # Monotonic reduction in underprediction gap
        self.assertLess(dawr_g05_disc, base_disc)
        self.assertLess(dawr_g10_disc, dawr_g05_disc)
        self.assertLess(dawr_g20_disc, dawr_g10_disc)

        # Total reduction at gamma=1.0 exceeds 0.8 points
        reduction = base_disc - dawr_g10_disc
        self.assertGreater(reduction, 0.8)

    # 3. Claims A, B, C evaluation table structure and finite-sample p-value
    def test_03_claims_analysis_structure_and_finite_sample_p(self):
        self.assertTrue(self.claims_path.exists())
        df = pd.read_csv(self.claims_path)

        claims = set(df["claim_id"].unique())
        self.assertEqual(claims, {"Claim_A", "Claim_B", "Claim_C"})

        # For Claim B, finite-sample corrected p-value must be in (0, 1]
        claim_b = df[df["claim_id"] == "Claim_B"]
        for _, row in claim_b.iterrows():
            pval = row["finite_sample_p_value"]
            self.assertTrue(np.isfinite(pval))
            self.assertGreater(pval, 0.0, "Finite-sample p-value must be strictly positive!")
            self.assertLessEqual(pval, 1.0)

    # 4. Presentation-ready plots exist and are non-empty
    def test_04_presentation_plots_exist(self):
        expected_plots = [
            PLOTS_DIR / "baseline_vs_proposed_comparison.png",
            PLOTS_DIR / "regression_to_the_mean_residual_bins.png",
            PLOTS_DIR / "high_phq_underprediction_gap.png",
            PLOTS_DIR / "permutation_null_distribution.png",
        ]
        for p in expected_plots:
            self.assertTrue(p.exists(), f"Missing plot: {p}")
            self.assertGreater(p.stat().st_size, 10000, f"Plot file suspiciously small: {p}")

    # 5. Summary JSON and Experiment Log exist
    def test_05_summary_and_log_exist(self):
        self.assertTrue(self.summary_path.exists())
        self.assertTrue(self.log_path.exists())
        self.assertGreater(self.summary_path.stat().st_size, 0)
        self.assertGreater(self.log_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
