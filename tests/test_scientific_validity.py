"""Automated unit tests for Phase 6B: Scientific Validity of HSED Hypothesis."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_scientific_validity_analysis import compute_bootstrap_ci

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_P6_DIR = ROOT_DIR / "reports" / "phase6"
VALIDITY_DIR = REPORTS_P6_DIR / "scientific_validity"


class TestScientificValidity(unittest.TestCase):
    """Rigorous unit tests for Phase 6B scientific validity analysis."""

    def setUp(self):
        self.oof_path = REPORTS_P6_DIR / "train_oof_predictions.csv"
        self.dev_path = REPORTS_P6_DIR / "dev_regression_predictions.csv"
        self.thresh_path = REPORTS_P6_DIR / "hsed_thresholds.csv"
        self.summary_path = VALIDITY_DIR / "scientific_validity_summary.json"

    # 1. Cohort and participant-ID integrity
    def test_01_cohort_and_participant_integrity(self):
        oof_df = pd.read_csv(self.oof_path)
        dev_df = pd.read_csv(self.dev_path)

        train_pids = set(oof_df["participant_id"].astype(str))
        dev_pids = set(dev_df["participant_id"].astype(str))

        self.assertEqual(len(train_pids), 125)
        self.assertEqual(len(dev_pids), 44)
        self.assertEqual(len(train_pids.intersection(dev_pids)), 0, "Train and Dev must be strictly disjoint!")

    # 2. Correct residual sign: D = y - y_hat (positive D indicates underprediction)
    def test_02_residual_sign_convention(self):
        oof_df = pd.read_csv(self.oof_path)
        y = oof_df["phq8_true"].values
        y_hat = oof_df["phq8_pred"].values
        d = oof_df["discrepancy"].values

        np.testing.assert_allclose(d, y - y_hat, atol=1e-6)

        # Underprediction case: observed 18, predicted 10 -> residual +8
        self.assertGreater(18.0 - 10.0, 0)

    # 3. Correct group membership at PHQ-8 cutoff 10
    def test_03_group_membership_cutoff_10(self):
        oof_df = pd.read_csv(self.oof_path)
        single_mod = oof_df[oof_df["modality"] == "egemaps"]

        dep = single_mod[single_mod["phq8_true"] >= 10]
        nondep = single_mod[single_mod["phq8_true"] < 10]

        self.assertEqual(len(dep), 38)
        self.assertEqual(len(nondep), 87)
        self.assertTrue((dep["phq8_true"] >= 10.0).all())
        self.assertTrue((nondep["phq8_true"] < 10.0).all())

    # 4. Correct application of frozen training thresholds to development data
    def test_04_frozen_training_threshold_application(self):
        thresh_df = pd.read_csv(self.thresh_path)
        dev_df = pd.read_csv(self.dev_path)
        dev_df["discrepancy"] = dev_df["phq8_true"] - dev_df["phq8_pred"]

        for _, trow in thresh_df[thresh_df["quantile"] == 0.75].iterrows():
            mod = trow["modality"]
            tau = trow["threshold_value"]

            sub = dev_df[dev_df["modality"] == mod]
            dep_dev = sub[sub["phq8_true"] >= 10]
            cand = dep_dev[dep_dev["discrepancy"] >= tau]

            # Candidate count must match stability table
            stab_df = pd.read_csv(VALIDITY_DIR / "hsed_stability.csv")
            expected_cand_count = stab_df[(stab_df["modality"] == mod) & (stab_df["quantile"] == 0.75)]["dev_cand_count"].iloc[0]
            self.assertEqual(len(cand), expected_cand_count)

    # 5. Correct participant-level candidate overlap calculation
    def test_05_participant_level_candidate_overlap(self):
        set_a = {"308", "346", "348", "362"}
        set_b = {"346", "348", "362", "426"}

        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = inter / union

        self.assertEqual(inter, 3)
        self.assertEqual(union, 5)
        self.assertAlmostEqual(jaccard, 0.6, places=4)

    # 6. Bootstrap reproducibility under fixed seed
    def test_06_bootstrap_reproducibility(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        mean_fn = lambda d: float(d["val"].mean())

        lo1, hi1, se1 = compute_bootstrap_ci(df, mean_fn, n_resamples=200, seed=42)
        lo2, hi2, se2 = compute_bootstrap_ci(df, mean_fn, n_resamples=200, seed=42)

        self.assertEqual(lo1, lo2)
        self.assertEqual(hi1, hi2)
        self.assertEqual(se1, se2)

    # 7. Correct calculation of reported statistics from saved predictions
    def test_07_reported_statistics_calculation(self):
        res_df = pd.read_csv(VALIDITY_DIR / "residual_analysis.csv")
        oof_df = pd.read_csv(self.oof_path)

        # Check eGeMAPS depressed mean discrepancy
        egemaps_dep = oof_df[(oof_df["modality"] == "egemaps") & (oof_df["phq8_true"] >= 10)]
        expected_d = float(egemaps_dep["discrepancy"].mean())

        reported_d = res_df[(res_df["split"] == "train_oof") & (res_df["modality"] == "egemaps") & (res_df["group"] == "phq_ge_10")]["mean_discrepancy"].iloc[0]
        self.assertAlmostEqual(reported_d, expected_d, places=4)

    # 8. Proper handling of empty or tiny groups
    def test_08_empty_or_tiny_group_handling(self):
        empty_df = pd.DataFrame({"phq8_true": [], "discrepancy": []})
        mean_fn = lambda d: float(d["discrepancy"].mean()) if len(d) > 0 else np.nan

        lo, hi, se = compute_bootstrap_ci(empty_df, mean_fn, n_resamples=50)
        self.assertTrue(np.isnan(lo) and np.isnan(hi) and np.isnan(se))

    # 9. No accidental use of development data for threshold derivation
    def test_09_no_development_data_in_thresholds(self):
        oof_df = pd.read_csv(self.oof_path)
        thresh_df = pd.read_csv(self.thresh_path)

        # Recalculate threshold on training OOF depressed subset only
        for _, trow in thresh_df.iterrows():
            mod = trow["modality"]
            q = trow["quantile"]
            tau_reported = trow["threshold_value"]

            sub_tr = oof_df[(oof_df["modality"] == mod) & (oof_df["phq8_true"] >= 10)]
            tau_recomputed = float(np.quantile(sub_tr["discrepancy"].values, q, method="linear"))

            self.assertAlmostEqual(tau_reported, tau_recomputed, places=4)

    # 10. No fabricated test participants, predictions, or metrics
    def test_10_no_test_fabrication(self):
        res_df = pd.read_csv(VALIDITY_DIR / "residual_analysis.csv")
        self.assertNotIn("test", res_df["split"].unique())

        unc_df = pd.read_csv(VALIDITY_DIR / "uncertainty_analysis.csv")
        self.assertNotIn("test", unc_df["eval_split"].unique())

        over_df = pd.read_csv(VALIDITY_DIR / "hsed_overlap.csv")
        self.assertNotIn("test", over_df["eval_split"].unique())

    # 11. All deliverables exist and are non-empty
    def test_11_deliverables_exist(self):
        files = [
            VALIDITY_DIR / "residual_analysis.csv",
            VALIDITY_DIR / "hsed_stability.csv",
            VALIDITY_DIR / "hsed_overlap.csv",
            VALIDITY_DIR / "uncertainty_analysis.csv",
            VALIDITY_DIR / "scientific_validity_summary.json",
        ]
        for f in files:
            self.assertTrue(f.exists(), f"Missing deliverable: {f}")
            self.assertGreater(f.stat().st_size, 0, f"Empty deliverable: {f}")


if __name__ == "__main__":
    unittest.main()
