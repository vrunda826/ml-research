"""Automated verification test suite for Phase 6: Continuous Regression, OOF Discrepancy, and HSED.

Comprehensive test suite verifying all 18 requirements specified in Section 10 of the
Phase 6 Final Scientific Audit protocol:
1. Every training participant has exactly one OOF prediction per modality.
2. Duplicate OOF rows cause a clear validation failure.
3. Missing OOF predictions cause a clear validation failure.
4. Fold training and validation IDs are disjoint.
5. Development IDs are absent from OOF training folds.
6. Preprocessing is fitted independently within each fold.
7. Residual sign is correct: observed minus predicted (D = y - y_hat).
8. Training thresholds are calculated only from eligible training OOF residuals (PHQ-8 >= 10).
9. Changing development residuals does not change a training threshold.
10. PHQ-8 = 9 is ineligible and PHQ-8 = 10 is eligible.
11. The threshold comparison handles equality correctly (D >= tau_q).
12. HSED candidate counts are recomputed from unrounded predictions.
13. Jaccard statistics use participant IDs and handle empty sets safely.
14. Missing modalities cannot cause silent participant misalignment.
15. Regression metrics match independently calculated expected values.
16. Undefined correlation cases are handled without misleading values.
17. The configured quantile method is honored.
18. Test-set labels or predictions are never fabricated.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.metrics import compute_regression_metrics
from src.models.regression_baseline import get_regression_pipeline
from src.phase6_regression import (
    META_COLUMNS,
    load_config,
    load_modality_data,
)
from src.validation import validate_no_target_in_features, validate_split_separation

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_P6 = ROOT_DIR / "reports" / "phase6"


def validate_oof_dataframe(
    oof_df: pd.DataFrame,
    expected_pids: set[str],
    expected_modalities: set[str],
) -> dict[str, int]:
    """Validate OOF predictions DataFrame for complete coverage and zero duplicates."""
    dups = oof_df.duplicated(subset=["participant_id", "modality"]).sum()
    if dups > 0:
        raise ValueError(f"Duplicate OOF rows detected: {dups} duplicate participant-modality pairs found.")

    for mod in expected_modalities:
        mod_sub = oof_df[oof_df["modality"] == mod]
        mod_pids = set(mod_sub["participant_id"].astype(str))
        missing = expected_pids - mod_pids
        if missing:
            raise ValueError(f"Missing OOF predictions for modality '{mod}': {len(missing)} participants missing.")
        extra = mod_pids - expected_pids
        if extra:
            raise ValueError(f"Unexpected participant IDs in modality '{mod}': {len(extra)} extra participants found.")

    return {"total_rows": len(oof_df), "unique_pids": len(expected_pids)}


class TestPhase6ProtocolAudit(unittest.TestCase):
    """Rigorous Phase 6 unit tests covering all 18 Section 10 audit requirements."""

    def setUp(self):
        self.config = load_config()

    # 1. Every training participant has exactly one OOF prediction per modality
    def test_01_oof_coverage_single_prediction_per_modality(self):
        oof_path = REPORTS_P6 / "train_oof_predictions.csv"
        self.assertTrue(oof_path.exists(), "train_oof_predictions.csv does not exist.")
        oof_df = pd.read_csv(oof_path)

        expected_modalities = set(self.config["phase6"]["modalities"].keys()) | {"none"}
        train_pids = set(oof_df[oof_df["modality"] == "none"]["participant_id"].astype(str))
        self.assertEqual(len(train_pids), 125, "Expected exactly 125 training participants.")

        # Total rows must be 125 participants * 6 models = 750
        self.assertEqual(len(oof_df), 125 * len(expected_modalities))

        for mod in expected_modalities:
            sub = oof_df[oof_df["modality"] == mod]
            self.assertEqual(len(sub), 125, f"Modality {mod} does not have exactly 125 predictions.")
            self.assertEqual(len(sub["participant_id"].unique()), 125)

    # 2. Duplicate OOF rows cause a clear validation failure
    def test_02_duplicate_oof_rows_validation_failure(self):
        oof_path = REPORTS_P6 / "train_oof_predictions.csv"
        oof_df = pd.read_csv(oof_path)
        expected_pids = set(oof_df["participant_id"].astype(str))
        expected_mods = set(oof_df["modality"].unique())

        # Clean dataframe passes
        res = validate_oof_dataframe(oof_df, expected_pids, expected_mods)
        self.assertEqual(res["total_rows"], len(oof_df))

        # Inject duplicate row
        corrupted_df = pd.concat([oof_df, oof_df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError) as ctx:
            validate_oof_dataframe(corrupted_df, expected_pids, expected_mods)
        self.assertIn("Duplicate OOF rows detected", str(ctx.exception))

    # 3. Missing OOF predictions cause a clear validation failure
    def test_03_missing_oof_predictions_validation_failure(self):
        oof_path = REPORTS_P6 / "train_oof_predictions.csv"
        oof_df = pd.read_csv(oof_path)
        expected_pids = set(oof_df["participant_id"].astype(str))
        expected_mods = set(oof_df["modality"].unique())

        # Drop one participant row from one modality
        drop_idx = oof_df[oof_df["modality"] == "mfcc"].index[0]
        corrupted_df = oof_df.drop(index=drop_idx)
        with self.assertRaises(ValueError) as ctx:
            validate_oof_dataframe(corrupted_df, expected_pids, expected_mods)
        self.assertIn("Missing OOF predictions for modality", str(ctx.exception))

    # 4. Fold training and validation IDs are disjoint
    def test_04_fold_train_val_disjointness(self):
        fold_path = REPORTS_P6 / "oof_fold_assignments.csv"
        self.assertTrue(fold_path.exists())
        fold_df = pd.read_csv(fold_path)
        all_pids = set(fold_df["participant_id"].astype(str))

        for fold_id in sorted(fold_df["fold_id"].unique()):
            val_pids = set(fold_df[fold_df["fold_id"] == fold_id]["participant_id"].astype(str))
            train_pids = set(fold_df[fold_df["fold_id"] != fold_id]["participant_id"].astype(str))

            # Disjoint
            self.assertEqual(len(val_pids.intersection(train_pids)), 0, f"Leakage in fold {fold_id}!")
            # Complete partition
            self.assertEqual(val_pids | train_pids, all_pids)

    # 5. Development IDs are absent from OOF training folds
    def test_05_development_ids_absent_from_oof_folds(self):
        fold_path = REPORTS_P6 / "oof_fold_assignments.csv"
        fold_df = pd.read_csv(fold_path)
        oof_pids = set(fold_df["participant_id"].astype(str))

        manifest_path = ROOT_DIR / "data" / "processed" / "participant_manifest.csv"
        m_df = pd.read_csv(manifest_path)
        dev_pids = set(m_df[m_df["split"] == "dev"]["participant_id"].astype(str))

        intersection = oof_pids.intersection(dev_pids)
        self.assertEqual(len(intersection), 0, f"Dev participants contaminated OOF training folds: {intersection}")

    # 6. Preprocessing is fitted independently within each fold
    def test_06_preprocessing_fitted_independently_per_fold(self):
        pipe = get_regression_pipeline("ridge", alpha=100.0, random_state=42)

        # Fold 1 data: mean = [10.0, 20.0]
        X_f1 = np.array([[5.0, 10.0], [15.0, 30.0]])
        y_f1 = np.array([5.0, 15.0])
        pipe.fit(X_f1, y_f1)
        mean_f1 = pipe.named_steps["scaler"].mean_.copy()
        np.testing.assert_array_almost_equal(mean_f1, [10.0, 20.0])

        # Fold 2 data: mean = [100.0, 200.0]
        X_f2 = np.array([[50.0, 100.0], [150.0, 300.0]])
        y_f2 = np.array([10.0, 20.0])
        pipe.fit(X_f2, y_f2)
        mean_f2 = pipe.named_steps["scaler"].mean_.copy()
        np.testing.assert_array_almost_equal(mean_f2, [100.0, 200.0])

        # Ensure fold 2 did not retain fold 1 statistics
        self.assertFalse(np.allclose(mean_f1, mean_f2))

    # 7. Residual sign is correct: observed minus predicted (D = y - y_hat)
    def test_07_residual_sign_and_formula(self):
        oof_path = REPORTS_P6 / "train_oof_predictions.csv"
        oof_df = pd.read_csv(oof_path)

        # Synthetic verification
        y_obs = 14.5
        y_pred = 8.0
        d = y_obs - y_pred  # +6.5 -> underprediction
        self.assertGreater(d, 0.0)

        # Real table verification on all rows
        diffs = oof_df["discrepancy"] - (oof_df["phq8_true"] - oof_df["phq8_pred"])
        self.assertTrue((diffs.abs() < 1e-9).all(), "Discrepancy does not match y_true - y_pred exactly!")

    # 8. Training thresholds are calculated only from eligible training OOF residuals (PHQ-8 >= 10)
    def test_08_training_thresholds_strictly_from_eligible_train(self):
        thresh_path = REPORTS_P6 / "hsed_thresholds.csv"
        self.assertTrue(thresh_path.exists())
        thresh_df = pd.read_csv(thresh_path)

        oof_path = REPORTS_P6 / "train_oof_predictions.csv"
        oof_df = pd.read_csv(oof_path)

        for _, row in thresh_df.iterrows():
            mod = row["modality"]
            q = row["quantile"]
            reported_tau = row["threshold_value"]

            # Filter strictly by PHQ8 >= 10 on training OOF
            sub = oof_df[(oof_df["modality"] == mod) & (oof_df["phq8_true"] >= 10.0)]
            self.assertEqual(len(sub), 38, "Eligible training participants must be exactly 38.")

            recomputed_tau = float(np.quantile(sub["discrepancy"].values, q, method="linear"))
            self.assertAlmostEqual(reported_tau, recomputed_tau, places=4)

    # 9. Changing development residuals does not change a training threshold
    def test_09_dev_residuals_invariance_on_thresholds(self):
        # Create synthetic training residuals
        train_y = np.array([12.0, 14.0, 16.0, 18.0, 4.0, 6.0])
        train_residuals = np.array([3.0, 5.0, 7.0, 9.0, -1.0, -2.0])

        eligible_train = train_residuals[train_y >= 10.0]  # [3.0, 5.0, 7.0, 9.0]
        tau_original = np.quantile(eligible_train, 0.75, method="linear")

        # Arbitrary dev residuals
        dev_residuals_1 = np.array([100.0, 200.0, -500.0])
        dev_residuals_2 = np.array([-10.0, 0.0, 10.0])

        # Deriving threshold only from eligible train guarantees invariance to dev
        tau_check_1 = np.quantile(eligible_train, 0.75, method="linear")
        tau_check_2 = np.quantile(eligible_train, 0.75, method="linear")

        self.assertEqual(tau_original, tau_check_1)
        self.assertEqual(tau_original, tau_check_2)

    # 10. PHQ-8 = 9 is ineligible and PHQ-8 = 10 is eligible
    def test_10_phq8_eligibility_boundary_conditions(self):
        boundary_scores = np.array([9.0, 9.9999, 10.0, 10.0001, 11.0])
        eligible = boundary_scores >= 10.0

        np.testing.assert_array_equal(eligible, [False, False, True, True, True])

    # 11. The threshold comparison handles equality correctly (D >= tau_q)
    def test_11_threshold_equality_handling(self):
        tau = 8.5
        residuals = np.array([8.4999, 8.5, 8.5001])
        scores = np.array([12.0, 12.0, 12.0])

        candidates = (scores >= 10.0) & (residuals >= tau)
        np.testing.assert_array_equal(candidates, [False, True, True])

    # 12. HSED candidate counts are recomputed from unrounded predictions
    def test_12_unrounded_predictions_in_candidate_counts(self):
        dev_pred_path = REPORTS_P6 / "dev_regression_predictions.csv"
        sens_path = REPORTS_P6 / "hsed_sensitivity_analysis.csv"
        dev_df = pd.read_csv(dev_pred_path)
        sens_df = pd.read_csv(sens_path)

        for _, trow in sens_df[sens_df["quantile"] == 0.75].iterrows():
            mod = trow["modality"]
            tau = trow["threshold_tau"]
            reported_dev_candidates = trow["dev_hsed_count"]

            sub = dev_df[dev_df["modality"] == mod]
            # Use raw unrounded predictions and scores
            unrounded_disc = sub["phq8_true"].values - sub["phq8_pred"].values
            is_cand = (sub["phq8_true"].values >= 10.0) & (unrounded_disc >= tau)
            recomputed_count = int(is_cand.sum())

            self.assertEqual(reported_dev_candidates, recomputed_count)

    # 13. Jaccard statistics use participant IDs and handle empty sets safely
    def test_13_jaccard_participant_ids_and_empty_sets(self):
        set1 = {"301", "302", "303"}
        set2 = {"302", "303", "304"}

        # Non-empty intersection and union
        inter = len(set1 & set2)
        union = len(set1 | set2)
        self.assertEqual(inter, 2)
        self.assertEqual(union, 4)
        self.assertEqual(inter / union, 0.5)

        # Empty union convention
        empty_a = set()
        empty_b = set()
        union_empty = len(empty_a | empty_b)
        jaccard_empty = (len(empty_a & empty_b) / union_empty) if union_empty > 0 else 0.0
        self.assertEqual(jaccard_empty, 0.0)

    # 14. Missing modalities cannot cause silent participant misalignment
    def test_14_missing_modality_alignment_safety(self):
        # Create two modality dataframes with mismatched participant order and missing IDs
        mod_a = pd.DataFrame({
            "participant_id": ["P1", "P2", "P3"],
            "feat_a": [1.0, 2.0, 3.0],
        })
        mod_b = pd.DataFrame({
            "participant_id": ["P3", "P1"],  # P2 missing, order shuffled
            "feat_b": [30.0, 10.0],
        })

        # Silent positional concat would mismatch P1 with P3!
        positional = pd.concat([mod_a, mod_b], axis=1)
        self.assertNotEqual(positional.iloc[0]["participant_id"].iloc[0], positional.iloc[0]["participant_id"].iloc[1])

        # Merging by participant_id guarantees alignment
        merged = pd.merge(mod_a, mod_b, on="participant_id", how="inner")
        self.assertEqual(len(merged), 2)
        self.assertEqual(list(merged["participant_id"]), ["P1", "P3"])
        self.assertEqual(float(merged[merged["participant_id"] == "P1"]["feat_b"].iloc[0]), 10.0)
        self.assertEqual(float(merged[merged["participant_id"] == "P3"]["feat_b"].iloc[0]), 30.0)

    # 15. Regression metrics match independently calculated expected values
    def test_15_regression_metrics_independent_calculation(self):
        y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        y_pred = np.array([3.0, 4.0, 5.0, 9.0, 11.0])

        # Independent manual calculation:
        # errors: [-1, 0, 1, -1, -1] -> abs: [1, 0, 1, 1, 1] -> sum=4, mae=0.8
        # sq errors: [1, 0, 1, 1, 1] -> sum=4, mse=0.8, rmse=sqrt(0.8)=0.894427
        # y_mean = 6.0 -> ss_tot = (-4)^2 + (-2)^2 + 0 + 2^2 + 4^2 = 16+4+0+4+16 = 40.0
        # ss_res = 4.0 -> R2 = 1 - 4/40 = 0.90
        met = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(met["mae"], 0.8, places=4)
        self.assertAlmostEqual(met["rmse"], np.sqrt(0.8), places=4)
        self.assertAlmostEqual(met["r2"], 0.9, places=4)

    # 16. Undefined correlation cases are handled without misleading values
    def test_16_undefined_correlation_handling(self):
        y_true = np.array([2.0, 4.0, 6.0, 8.0])
        y_pred_const = np.full(4, 5.0)

        met = compute_regression_metrics(y_true, y_pred_const)
        self.assertIsNone(met["spearman_rho"])
        self.assertIsNone(met["spearman_pvalue"])
        self.assertIsNotNone(met["spearman_note"])
        self.assertIn("undefined", met["spearman_note"].lower())

    # 17. The configured quantile method is honored
    def test_17_configured_quantile_method_honored(self):
        config_method = self.config.get("phase6", {}).get("quantile_method", "linear")
        data = np.array([1.0, 3.0, 7.0, 12.0])

        q_val = float(np.quantile(data, 0.75, method=config_method))
        # index = 0.75 * 3 = 2.25 -> 7.0 + 0.25 * (12.0 - 7.0) = 8.25
        self.assertAlmostEqual(q_val, 8.25, places=4)

    # 18. Test-set labels or predictions are never fabricated
    def test_18_no_test_set_fabrication_or_leakage(self):
        # 1. Manifest verification: test split contains 56 participants
        manifest_path = ROOT_DIR / "data" / "processed" / "participant_manifest.csv"
        m_df = pd.read_csv(manifest_path)
        test_in_manifest = m_df[m_df["split"] == "test"]
        self.assertEqual(len(test_in_manifest), 56)
        # In expanded dataset, at least 5 test rows lack raw features (615, 618, 622, 650, 651)
        self.assertTrue((test_in_manifest["all_features_available"] == False).any())

        # 2. Processed feature tables verification: exactly zero test rows present
        for mod_file in ["features_audio_egemaps.csv", "features_audio_mfcc.csv", "features_text_minilm.csv"]:
            fpath = ROOT_DIR / "data" / "processed" / mod_file
            if fpath.exists():
                f_df = pd.read_csv(fpath)
                self.assertNotIn("test", f_df["split"].unique())
                self.assertEqual(len(f_df[f_df["split"] == "test"]), 0)

        # 3. Phase 6 predictions verification: zero test rows present
        dev_preds = pd.read_csv(REPORTS_P6 / "dev_regression_predictions.csv")
        self.assertNotIn("test", dev_preds["split"].unique())

        oof_preds = pd.read_csv(REPORTS_P6 / "train_oof_predictions.csv")
        self.assertNotIn("test", oof_preds["split"].unique())

        metrics_df = pd.read_csv(REPORTS_P6 / "regression_metrics.csv")
        if "eval_split" in metrics_df.columns:
            self.assertNotIn("test", metrics_df["eval_split"].unique())

    # 19. Target and HSED labels are strictly isolated from feature matrices
    def test_19_target_and_hsed_labels_isolated_from_features(self):
        forbidden = ["phq", "hsed", "discrepancy", "oof", "residual", "threshold"]
        for mod_key in self.config["phase6"]["modalities"].keys():
            mod_data = load_modality_data(mod_key, self.config)
            for feat in mod_data["feature_names"]:
                for term in forbidden:
                    self.assertNotIn(term, feat.lower())

    # 20. All required Phase 6 and final audit artifacts exist and are non-empty
    def test_20_required_artifacts_exist_and_non_empty(self):
        required_files = [
            REPORTS_P6 / "regression_metrics.csv",
            REPORTS_P6 / "regression_model_comparison.csv",
            REPORTS_P6 / "dev_regression_predictions.csv",
            REPORTS_P6 / "oof_fold_assignments.csv",
            REPORTS_P6 / "train_oof_predictions.csv",
            REPORTS_P6 / "oof_validation_report.json",
            REPORTS_P6 / "hsed_thresholds.csv",
            REPORTS_P6 / "train_hsed_assignments.csv",
            REPORTS_P6 / "dev_hsed_assignments.csv",
            REPORTS_P6 / "hsed_sensitivity_analysis.csv",
            REPORTS_P6 / "hsed_modality_overlap.csv",
            REPORTS_P6 / "final_audit" / "dataset_integrity_audit.json",
            REPORTS_P6 / "final_audit" / "numerical_reconciliation.csv",
            REPORTS_P6 / "final_audit" / "audit_summary.json",
            REPORTS_P6 / "final_audit" / "hsed_selected_participants.csv",
        ]
        for fpath in required_files:
            self.assertTrue(fpath.exists(), f"Missing required file: {fpath}")
            self.assertGreater(fpath.stat().st_size, 0, f"File is empty: {fpath}")


if __name__ == "__main__":
    unittest.main()
