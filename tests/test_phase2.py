"""Automated unit tests for Phase 2 Experimental Protocol, Validation, and Metrics."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.validation import (
    validate_split_separation,
    validate_feature_matrix_alignment,
    validate_label_integrity,
    assert_no_test_labels_used,
)
from src.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_hsed_metrics,
)


class TestValidationModule(unittest.TestCase):
    """Test data leakage prevention and split separation rules."""

    def test_split_separation_valid(self):
        train = ["300", "301", "302"]
        dev = ["303", "304"]
        test = ["305", "306"]
        res = validate_split_separation(train, dev, test)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["n_train"], 3)
        self.assertEqual(res["n_dev"], 2)
        self.assertEqual(res["n_test"], 2)

    def test_split_overlap_detection(self):
        train = ["300", "301", "302"]
        dev = ["302", "303"]  # '302' appears in both!
        with self.assertRaises(ValueError) as ctx:
            validate_split_separation(train, dev)
        self.assertIn("Data leakage detected", str(ctx.exception))

    def test_within_split_duplicates(self):
        train = ["300", "300", "301"]  # duplicate 300
        dev = ["302", "303"]
        with self.assertRaises(ValueError) as ctx:
            validate_split_separation(train, dev)
        self.assertIn("Duplicate participant IDs found in training split", str(ctx.exception))

    def test_feature_matrix_alignment(self):
        pids = ["300", "301"]
        X_valid = np.array([[1.0, 2.0], [3.0, 4.0]])
        validate_feature_matrix_alignment(pids, X_valid)

        # Mismatched rows
        X_mismatch = np.array([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            validate_feature_matrix_alignment(pids, X_mismatch)

        # Non-finite values
        X_nan = np.array([[1.0, np.nan], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            validate_feature_matrix_alignment(pids, X_nan)

    def test_label_integrity(self):
        y = pd.Series([0, 1, 0, 1, 0])
        res = validate_label_integrity(y, "dev")
        self.assertEqual(res["count_0_non_depressed"], 3)
        self.assertEqual(res["count_1_depressed"], 2)

        # Invalid label value
        y_invalid = pd.Series([0, 2, 0])
        with self.assertRaises(ValueError):
            validate_label_integrity(y_invalid, "dev")

    def test_no_test_labels_used(self):
        # Valid selection info
        assert_no_test_labels_used({"selection_dataset": "dev", "dev_macro_f1": 0.72})

        # Test set used for selection violates protocol
        with self.assertRaises(ValueError):
            assert_no_test_labels_used({"selection_dataset": "test"})


class TestMetricsModule(unittest.TestCase):
    """Test evaluation metrics and edge case safety."""

    def test_classification_metrics_standard(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 0]  # 1 false negative
        y_prob = [0.1, 0.9, 0.2, 0.4]

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        self.assertEqual(metrics["n_eval"], 4)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 2)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 0)
        self.assertEqual(metrics["depression_recall"], 0.5)
        self.assertIsNotNone(metrics["roc_auc"])
        self.assertIsNotNone(metrics["pr_auc"])

    def test_single_class_edge_case(self):
        # All samples are non-depressed (class 0)
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        y_prob = [0.1, 0.2, 0.15, 0.05]

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        self.assertEqual(metrics["accuracy"], 1.0)
        # ROC-AUC and PR-AUC must be safely None, not fabricated or crashing
        self.assertIsNone(metrics["roc_auc"])
        self.assertIn("Only one class present", metrics["roc_auc_unavailable_reason"])
        self.assertIsNone(metrics["pr_auc"])
        self.assertIn("Only one class present", metrics["pr_auc_unavailable_reason"])

    def test_regression_metrics(self):
        y_true = [5.0, 12.0, 3.0]
        y_pred = [4.0, 10.0, 4.0]
        reg = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(reg["mae"], (1.0 + 2.0 + 1.0) / 3.0, places=4)

    def test_hsed_metrics_pending(self):
        # When regression predictions are not provided
        res = compute_hsed_metrics(y_true_scores=[10.0, 4.0, 15.0])
        self.assertEqual(res["status"], "PENDING")

    def test_train_fitted_preprocessing_leakage(self):
        # Verify that preprocessor statistics are derived only from training data
        X_train = np.array([[10.0], [20.0], [30.0]])
        X_dev = np.array([[100.0], [200.0]])

        scaler = StandardScaler()
        scaler.fit(X_train)

        # Mean must be 20.0 (from train), not affected by X_dev
        self.assertEqual(scaler.mean_[0], 20.0)
        X_dev_scaled = scaler.transform(X_dev)
        # Verify transform uses train stats: (100 - 20) / std
        expected_scaled_0 = (100.0 - 20.0) / scaler.scale_[0]
        self.assertAlmostEqual(X_dev_scaled[0, 0], expected_scaled_0, places=4)


if __name__ == "__main__":
    unittest.main()
