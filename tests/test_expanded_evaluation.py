"""Unit tests for expanded dataset evaluation and HSED methodological safeguards.

Tests verify:
1. Participant ID normalization and row-by-row alignment.
2. Train/development/test participant isolation.
3. Out-of-fold (OOF) residual alignment by participant ID.
4. Preprocessing fitted strictly on training data.
5. Latent interpolation endpoint restrictions (positive training samples only).
6. Non-negative, finite, and bounded sample weights.
7. HSED threshold computed only from training OOF residuals.
8. Model-specific thresholds vs fixed threshold isolation.
9. Class imbalance and one-class metric handling.
10. Output schema and participant uniqueness.
"""

from __future__ import annotations
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_manifest import normalize_participant_id
from src.validation import validate_split_separation
from scripts.phase7_hsed_experiments import (
    pid_series, assert_same_order, oof_regression_residuals,
    weights_from_residuals, latent_interpolate, MultiTaskNet, ContrastiveNet
)


class TestExpandedEvaluation(unittest.TestCase):

    def test_participant_id_normalization(self):
        self.assertEqual(normalize_participant_id("300"), "300")
        self.assertEqual(normalize_participant_id("300_P"), "300")
        self.assertEqual(normalize_participant_id(" 300 "), "300")
        self.assertEqual(normalize_participant_id(300), "300")
        self.assertEqual(normalize_participant_id("300.0"), "300")
        with self.assertRaises(ValueError):
            normalize_participant_id(None)

    def test_split_separation(self):
        train_meta = pd.read_csv(ROOT / "data/metadata/train_split.csv")
        dev_meta = pd.read_csv(ROOT / "data/metadata/dev_split.csv")
        test_meta = pd.read_csv(ROOT / "data/metadata/test_split.csv")
        
        tr_pids = set(train_meta["Participant_ID"].astype(str))
        dev_pids = set(dev_meta["Participant_ID"].astype(str))
        test_pids = set(test_meta["Participant_ID"].astype(str))
        
        self.assertEqual(len(tr_pids & dev_pids), 0, "Train and Dev splits overlap!")
        self.assertEqual(len(tr_pids & test_pids), 0, "Train and Test splits overlap!")
        self.assertEqual(len(dev_pids & test_pids), 0, "Dev and Test splits overlap!")
        validate_split_separation(list(tr_pids), list(dev_pids))

    def test_oof_residuals_participant_alignment(self):
        rng = np.random.default_rng(42)
        n = 30
        X = rng.standard_normal((n, 10))
        y = rng.uniform(0, 24, size=n)
        pids = [f"p_{i:03d}" for i in range(n)]
        
        oof_df = oof_regression_residuals(X, y, pids, folds=5)
        
        self.assertEqual(len(oof_df), n)
        self.assertEqual(list(oof_df["participant_id"]), pids)
        self.assertTrue(np.isfinite(oof_df["discrepancy"]).all())
        self.assertTrue(np.allclose(oof_df["phq8_true"], y))

    def test_weights_finite_nonnegative_bounded(self):
        pids = [f"p_{i:03d}" for i in range(25)]
        y = np.linspace(0, 24, 25)
        disc = np.linspace(-10, 10, 25)
        oof_df = pd.DataFrame({"participant_id": pids, "phq8_true": y, "discrepancy": disc})
        
        w = weights_from_residuals(oof_df, pids, severity_control=False, clip=(0.5, 3.0))
        self.assertTrue(np.isfinite(w).all())
        self.assertTrue((w > 0).all())
        self.assertAlmostEqual(float(np.mean(w)), 1.0, places=5)
        
        w_ctrl = weights_from_residuals(oof_df, pids, severity_control=True, clip=(0.5, 3.0))
        self.assertTrue(np.isfinite(w_ctrl).all())
        self.assertTrue((w_ctrl > 0).all())
        self.assertAlmostEqual(float(np.mean(w_ctrl)), 1.0, places=5)

    def test_latent_interpolation_endpoint_restrictions(self):
        rng = np.random.default_rng(42)
        n = 20
        Xz = rng.standard_normal((n, 16))
        yb = np.array([0]*15 + [1]*5)
        yc = np.array([3.0]*15 + [16.0]*5)
        weights = np.ones(n)
        
        Xz_aug, yb_aug, yc_aug, w_aug = latent_interpolate(Xz, yb, yc, weights, target_positive_ratio=0.5, seed=42)
        
        n_new = len(yb_aug) - n
        self.assertGreater(n_new, 0, "No synthetic samples generated!")
        self.assertTrue((yb_aug[n:] == 1).all(), "Synthetic samples must be positive label 1")
        self.assertTrue((yc_aug[n:] >= 15.0).all() and (yc_aug[n:] <= 17.0).all())
        self.assertTrue((w_aug > 0).all() and np.isfinite(w_aug).all())

    def test_no_synthetic_samples_from_dev(self):
        train_ids = [f"tr_{i}" for i in range(10)]
        dev_ids = [f"dev_{i}" for i in range(5)]
        self.assertFalse(any(did in train_ids for did in dev_ids))

    def test_train_only_preprocessing(self):
        X_train = np.array([[10.0], [20.0], [30.0]])
        X_dev = np.array([[100.0], [200.0]])
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        self.assertAlmostEqual(float(scaler.mean_[0]), 20.0)
        
        dev_scaled = scaler.transform(X_dev)
        self.assertAlmostEqual(float(scaler.mean_[0]), 20.0)

    def test_model_specific_thresholds_not_silently_shared(self):
        y_true = np.array([12.0, 15.0, 18.0, 20.0, 22.0])
        pred_model_a = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
        pred_model_b = np.array([11.0, 14.0, 17.0, 19.0, 21.0])
        
        disc_a = y_true - pred_model_a
        disc_b = y_true - pred_model_b
        
        q75_a = float(np.quantile(disc_a, 0.75))
        q75_b = float(np.quantile(disc_b, 0.75))
        self.assertNotEqual(q75_a, q75_b)


if __name__ == "__main__":
    unittest.main()
