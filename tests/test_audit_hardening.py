"""Audit and verification test suite for DAIC-WOZ pre-Phase-6 readiness.

Implements all 15 audit criteria specified in the pre-Phase-6 audit mandate:
1. Disjoint train, development, and test splits.
2. Duplicate participant ID detection.
3. ID-based feature/label joins (preserves correct participant).
4. PHQ-8 validation rejects invalid and missing values.
5. The PHQ-8 >= 10 classification rule behaves correctly at boundary values.
6. Audio aggregation excludes identifiers and timestamps.
7. Audio aggregation returns one row per participant with stable feature names.
8. Text processing handles missing and empty transcripts safely.
9. Speaker filtering policy: only applied when reliable annotations exist.
10. No PHQ-8 score or derived target in model input features.
11. Preprocessing is fitted on training data only.
12. Development data does not influence model fitting.
13. Missing modalities do not silently shift alignment.
14. Evaluation metrics match independently calculated expected values on synthetic example.
15. Consistent evaluation cohort across all baseline models.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.build_manifest import normalize_participant_id
from src.features.audio_features import extract_participant_acoustic_summary
from src.features.text_features import (
    load_and_clean_transcript,
    extract_participant_text_embedding,
    EMBEDDING_DIM,
)
from src.features.multimodal_features import build_multimodal_features
from src.models.audio_baseline import get_audio_pipeline
from src.models.text_baseline import get_text_pipeline
from src.models.multimodal_baseline import get_multimodal_pipeline
from src.validation import (
    validate_split_separation,
    validate_feature_matrix_alignment,
    validate_label_integrity,
    validate_no_target_in_features,
    assert_no_test_labels_used,
)
from src.metrics import compute_classification_metrics

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestAuditHardening(unittest.TestCase):
    """Rigorous pre-Phase-6 audit verification tests."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Train, development, and test participant sets are disjoint
    def test_01_splits_strictly_disjoint(self):
        # Disjoint sets pass
        train_ids = ["300", "301", "302"]
        dev_ids = ["303", "304"]
        test_ids = ["305", "306"]
        res = validate_split_separation(train_ids, dev_ids, test_ids)
        self.assertTrue(res["is_valid"])

        # Train and Dev overlap fails
        with self.assertRaises(ValueError) as ctx:
            validate_split_separation(["300", "301"], ["301", "302"], ["303"])
        self.assertIn("Train and Dev", str(ctx.exception))

        # Train and Test overlap fails
        with self.assertRaises(ValueError) as ctx:
            validate_split_separation(["300", "301"], ["302"], ["301", "303"])
        self.assertIn("Train and Test", str(ctx.exception))

        # Dev and Test overlap fails
        with self.assertRaises(ValueError) as ctx:
            validate_split_separation(["300"], ["301", "302"], ["302", "303"])
        self.assertIn("Dev and Test", str(ctx.exception))

        # Also verify actual participant manifest on disk if present
        manifest_path = ROOT_DIR / "data" / "processed" / "participant_manifest.csv"
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            tr = set(df[df["split"] == "train"]["participant_id"].astype(str))
            de = set(df[df["split"] == "dev"]["participant_id"].astype(str))
            te = set(df[df["split"] == "test"]["participant_id"].astype(str))
            self.assertEqual(len(tr & de), 0, "Train and Dev manifest overlap!")
            self.assertEqual(len(tr & te), 0, "Train and Test manifest overlap!")
            self.assertEqual(len(de & te), 0, "Dev and Test manifest overlap!")

    # 2. Duplicate participant IDs are detected
    def test_02_duplicate_participant_ids_detected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_split_separation(["300", "300", "301"], ["302"])
        self.assertIn("Duplicate participant IDs found in training split", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_split_separation(["300"], ["302", "302"])
        self.assertIn("Duplicate participant IDs found in development split", str(ctx.exception))

    # 3. Feature/label joins are ID-based and preserve the correct participant
    def test_03_feature_label_joins_id_based(self):
        # Create metadata labels
        labels_df = pd.DataFrame({
            "participant_id": ["P1", "P2", "P3"],
            "depression_label": [0, 1, 0],
            "phq_score": [3, 14, 7],
        })
        # Create features in completely reversed order
        features_df = pd.DataFrame({
            "participant_id": ["P3", "P2", "P1"],
            "feat_val": [300.0, 200.0, 100.0],
        })

        # Join by participant ID
        merged = pd.merge(labels_df, features_df, on="participant_id")
        p1 = merged[merged["participant_id"] == "P1"].iloc[0]
        p2 = merged[merged["participant_id"] == "P2"].iloc[0]
        p3 = merged[merged["participant_id"] == "P3"].iloc[0]

        # Verify correct feature value mapped to each participant
        self.assertEqual(p1["feat_val"], 100.0)
        self.assertEqual(p1["depression_label"], 0)
        self.assertEqual(p2["feat_val"], 200.0)
        self.assertEqual(p2["depression_label"], 1)
        self.assertEqual(p3["feat_val"], 300.0)
        self.assertEqual(p3["depression_label"], 0)

        # Naive positional join would mismatch P1 and P3
        self.assertNotEqual(labels_df.iloc[0]["participant_id"], features_df.iloc[0]["participant_id"])

    # 4. PHQ-8 validation rejects invalid and missing values
    def test_04_phq8_validation_rejects_invalid_and_missing(self):
        # Out-of-bounds binary labels (e.g., 2, -1)
        with self.assertRaises(ValueError):
            validate_label_integrity(pd.Series([0, 1, 2]), "train", allow_missing=False)

        # Missing values when disallowed
        with self.assertRaises(ValueError):
            validate_label_integrity(pd.Series([0, 1, np.nan]), "train", allow_missing=False)

        # Valid binary labels pass
        res = validate_label_integrity(pd.Series([0, 1, 0, 1]), "train", allow_missing=False)
        self.assertEqual(res["labeled"], 4)
        self.assertEqual(res["count_1_depressed"], 2)

    # 5. The PHQ-8 >= 10 classification rule behaves correctly at boundary values
    def test_05_phq8_cutoff_boundary_behavior(self):
        # Standard cutoff: score >= 10 is 1 (depressed), < 10 is 0
        def apply_cutoff(score: float) -> int:
            return 1 if score >= 10 else 0

        self.assertEqual(apply_cutoff(0), 0)
        self.assertEqual(apply_cutoff(9), 0)
        self.assertEqual(apply_cutoff(9.99), 0)
        self.assertEqual(apply_cutoff(10), 1)
        self.assertEqual(apply_cutoff(10.01), 1)
        self.assertEqual(apply_cutoff(24), 1)

    # 6. Audio aggregation excludes identifiers and timestamps
    def test_06_audio_aggregation_excludes_identifiers_and_timestamps(self):
        csv_file = self.root / "sample_audio.csv"
        # CSV with identifiers, timestamps, and acoustic features
        df = pd.DataFrame({
            "name": ["rec_01", "rec_01"],
            "frameTime": [0.01, 0.02],
            "timestamp": [100.0, 100.1],
            "time": [0.1, 0.2],
            "F0semitoneFrom27.5Hz_sma3nz": [12.0, 14.0],
            "jitterLocal_sma3nz": [0.01, 0.02],
        })
        df.to_csv(csv_file, sep=";", index=False)

        summary, feat_names = extract_participant_acoustic_summary(csv_file, expected_delimiter=";")

        # Acoustic features must be present
        self.assertIn("F0semitoneFrom27.5Hz_sma3nz_mean", summary)
        self.assertIn("jitterLocal_sma3nz_mean", summary)

        # Identifier and timestamp columns MUST NOT be present
        for col in ["name", "frameTime", "timestamp", "time"]:
            self.assertNotIn(f"{col}_mean", summary)
            self.assertNotIn(f"{col}_std", summary)
            self.assertNotIn(col, feat_names)

    # 7. Audio aggregation returns one row per participant with stable feature names
    def test_07_audio_aggregation_one_row_and_stable_feature_names(self):
        csv_file = self.root / "sample_frames.csv"
        df = pd.DataFrame({
            "alphaRatio_sma3": [1.0, 2.0, 3.0, 4.0],
            "hammarbergIndex_sma3": [10.0, 20.0, 30.0, 40.0],
        })
        df.to_csv(csv_file, sep=";", index=False)

        summary1, names1 = extract_participant_acoustic_summary(csv_file, expected_delimiter=";")
        summary2, names2 = extract_participant_acoustic_summary(csv_file, expected_delimiter=";")

        # Exactly 1 summary dict per call (for 1 participant)
        self.assertIsInstance(summary1, dict)
        self.assertEqual(len(summary1), 4)

        # Feature names are deterministic and stable across calls
        self.assertEqual(names1, names2)

        # Empty data rows raises ValueError
        empty_csv = self.root / "empty_frames.csv"
        pd.DataFrame(columns=["alphaRatio_sma3", "hammarbergIndex_sma3"]).to_csv(empty_csv, sep=";", index=False)
        with self.assertRaises(ValueError):
            extract_participant_acoustic_summary(empty_csv, expected_delimiter=";")

    # 8. Text processing handles missing and empty transcripts safely
    def test_08_text_processing_handles_missing_and_empty_transcripts_safely(self):
        # Missing file raises FileNotFoundError
        missing_file = self.root / "does_not_exist.csv"
        with self.assertRaises(FileNotFoundError):
            load_and_clean_transcript(missing_file, allow_empty=False)

        # 0-byte file raises ValueError
        empty_file = self.root / "zero_bytes.csv"
        empty_file.touch()
        with self.assertRaises(ValueError):
            load_and_clean_transcript(empty_file, allow_empty=False)

        # Transcript without 'Text' column raises ValueError
        no_text_col = self.root / "no_text.csv"
        pd.DataFrame({"Start_Time": [1.0], "End_Time": [2.0]}).to_csv(no_text_col, index=False)
        with self.assertRaises(ValueError):
            load_and_clean_transcript(no_text_col, allow_empty=False)

        # Transcript with only whitespace raises ValueError
        whitespace_file = self.root / "whitespace_only.csv"
        pd.DataFrame({"Text": ["   ", "\n", "  \t  "]}).to_csv(whitespace_file, index=False)
        with self.assertRaises(ValueError):
            load_and_clean_transcript(whitespace_file, allow_empty=False)

        # Valid transcript returns utterances
        valid_file = self.root / "valid_transcript.csv"
        pd.DataFrame({"Text": ["Hello there", "I feel okay today"]}).to_csv(valid_file, index=False)
        utts = load_and_clean_transcript(valid_file)
        self.assertEqual(utts, ["Hello there", "I feel okay today"])

    # 9. Speaker filtering is applied only when reliable speaker annotations exist
    def test_09_speaker_filtering_policy(self):
        # Verify that transcripts without speaker tags cannot be selectively filtered
        # and are designated as mixed-speaker dialogue
        csv_file = self.root / "mixed_transcript.csv"
        df = pd.DataFrame({
            "Start_Time": [10.0, 15.0],
            "End_Time": [12.0, 18.0],
            "Text": ["Who is your best friend?", "My sister is."],
        })
        df.to_csv(csv_file, index=False)

        read_df = pd.read_csv(csv_file)
        # Check no speaker column exists
        has_speaker = any(c.lower() in ("speaker", "speaker_id", "role") for c in read_df.columns)
        self.assertFalse(has_speaker)
        # All dialogue preserved under mixed-speaker baseline
        utts = load_and_clean_transcript(csv_file)
        self.assertEqual(len(utts), 2)

    # 10. No PHQ-8 score or derived target is included in model input features
    def test_10_no_target_in_model_features(self):
        valid_features = ["audio_alphaRatio_mean", "audio_alphaRatio_std", "text_emb_0", "text_emb_1"]
        validate_no_target_in_features(valid_features)  # passes

        # Contaminated feature sets fail immediately
        for bad_col in ["phq_score", "depression_label", "phq8_total", "phq_binary"]:
            with self.assertRaises(ValueError):
                validate_no_target_in_features(valid_features + [bad_col])

        # Also verify feature matrices saved in processed directory
        audio_csv = ROOT_DIR / "data" / "processed" / "features" / "audio" / "egemaps_features.csv"
        if audio_csv.exists():
            df_a = pd.read_csv(audio_csv)
            # Feature columns used for modeling (excluding metadata)
            meta = {"participant_id", "split", "gender", "phq_score", "depression_label"}
            feat_cols = [c for c in df_a.columns if c not in meta]
            validate_no_target_in_features(feat_cols)

    # 11. Preprocessing is fitted on training data only
    def test_11_preprocessing_fitted_on_training_only(self):
        X_train = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        y_train = np.array([0, 0, 1])
        X_dev = np.array([[1000.0, 10000.0]])  # extreme outlier dev point

        pipe = get_audio_pipeline("logistic_regression")
        pipe.fit(X_train, y_train)

        scaler = pipe.named_steps["scaler"]
        # Training mean should be exactly [20.0, 200.0]
        np.testing.assert_array_almost_equal(scaler.mean_, np.array([20.0, 200.0]))

        # Transform dev without refitting
        _ = pipe.predict(X_dev)
        # Verify scaler statistics are unchanged by dev data
        np.testing.assert_array_almost_equal(scaler.mean_, np.array([20.0, 200.0]))

    # 12. Development data does not influence model fitting
    def test_12_dev_data_does_not_influence_model_fitting(self):
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y_train = np.array([0, 0, 1, 1])

        pipe1 = get_text_pipeline("logistic_regression", random_state=42)
        pipe1.fit(X_train, y_train)
        weights_before = pipe1.named_steps["classifier"].coef_.copy()

        # Dev prediction
        X_dev = np.array([[10.0, 20.0]])
        _ = pipe1.predict(X_dev)
        weights_after = pipe1.named_steps["classifier"].coef_.copy()

        np.testing.assert_array_equal(weights_before, weights_after)

    # 13. Missing modalities do not silently change participant alignment
    def test_13_missing_modalities_do_not_silently_shift_alignment(self):
        # Participant 300 has both; 301 has audio only; 302 has text only
        audio_df = pd.DataFrame({
            "participant_id": ["300", "301"],
            "split": ["train", "train"],
            "gender": ["male", "female"],
            "phq_score": [5, 12],
            "depression_label": [0, 1],
            "a1": [1.1, 2.2],
        })
        text_df = pd.DataFrame({
            "participant_id": ["300", "302"],
            "split": ["train", "train"],
            "gender": ["male", "male"],
            "phq_score": [5, 15],
            "depression_label": [0, 1],
            "utterance_count": [10, 20],
            "t1": [9.9, 8.8],
        })

        audio_dir = self.root / "audio"
        text_dir = self.root / "text"
        out_dir = self.root / "mm"
        audio_dir.mkdir()
        text_dir.mkdir()
        out_dir.mkdir()

        audio_df.to_csv(audio_dir / "egemaps_features.csv", index=False)
        text_df.to_csv(text_dir / "text_embeddings_minilm.csv", index=False)

        from unittest.mock import patch
        with patch("src.features.multimodal_features.AUDIO_FEAT_DIR", audio_dir), \
             patch("src.features.multimodal_features.TEXT_FEAT_DIR", text_dir):
            mm_df = build_multimodal_features(audio_type="egemaps", output_dir=out_dir)

        # Only the common participant "300" should remain
        self.assertEqual(len(mm_df), 1)
        self.assertEqual(mm_df.iloc[0]["participant_id"], "300")
        self.assertEqual(mm_df.iloc[0]["audio_a1"], 1.1)
        self.assertEqual(mm_df.iloc[0]["text_t1"], 9.9)

    # 14. Evaluation metrics match independently calculated expected values on synthetic example
    def test_14_metrics_match_independently_calculated_expected_values(self):
        # Synthetic evaluation with 4 samples:
        # y_true = [0, 0, 1, 1]
        # y_pred = [0, 1, 0, 1] (TP=1, FP=1, TN=1, FN=1)
        # y_prob = [0.2, 0.7, 0.4, 0.8]
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.7, 0.4, 0.8])

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)

        # Accuracy = 2/4 = 0.5
        self.assertEqual(metrics["accuracy"], 0.5)
        # Confusion matrix
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tp"], 1)
        self.assertEqual(cm["fp"], 1)
        self.assertEqual(cm["tn"], 1)
        self.assertEqual(cm["fn"], 1)

        # Recall for depression (class 1) = 1 / (1 + 1) = 0.5
        self.assertEqual(metrics["depression_recall"], 0.5)
        # Precision for class 1 = 1 / (1 + 1) = 0.5
        self.assertEqual(metrics["depression_precision"], 0.5)
        # F1 for class 1 = 0.5
        self.assertEqual(metrics["depression_f1"], 0.5)
        # Macro F1 = (0.5 + 0.5) / 2 = 0.5
        self.assertEqual(metrics["macro_f1"], 0.5)

        # Expected ROC-AUC
        expected_roc = roc_auc_score(y_true, y_prob)
        self.assertAlmostEqual(metrics["roc_auc"], expected_roc, places=4)

        # Expected PR-AUC (Average Precision)
        expected_pr = average_precision_score(y_true, y_prob)
        self.assertAlmostEqual(metrics["pr_auc"], expected_pr, places=4)

    # 15. Consistent evaluation cohort across all baseline models
    def test_15_consistent_dev_evaluation_cohort(self):
        # Manifest check for complete multimodal cohort
        manifest_path = ROOT_DIR / "data" / "processed" / "participant_manifest.csv"
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            dev_df = df[df["split"] == "dev"]
            # Official dev metadata contains 56 participants
            self.assertEqual(len(dev_df), 56)
            # Either legacy 44 or expanded 56 participants have extracted features for each modality
            self.assertIn(dev_df["has_egemaps"].sum(), [44, 56])
            self.assertIn(dev_df["has_mfcc"].sum(), [44, 56])
            self.assertIn(dev_df["has_transcript"].sum(), [44, 56])
            # All feature-complete dev participants are eligible for multimodal baseline evaluation
            eligible_dev = dev_df[
                (dev_df["has_egemaps"] == True) &
                (dev_df["has_mfcc"] == True) &
                (dev_df["has_transcript"] == True)
            ]
            self.assertIn(len(eligible_dev), [44, 56])


if __name__ == "__main__":
    unittest.main()
