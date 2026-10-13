"""Automated tests for Phase 3, 4, 5 baselines: aggregation, alignment, and artifact integrity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.audio_features import extract_participant_acoustic_summary
from src.features.text_features import extract_participant_text_embedding
from src.features.multimodal_features import build_multimodal_features
from src.models.audio_baseline import get_audio_pipeline
from src.models.text_baseline import get_text_pipeline
from src.models.multimodal_baseline import get_multimodal_pipeline


class TestAudioFeatureAggregation(unittest.TestCase):
    """Test acoustic feature aggregation rules (items 6, 7, 8)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_feature_aggregation_one_row_and_excludes_timestamps(self):
        # Create synthetic OpenSMILE CSV with name, frameTime, and acoustic features
        csv_path = self.root / "300_sample.csv"
        df = pd.DataFrame({
            "name": ["rec1", "rec1", "rec1"],
            "frameTime": [0.01, 0.02, 0.03],
            "Loudness_sma3": [1.0, 2.0, 3.0],
            "alphaRatio_sma3": [10.0, 20.0, 30.0],
        })
        df.to_csv(csv_path, sep=";", index=False)

        summary, feat_names = extract_participant_acoustic_summary(csv_path, expected_delimiter=";")

        # Exactly 1 summary dict returned with 4 features (2 stats * 2 features)
        self.assertEqual(len(summary), 4)
        self.assertIn("Loudness_sma3_mean", summary)
        self.assertIn("Loudness_sma3_std", summary)
        self.assertIn("alphaRatio_sma3_mean", summary)
        self.assertIn("alphaRatio_sma3_std", summary)

        # Ensure timestamps and names are NEVER aggregated as acoustic features
        self.assertNotIn("name_mean", summary)
        self.assertNotIn("frameTime_mean", summary)
        self.assertNotIn("frameTime_std", summary)

        # Verify numerical accuracy
        self.assertEqual(summary["Loudness_sma3_mean"], 2.0)
        self.assertAlmostEqual(summary["Loudness_sma3_std"], 1.0, places=4)


class TestMultimodalAlignment(unittest.TestCase):
    """Test participant-ID-based multimodal joining (items 9, 10)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multimodal_join_by_participant_id_not_row_order(self):
        # Audio table with participants [300, 301, 302]
        audio_df = pd.DataFrame({
            "participant_id": ["300", "301", "302"],
            "split": ["train", "train", "dev"],
            "gender": ["male", "female", "male"],
            "phq_score": [5, 12, 3],
            "depression_label": [0, 1, 0],
            "f1": [1.0, 2.0, 3.0],
        })

        # Text table in REVERSED row order [302, 301, 300]
        text_df = pd.DataFrame({
            "participant_id": ["302", "301", "300"],
            "split": ["dev", "train", "train"],
            "gender": ["male", "female", "male"],
            "phq_score": [3, 12, 5],
            "depression_label": [0, 1, 0],
            "utterance_count": [10, 20, 30],
            "t1": [30.0, 20.0, 10.0],
        })

        audio_dir = self.root / "audio"
        text_dir = self.root / "text"
        out_dir = self.root / "mm"
        audio_dir.mkdir()
        text_dir.mkdir()
        out_dir.mkdir()

        audio_df.to_csv(audio_dir / "egemaps_features.csv", index=False)
        text_df.to_csv(text_dir / "text_embeddings_minilm.csv", index=False)

        # Mock paths
        from unittest.mock import patch
        with patch("src.features.multimodal_features.AUDIO_FEAT_DIR", audio_dir), \
             patch("src.features.multimodal_features.TEXT_FEAT_DIR", text_dir):
            mm_df = build_multimodal_features(audio_type="egemaps", output_dir=out_dir)

        # Verify participant alignment: 300 must have f1=1.0 and t1=10.0 despite reversed row order
        p300 = mm_df[mm_df["participant_id"] == "300"].iloc[0]
        self.assertEqual(p300["audio_f1"], 1.0)
        self.assertEqual(p300["text_t1"], 10.0)

        p302 = mm_df[mm_df["participant_id"] == "302"].iloc[0]
        self.assertEqual(p302["audio_f1"], 3.0)
        self.assertEqual(p302["text_t1"], 30.0)


class TestArtifactPreservation(unittest.TestCase):
    """Test model artifact persistence, pipeline structure, and feature preservation (item 13)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pipeline_artifact_preservation(self):
        # Synthetic training data
        X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        y_train = np.array([0, 0, 1, 1])

        pipe = get_audio_pipeline("logistic_regression", C=1.0, random_state=42)
        pipe.fit(X_train, y_train)

        # Save and reload artifact
        artifact_path = self.root / "model.joblib"
        joblib.dump(pipe, artifact_path)

        loaded_pipe = joblib.load(artifact_path)
        # Predictions match
        preds_orig = pipe.predict(X_train)
        preds_loaded = loaded_pipe.predict(X_train)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

        # Verify learned scaler stats preserved
        scaler = loaded_pipe.named_steps["scaler"]
        np.testing.assert_array_almost_equal(scaler.mean_, np.array([2.5, 25.0]))


if __name__ == "__main__":
    unittest.main()
