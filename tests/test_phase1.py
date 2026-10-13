"""Focused unit tests for Phase 1 DAIC-WOZ data preparation pipeline.

Tests core data validation rules using small synthetic fixtures:
- Participant ID matching and normalization
- Missing required feature file detection
- Duplicate metadata ID detection
- PHQ-8 threshold handling (cutoff >= 10)
- Unlabeled participant handling (no silent negative assignment)
- Split overlap detection
- Empty and malformed CSV detection
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit_dataset import (
    REQUIRED_FILE_SPECS,
    audit_participant,
    check_csv_file,
)
from src.build_manifest import (
    build_participant_manifest,
    load_official_metadata,
    normalize_participant_id,
    validate_manifest,
)


class TestParticipantIdNormalization(unittest.TestCase):
    """Test safe participant ID normalization."""

    def test_participant_id_matching(self):
        self.assertEqual(normalize_participant_id(300), "300")
        self.assertEqual(normalize_participant_id("300"), "300")
        self.assertEqual(normalize_participant_id("300_P"), "300")
        self.assertEqual(normalize_participant_id("300_p"), "300")
        self.assertEqual(normalize_participant_id(" 300 "), "300")
        self.assertEqual(normalize_participant_id("300.0"), "300")

        # Distinct IDs must remain distinct
        self.assertNotEqual(normalize_participant_id("300"), normalize_participant_id("301"))
        self.assertNotEqual(normalize_participant_id(300), normalize_participant_id(301))

    def test_invalid_participant_ids(self):
        with self.assertRaises(ValueError):
            normalize_participant_id(None)
        with self.assertRaises(ValueError):
            normalize_participant_id(np.nan)
        with self.assertRaises(ValueError):
            normalize_participant_id("   ")


class TestFeatureAudit(unittest.TestCase):
    """Test feature audit and structural CSV checks."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_required_feature_files(self):
        p_dir = self.root / "300"
        p_dir.mkdir(parents=True)

        # Only write transcript, omit MFCC and eGeMAPS
        t_file = p_dir / "300_Transcript.csv"
        with t_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Start_Time", "End_Time", "Text", "Confidence"])
            writer.writerow([1.0, 2.0, "Hello", 0.95])

        result = audit_participant(p_dir)
        self.assertFalse(result.is_complete)
        self.assertEqual(result.files["transcript"].status, "OK")
        self.assertEqual(result.files["mfcc"].status, "MISSING")
        self.assertEqual(result.files["egemaps"].status, "MISSING")
        self.assertIn("300_OpenSMILE2.3.0_mfcc.csv", result.missing_files)
        self.assertIn("300_OpenSMILE2.3.0_egemaps.csv", result.missing_files)

    def test_empty_and_malformed_csv_detection(self):
        p_dir = self.root / "test_csvs"
        p_dir.mkdir(parents=True)

        # 1. 0-byte file
        zero_file = p_dir / "zero.csv"
        zero_file.touch()
        res_zero = check_csv_file(zero_file, "transcript", ",")
        self.assertEqual(res_zero.status, "EMPTY")

        # 2. Header only, 0 data rows
        header_only = p_dir / "header_only.csv"
        with header_only.open("w", newline="", encoding="utf-8") as f:
            f.write("col1,col2,col3\n")
        res_hdr = check_csv_file(header_only, "transcript", ",")
        self.assertEqual(res_hdr.status, "EMPTY")

        # 3. Delimiter mismatch (expected semicolon, got comma)
        delim_mismatch = p_dir / "delim.csv"
        with delim_mismatch.open("w", newline="", encoding="utf-8") as f:
            f.write("name,frameTime,val1,val2\n1.0,2.0,3.0,4.0\n")
        res_delim = check_csv_file(delim_mismatch, "mfcc", ";")
        self.assertEqual(res_delim.status, "MALFORMED")

        # 4. Valid semicolon CSV
        valid_semi = p_dir / "valid.csv"
        with valid_semi.open("w", newline="", encoding="utf-8") as f:
            f.write("name;frameTime;val1;val2\nrec1;0.01;1.5;2.5\nrec1;0.02;1.6;2.6\n")
        res_valid = check_csv_file(valid_semi, "mfcc", ";")
        self.assertEqual(res_valid.status, "OK")
        self.assertEqual(res_valid.row_count, 2)
        self.assertEqual(res_valid.column_count, 4)


class TestMetadataAndManifest(unittest.TestCase):
    """Test metadata loading, validation, and manifest generation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.meta_dir = self.root / "metadata"
        self.meta_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_split_file(self, filename: str, rows: list[dict]):
        path = self.meta_dir / filename
        cols = ["Participant_ID", "Gender", "PHQ_Score", "PHQ_Binary"]
        if rows:
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(columns=cols)
        df.to_csv(path, index=False)
        return path

    def test_duplicate_metadata_ids(self):
        # Create train split with duplicate ID 302
        self._create_split_file("train_split.csv", [
            {"Participant_ID": 302, "Gender": "male", "PHQ_Score": 4, "PHQ_Binary": 0},
            {"Participant_ID": 302, "Gender": "male", "PHQ_Score": 4, "PHQ_Binary": 0},
        ])
        self._create_split_file("dev_split.csv", [
            {"Participant_ID": 300, "Gender": "male", "PHQ_Score": 2, "PHQ_Binary": 0},
        ])
        self._create_split_file("test_split.csv", [
            {"Participant_ID": 600, "Gender": "female", "PHQ_Score": 5, "PHQ_Binary": 0},
        ])

        with self.assertRaises(ValueError) as ctx:
            load_official_metadata(self.meta_dir, mapped_csv_path=None)
        self.assertIn("Duplicate participant ID 302", str(ctx.exception))

    def test_split_overlap_detection(self):
        # ID 305 appears in both train and dev
        self._create_split_file("train_split.csv", [
            {"Participant_ID": 302, "Gender": "male", "PHQ_Score": 4, "PHQ_Binary": 0},
            {"Participant_ID": 305, "Gender": "male", "PHQ_Score": 7, "PHQ_Binary": 0},
        ])
        self._create_split_file("dev_split.csv", [
            {"Participant_ID": 305, "Gender": "male", "PHQ_Score": 7, "PHQ_Binary": 0},
            {"Participant_ID": 300, "Gender": "male", "PHQ_Score": 2, "PHQ_Binary": 0},
        ])
        self._create_split_file("test_split.csv", [
            {"Participant_ID": 600, "Gender": "female", "PHQ_Score": 5, "PHQ_Binary": 0},
        ])

        with self.assertRaises(ValueError) as ctx:
            load_official_metadata(self.meta_dir, mapped_csv_path=None)
        self.assertIn("Split overlap detected", str(ctx.exception))

    def test_phq8_threshold_handling(self):
        # Test boundary cases: 0, 9, 10, 11, 24
        scores = [0, 9, 10, 11, 24]
        rows = [
            {"Participant_ID": 100 + i, "Gender": "male", "PHQ_Score": score, "PHQ_Binary": 0}
            for i, score in enumerate(scores)
        ]
        self._create_split_file("train_split.csv", rows)
        self._create_split_file("dev_split.csv", [])
        self._create_split_file("test_split.csv", [])

        meta_df = load_official_metadata(self.meta_dir, mapped_csv_path=None)
        manifest_df = build_participant_manifest(meta_df, features_dict={})

        res_map = dict(zip(manifest_df["phq_score"], manifest_df["phq_binary_cutoff10"]))
        self.assertEqual(res_map[0], 0)
        self.assertEqual(res_map[9], 0)
        self.assertEqual(res_map[10], 1)
        self.assertEqual(res_map[11], 1)
        self.assertEqual(res_map[24], 1)

    def test_unlabeled_participants(self):
        # Participant with missing score (NaN)
        self._create_split_file("train_split.csv", [
            {"Participant_ID": 302, "Gender": "male", "PHQ_Score": 4, "PHQ_Binary": 0},
        ])
        self._create_split_file("dev_split.csv", [])
        self._create_split_file("test_split.csv", [
            {"Participant_ID": 600, "Gender": "female", "PHQ_Score": np.nan, "PHQ_Binary": np.nan},
        ])

        meta_df = load_official_metadata(self.meta_dir, mapped_csv_path=None)
        manifest_df = build_participant_manifest(meta_df, features_dict={})

        unlabeled = manifest_df[manifest_df["participant_id"] == "600"].iloc[0]
        self.assertFalse(bool(unlabeled["has_ground_truth_phq"]))
        self.assertTrue(pd.isna(unlabeled["phq_score"]))
        # Unlabeled must NOT be treated as negative (0)
        self.assertTrue(pd.isna(unlabeled["phq_binary_cutoff10"]))
        self.assertNotEqual(unlabeled["phq_binary_cutoff10"], 0)

    def test_manifest_validation_pass(self):
        self._create_split_file("train_split.csv", [
            {"Participant_ID": 302, "Gender": "male", "PHQ_Score": 4, "PHQ_Binary": 0},
            {"Participant_ID": 303, "Gender": "female", "PHQ_Score": 12, "PHQ_Binary": 1},
        ])
        self._create_split_file("dev_split.csv", [
            {"Participant_ID": 300, "Gender": "male", "PHQ_Score": 2, "PHQ_Binary": 0},
        ])
        self._create_split_file("test_split.csv", [
            {"Participant_ID": 600, "Gender": "female", "PHQ_Score": 8, "PHQ_Binary": 0},
        ])

        features = {
            "302": {"has_transcript": True, "has_mfcc": True, "has_egemaps": True},
            "303": {"has_transcript": True, "has_mfcc": True, "has_egemaps": True},
        }

        meta_df = load_official_metadata(self.meta_dir, mapped_csv_path=None)
        manifest_df = build_participant_manifest(meta_df, features)
        report = validate_manifest(manifest_df)

        self.assertTrue(report["validation_passed"])
        self.assertEqual(report["total_manifest_participants"], 4)
        self.assertEqual(report["duplicate_id_count"], 0)
        self.assertFalse(report["has_split_overlap"])
        self.assertEqual(report["extracted_features_summary"]["total_with_extracted_features"], 2)


if __name__ == "__main__":
    unittest.main()
