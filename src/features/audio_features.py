"""Audio feature aggregation for DAIC-WOZ acoustic baselines.

Extracts frame-level OpenSMILE features (eGeMAPS, MFCC) participant-by-participant,
computes compact participant-level summary statistics (mean, std), and outputs
structured, leakage-free feature matrices.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.build_manifest import normalize_participant_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LIGHTWEIGHT_DIR = ROOT_DIR / "data" / "processed" / "lightweight"
DEFAULT_MANIFEST_PATH = ROOT_DIR / "data" / "processed" / "participant_manifest.csv"
DEFAULT_AUDIO_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "features" / "audio"

NON_ACOUSTIC_COLUMNS = {"name", "frametime", "frame_time", "timestamp", "time"}


def extract_participant_acoustic_summary(
    csv_path: Path,
    expected_delimiter: str = ";",
) -> tuple[dict[str, float], list[str]]:
    """Compute mean and std for all numeric acoustic features in a frame-level CSV.
    
    Excludes non-acoustic columns (identifiers, timestamps).
    Handles empty/non-finite values safely.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise FileNotFoundError(f"Feature file missing or empty: {csv_path}")

    # Inspect header
    with csv_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f, delimiter=expected_delimiter)
        header = next(reader, None)

    if not header:
        raise ValueError(f"Empty header in {csv_path}")

    # Determine numeric acoustic feature columns
    acoustic_cols: list[str] = []
    for col in header:
        clean_col = col.strip()
        if clean_col.lower() not in NON_ACOUSTIC_COLUMNS:
            acoustic_cols.append(clean_col)

    if not acoustic_cols:
        raise ValueError(f"No numeric acoustic feature columns discovered in {csv_path}")

    # Read numeric columns
    df = pd.read_csv(csv_path, sep=expected_delimiter, usecols=acoustic_cols)
    if len(df) == 0:
        raise ValueError(f"Acoustic feature file {csv_path} contains 0 data rows.")

    # Coerce to numeric and replace non-finite
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Compute summary statistics
    means = df.mean(axis=0)
    stds = df.std(axis=0).fillna(0.0)

    summary: dict[str, float] = {}
    feature_names: list[str] = []

    for col in acoustic_cols:
        mean_name = f"{col}_mean"
        std_name = f"{col}_std"

        m_val = float(means.get(col, 0.0))
        s_val = float(stds.get(col, 0.0))

        summary[mean_name] = m_val if np.isfinite(m_val) else 0.0
        summary[std_name] = s_val if np.isfinite(s_val) else 0.0

        feature_names.extend([mean_name, std_name])

    return summary, feature_names


def build_audio_features(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    lightweight_dir: Path = DEFAULT_LIGHTWEIGHT_DIR,
    feature_type: str = "egemaps",
    output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR,
) -> pd.DataFrame:
    """Build and save participant-level acoustic features for all eligible participants."""
    manifest_df = pd.read_csv(manifest_path)

    # Filter to participants that have this feature
    feature_flag = f"has_{feature_type}"
    if feature_flag not in manifest_df.columns:
        raise KeyError(f"Column {feature_flag} not found in manifest: {manifest_df.columns.tolist()}")

    eligible_df = manifest_df[manifest_df[feature_flag] == True].copy()
    logger.info(f"Extracting {feature_type} features for {len(eligible_df)} participants...")

    file_suffix = "_OpenSMILE2.3.0_egemaps.csv" if feature_type == "egemaps" else "_OpenSMILE2.3.0_mfcc.csv"

    records: list[dict[str, Any]] = []
    canonical_feature_names: Optional[list[str]] = None

    for _, row in tqdm(eligible_df.iterrows(), total=len(eligible_df), desc=f"Extracting {feature_type}"):
        pid = normalize_participant_id(row["participant_id"])
        file_path = lightweight_dir / pid / f"{pid}{file_suffix}"

        try:
            summary, feat_names = extract_participant_acoustic_summary(file_path, expected_delimiter=";")
            if canonical_feature_names is None:
                canonical_feature_names = feat_names

            rec: dict[str, Any] = {
                "participant_id": pid,
                "split": row["split"],
                "gender": row.get("gender"),
                "phq_score": row.get("phq_score"),
                "depression_label": row.get("phq_binary_cutoff10"),
            }
            rec.update(summary)
            records.append(rec)
        except Exception as exc:
            logger.error(f"Failed to process participant {pid} ({file_path}): {exc}")

    feat_df = pd.DataFrame(records)

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{feature_type}_features.csv"
    feat_df.to_csv(out_file, index=False)

    logger.info(
        f"Saved {feature_type} features: {len(feat_df)} rows, "
        f"{len(canonical_feature_names or [])} acoustic features to {out_file}"
    )
    return feat_df


if __name__ == "__main__":
    # Extract both eGeMAPS and MFCC
    build_audio_features(feature_type="egemaps")
    build_audio_features(feature_type="mfcc")
