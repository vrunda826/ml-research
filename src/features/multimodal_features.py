"""Multimodal feature alignment and fusion for DAIC-WOZ (B3).

Joins acoustic summary features (eGeMAPS, MFCC) and text embeddings strictly by
participant ID (never row order), validates label/split alignment, and outputs
leakage-free multimodal feature matrices.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.build_manifest import normalize_participant_id
from src.validation import validate_split_separation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

AUDIO_FEAT_DIR = ROOT_DIR / "data" / "processed" / "features" / "audio"
TEXT_FEAT_DIR = ROOT_DIR / "data" / "processed" / "features" / "text"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "features" / "multimodal"


def build_multimodal_features(
    audio_type: str = "egemaps",
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Align and concatenate audio and text representations by participant ID."""
    audio_path = AUDIO_FEAT_DIR / f"{audio_type}_features.csv"
    text_path = TEXT_FEAT_DIR / "text_embeddings_minilm.csv"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio features missing: {audio_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"Text features missing: {text_path}")

    audio_df = pd.read_csv(audio_path)
    text_df = pd.read_csv(text_path)

    # Normalize participant IDs
    audio_df["norm_id"] = audio_df["participant_id"].apply(normalize_participant_id)
    text_df["norm_id"] = text_df["participant_id"].apply(normalize_participant_id)

    # Check for duplicates within each table
    if audio_df["norm_id"].duplicated().any():
        raise ValueError(f"Duplicate participant IDs found in {audio_path}")
    if text_df["norm_id"].duplicated().any():
        raise ValueError(f"Duplicate participant IDs found in {text_path}")

    # Find common participant intersection
    audio_ids = set(audio_df["norm_id"])
    text_ids = set(text_df["norm_id"])
    common_ids = sorted(audio_ids.intersection(text_ids))

    logger.info(
        f"Aligning {audio_type.upper()} ({len(audio_ids)} participants) and "
        f"Text MiniLM ({len(text_ids)} participants) -> Common: {len(common_ids)} participants."
    )

    if not common_ids:
        raise ValueError("No common participants found between audio and text feature tables!")

    # Align strictly by participant ID
    audio_sub = audio_df.set_index("norm_id").loc[common_ids].reset_index()
    text_sub = text_df.set_index("norm_id").loc[common_ids].reset_index()

    # Validate label and split agreement
    if not (audio_sub["norm_id"] == text_sub["norm_id"]).all():
        raise ValueError("Critical alignment error: Participant IDs do not match after join!")
    if not (audio_sub["split"] == text_sub["split"]).all():
        raise ValueError("Split mismatch between audio and text features!")
    if not (audio_sub["depression_label"] == text_sub["depression_label"]).all():
        raise ValueError("Depression target label mismatch between audio and text features!")

    # Separate feature columns
    meta_cols_audio = {"norm_id", "participant_id", "split", "gender", "phq_score", "depression_label"}
    audio_feat_cols = [c for c in audio_df.columns if c not in meta_cols_audio]

    meta_cols_text = {"norm_id", "participant_id", "split", "gender", "phq_score", "depression_label", "utterance_count"}
    text_feat_cols = [c for c in text_df.columns if c not in meta_cols_text]

    # Construct aligned DataFrame via vectorized concatenation
    meta_df = pd.DataFrame({
        "participant_id": common_ids,
        "split": audio_sub["split"].values,
        "gender": audio_sub["gender"].values,
        "phq_score": audio_sub["phq_score"].values,
        "depression_label": audio_sub["depression_label"].astype(int).values,
    })

    audio_features_df = audio_sub[audio_feat_cols].copy()
    audio_features_df.columns = [f"audio_{col}" for col in audio_feat_cols]

    text_features_df = text_sub[text_feat_cols].copy()
    text_features_df.columns = [f"text_{col}" for col in text_feat_cols]

    mm_df = pd.concat([meta_df, audio_features_df, text_features_df], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"multimodal_{audio_type}_text.csv"
    mm_df.to_csv(out_file, index=False)

    n_feats = len(audio_feat_cols) + len(text_feat_cols)
    logger.info(
        f"Saved aligned multimodal features ({audio_type} + text): "
        f"{len(mm_df)} participants, {n_feats} features ({len(audio_feat_cols)} audio + {len(text_feat_cols)} text) to {out_file}"
    )
    return mm_df


if __name__ == "__main__":
    build_multimodal_features(audio_type="egemaps")
    build_multimodal_features(audio_type="mfcc")
