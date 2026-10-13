"""Participant-level text representation extraction for DAIC-WOZ (B2).

Loads participant transcripts, performs conservative text cleaning, generates
utterance sentence embeddings via MiniLM, and pools into participant representations.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.build_manifest import normalize_participant_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LIGHTWEIGHT_DIR = ROOT_DIR / "data" / "processed" / "lightweight"
DEFAULT_MANIFEST_PATH = ROOT_DIR / "data" / "processed" / "participant_manifest.csv"
DEFAULT_TEXT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "features" / "text"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_and_clean_transcript(csv_path: Path, allow_empty: bool = False) -> list[str]:
    """Load transcript CSV, validate columns, and apply conservative text cleaning.
    
    Preserves negations, punctuation, and linguistic content. Drops empty lines.
    Note: DAIC-WOZ lightweight transcripts do not contain speaker IDs; utterances
    represent full conversation dialogue.
    """
    if not csv_path.exists():
        if allow_empty:
            return []
        raise FileNotFoundError(f"Transcript file missing: {csv_path}")

    if csv_path.stat().st_size == 0:
        if allow_empty:
            return []
        raise ValueError(f"Transcript file is empty (0 bytes): {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        if allow_empty:
            logger.warning(f"Failed to read transcript {csv_path}: {exc}")
            return []
        raise ValueError(f"Failed to read transcript {csv_path}: {exc}") from exc

    if "Text" not in df.columns:
        if allow_empty:
            logger.warning(f"No 'Text' column in {csv_path}. Columns: {df.columns.tolist()}")
            return []
        raise ValueError(f"No 'Text' column in {csv_path}. Columns: {df.columns.tolist()}")

    utterances: list[str] = []
    for raw_text in df["Text"].dropna():
        cleaned = str(raw_text).strip()
        if cleaned:
            utterances.append(cleaned)

    if not utterances and not allow_empty:
        raise ValueError(f"Transcript {csv_path} contains 0 non-empty utterances.")

    return utterances


def extract_participant_text_embedding(
    utterances: list[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    allow_empty: bool = False,
) -> np.ndarray:
    """Encode utterances in batches and compute mean pooling over session."""
    if not utterances:
        if allow_empty:
            return np.zeros(EMBEDDING_DIM, dtype=float)
        raise ValueError("Cannot extract text embeddings from empty utterance list.")

    # Encode utterances with batching to control memory
    embeddings = model.encode(
        utterances,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Mean pooling across utterances
    pooled = np.mean(embeddings, axis=0)
    return pooled.astype(float)


def build_text_features(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    lightweight_dir: Path = DEFAULT_LIGHTWEIGHT_DIR,
    output_dir: Path = DEFAULT_TEXT_OUTPUT_DIR,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Build and save participant-level text embeddings for all participants with transcripts."""
    manifest_df = pd.read_csv(manifest_path)

    eligible_df = manifest_df[manifest_df["has_transcript"] == True].copy()
    logger.info(f"Generating sentence embeddings for {len(eligible_df)} participants using {model_name}...")

    # Load frozen sentence-transformer model
    model = SentenceTransformer(model_name)

    records: list[dict[str, Any]] = []
    emb_cols = [f"text_emb_{i}" for i in range(EMBEDDING_DIM)]

    for _, row in tqdm(eligible_df.iterrows(), total=len(eligible_df), desc="Encoding Transcripts"):
        pid = normalize_participant_id(row["participant_id"])
        transcript_path = lightweight_dir / pid / f"{pid}_Transcript.csv"

        utterances = load_and_clean_transcript(transcript_path)
        emb_vector = extract_participant_text_embedding(utterances, model)

        rec: dict[str, Any] = {
            "participant_id": pid,
            "split": row["split"],
            "gender": row.get("gender"),
            "phq_score": row.get("phq_score"),
            "depression_label": row.get("phq_binary_cutoff10"),
            "utterance_count": len(utterances),
        }
        for i, val in enumerate(emb_vector):
            rec[emb_cols[i]] = float(val)

        records.append(rec)

    text_df = pd.DataFrame(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "text_embeddings_minilm.csv"
    text_df.to_csv(out_file, index=False)

    logger.info(
        f"Saved text embeddings: {len(text_df)} participants, "
        f"{EMBEDDING_DIM} embedding dimensions to {out_file}"
    )
    return text_df


if __name__ == "__main__":
    build_text_features()
