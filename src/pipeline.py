"""End-to-end pipeline runner for DAIC-WOZ project (Phases 1 through 5).

Orchestrates data audit, manifest generation, feature extraction, and
training for Audio (B1), Text (B2), and Multimodal (B3) baselines.

Usage:
    python src/pipeline.py [--all | --phase {1,3,4,5}]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.audit_dataset import run_dataset_audit
from src.build_manifest import build_and_validate
from src.features.audio_features import build_audio_features
from src.features.text_features import build_text_features
from src.features.multimodal_features import build_multimodal_features
from src.train_audio_baseline import run_audio_experiments
from src.train_text_baseline import run_text_experiments
from src.train_multimodal_baseline import run_multimodal_experiments

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_phase1() -> bool:
    """Run Phase 1: Audit and Manifest."""
    logger.info("\n=== RUNNING PHASE 1: DATA AUDIT & MANIFEST ===")
    audit_report = run_dataset_audit()
    if audit_report["incomplete_participants"] > 0:
        logger.warning(f"Incomplete participants: {audit_report['incomplete_participant_ids']}")
    manifest_df, val_report = build_and_validate()
    return bool(val_report["validation_passed"])


def run_phase3() -> bool:
    """Run Phase 3: Audio Features and Baseline Experiments (B0, B1)."""
    logger.info("\n=== RUNNING PHASE 3: AUDIO BASELINES (B0, B1) ===")
    egemaps_file = ROOT_DIR / "data" / "processed" / "features" / "audio" / "egemaps_features.csv"
    mfcc_file = ROOT_DIR / "data" / "processed" / "features" / "audio" / "mfcc_features.csv"

    if not egemaps_file.exists():
        build_audio_features(feature_type="egemaps")
    if not mfcc_file.exists():
        build_audio_features(feature_type="mfcc")

    run_audio_experiments()
    return True


def run_phase4() -> bool:
    """Run Phase 4: Text Features and Baseline Experiments (B2)."""
    logger.info("\n=== RUNNING PHASE 4: TEXT BASELINE (B2) ===")
    text_file = ROOT_DIR / "data" / "processed" / "features" / "text" / "text_embeddings_minilm.csv"
    if not text_file.exists():
        build_text_features()

    run_text_experiments()
    return True


def run_phase5() -> bool:
    """Run Phase 5: Multimodal Features and Baseline Experiments (B3)."""
    logger.info("\n=== RUNNING PHASE 5: MULTIMODAL BASELINE (B3) ===")
    mm_egemaps = ROOT_DIR / "data" / "processed" / "features" / "multimodal" / "multimodal_egemaps_text.csv"
    mm_mfcc = ROOT_DIR / "data" / "processed" / "features" / "multimodal" / "multimodal_mfcc_text.csv"

    if not mm_egemaps.exists():
        build_multimodal_features(audio_type="egemaps")
    if not mm_mfcc.exists():
        build_multimodal_features(audio_type="mfcc")

    run_multimodal_experiments()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="DAIC-WOZ Research Pipeline")
    parser.add_argument("--phase", choices=["1", "3", "4", "5", "6", "all"], default="all", help="Phase to execute")
    args = parser.parse_args()

    phase = args.phase
    logger.info("=" * 65)
    logger.info("DAIC-WOZ DEPRESSION RESEARCH PIPELINE")
    logger.info("=" * 65)

    try:
        if phase in ["1", "all"]:
            if not run_phase1():
                return 1
        if phase in ["3", "all"]:
            if not run_phase3():
                return 1
        if phase in ["4", "all"]:
            if not run_phase4():
                return 1
        if phase in ["5", "all"]:
            if not run_phase5():
                return 1
        if phase in ["6", "all"]:
            logger.info("\n=== RUNNING PHASE 6: PHQ-8 REGRESSION & HSED ANALYSIS ===")
            from src.phase6_regression import run_all_phase6
            if not run_all_phase6():
                return 1

        logger.info("\n" + "=" * 65)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Manifest:          {ROOT_DIR / 'data' / 'processed' / 'participant_manifest.csv'}")
        logger.info(f"Audio Results:     {ROOT_DIR / 'reports' / 'audio_baseline_results.csv'}")
        logger.info(f"Text Results:      {ROOT_DIR / 'reports' / 'text_baseline_results.csv'}")
        logger.info(f"Multimodal Results:{ROOT_DIR / 'reports' / 'multimodal_baseline_results.csv'}")
        logger.info(f"Unified Comparison:{ROOT_DIR / 'reports' / 'baseline_comparison.csv'}")
        logger.info(f"Phase 6 Reports:   {ROOT_DIR / 'reports' / 'phase6'}")
        logger.info("=" * 65)
        return 0

    except Exception as exc:
        logger.error(f"Pipeline failed with exception: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
