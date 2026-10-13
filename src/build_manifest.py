"""Build and validate participant-level manifest for DAIC-WOZ dataset.

Joins official metadata splits with extracted lightweight features,
applies the standardized PHQ-8 >= 10 depression label, verifies data integrity
(split overlaps, valid ranges, missing features, duplicate IDs), and generates
both CSV manifest and JSON validation reports.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = ROOT_DIR / "data" / "metadata"
LIGHTWEIGHT_DIR = ROOT_DIR / "data" / "processed" / "lightweight"
ARCHIVE_LABELS_TAR = ROOT_DIR / "labels2019.tar.gz"
MAPPED_METADATA_CSV = ROOT_DIR / "metadata_mapped.csv"

DEFAULT_OUTPUT_MANIFEST = ROOT_DIR / "data" / "processed" / "participant_manifest.csv"
DEFAULT_OUTPUT_REPORT = ROOT_DIR / "data" / "processed" / "manifest_validation_report.json"
RESULTS_OUTPUT_REPORT = ROOT_DIR / "results" / "manifest_validation_report.json"

PHQ8_MIN_SCORE = 0
PHQ8_MAX_SCORE = 24
PHQ8_DEPRESSION_CUTOFF = 10


def normalize_participant_id(raw_id: Any) -> str:
    """Safely normalize a participant identifier without altering its numeric identity.
    
    Examples:
        300 -> '300'
        '300' -> '300'
        '300_P' -> '300'
        ' 300 ' -> '300'
    """
    if raw_id is None or (isinstance(raw_id, float) and np.isnan(raw_id)):
        raise ValueError("Participant ID cannot be null or NaN.")

    s = str(raw_id).strip()
    # Remove trailing _P or _p if present from archive naming
    s = re.sub(r"_[Pp]$", "", s)

    # Check for integer representation (e.g. '300.0' from float conversion)
    if re.match(r"^\d+\.0$", s):
        s = s.split(".")[0]

    if not s:
        raise ValueError(f"Participant ID normalized to empty string from '{raw_id}'.")

    return s


def ensure_metadata_extracted(
    metadata_dir: Path = METADATA_DIR,
    archive_tar: Path = ARCHIVE_LABELS_TAR,
) -> Path:
    """Ensure official metadata CSV files are present in metadata_dir.
    
    Extracts from archive_tar if necessary.
    """
    metadata_dir.mkdir(parents=True, exist_ok=True)
    required_splits = ["train_split.csv", "dev_split.csv", "test_split.csv"]

    missing_files = [f for f in required_splits if not (metadata_dir / f).exists()]
    if missing_files and archive_tar.exists():
        logger.info(f"Extracting missing metadata files from {archive_tar} to {metadata_dir}...")
        with tarfile.open(archive_tar, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".csv"):
                    target_name = Path(member.name).name
                    dest = metadata_dir / target_name
                    with tar.extractfile(member) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    logger.info(f"Extracted: {target_name}")

    # Recheck
    missing_after = [f for f in required_splits if not (metadata_dir / f).exists()]
    if missing_after:
        raise FileNotFoundError(
            f"Required metadata files are missing: {missing_after}. "
            f"Please ensure they are placed in {metadata_dir} or {archive_tar} is available."
        )

    return metadata_dir


def load_official_metadata(
    metadata_dir: Path = METADATA_DIR,
    mapped_csv_path: Optional[Path] = MAPPED_METADATA_CSV,
) -> pd.DataFrame:
    """Load and normalize official train, dev, and test split metadata."""
    splits_spec = [
        ("train", metadata_dir / "train_split.csv"),
        ("dev", metadata_dir / "dev_split.csv"),
        ("test", metadata_dir / "test_split.csv"),
    ]

    records: list[dict[str, Any]] = []
    seen_ids_by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}

    for split_name, split_path in splits_spec:
        if not split_path.exists():
            raise FileNotFoundError(f"Missing official split file: {split_path}")

        try:
            df = pd.read_csv(split_path)
        except pd.errors.EmptyDataError:
            logger.warning(f"Split file {split_path} is empty. Skipping.")
            continue

        if df.empty:
            continue

        # Discover participant identifier column
        id_col = None
        for col in ["Participant_ID", "participant_id", "ParticipantID", "ID"]:
            if col in df.columns:
                id_col = col
                break
        if id_col is None:
            raise KeyError(f"Could not discover participant ID column in {split_path}. Found: {df.columns.tolist()}")

        for _, row in df.iterrows():
            norm_id = normalize_participant_id(row[id_col])

            if norm_id in seen_ids_by_split[split_name]:
                raise ValueError(f"Duplicate participant ID {norm_id} found in {split_name} split!")
            seen_ids_by_split[split_name].add(norm_id)

            # Gender normalization
            raw_gender = row.get("Gender")
            gender = str(raw_gender).strip().lower() if pd.notna(raw_gender) else None

            # PHQ-8 score and official binary label
            raw_score = row.get("PHQ_Score")
            raw_binary = row.get("PHQ_Binary")

            phq_score: Optional[int] = None
            if pd.notna(raw_score):
                phq_score = int(round(float(raw_score)))

            official_binary: Optional[int] = None
            if pd.notna(raw_binary):
                official_binary = int(round(float(raw_binary)))

            # PTSD scores if present
            pcl_ptsd = row.get("PCL-C (PTSD)")
            ptsd_sev = row.get("PTSD Severity")

            records.append({
                "participant_id": norm_id,
                "split": split_name,
                "gender": gender,
                "phq_score": phq_score,
                "official_phq_binary": official_binary,
                "pcl_ptsd": int(round(float(pcl_ptsd))) if pd.notna(pcl_ptsd) else None,
                "ptsd_severity": float(ptsd_sev) if pd.notna(ptsd_sev) else None,
            })

    meta_df = pd.DataFrame(records)

    # Check for split overlaps
    train_ids = seen_ids_by_split["train"]
    dev_ids = seen_ids_by_split["dev"]
    test_ids = seen_ids_by_split["test"]

    train_dev = train_ids.intersection(dev_ids)
    train_test = train_ids.intersection(test_ids)
    dev_test = dev_ids.intersection(test_ids)

    if train_dev or train_test or dev_test:
        raise ValueError(
            f"Split overlap detected! Train/Dev: {train_dev}, "
            f"Train/Test: {train_test}, Dev/Test: {dev_test}"
        )

    # Attach optional AVECParticipant_ID from mapped CSV if present
    if mapped_csv_path and mapped_csv_path.exists():
        try:
            mapped_df = pd.read_csv(mapped_csv_path)
            if "Participant_ID" in mapped_df.columns and "AVECParticipant_ID" in mapped_df.columns:
                mapped_df["norm_id"] = mapped_df["Participant_ID"].apply(normalize_participant_id)
                mapping = dict(zip(mapped_df["norm_id"], mapped_df["AVECParticipant_ID"]))
                meta_df["avec_id"] = meta_df["participant_id"].map(mapping)
        except Exception as exc:
            logger.warning(f"Could not load AVECParticipant_ID mapping: {exc}")
            meta_df["avec_id"] = None
    else:
        meta_df["avec_id"] = None

    return meta_df


def discover_extracted_features(lightweight_dir: Path = LIGHTWEIGHT_DIR) -> dict[str, dict[str, Any]]:
    """Discover feature file availability for participants in lightweight_dir."""
    if not lightweight_dir.exists():
        return {}

    feature_info: dict[str, dict[str, Any]] = {}

    for folder in lightweight_dir.iterdir():
        if not folder.is_dir():
            continue

        pid = normalize_participant_id(folder.name)

        transcript_file = folder / f"{pid}_Transcript.csv"
        mfcc_file = folder / f"{pid}_OpenSMILE2.3.0_mfcc.csv"
        egemaps_file = folder / f"{pid}_OpenSMILE2.3.0_egemaps.csv"

        has_t = transcript_file.exists() and transcript_file.stat().st_size > 0
        has_m = mfcc_file.exists() and mfcc_file.stat().st_size > 0
        has_e = egemaps_file.exists() and egemaps_file.stat().st_size > 0

        feature_info[pid] = {
            "has_transcript": has_t,
            "has_mfcc": has_m,
            "has_egemaps": has_e,
            "all_features_available": has_t and has_m and has_e,
            "feature_dir": folder,
        }

    return feature_info


def build_participant_manifest(
    metadata_df: pd.DataFrame,
    features_dict: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Create complete participant manifest by joining metadata and feature availability."""
    meta_ids = set(metadata_df["participant_id"].unique())
    feature_ids = set(features_dict.keys())
    all_pids = meta_ids.union(feature_ids)

    # Index metadata by participant_id
    meta_by_id = metadata_df.set_index("participant_id").to_dict(orient="index")

    manifest_rows: list[dict[str, Any]] = []

    for pid in all_pids:
        meta = meta_by_id.get(pid, {})
        feat = features_dict.get(pid, {})

        split = meta.get("split", "unassigned")
        gender = meta.get("gender")
        raw_phq_score = meta.get("phq_score")
        raw_official_binary = meta.get("official_phq_binary")
        avec_id = meta.get("avec_id")

        has_ground_truth_score = bool(pd.notna(raw_phq_score) and raw_phq_score is not None)
        phq_score = int(round(float(raw_phq_score))) if has_ground_truth_score else None

        has_official_binary = bool(pd.notna(raw_official_binary) and raw_official_binary is not None)
        official_binary = int(round(float(raw_official_binary))) if has_official_binary else None

        # Binary depression label using PHQ-8 >= 10 ONLY when a valid score exists
        if has_ground_truth_score and phq_score is not None:
            phq_binary = 1 if phq_score >= PHQ8_DEPRESSION_CUTOFF else 0
        else:
            phq_binary = None

        has_t = feat.get("has_transcript", False)
        has_m = feat.get("has_mfcc", False)
        has_e = feat.get("has_egemaps", False)
        all_features = has_t and has_m and has_e

        missing_list: list[str] = []
        if not has_t:
            missing_list.append("transcript")
        if not has_m:
            missing_list.append("mfcc")
        if not has_e:
            missing_list.append("egemaps")
        missing_features_str = ", ".join(missing_list) if missing_list else "none"

        # Explicit data status
        if all_features and has_ground_truth_score:
            data_status = "COMPLETE"
        elif all_features and not has_ground_truth_score:
            data_status = "FEATURES_ONLY_UNLABELED"
        elif pid not in meta_ids:
            data_status = "UNMATCHED_FEATURES_NO_METADATA"
        elif not has_t and not has_m and not has_e:
            data_status = "METADATA_ONLY_NO_FEATURES"
        else:
            data_status = "PARTIAL_FEATURES"

        manifest_rows.append({
            "participant_id": pid,
            "split": split,
            "gender": gender,
            "has_ground_truth_phq": has_ground_truth_score,
            "phq_score": phq_score,
            "phq_binary_cutoff10": phq_binary,
            "official_phq_binary": official_binary,
            "has_transcript": has_t,
            "has_mfcc": has_m,
            "has_egemaps": has_e,
            "all_features_available": all_features,
            "missing_features": missing_features_str,
            "data_status": data_status,
            "avec_id": avec_id,
        })

    manifest_df = pd.DataFrame(manifest_rows)

    # Sort deterministically: split priority ('train', 'dev', 'test', 'unassigned'), then numeric ID
    split_order = {"train": 0, "dev": 1, "test": 2, "unassigned": 3}
    manifest_df["_split_sort"] = manifest_df["split"].map(lambda s: split_order.get(s, 99))
    manifest_df["_id_sort"] = manifest_df["participant_id"].apply(
        lambda x: (0, int(x)) if x.isdigit() else (1, x)
    )
    manifest_df = manifest_df.sort_values(by=["_split_sort", "_id_sort"]).reset_index(drop=True)
    manifest_df = manifest_df.drop(columns=["_split_sort", "_id_sort"])

    return manifest_df


def validate_manifest(manifest_df: pd.DataFrame) -> dict[str, Any]:
    """Perform rigorous validation on the participant manifest."""
    total_participants = len(manifest_df)
    unique_ids = manifest_df["participant_id"].nunique()
    duplicate_count = total_participants - unique_ids

    # 1. Split distribution
    splits = ["train", "dev", "test"]
    split_summary: dict[str, dict[str, Any]] = {}

    for split in splits + ["unassigned"]:
        sub = manifest_df[manifest_df["split"] == split]
        if sub.empty and split == "unassigned":
            continue

        n_total = len(sub)
        n_labeled = sub["has_ground_truth_phq"].sum()
        n_complete_features = sub["all_features_available"].sum()
        n_fully_complete = (sub["data_status"] == "COMPLETE").sum()

        labeled_sub = sub[sub["has_ground_truth_phq"]]
        if not labeled_sub.empty:
            depressed_count = (labeled_sub["phq_binary_cutoff10"] == 1).sum()
            non_depressed_count = (labeled_sub["phq_binary_cutoff10"] == 0).sum()
            prevalence = round(float(depressed_count / len(labeled_sub)), 4)
        else:
            depressed_count = 0
            non_depressed_count = 0
            prevalence = 0.0

        # Extracted complete subset for this split
        extracted_sub = sub[sub["all_features_available"]]
        extracted_depressed = (extracted_sub["phq_binary_cutoff10"] == 1).sum()
        extracted_non_depressed = (extracted_sub["phq_binary_cutoff10"] == 0).sum()

        split_summary[split] = {
            "total_metadata_participants": int(n_total),
            "with_extracted_features": int(n_complete_features),
            "missing_extracted_features": int(n_total - n_complete_features),
            "fully_complete_ready_for_experiments": int(n_fully_complete),
            "labeled_participants": int(n_labeled),
            "unlabeled_participants": int(n_total - n_labeled),
            "class_distribution_cutoff10": {
                "depressed_phq_ge_10": int(depressed_count),
                "non_depressed_phq_lt_10": int(non_depressed_count),
                "prevalence_rate": prevalence,
            },
            "extracted_features_class_distribution": {
                "depressed_phq_ge_10": int(extracted_depressed),
                "non_depressed_phq_lt_10": int(extracted_non_depressed),
                "total_extracted": int(len(extracted_sub)),
            },
        }

    # 2. PHQ-8 score validation
    labeled_rows = manifest_df[manifest_df["has_ground_truth_phq"]]
    invalid_scores = []
    for _, row in labeled_rows.iterrows():
        score = row["phq_score"]
        if score < PHQ8_MIN_SCORE or score > PHQ8_MAX_SCORE:
            invalid_scores.append({"participant_id": row["participant_id"], "score": score})

    # 3. Discrepancies between cutoff >= 10 and official_phq_binary
    official_discrepancies: list[dict[str, Any]] = []
    for _, row in labeled_rows.iterrows():
        c10 = row["phq_binary_cutoff10"]
        off = row["official_phq_binary"]
        if off is not None and c10 != off:
            official_discrepancies.append({
                "participant_id": row["participant_id"],
                "split": row["split"],
                "phq_score": row["phq_score"],
                "cutoff10_label": c10,
                "official_binary_label": off,
            })

    # 4. Check for split overlaps
    train_set = set(manifest_df[manifest_df["split"] == "train"]["participant_id"])
    dev_set = set(manifest_df[manifest_df["split"] == "dev"]["participant_id"])
    test_set = set(manifest_df[manifest_df["split"] == "test"]["participant_id"])

    overlap_train_dev = list(train_set.intersection(dev_set))
    overlap_train_test = list(train_set.intersection(test_set))
    overlap_dev_test = list(dev_set.intersection(test_set))
    has_split_overlap = bool(overlap_train_dev or overlap_train_test or overlap_dev_test)

    # 5. Missing features summary
    extracted_total = manifest_df["all_features_available"].sum()
    metadata_total = len(manifest_df[manifest_df["split"].isin(["train", "dev", "test"])])
    unmatched_feature_count = len(manifest_df[manifest_df["split"] == "unassigned"])

    # Convert numpy types to native Python types
    is_valid = bool(
        duplicate_count == 0
        and not has_split_overlap
        and len(invalid_scores) == 0
        and extracted_total > 0
    )

    validation_report: dict[str, Any] = {
        "validation_passed": bool(is_valid),
        "total_manifest_participants": int(total_participants),
        "unique_participant_ids": int(unique_ids),
        "duplicate_id_count": int(duplicate_count),
        "has_split_overlap": bool(has_split_overlap),
        "split_overlaps": {
            "train_dev": list(overlap_train_dev),
            "train_test": list(overlap_train_test),
            "dev_test": list(overlap_dev_test),
        },
        "score_validation": {
            "expected_range": f"{PHQ8_MIN_SCORE}-{PHQ8_MAX_SCORE}",
            "invalid_score_count": int(len(invalid_scores)),
            "invalid_scores": invalid_scores,
        },
        "official_binary_vs_cutoff10_discrepancies": {
            "count": int(len(official_discrepancies)),
            "description": (
                "Participants where PHQ_Score >= 10 but official AVEC PHQ_Binary == 0 "
                "due to clinical DSM item criteria in official AVEC challenge metadata."
            ),
            "discrepancies": official_discrepancies,
        },
        "split_summary": split_summary,
        "extracted_features_summary": {
            "total_with_extracted_features": int(extracted_total),
            "total_in_official_metadata": int(metadata_total),
            "unmatched_participants_with_features": int(unmatched_feature_count),
            "complete_train_participants": int(split_summary.get("train", {}).get("with_extracted_features", 0)),
            "complete_dev_participants": int(split_summary.get("dev", {}).get("with_extracted_features", 0)),
            "complete_test_participants": int(split_summary.get("test", {}).get("with_extracted_features", 0)),
        },
    }

    return validation_report


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def print_manifest_summary(report: dict[str, Any], manifest_path: Path, report_path: Path) -> None:
    """Print readable terminal validation report."""
    print("\n" + "=" * 70)
    print("      DAIC-WOZ PARTICIPANT MANIFEST VALIDATION REPORT")
    print("=" * 70)
    print(f"Validation Status:            {'PASS' if report['validation_passed'] else 'FAIL'}")
    print(f"Total Participants in Manifest: {report['total_manifest_participants']}")
    print(f"Unique Participant IDs:        {report['unique_participant_ids']}")
    print(f"Duplicate IDs:                 {report['duplicate_id_count']}")
    print(f"Split Overlap Detected:        {report['has_split_overlap']}")
    print(f"Invalid PHQ-8 Scores:          {report['score_validation']['invalid_score_count']}")
    print("-" * 70)
    print("Split Membership & Feature Availability Breakdown:")
    print(f"{'Split':<8} | {'Total Meta':<10} | {'Features OK':<11} | {'Features Missing':<16} | {'Labeled':<7}")
    print("-" * 70)
    for sp in ["train", "dev", "test"]:
        s_data = report["split_summary"].get(sp, {})
        print(
            f"{sp:<8} | {s_data.get('total_metadata_participants', 0):<10} | "
            f"{s_data.get('with_extracted_features', 0):<11} | "
            f"{s_data.get('missing_extracted_features', 0):<16} | "
            f"{s_data.get('labeled_participants', 0):<7}"
        )
    print("-" * 70)
    print("Class Distribution for Extracted Features (PHQ-8 >= 10 Cutoff):")
    for sp in ["train", "dev", "test"]:
        s_data = report["split_summary"].get(sp, {})
        ext_cls = s_data.get("extracted_features_class_distribution", {})
        n_ext = ext_cls.get("total_extracted", 0)
        n_dep = ext_cls.get("depressed_phq_ge_10", 0)
        n_ndep = ext_cls.get("non_depressed_phq_lt_10", 0)
        rate = round(n_dep / n_ext, 4) if n_ext > 0 else 0.0
        print(
            f"  {sp:<6}: Total Extracted={n_ext:<3} | Depressed(>=10)={n_dep:<2} | "
            f"Non-Depressed(<10)={n_ndep:<2} | Prev={rate:.1%}"
        )
    print("-" * 70)
    print(
        f"Official AVEC PHQ_Binary vs Cutoff>=10 Mismatches: "
        f"{report['official_binary_vs_cutoff10_discrepancies']['count']} participants"
    )
    print(f"Manifest CSV saved to:         {manifest_path}")
    print(f"Validation JSON saved to:      {report_path}")
    print("=" * 70 + "\n")


def build_and_validate(
    metadata_dir: Path = METADATA_DIR,
    lightweight_dir: Path = LIGHTWEIGHT_DIR,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    output_report: Path = DEFAULT_OUTPUT_REPORT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run full build and validation pipeline."""
    # Ensure metadata is available
    ensure_metadata_extracted(metadata_dir)

    # Load metadata
    metadata_df = load_official_metadata(metadata_dir)

    # Discover features
    features_dict = discover_extracted_features(lightweight_dir)

    # Construct manifest
    manifest_df = build_participant_manifest(metadata_df, features_dict)

    # Validate
    validation_report = validate_manifest(manifest_df)

    # Save manifest CSV
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_manifest, index=False)

    # Save validation reports
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as jf:
        json.dump(validation_report, jf, indent=2, default=_json_default)

    RESULTS_OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_OUTPUT_REPORT.open("w", encoding="utf-8") as rjf:
        json.dump(validation_report, rjf, indent=2, default=_json_default)

    # Print summary
    print_manifest_summary(validation_report, output_manifest, output_report)

    return manifest_df, validation_report


if __name__ == "__main__":
    build_and_validate()
