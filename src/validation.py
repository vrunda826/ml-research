"""Shared validation and data leakage prevention module for DAIC-WOZ experiments.

Provides strict assertions for participant-level split separation,
feature alignment, target integrity, and leakage prevention.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_split_separation(
    train_ids: Iterable[str],
    dev_ids: Iterable[str],
    test_ids: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Verify strict participant-level split disjointness and absence of duplicate IDs.
    
    Raises ValueError if any participant appears in more than one split or if duplicates exist.
    """
    train_list = [str(x).strip() for x in train_ids]
    dev_list = [str(x).strip() for x in dev_ids]
    test_list = [str(x).strip() for x in test_ids] if test_ids is not None else []

    # Check within-split uniqueness
    train_set = set(train_list)
    dev_set = set(dev_list)
    test_set = set(test_list)

    train_dups = len(train_list) - len(train_set)
    dev_dups = len(dev_list) - len(dev_set)
    test_dups = len(test_list) - len(test_set)

    if train_dups > 0:
        raise ValueError(f"Duplicate participant IDs found in training split ({train_dups} duplicates).")
    if dev_dups > 0:
        raise ValueError(f"Duplicate participant IDs found in development split ({dev_dups} duplicates).")
    if test_dups > 0:
        raise ValueError(f"Duplicate participant IDs found in test split ({test_dups} duplicates).")

    # Check pairwise disjointness
    train_dev = train_set.intersection(dev_set)
    train_test = train_set.intersection(test_set)
    dev_test = dev_set.intersection(test_set)

    if train_dev:
        raise ValueError(f"Data leakage detected! Participants appear in both Train and Dev: {sorted(train_dev)}")
    if train_test:
        raise ValueError(f"Data leakage detected! Participants appear in both Train and Test: {sorted(train_test)}")
    if dev_test:
        raise ValueError(f"Data leakage detected! Participants appear in both Dev and Test: {sorted(dev_test)}")

    return {
        "is_valid": True,
        "n_train": len(train_set),
        "n_dev": len(dev_set),
        "n_test": len(test_set),
        "total_unique": len(train_set | dev_set | test_set),
    }


def validate_feature_matrix_alignment(
    participant_ids: list[str],
    feature_matrix: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> None:
    """Validate that feature matrix matches participant count and contains no non-finite values."""
    if len(participant_ids) != feature_matrix.shape[0]:
        raise ValueError(
            f"Shape mismatch: {len(participant_ids)} participant IDs but feature matrix has {feature_matrix.shape[0]} rows."
        )

    # Check for NaN / Inf
    if not np.isfinite(feature_matrix).all():
        nan_count = np.isnan(feature_matrix).sum()
        inf_count = np.isinf(feature_matrix).sum()
        raise ValueError(
            f"Feature matrix contains non-finite values (NaN: {nan_count}, Inf: {inf_count}). Imputation required."
        )

    if feature_names is not None and len(feature_names) != feature_matrix.shape[1]:
        raise ValueError(
            f"Feature names count ({len(feature_names)}) does not match feature columns ({feature_matrix.shape[1]})."
        )


def validate_label_integrity(
    labels: np.ndarray | pd.Series,
    split_name: str,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Validate binary depression labels (must be 0 or 1, no NaNs unless explicitly allowed)."""
    s = pd.Series(labels)

    if not allow_missing and s.isnull().any():
        null_count = s.isnull().sum()
        raise ValueError(f"Found {null_count} missing labels in {split_name} split where missing labels are disallowed.")

    valid_s = s.dropna()
    unique_vals = set(valid_s.unique())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Labels in {split_name} contain values outside {{0, 1}}: {unique_vals}")

    counts = valid_s.value_counts().to_dict()
    n_0 = int(counts.get(0, counts.get(0.0, 0)))
    n_1 = int(counts.get(1, counts.get(1.0, 0)))

    return {
        "split": split_name,
        "total": len(s),
        "labeled": len(valid_s),
        "unlabeled": len(s) - len(valid_s),
        "count_0_non_depressed": n_0,
        "count_1_depressed": n_1,
        "prevalence": round(n_1 / len(valid_s), 4) if len(valid_s) > 0 else 0.0,
    }


def assert_no_test_labels_used(
    model_selection_info: dict[str, Any],
) -> None:
    """Ensure test set labels were never accessed during model tuning or selection."""
    if "test_metrics" in model_selection_info:
        raise ValueError(
            "Protocol violation: Test metrics cannot be present in model selection metadata!"
        )
    if model_selection_info.get("selection_dataset") == "test":
        raise ValueError(
            "Protocol violation: Model selection must be conducted on development set, never test set!"
        )


FORBIDDEN_TARGET_SUBSTRINGS = {
    "phq",
    "depression_label",
    "phq_score",
    "phq8",
    "phq_binary",
    "ground_truth",
}


def validate_no_target_in_features(feature_names: Iterable[str]) -> None:
    """Ensure no target labels or PHQ-8 scores accidentally leak into feature sets.
    
    Raises ValueError if any feature column name suggests a label or ground-truth score.
    """
    leaked_cols = []
    for col in feature_names:
        col_lower = str(col).strip().lower()
        for forbidden in FORBIDDEN_TARGET_SUBSTRINGS:
            if forbidden in col_lower:
                leaked_cols.append(col)
                break

    if leaked_cols:
        raise ValueError(
            f"Target leakage detected! Feature columns contain target-derived variables: {leaked_cols}"
        )
