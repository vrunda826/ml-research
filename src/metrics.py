"""Comprehensive evaluation metrics for DAIC-WOZ baseline models.

Calculates classification, regression, and HSED research discrepancy metrics
with strict handling of single-class edge cases and missing predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray | list[int] | pd.Series,
    y_pred: np.ndarray | list[int] | pd.Series,
    y_prob: Optional[np.ndarray | list[float] | pd.Series] = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics for binary depression detection.
    
    Target: 1 = depressed (PHQ-8 >= 10), 0 = non-depressed (PHQ-8 < 10).
    Safely handles edge cases (e.g. single class present in evaluation subset).
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    n_samples = len(y_t)

    if n_samples == 0:
        raise ValueError("Cannot compute metrics on empty evaluation set.")

    if len(y_p) != n_samples:
        raise ValueError(f"Shape mismatch: {n_samples} true labels vs {len(y_p)} predictions.")

    unique_true = np.unique(y_t)
    has_both_classes = len(unique_true) > 1

    # Standard metrics
    acc = float(accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    # Depression class (1) metrics
    prec_1 = float(precision_score(y_t, y_p, pos_label=1, zero_division=0))
    rec_1 = float(recall_score(y_t, y_p, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))

    # Non-depressed class (0) metrics
    prec_0 = float(precision_score(y_t, y_p, pos_label=0, zero_division=0))
    rec_0 = float(recall_score(y_t, y_p, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_t, y_p, pos_label=0, zero_division=0))

    # Confusion matrix
    # Format: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_t, y_p, labels=[0, 1]).tolist()

    # ROC-AUC and PR-AUC require probabilities and both classes to be represented
    roc_auc: Optional[float] = None
    roc_auc_reason: Optional[str] = None
    pr_auc: Optional[float] = None
    pr_auc_reason: Optional[str] = None

    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=float)
        if len(y_pr) != n_samples:
            raise ValueError(f"Shape mismatch: {n_samples} true labels vs {len(y_pr)} predicted probabilities.")

        if not has_both_classes:
            roc_auc_reason = f"Only one class present in evaluation labels: {unique_true.tolist()}"
            pr_auc_reason = f"Only one class present in evaluation labels: {unique_true.tolist()}"
        else:
            try:
                roc_auc = float(roc_auc_score(y_t, y_pr))
            except Exception as exc:
                roc_auc_reason = f"ROC-AUC computation failed: {exc}"

            try:
                pr_auc = float(average_precision_score(y_t, y_pr))
            except Exception as exc:
                pr_auc_reason = f"PR-AUC computation failed: {exc}"
    else:
        roc_auc_reason = "Predicted probabilities not provided"
        pr_auc_reason = "Predicted probabilities not provided"

    return {
        "n_eval": n_samples,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "depression_precision": round(prec_1, 4),
        "depression_recall": round(rec_1, 4),
        "depression_f1": round(f1_1, 4),
        "non_depressed_precision": round(prec_0, 4),
        "non_depressed_recall": round(rec_0, 4),
        "non_depressed_f1": round(f1_0, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "roc_auc_unavailable_reason": roc_auc_reason,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "pr_auc_unavailable_reason": pr_auc_reason,
        "confusion_matrix": {
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
            "matrix": cm,
        },
    }


def compute_regression_metrics(
    y_true_scores: np.ndarray | list[float] | pd.Series,
    y_pred_scores: np.ndarray | list[float] | pd.Series,
) -> dict[str, Any]:
    """Compute continuous PHQ-8 score regression metrics (MAE, RMSE, R2, Spearman).
    
    Explicitly handles constant inputs and zero-variance edge cases without silent zero fallbacks.
    """
    y_t = np.asarray(y_true_scores, dtype=float)
    y_p = np.asarray(y_pred_scores, dtype=float)

    if len(y_t) == 0:
        raise ValueError("Cannot compute metrics on empty evaluation set.")

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: {len(y_t)} true scores vs {len(y_p)} predictions.")

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # R-squared: handle edge cases (e.g. len < 2 or single distinct true value)
    r2: Optional[float] = None
    r2_note: Optional[str] = None
    if len(y_t) >= 2 and np.var(y_t) > 1e-10:
        r2 = float(r2_score(y_t, y_p))
    else:
        r2_note = "R2 undefined: evaluation set has zero target variance or fewer than 2 samples."

    # Spearman rank correlation: handle constant inputs explicitly
    spearman_rho: Optional[float] = None
    spearman_p: Optional[float] = None
    spearman_note: Optional[str] = None

    if len(y_t) >= 2 and np.var(y_t) > 1e-10 and np.var(y_p) > 1e-10:
        try:
            res = spearmanr(y_t, y_p)
            rho_val = float(res.statistic) if hasattr(res, "statistic") else float(res.correlation)
            p_val = float(res.pvalue)
            if np.isfinite(rho_val):
                spearman_rho = rho_val
                spearman_p = p_val
            else:
                spearman_note = "Spearman rank correlation produced non-finite output."
        except Exception as exc:
            spearman_note = f"Spearman rank correlation failed: {exc}"
    else:
        spearman_note = "Spearman rank correlation undefined on constant inputs (zero variance in true or predicted values)."

    return {
        "n_eval": len(y_t),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if r2 is not None else None,
        "r2_note": r2_note,
        "spearman_rho": round(spearman_rho, 4) if spearman_rho is not None else None,
        "spearman_pvalue": round(spearman_p, 4) if spearman_p is not None else None,
        "spearman_note": spearman_note,
    }


def compute_hsed_metrics(
    y_true_scores: np.ndarray | list[float],
    y_pred_scores: Optional[np.ndarray | list[float]] = None,
    train_discrepancies: Optional[np.ndarray | list[float]] = None,
    percentiles: tuple[int, ...] = (70, 75, 80),
) -> dict[str, Any]:
    """Compute preliminary High Symptom-Expression Discrepancy (HSED) metrics.
    
    Discrepancy: D_i = PHQ8_i - predicted_PHQ8_i.
    Positive D_i indicates symptom score is higher than model prediction.
    Restricted to participants with PHQ-8 >= 10.
    
    If regression predictions are not provided, returns a documented pending status.
    """
    if y_pred_scores is None or train_discrepancies is None:
        return {
            "status": "PENDING",
            "reason": (
                "HSED requires out-of-fold PHQ-8 regression predictions and training-derived "
                "discrepancy thresholds, which are not produced by binary classification baselines."
            ),
        }

    y_t = np.asarray(y_true_scores, dtype=float)
    y_p = np.asarray(y_pred_scores, dtype=float)
    d_train = np.asarray(train_discrepancies, dtype=float)

    # Restrict to participants with PHQ-8 >= 10
    depressed_mask = y_t >= 10.0
    if not depressed_mask.any():
        return {
            "status": "NO_DEPRESSED_PARTICIPANTS",
            "n_depressed": 0,
        }

    y_t_sub = y_t[depressed_mask]
    y_p_sub = y_p[depressed_mask]
    eval_discrepancies = y_t_sub - y_p_sub

    threshold_results: dict[str, Any] = {}
    for p in percentiles:
        tau = float(np.percentile(d_train, p))
        is_hsed = eval_discrepancies >= tau
        threshold_results[f"p{p}"] = {
            "threshold_tau": round(tau, 4),
            "hsed_count": int(is_hsed.sum()),
            "total_depressed": int(len(eval_discrepancies)),
            "hsed_proportion": round(float(is_hsed.mean()), 4),
        }

    return {
        "status": "COMPLETED",
        "n_depressed_evaluated": int(len(y_t_sub)),
        "mean_discrepancy": round(float(np.mean(eval_discrepancies)), 4),
        "std_discrepancy": round(float(np.std(eval_discrepancies)), 4),
        "threshold_evaluations": threshold_results,
    }
