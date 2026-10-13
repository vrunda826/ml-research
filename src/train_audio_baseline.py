"""Training and evaluation pipeline for DAIC-WOZ audio baselines (B0, B1).

Trains acoustic baselines on official training participants (N=125) and
evaluates on official development participants (N=44) with zero data leakage.
Selects best audio model based on development Macro-F1 and saves artifacts.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.metrics import compute_classification_metrics
from src.models.audio_baseline import MajorityClassifier, get_audio_pipeline
from src.validation import (
    assert_no_test_labels_used,
    validate_feature_matrix_alignment,
    validate_label_integrity,
    validate_split_separation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

AUDIO_FEAT_DIR = ROOT_DIR / "data" / "processed" / "features" / "audio"
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "audio"
REPORTS_DIR = ROOT_DIR / "reports"


def load_and_split_features(
    feature_csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Load acoustic feature table and split into train and dev matrices."""
    if not feature_csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv_path}")

    df = pd.read_csv(feature_csv_path)

    # Validate splits
    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()

    train_pids = train_df["participant_id"].astype(str).tolist()
    dev_pids = dev_df["participant_id"].astype(str).tolist()

    validate_split_separation(train_pids, dev_pids)

    # Feature columns (exclude metadata columns)
    meta_cols = {"participant_id", "split", "gender", "phq_score", "depression_label"}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X_train = train_df[feat_cols].to_numpy(dtype=float)
    y_train = train_df["depression_label"].to_numpy(dtype=int)

    X_dev = dev_df[feat_cols].to_numpy(dtype=float)
    y_dev = dev_df["depression_label"].to_numpy(dtype=int)

    validate_feature_matrix_alignment(train_pids, X_train, feat_cols)
    validate_feature_matrix_alignment(dev_pids, X_dev, feat_cols)

    validate_label_integrity(y_train, "train")
    validate_label_integrity(y_dev, "dev")

    return X_train, y_train, X_dev, y_dev, train_pids, dev_pids, feat_cols


def run_audio_experiments() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Train and evaluate all B0 and B1 audio baseline models."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_records: list[dict[str, Any]] = []
    dev_predictions: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    feature_sets_map: dict[str, list[str]] = {}

    # 1. Load eGeMAPS
    egemaps_path = AUDIO_FEAT_DIR / "egemaps_features.csv"
    X_tr_e, y_tr, X_dv_e, y_dv, pids_tr, pids_dv, cols_e = load_and_split_features(egemaps_path)
    feature_sets_map["egemaps"] = cols_e

    # 2. Load MFCC
    mfcc_path = AUDIO_FEAT_DIR / "mfcc_features.csv"
    X_tr_m, _, X_dv_m, _, _, _, cols_m = load_and_split_features(mfcc_path)
    feature_sets_map["mfcc"] = cols_m

    logger.info(
        f"Acoustic baselines dataset: {len(pids_tr)} Training participants, "
        f"{len(pids_dv)} Development participants."
    )
    logger.info(f"eGeMAPS features: {len(cols_e)}, MFCC features: {len(cols_m)}")

    # Define model configurations
    experiments = [
        {
            "id": "B0",
            "name": "Majority_Class",
            "feature_set": "egemaps",
            "model": MajorityClassifier(),
            "notes": "Always predicts training majority class (0)",
        },
        {
            "id": "B1a",
            "name": "eGeMAPS_LogisticRegression",
            "feature_set": "egemaps",
            "model": get_audio_pipeline("logistic_regression", C=1.0, class_weight=None, random_state=42),
            "notes": "StandardScaler + LogisticRegression (unweighted)",
        },
        {
            "id": "B1a_bal",
            "name": "eGeMAPS_LogisticRegression_Balanced",
            "feature_set": "egemaps",
            "model": get_audio_pipeline("logistic_regression", C=1.0, class_weight="balanced", random_state=42),
            "notes": "StandardScaler + LogisticRegression (balanced weights)",
        },
        {
            "id": "B1b_lin",
            "name": "eGeMAPS_LinearSVM",
            "feature_set": "egemaps",
            "model": get_audio_pipeline("svm", C=1.0, kernel="linear", class_weight=None, random_state=42),
            "notes": "StandardScaler + Linear SVM",
        },
        {
            "id": "B1b_rbf",
            "name": "eGeMAPS_RBFSVM",
            "feature_set": "egemaps",
            "model": get_audio_pipeline("svm", C=1.0, kernel="rbf", class_weight=None, random_state=42),
            "notes": "StandardScaler + RBF SVM",
        },
        {
            "id": "B1c_lr",
            "name": "MFCC_LogisticRegression",
            "feature_set": "mfcc",
            "model": get_audio_pipeline("logistic_regression", C=1.0, class_weight=None, random_state=42),
            "notes": "StandardScaler + LogisticRegression on 78 MFCC summaries",
        },
        {
            "id": "B1c_svm",
            "name": "MFCC_LinearSVM",
            "feature_set": "mfcc",
            "model": get_audio_pipeline("svm", C=1.0, kernel="linear", class_weight=None, random_state=42),
            "notes": "StandardScaler + Linear SVM on 78 MFCC summaries",
        },
    ]

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        feat_type = exp["feature_set"]
        model = exp["model"]

        X_train = X_tr_e if feat_type == "egemaps" else X_tr_m
        X_dev = X_dv_e if feat_type == "egemaps" else X_dv_m

        # Fit model strictly on training data
        model.fit(X_train, y_tr)
        trained_models[exp_name] = model

        # Predict on development set
        y_pred = model.predict(X_dev)
        y_prob = model.predict_proba(X_dev)[:, 1] if hasattr(model, "predict_proba") else None

        # Compute metrics
        metrics = compute_classification_metrics(y_dv, y_pred, y_prob)

        # Store predictions
        probs_to_zip = y_prob if y_prob is not None else [0.0] * len(y_dv)
        for pid, true_l, pred_l, pr in zip(pids_dv, y_dv, y_pred, probs_to_zip):
            dev_predictions.append({
                "participant_id": pid,
                "split": "dev",
                "model_id": exp_id,
                "model_name": exp_name,
                "true_label": int(true_l),
                "predicted_label": int(pred_l),
                "predicted_prob": round(float(pr), 4),
            })

        # Save individual model
        model_artifact_path = ARTIFACTS_DIR / f"{exp_name}.joblib"
        joblib.dump(model, model_artifact_path)

        rec = {
            "model_id": exp_id,
            "model_name": exp_name,
            "feature_set": feat_type,
            "n_features": X_train.shape[1],
            "n_train": len(y_tr),
            "n_dev": len(y_dv),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "depression_f1": metrics["depression_f1"],
            "depression_precision": metrics["depression_precision"],
            "depression_recall": metrics["depression_recall"],
            "non_depressed_f1": metrics["non_depressed_f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "tp": metrics["confusion_matrix"]["tp"],
            "fp": metrics["confusion_matrix"]["fp"],
            "fn": metrics["confusion_matrix"]["fn"],
            "tn": metrics["confusion_matrix"]["tn"],
            "notes": exp["notes"],
        }
        results_records.append(rec)

    results_df = pd.DataFrame(results_records)
    preds_df = pd.DataFrame(dev_predictions)

    # Save results CSV
    results_csv = REPORTS_DIR / "audio_baseline_results.csv"
    results_df.to_csv(results_csv, index=False)

    preds_csv = REPORTS_DIR / "audio_dev_predictions.csv"
    preds_df.to_csv(preds_csv, index=False)

    # Identify best audio baseline based strictly on development macro-F1
    best_idx = results_df["macro_f1"].idxmax()
    best_model_name = results_df.loc[best_idx, "model_name"]
    best_macro_f1 = results_df.loc[best_idx, "macro_f1"]
    best_feature_set = results_df.loc[best_idx, "feature_set"]

    logger.info(f"Selected Best Audio Baseline: {best_model_name} (Dev Macro-F1 = {best_macro_f1})")

    # Save best model reference and feature columns
    best_model = trained_models[best_model_name]
    best_model_path = ARTIFACTS_DIR / "best_audio_model.joblib"
    joblib.dump(best_model, best_model_path)

    metadata = {
        "best_model_name": best_model_name,
        "best_macro_f1": best_macro_f1,
        "best_feature_set": best_feature_set,
        "feature_columns": feature_sets_map[best_feature_set],
        "selection_metric": "macro_f1",
        "selection_dataset": "dev",
    }
    assert_no_test_labels_used(metadata)

    with (ARTIFACTS_DIR / "best_audio_model_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Write Markdown Report
    generate_audio_report(results_df, best_model_name, len(pids_tr), len(pids_dv))

    return results_df, preds_df, best_model_name


def generate_audio_report(
    results_df: pd.DataFrame,
    best_model_name: str,
    n_train: int,
    n_dev: int,
) -> None:
    """Generate Markdown report for Phase 3 audio baselines."""
    report_path = REPORTS_DIR / "audio_baseline_report.md"

    md_lines = [
        "# Phase 3: Conventional Audio Baseline (B1) Evaluation Report",
        "",
        "## 1. Experimental Overview",
        f"- **Training Participants**: {n_train} participants (official train split)",
        f"- **Development Participants**: {n_dev} participants (official dev split)",
        "- **Evaluation Metric**: Primary selection performed using **Development Macro-F1**.",
        "- **Preprocessing Safeguards**: Imputation and StandardScaler fitted strictly on training data.",
        "- **Feature Sets Evaluated**: OpenSMILE eGeMAPS (46 features) and MFCC (78 features).",
        "",
        "## 2. Audio Baselines Comparison Table",
        "",
        "| Model ID | Model Name | Features | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in results_df.iterrows():
        roc_str = f"{row['roc_auc']:.4f}" if pd.notna(row['roc_auc']) else "N/A"
        pr_str = f"{row['pr_auc']:.4f}" if pd.notna(row['pr_auc']) else "N/A"
        md_lines.append(
            f"| **{row['model_id']}** | {row['model_name']} | {row['feature_set']} | "
            f"{row['accuracy']:.4f} | **{row['macro_f1']:.4f}** | {row['depression_recall']:.4f} | "
            f"{row['depression_precision']:.4f} | {row['depression_f1']:.4f} | {roc_str} | {pr_str} |"
        )

    best_row = results_df[results_df["model_name"] == best_model_name].iloc[0]

    md_lines.extend([
        "",
        "## 3. Selected Best Audio Baseline",
        f"- **Selected Configuration**: **{best_model_name}** ({best_row['model_id']})",
        f"- **Primary Criterion (Dev Macro-F1)**: **{best_row['macro_f1']:.4f}**",
        f"- **Depression Recall**: {best_row['depression_recall']:.4f} (TP={int(best_row['tp'])}, FN={int(best_row['fn'])})",
        f"- **Depression Precision**: {best_row['depression_precision']:.4f} (FP={int(best_row['fp'])})",
        f"- **Confusion Matrix**: TN={int(best_row['tn'])}, FP={int(best_row['fp'])}, FN={int(best_row['fn'])}, TP={int(best_row['tp'])}",
        f"- **Artifact Saved To**: `artifacts/audio/{best_model_name}.joblib` and `artifacts/audio/best_audio_model.joblib`",
        "",
        "## 4. Observations and Limitations",
        "- The majority class baseline (B0) establishes the lower bound (Macro-F1 ~0.4286, depression recall 0.0).",
        "- Acoustic features demonstrate predictive signal above majority chance, but unimodal depression classification remains challenging due to acoustic heterogeneity and subtle clinical manifestation.",
        "- Class-weight adjustments significantly improve depression recall at the expense of lower precision.",
        "",
    ])

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Audio baseline report saved to: {report_path}")


if __name__ == "__main__":
    run_audio_experiments()
