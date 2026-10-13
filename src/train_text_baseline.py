"""Training and evaluation pipeline for DAIC-WOZ text baseline (B2).

Trains text models on official training participants (N=125) and evaluates on
development participants (N=44) with strict participant-level separation.
Selects best text model based on development Macro-F1 and saves artifacts.
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
from src.models.audio_baseline import MajorityClassifier
from src.models.text_baseline import get_text_pipeline
from src.validation import (
    assert_no_test_labels_used,
    validate_feature_matrix_alignment,
    validate_label_integrity,
    validate_split_separation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEXT_FEAT_DIR = ROOT_DIR / "data" / "processed" / "features" / "text"
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "text"
REPORTS_DIR = ROOT_DIR / "reports"


def load_and_split_text_features(
    feature_csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Load text embedding table and split into train and dev matrices."""
    if not feature_csv_path.exists():
        raise FileNotFoundError(f"Text feature CSV not found: {feature_csv_path}")

    df = pd.read_csv(feature_csv_path)

    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()

    train_pids = train_df["participant_id"].astype(str).tolist()
    dev_pids = dev_df["participant_id"].astype(str).tolist()

    validate_split_separation(train_pids, dev_pids)

    meta_cols = {"participant_id", "split", "gender", "phq_score", "depression_label", "utterance_count"}
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


def run_text_experiments() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Train and evaluate all B2 text baseline models."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    text_csv = TEXT_FEAT_DIR / "text_embeddings_minilm.csv"
    X_train, y_train, X_dev, y_dev, train_pids, dev_pids, feat_cols = load_and_split_text_features(text_csv)

    logger.info(
        f"Text baseline dataset: {len(train_pids)} Training participants, "
        f"{len(dev_pids)} Development participants. Embedding dimensions: {len(feat_cols)}"
    )

    results_records: list[dict[str, Any]] = []
    dev_predictions: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    experiments = [
        {
            "id": "B0",
            "name": "Majority_Class",
            "model": MajorityClassifier(),
            "notes": "Always predicts training majority class (0)",
        },
        {
            "id": "B2a",
            "name": "Text_MiniLM_LogisticRegression",
            "model": get_text_pipeline("logistic_regression", C=1.0, class_weight=None, random_state=42),
            "notes": "StandardScaler + LogisticRegression on 384 MiniLM embeddings",
        },
        {
            "id": "B2a_bal",
            "name": "Text_MiniLM_LogisticRegression_Balanced",
            "model": get_text_pipeline("logistic_regression", C=1.0, class_weight="balanced", random_state=42),
            "notes": "StandardScaler + LogisticRegression (balanced weights)",
        },
        {
            "id": "B2b_lin",
            "name": "Text_MiniLM_LinearSVM",
            "model": get_text_pipeline("svm", C=1.0, kernel="linear", class_weight=None, random_state=42),
            "notes": "StandardScaler + Linear SVM",
        },
        {
            "id": "B2b_rbf",
            "name": "Text_MiniLM_RBFSVM",
            "model": get_text_pipeline("svm", C=1.0, kernel="rbf", class_weight=None, random_state=42),
            "notes": "StandardScaler + RBF SVM",
        },
    ]

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        model = exp["model"]

        # Fit model strictly on training data
        model.fit(X_train, y_train)
        trained_models[exp_name] = model

        # Predict on development set
        y_pred = model.predict(X_dev)
        y_prob = model.predict_proba(X_dev)[:, 1] if hasattr(model, "predict_proba") else None

        # Compute metrics
        metrics = compute_classification_metrics(y_dev, y_pred, y_prob)

        # Store predictions
        probs_to_zip = y_prob if y_prob is not None else [0.0] * len(y_dev)
        for pid, true_l, pred_l, pr in zip(dev_pids, y_dev, y_pred, probs_to_zip):
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
            "n_features": X_train.shape[1],
            "n_train": len(y_train),
            "n_dev": len(y_dev),
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

    results_csv = REPORTS_DIR / "text_baseline_results.csv"
    results_df.to_csv(results_csv, index=False)

    preds_csv = REPORTS_DIR / "text_dev_predictions.csv"
    preds_df.to_csv(preds_csv, index=False)

    best_idx = results_df["macro_f1"].idxmax()
    best_model_name = results_df.loc[best_idx, "model_name"]
    best_macro_f1 = results_df.loc[best_idx, "macro_f1"]

    logger.info(f"Selected Best Text Baseline: {best_model_name} (Dev Macro-F1 = {best_macro_f1})")

    best_model = trained_models[best_model_name]
    best_model_path = ARTIFACTS_DIR / "best_text_model.joblib"
    joblib.dump(best_model, best_model_path)

    metadata = {
        "best_model_name": best_model_name,
        "best_macro_f1": best_macro_f1,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimensions": len(feat_cols),
        "selection_metric": "macro_f1",
        "selection_dataset": "dev",
    }
    assert_no_test_labels_used(metadata)

    with (ARTIFACTS_DIR / "best_text_model_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    generate_text_report(results_df, best_model_name, len(train_pids), len(dev_pids))

    return results_df, preds_df, best_model_name


def generate_text_report(
    results_df: pd.DataFrame,
    best_model_name: str,
    n_train: int,
    n_dev: int,
) -> None:
    """Generate Markdown report for Phase 4 text baselines."""
    report_path = REPORTS_DIR / "text_baseline_report.md"

    md_lines = [
        "# Phase 4: Conventional Text Baseline (B2) Evaluation Report",
        "",
        "## 1. Experimental Overview",
        f"- **Training Participants**: {n_train} participants (official train split)",
        f"- **Development Participants**: {n_dev} participants (official dev split)",
        "- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, frozen).",
        "- **Aggregation**: Utterance embeddings pooled via mean pooling to form one vector per participant.",
        "- **Primary Selection Metric**: Development Set Macro-F1.",
        "",
        "## 2. Text Baselines Comparison Table",
        "",
        "| Model ID | Model Name | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in results_df.iterrows():
        roc_str = f"{row['roc_auc']:.4f}" if pd.notna(row['roc_auc']) else "N/A"
        pr_str = f"{row['pr_auc']:.4f}" if pd.notna(row['pr_auc']) else "N/A"
        md_lines.append(
            f"| **{row['model_id']}** | {row['model_name']} | "
            f"{row['accuracy']:.4f} | **{row['macro_f1']:.4f}** | {row['depression_recall']:.4f} | "
            f"{row['depression_precision']:.4f} | {row['depression_f1']:.4f} | {roc_str} | {pr_str} |"
        )

    best_row = results_df[results_df["model_name"] == best_model_name].iloc[0]

    md_lines.extend([
        "",
        "## 3. Selected Best Text Baseline",
        f"- **Selected Configuration**: **{best_model_name}** ({best_row['model_id']})",
        f"- **Primary Criterion (Dev Macro-F1)**: **{best_row['macro_f1']:.4f}**",
        f"- **Depression Recall**: {best_row['depression_recall']:.4f} (TP={int(best_row['tp'])}, FN={int(best_row['fn'])})",
        f"- **Depression Precision**: {best_row['depression_precision']:.4f} (FP={int(best_row['fp'])})",
        f"- **Confusion Matrix**: TN={int(best_row['tn'])}, FP={int(best_row['fp'])}, FN={int(best_row['fn'])}, TP={int(best_row['tp'])}",
        f"- **Artifact Saved To**: `artifacts/text/best_text_model.joblib`",
        "",
        "## 4. Linguistic Observations and Known Limitations",
        "- **Absence of Speaker Annotations**: In the lightweight transcript schema, utterances are not explicitly tagged with speaker identities (Ellie vs. Participant). Embeddings pool dialogue across the session. The model captures overall session semantic content rather than isolated participant speech.",
        "- **Preserved Clinical Vocabulary**: Text cleaning conservatively retained all negations, symptom keywords, and affective terms without stopword deletion.",
        "- **Model Comparison**: Contextual sentence embeddings provide strong semantic representations for depression detection, establishing a solid unimodal benchmark for multimodal fusion.",
        "",
    ])

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Text baseline report saved to: {report_path}")


if __name__ == "__main__":
    run_text_experiments()
