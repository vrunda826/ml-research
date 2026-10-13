"""Phase 6 Engine: Continuous PHQ-8 Regression, OOF Discrepancy, and HSED Subgroup Analysis.

Implements Phase 6A (Regression Baselines), Phase 6B (Out-of-Fold Discrepancy Modeling),
and Phase 6C (HSED Candidate Subgroup Derivation and Stability Analysis).
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
from sklearn.model_selection import KFold

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.build_manifest import normalize_participant_id
from src.metrics import compute_regression_metrics
from src.models.regression_baseline import get_regression_pipeline
from src.validation import (
    validate_feature_matrix_alignment,
    validate_no_target_in_features,
    validate_split_separation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT_DIR / "configs" / "experiment_config.json"
REPORTS_PHASE6_DIR = ROOT_DIR / "reports" / "phase6"
ARTIFACTS_PHASE6_DIR = ROOT_DIR / "artifacts" / "phase6" / "regression"

META_COLUMNS = {
    "participant_id",
    "split",
    "gender",
    "phq_score",
    "depression_label",
    "utterance_count",
}


def load_config() -> dict[str, Any]:
    """Load experimental configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_modality_data(
    modality_key: str,
    config: Optional[dict[str, Any]] = None,
    root_dir: Path = ROOT_DIR,
) -> dict[str, Any]:
    """Load, align, and validate feature matrices and continuous targets for a modality.
    
    Returns
    -------
    dict
        Dictionary containing X_train, y_train, train_pids, X_dev, y_dev, dev_pids,
        feature_names, and metadata.
    """
    if config is None:
        config = load_config()

    phase6_cfg = config.get("phase6", {})
    modalities_cfg = phase6_cfg.get("modalities", {})

    if modality_key not in modalities_cfg:
        raise KeyError(f"Modality '{modality_key}' not found in configuration: {list(modalities_cfg.keys())}")

    mod_info = modalities_cfg[modality_key]
    csv_path = root_dir / mod_info["feature_file"]

    if not csv_path.exists():
        raise FileNotFoundError(f"Feature file missing for {modality_key}: {csv_path}")

    df = pd.read_csv(csv_path)
    df["participant_id"] = [normalize_participant_id(pid) for pid in df["participant_id"]]

    if df["participant_id"].duplicated().any():
        raise ValueError(f"Duplicate participant IDs discovered in {csv_path}")

    # Separate train and dev splits
    train_df = df[df["split"] == "train"].copy().sort_values("participant_id").reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].copy().sort_values("participant_id").reset_index(drop=True)

    train_pids = train_df["participant_id"].tolist()
    dev_pids = dev_df["participant_id"].tolist()

    # Enforce strict participant disjointness
    validate_split_separation(train_pids, dev_pids)

    # Extract feature columns and verify zero target leakage
    feat_cols = [c for c in df.columns if c not in META_COLUMNS]
    validate_no_target_in_features(feat_cols)

    X_train = train_df[feat_cols].to_numpy(dtype=float)
    y_train = train_df["phq_score"].to_numpy(dtype=float)

    X_dev = dev_df[feat_cols].to_numpy(dtype=float)
    y_dev = dev_df["phq_score"].to_numpy(dtype=float)

    # Validate matrix dimensions and finiteness
    validate_feature_matrix_alignment(train_pids, X_train, feat_cols)
    validate_feature_matrix_alignment(dev_pids, X_dev, feat_cols)

    return {
        "modality_key": modality_key,
        "modality_name": mod_info["name"],
        "modality_type": mod_info["type"],
        "feature_names": feat_cols,
        "n_features": len(feat_cols),
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": train_pids,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "dev_pids": dev_pids,
        "train_df": train_df,
        "dev_df": dev_df,
    }


# ==============================================================================
# PHASE 6A: CONTINUOUS PHQ-8 REGRESSION BASELINES
# ==============================================================================

def run_phase6a_regression_baselines(
    config: Optional[dict[str, Any]] = None,
    output_dir: Path = REPORTS_PHASE6_DIR,
    artifacts_dir: Path = ARTIFACTS_PHASE6_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute Phase 6A: Train and evaluate continuous PHQ-8 regression baselines.
    
    Trains DummyRegressor and Ridge regression (across alpha grid and default alpha),
    computes evaluation metrics on the common development cohort (N=44), saves predictions,
    and persists fitted model artifacts.
    """
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 6A: CONTINUOUS PHQ-8 REGRESSION BASELINES")
    logger.info("=" * 65)

    if config is None:
        config = load_config()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    phase6_cfg = config.get("phase6", {})
    modalities_cfg = phase6_cfg.get("modalities", {})
    alpha_grid = phase6_cfg.get("alpha_grid", [0.1, 1.0, 10.0, 100.0])
    default_alpha = phase6_cfg.get("default_alpha", 100.0)
    seed = phase6_cfg.get("random_seed", 42)

    primary_metrics_records: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []
    dev_predictions_records: list[dict[str, Any]] = []

    # 1. Evaluate DummyRegressor Baseline (once per evaluation split)
    first_mod_key = list(modalities_cfg.keys())[0]
    base_data = load_modality_data(first_mod_key, config)
    dummy_pipe = get_regression_pipeline("dummy")
    dummy_pipe.fit(base_data["X_train"], base_data["y_train"])
    dummy_dev_pred = dummy_pipe.predict(base_data["X_dev"])
    dummy_metrics = compute_regression_metrics(base_data["y_dev"], dummy_dev_pred)

    dummy_rec = {
        "model_id": "B0_reg",
        "modality": "none",
        "model_name": "Dummy_Mean_Regressor",
        "n_features": 0,
        "n_train": len(base_data["train_pids"]),
        "n_dev": len(base_data["dev_pids"]),
        "mae": dummy_metrics["mae"],
        "rmse": dummy_metrics["rmse"],
        "r2": dummy_metrics["r2"],
        "spearman_rho": dummy_metrics["spearman_rho"],
        "spearman_pvalue": dummy_metrics["spearman_pvalue"],
        "spearman_note": dummy_metrics["spearman_note"],
        "notes": "Always predicts training PHQ-8 mean (6.472)",
        "baseline_tier": "B0_Reg (Dummy)",
    }
    primary_metrics_records.append(dummy_rec)
    comparison_records.append(dummy_rec)

    for pid, y_true, y_pred in zip(base_data["dev_pids"], base_data["y_dev"], dummy_dev_pred):
        dev_predictions_records.append({
            "participant_id": pid,
            "split": "dev",
            "modality": "none",
            "model_name": "Dummy_Mean_Regressor",
            "phq8_true": float(y_true),
            "phq8_pred": float(y_pred),
            "absolute_error": float(abs(y_true - y_pred)),
            "squared_error": float((y_true - y_pred) ** 2),
        })

    # Save Dummy model artifact
    joblib.dump(dummy_pipe, artifacts_dir / "dummy_mean_regressor.joblib")

    # 2. Evaluate Ridge models across all 5 modalities
    for mod_idx, mod_key in enumerate(modalities_cfg.keys(), start=1):
        mod_data = load_modality_data(mod_key, config)
        mod_name = mod_data["modality_name"]
        n_feats = mod_data["n_features"]

        logger.info(f"Training Regression Baselines for {mod_name} ({n_feats} features)...")

        # Evaluate alpha grid for model comparison
        for alpha in alpha_grid:
            pipe = get_regression_pipeline("ridge", alpha=alpha, random_state=seed)
            pipe.fit(mod_data["X_train"], mod_data["y_train"])
            pred = pipe.predict(mod_data["X_dev"])
            met = compute_regression_metrics(mod_data["y_dev"], pred)

            rec = {
                "model_id": f"Ridge_{mod_key}_a{alpha}",
                "modality": mod_key,
                "model_name": f"Ridge_{mod_name}_a{alpha}",
                "alpha": alpha,
                "n_features": n_feats,
                "n_train": len(mod_data["train_pids"]),
                "n_dev": len(mod_data["dev_pids"]),
                "mae": met["mae"],
                "rmse": met["rmse"],
                "r2": met["r2"],
                "spearman_rho": met["spearman_rho"],
                "spearman_pvalue": met["spearman_pvalue"],
                "spearman_note": met["spearman_note"],
                "is_default": (alpha == default_alpha),
                "baseline_tier": f"Ridge_a{alpha}",
            }
            comparison_records.append(rec)

            # Default alpha model is selected as the primary baseline for Phase 6
            if alpha == default_alpha:
                prim_rec = {
                    "model_id": f"B{mod_idx}_reg",
                    "modality": mod_key,
                    "model_name": f"Ridge_{mod_name}",
                    "alpha": alpha,
                    "n_features": n_feats,
                    "n_train": len(mod_data["train_pids"]),
                    "n_dev": len(mod_data["dev_pids"]),
                    "mae": met["mae"],
                    "rmse": met["rmse"],
                    "r2": met["r2"],
                    "spearman_rho": met["spearman_rho"],
                    "spearman_pvalue": met["spearman_pvalue"],
                    "spearman_note": met["spearman_note"],
                    "notes": f"StandardScaler + Ridge(alpha={alpha}) on {n_feats} features",
                    "baseline_tier": f"B{mod_idx}_Reg ({mod_name})",
                }
                primary_metrics_records.append(prim_rec)

                # Persist model artifact
                artifact_path = artifacts_dir / f"{mod_key}_ridge_a{int(alpha)}.joblib"
                joblib.dump(pipe, artifact_path)

                # Record dev predictions
                for pid, y_t, y_p in zip(mod_data["dev_pids"], mod_data["y_dev"], pred):
                    dev_predictions_records.append({
                        "participant_id": pid,
                        "split": "dev",
                        "modality": mod_key,
                        "model_name": f"Ridge_{mod_name}",
                        "phq8_true": float(y_t),
                        "phq8_pred": float(y_p),
                        "absolute_error": float(abs(y_t - y_p)),
                        "squared_error": float((y_t - y_p) ** 2),
                    })

    # Save output artifacts
    primary_df = pd.DataFrame(primary_metrics_records)
    comparison_df = pd.DataFrame(comparison_records)
    dev_preds_df = pd.DataFrame(dev_predictions_records)

    primary_df.to_csv(output_dir / "regression_metrics.csv", index=False)
    comparison_df.to_csv(output_dir / "regression_model_comparison.csv", index=False)
    dev_preds_df.to_csv(output_dir / "dev_regression_predictions.csv", index=False)

    logger.info(f"Saved primary regression metrics to: {output_dir / 'regression_metrics.csv'}")
    logger.info(f"Saved model comparison table to: {output_dir / 'regression_model_comparison.csv'}")
    logger.info(f"Saved dev predictions ({len(dev_preds_df)} rows) to: {output_dir / 'dev_regression_predictions.csv'}")

    return primary_df, comparison_df, dev_preds_df


# ==============================================================================
# PHASE 6B: OUT-OF-FOLD (OOF) TRAINING PREDICTIONS
# ==============================================================================

def run_phase6b_oof_predictions(
    config: Optional[dict[str, Any]] = None,
    output_dir: Path = REPORTS_PHASE6_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute Phase 6B: Generate valid, participant-level Out-of-Fold training predictions.
    
    Uses 5-fold cross-validation on 125 training participants with strictly fold-fitted
    imputation and scaling. Calculates residuals D_i = y_i - y_hat_i.
    """
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 6B: OUT-OF-FOLD (OOF) TRAINING PREDICTIONS")
    logger.info("=" * 65)

    if config is None:
        config = load_config()

    output_dir.mkdir(parents=True, exist_ok=True)

    phase6_cfg = config.get("phase6", {})
    modalities_cfg = phase6_cfg.get("modalities", {})
    n_folds = phase6_cfg.get("n_folds", 5)
    seed = phase6_cfg.get("random_seed", 42)
    default_alpha = phase6_cfg.get("default_alpha", 100.0)

    # 1. Deterministic fold partitioning across training participants
    first_mod_key = list(modalities_cfg.keys())[0]
    base_data = load_modality_data(first_mod_key, config)
    train_pids = np.array(base_data["train_pids"])
    n_train = len(train_pids)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits: list[tuple[np.ndarray, np.ndarray]] = list(kf.split(train_pids))

    # Save fold assignments table
    fold_assignment_records: list[dict[str, Any]] = []
    pid_to_fold: dict[str, int] = {}

    for fold_id, (_, val_idx) in enumerate(fold_splits, start=1):
        for idx in val_idx:
            pid = str(train_pids[idx])
            pid_to_fold[pid] = fold_id
            fold_assignment_records.append({
                "participant_id": pid,
                "split": "train",
                "fold_id": fold_id,
            })

    fold_df = pd.DataFrame(fold_assignment_records).sort_values("participant_id").reset_index(drop=True)
    fold_df.to_csv(output_dir / "oof_fold_assignments.csv", index=False)
    logger.info(f"Saved OOF fold assignments ({len(fold_df)} participants) to: {output_dir / 'oof_fold_assignments.csv'}")

    oof_predictions_records: list[dict[str, Any]] = []
    oof_metrics_summary: dict[str, Any] = {}

    # 2. Generate OOF predictions for Dummy baseline
    dummy_oof_pred = np.zeros(n_train, dtype=float)
    dummy_fold_id = np.zeros(n_train, dtype=int)
    for fold_id, (tr_idx, val_idx) in enumerate(fold_splits, start=1):
        pipe_dummy = get_regression_pipeline("dummy")
        pipe_dummy.fit(base_data["X_train"][tr_idx], base_data["y_train"][tr_idx])
        val_pred = pipe_dummy.predict(base_data["X_train"][val_idx])
        dummy_oof_pred[val_idx] = val_pred
        dummy_fold_id[val_idx] = fold_id

    disc_dummy = base_data["y_train"] - dummy_oof_pred
    for i, pid in enumerate(train_pids):
        oof_predictions_records.append({
            "participant_id": pid,
            "split": "train",
            "fold_id": int(dummy_fold_id[i]),
            "modality": "none",
            "model_name": "Dummy_Mean_Regressor",
            "phq8_true": float(base_data["y_train"][i]),
            "phq8_pred": float(dummy_oof_pred[i]),
            "discrepancy": float(disc_dummy[i]),
            "absolute_error": float(abs(disc_dummy[i])),
            "squared_error": float(disc_dummy[i] ** 2),
        })

    met_dummy = compute_regression_metrics(base_data["y_train"], dummy_oof_pred)
    oof_metrics_summary["dummy"] = {
        "modality_name": "Dummy_Mean_Regressor",
        "n_train": n_train,
        "mae": met_dummy["mae"],
        "rmse": met_dummy["rmse"],
        "r2": met_dummy["r2"],
        "spearman_rho": met_dummy["spearman_rho"],
        "spearman_pvalue": met_dummy["spearman_pvalue"],
        "mean_discrepancy": round(float(np.mean(disc_dummy)), 4),
        "std_discrepancy": round(float(np.std(disc_dummy)), 4),
    }

    # 3. Generate OOF predictions for each feature modality
    for mod_key in modalities_cfg.keys():
        mod_data = load_modality_data(mod_key, config)
        mod_name = mod_data["modality_name"]
        X_tr = mod_data["X_train"]
        y_tr = mod_data["y_train"]

        oof_pred_arr = np.zeros(n_train, dtype=float)
        fold_id_arr = np.zeros(n_train, dtype=int)

        for fold_id, (tr_idx, val_idx) in enumerate(fold_splits, start=1):
            # Verify disjoint fold separation
            tr_pids_set = set(train_pids[tr_idx])
            val_pids_set = set(train_pids[val_idx])
            if tr_pids_set.intersection(val_pids_set):
                raise ValueError(f"Cross-fold contamination in fold {fold_id}!")

            # Fit pipeline strictly on training fold
            pipe = get_regression_pipeline("ridge", alpha=default_alpha, random_state=seed)
            pipe.fit(X_tr[tr_idx], y_tr[tr_idx])

            # Predict on held-out validation fold
            val_pred = pipe.predict(X_tr[val_idx])
            oof_pred_arr[val_idx] = val_pred
            fold_id_arr[val_idx] = fold_id

        # Compute unrounded discrepancies and errors
        discrepancies = y_tr - oof_pred_arr
        abs_errors = np.abs(discrepancies)
        sq_errors = discrepancies ** 2

        # Record each participant's OOF prediction
        for i, pid in enumerate(train_pids):
            oof_predictions_records.append({
                "participant_id": pid,
                "split": "train",
                "fold_id": int(fold_id_arr[i]),
                "modality": mod_key,
                "model_name": f"Ridge_{mod_name}_a{int(default_alpha)}",
                "phq8_true": float(y_tr[i]),
                "phq8_pred": float(oof_pred_arr[i]),
                "discrepancy": float(discrepancies[i]),
                "absolute_error": float(abs_errors[i]),
                "squared_error": float(sq_errors[i]),
            })

        # Calculate overall OOF regression metrics
        met = compute_regression_metrics(y_tr, oof_pred_arr)
        oof_metrics_summary[mod_key] = {
            "modality_name": mod_name,
            "n_train": n_train,
            "mae": met["mae"],
            "rmse": met["rmse"],
            "r2": met["r2"],
            "spearman_rho": met["spearman_rho"],
            "spearman_pvalue": met["spearman_pvalue"],
            "mean_discrepancy": round(float(np.mean(discrepancies)), 4),
            "std_discrepancy": round(float(np.std(discrepancies)), 4),
        }

    oof_preds_df = pd.DataFrame(oof_predictions_records)
    oof_preds_df.to_csv(output_dir / "train_oof_predictions.csv", index=False)

    total_expected_models = len(modalities_cfg) + 1  # 5 modalities + 1 dummy baseline
    validation_report = {
        "n_training_participants": n_train,
        "n_folds": n_folds,
        "random_seed": seed,
        "default_alpha": default_alpha,
        "participants_per_fold": {
            f"fold_{f}": int((fold_df["fold_id"] == f).sum()) for f in range(1, n_folds + 1)
        },
        "oof_verification": {
            "all_training_participants_predicted": len(oof_preds_df["participant_id"].unique()) == n_train,
            "one_prediction_per_participant_modality": len(oof_preds_df) == (n_train * total_expected_models),
            "disjoint_folds": True,
            "zero_leakage_from_dev": True,
        },
        "oof_regression_metrics": oof_metrics_summary,
    }

    with open(output_dir / "oof_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    logger.info(f"Saved train OOF predictions ({len(oof_preds_df)} rows) to: {output_dir / 'train_oof_predictions.csv'}")
    logger.info(f"Saved OOF validation report to: {output_dir / 'oof_validation_report.json'}")

    return oof_preds_df, fold_df, validation_report


# ==============================================================================
# PHASE 6C: DERIVE AND EVALUATE HSED CANDIDATE SUBGROUP
# ==============================================================================

def run_phase6c_hsed_analysis(
    config: Optional[dict[str, Any]] = None,
    output_dir: Path = REPORTS_PHASE6_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute Phase 6C: Derive training HSED thresholds and evaluate candidate subgroups.
    
    Thresholds tau_q are estimated exclusively from training participants with PHQ-8 >= 10
    using unrounded OOF discrepancies. Frozen thresholds are applied to development predictions.
    Computes sensitivity analysis and cross-modality subgroup overlap.
    """
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 6C: HSED CANDIDATE SUBGROUP DERIVATION & ANALYSIS")
    logger.info("=" * 65)

    if config is None:
        config = load_config()

    output_dir.mkdir(parents=True, exist_ok=True)

    phase6_cfg = config.get("phase6", {})
    modalities_cfg = phase6_cfg.get("modalities", {})
    quantiles = phase6_cfg.get("hsed_quantiles", [0.70, 0.75, 0.80])
    default_alpha = phase6_cfg.get("default_alpha", 100.0)
    seed = phase6_cfg.get("random_seed", 42)
    quantile_method = phase6_cfg.get("quantile_method", "linear")

    oof_preds_path = output_dir / "train_oof_predictions.csv"
    if not oof_preds_path.exists():
        raise FileNotFoundError(f"OOF predictions file not found: {oof_preds_path}")

    oof_df = pd.read_csv(oof_preds_path)

    thresholds_records: list[dict[str, Any]] = []
    train_hsed_records: list[dict[str, Any]] = []
    dev_hsed_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []

    # Map for modality overlap calculations: (modality, quantile, split) -> set of PIDs
    selected_subgroups: dict[tuple[str, float, str], set[str]] = {}

    for mod_key in modalities_cfg.keys():
        mod_data = load_modality_data(mod_key, config)
        mod_name = mod_data["modality_name"]
        model_name = f"Ridge_{mod_name}_a{int(default_alpha)}"

        # 1. Filter training OOF predictions for this modality
        mod_oof = oof_df[oof_df["modality"] == mod_key].copy()
        depressed_train = mod_oof[mod_oof["phq8_true"] >= 10.0].copy()
        n_dep_train = len(depressed_train)

        if n_dep_train == 0:
            raise ValueError(f"No depressed training participants found for {mod_key}!")

        dep_discrepancies = depressed_train["discrepancy"].to_numpy(dtype=float)

        # 2. Fit full regression model on all training participants
        full_pipe = get_regression_pipeline("ridge", alpha=default_alpha, random_state=seed)
        full_pipe.fit(mod_data["X_train"], mod_data["y_train"])

        # 3. Generate development predictions and discrepancies
        dev_pred = full_pipe.predict(mod_data["X_dev"])
        dev_discrepancies = mod_data["y_dev"] - dev_pred

        dev_df_records = []
        for i, pid in enumerate(mod_data["dev_pids"]):
            dev_df_records.append({
                "participant_id": pid,
                "phq8_true": float(mod_data["y_dev"][i]),
                "predicted_phq8": float(dev_pred[i]),
                "discrepancy": float(dev_discrepancies[i]),
            })
        dev_pred_df = pd.DataFrame(dev_df_records)

        # 4. Derive tau_q exclusively from S_train OOF discrepancies
        for q in quantiles:
            tau_q = float(np.quantile(dep_discrepancies, q, method=quantile_method))

            # Training HSED selection: y_i >= 10 and D_i >= tau_q
            train_selected_pids = set()
            for _, row in mod_oof.iterrows():
                is_candidate = bool(row["phq8_true"] >= 10.0 and row["discrepancy"] >= tau_q)
                if is_candidate:
                    train_selected_pids.add(row["participant_id"])
                train_hsed_records.append({
                    "participant_id": row["participant_id"],
                    "split": "train",
                    "modality": mod_key,
                    "model_name": model_name,
                    "phq8_true": float(row["phq8_true"]),
                    "oof_predicted_phq8": float(row["phq8_pred"]),
                    "oof_discrepancy": float(row["discrepancy"]),
                    "quantile": float(q),
                    "threshold_value": round(tau_q, 4),
                    "hsed_candidate": int(is_candidate),
                })

            n_train_selected = len(train_selected_pids)
            train_fraction = n_train_selected / n_dep_train

            # Apply frozen threshold to development set: y_j >= 10 and D_j >= tau_q
            dev_selected_pids = set()
            for _, row in dev_pred_df.iterrows():
                is_candidate = bool(row["phq8_true"] >= 10.0 and row["discrepancy"] >= tau_q)
                if is_candidate:
                    dev_selected_pids.add(row["participant_id"])
                dev_hsed_records.append({
                    "participant_id": row["participant_id"],
                    "split": "dev",
                    "modality": mod_key,
                    "model_name": model_name,
                    "phq8_true": float(row["phq8_true"]),
                    "predicted_phq8": float(row["predicted_phq8"]),
                    "discrepancy": float(row["discrepancy"]),
                    "quantile": float(q),
                    "training_derived_threshold": round(tau_q, 4),
                    "hsed_candidate": int(is_candidate),
                })

            n_dev_dep = int((mod_data["y_dev"] >= 10.0).sum())
            n_dev_selected = len(dev_selected_pids)
            dev_fraction = n_dev_selected / n_dev_dep if n_dev_dep > 0 else 0.0

            # Store for overlap analysis
            selected_subgroups[(mod_key, q, "train")] = train_selected_pids
            selected_subgroups[(mod_key, q, "dev")] = dev_selected_pids

            thresholds_records.append({
                "modality": mod_key,
                "model_name": model_name,
                "quantile": q,
                "threshold_value": round(tau_q, 4),
                "n_train_phq8_ge_10": n_dep_train,
                "n_train_selected": n_train_selected,
                "train_selected_fraction": round(train_fraction, 4),
                "quantile_method": quantile_method,
            })

            sensitivity_records.append({
                "modality": mod_key,
                "quantile": q,
                "threshold_tau": round(tau_q, 4),
                "train_depressed_total": n_dep_train,
                "train_hsed_count": n_train_selected,
                "train_hsed_pct": round(train_fraction * 100, 2),
                "dev_depressed_total": n_dev_dep,
                "dev_hsed_count": n_dev_selected,
                "dev_hsed_pct": round(dev_fraction * 100, 2),
            })

    thresholds_df = pd.DataFrame(thresholds_records)
    train_hsed_df = pd.DataFrame(train_hsed_records)
    dev_hsed_df = pd.DataFrame(dev_hsed_records)
    sensitivity_df = pd.DataFrame(sensitivity_records)

    # 5. Compute cross-modality subgroup overlap and Jaccard similarity
    modality_keys = list(modalities_cfg.keys())
    overlap_records: list[dict[str, Any]] = []

    for split in ["train", "dev"]:
        for q in quantiles:
            for i in range(len(modality_keys)):
                for j in range(i + 1, len(modality_keys)):
                    mod_a = modality_keys[i]
                    mod_b = modality_keys[j]
                    set_a = selected_subgroups[(mod_a, q, split)]
                    set_b = selected_subgroups[(mod_b, q, split)]

                    intersection = len(set_a.intersection(set_b))
                    union = len(set_a.union(set_b))
                    jaccard = round(intersection / union, 4) if union > 0 else 0.0

                    overlap_records.append({
                        "split": split,
                        "quantile": q,
                        "modality_a": mod_a,
                        "modality_b": mod_b,
                        "count_a": len(set_a),
                        "count_b": len(set_b),
                        "intersection_count": intersection,
                        "union_count": union,
                        "jaccard_similarity": jaccard,
                    })

    overlap_df = pd.DataFrame(overlap_records)

    # 6. Save audit table containing actual selected participant IDs
    final_audit_dir = output_dir / "final_audit"
    final_audit_dir.mkdir(parents=True, exist_ok=True)

    selected_records: list[dict[str, Any]] = []
    selected_pids_dict: dict[str, Any] = {}

    for (mod_key, q, split), pids in selected_subgroups.items():
        sorted_pids = sorted([str(p) for p in pids])
        key_str = f"{split}_{mod_key}_q{int(round(q*100))}"
        selected_pids_dict[key_str] = sorted_pids
        selected_records.append({
            "split": split,
            "modality": mod_key,
            "quantile": q,
            "n_selected": len(sorted_pids),
            "selected_participant_ids": ",".join(sorted_pids),
        })

    pd.DataFrame(selected_records).to_csv(final_audit_dir / "hsed_selected_participants.csv", index=False)
    with open(final_audit_dir / "hsed_selected_participants.json", "w", encoding="utf-8") as f:
        json.dump(selected_pids_dict, f, indent=2)

    # Persist all Phase 6C artifacts
    thresholds_df.to_csv(output_dir / "hsed_thresholds.csv", index=False)
    train_hsed_df.to_csv(output_dir / "train_hsed_assignments.csv", index=False)
    dev_hsed_df.to_csv(output_dir / "dev_hsed_assignments.csv", index=False)
    sensitivity_df.to_csv(output_dir / "hsed_sensitivity_analysis.csv", index=False)
    overlap_df.to_csv(output_dir / "hsed_modality_overlap.csv", index=False)

    logger.info(f"Saved HSED thresholds to: {output_dir / 'hsed_thresholds.csv'}")
    logger.info(f"Saved train HSED assignments to: {output_dir / 'train_hsed_assignments.csv'}")
    logger.info(f"Saved dev HSED assignments to: {output_dir / 'dev_hsed_assignments.csv'}")
    logger.info(f"Saved sensitivity analysis to: {output_dir / 'hsed_sensitivity_analysis.csv'}")
    logger.info(f"Saved modality overlap analysis to: {output_dir / 'hsed_modality_overlap.csv'}")
    logger.info(f"Saved candidate participant IDs to: {final_audit_dir / 'hsed_selected_participants.csv'}")

    return thresholds_df, train_hsed_df, dev_hsed_df, sensitivity_df, overlap_df


def run_all_phase6() -> bool:
    """Run Phase 6 end-to-end (6A, 6B, 6C)."""
    cfg = load_config()
    run_phase6a_regression_baselines(cfg)
    run_phase6b_oof_predictions(cfg)
    run_phase6c_hsed_analysis(cfg)
    logger.info("\n=== PHASE 6 EXECUTION COMPLETE ===")
    return True


if __name__ == "__main__":
    run_all_phase6()
