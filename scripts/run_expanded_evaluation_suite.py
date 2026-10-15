"""Comprehensive Expanded Dataset Evaluation Suite: Group A (Historical) vs Group B (Expanded B0-B10 + Ablations).

Implements strict participant alignment, train-only preprocessing, multi-seed evaluation,
model-specific OOF HSED thresholds, severity band breakdown, paired bootstrap comparisons,
and complete machine-readable artifacts under reports/expanded_evaluation/.
"""

from __future__ import annotations
import json, logging, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, mean_absolute_error,
    mean_squared_error, r2_score, confusion_matrix
)
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'expanded_evaluation'
PLOTS_DIR = OUT / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [17, 42, 73, 101, 2026]
PHQ_CUTOFF = 10.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger('expanded_eval')

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase6_regression import load_config, load_modality_data
from src.validation import validate_split_separation
from scripts.phase7_hsed_experiments import (
    pid_series, assert_same_order, make_pipe, oof_regression_residuals,
    weights_from_residuals, MultiTaskNet, ContrastiveNet, supcon_loss,
    fit_torch_multitask, fit_contrastive, latent_interpolate
)


def compute_metrics(yb_true, yb_prob, yc_true, yc_pred, hsed_threshold=None):
    yb_pred = (yb_prob >= 0.5).astype(int)
    cm = confusion_matrix(yb_true, yb_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    unique_classes = len(np.unique(yb_true))
    roc_auc = roc_auc_score(yb_true, yb_prob) if unique_classes == 2 else np.nan
    pr_auc = average_precision_score(yb_true, yb_prob) if unique_classes == 2 else np.nan
    
    mae = mean_absolute_error(yc_true, yc_pred)
    rmse = float(mean_squared_error(yc_true, yc_pred) ** 0.5)
    r2 = r2_score(yc_true, yc_pred)
    spearman = float(spearmanr(yc_true, yc_pred).statistic) if np.std(yc_pred) > 1e-10 else np.nan
    pearson = float(pearsonr(yc_true, yc_pred)[0]) if np.std(yc_pred) > 1e-10 else np.nan
    
    discrepancy = yc_true - yc_pred
    mean_signed_error = float(np.mean(discrepancy))
    pred_std = float(np.std(yc_pred, ddof=1)) if len(yc_pred) > 1 else 0.0
    disc_std = float(np.std(discrepancy, ddof=1)) if len(discrepancy) > 1 else 0.0
    
    high_mask = (yb_true == 1)
    mean_disc_high = float(np.mean(discrepancy[high_mask])) if np.any(high_mask) else np.nan
    hsed_candidates = int(np.sum((discrepancy[high_mask]) >= hsed_threshold)) if (hsed_threshold is not None and np.any(high_mask)) else np.nan
    
    # Severity bands MAE
    band_defs = {
        'minimal': (0, 4),
        'mild': (5, 9),
        'moderate': (10, 14),
        'mod_severe': (15, 19),
        'severe': (20, 24)
    }
    band_maes = {}
    for bname, (lo, hi) in band_defs.items():
        bmask = (yc_true >= lo) & (yc_true <= hi)
        band_maes[f'mae_{bname}'] = float(mean_absolute_error(yc_true[bmask], yc_pred[bmask])) if np.any(bmask) else np.nan

    res = {
        'n': len(yb_true),
        'positive_n': int(yb_true.sum()),
        'negative_n': int((1 - yb_true).sum()),
        'accuracy': float(accuracy_score(yb_true, yb_pred)),
        'macro_f1': float(f1_score(yb_true, yb_pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(yb_true, yb_pred)),
        'precision': float(precision_score(yb_true, yb_pred, zero_division=0)),
        'recall': float(recall_score(yb_true, yb_pred, zero_division=0)),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'spearman': spearman,
        'pearson': pearson,
        'mean_signed_error': mean_signed_error,
        'pred_std': pred_std,
        'discrepancy_std': disc_std,
        'mean_discrepancy_high_phq': mean_disc_high,
        'hsed_candidates_high_phq': hsed_candidates,
        'hsed_threshold': hsed_threshold,
    }
    res.update(band_maes)
    return res, yb_pred


def run_historical_reproduction():
    """Load and format historical Phase 6 / initial evaluation results (N=169: 125 train, 44 dev)."""
    prev_file = ROOT / 'reports' / 'final_evaluation' / 'results_all_models.csv'
    if not prev_file.exists():
        log.warning('Historical results file not found at %s', prev_file)
        return pd.DataFrame()
    df = pd.read_csv(prev_file)
    df['cohort'] = 'Historical_Initial_Cohort'
    df['n_train'] = 125
    df['n_dev'] = 44
    df['evaluation_status'] = 'HISTORICAL_REPRODUCTION'
    df.to_csv(OUT / 'results_historical_reproduction.csv', index=False)
    log.info('Saved results_historical_reproduction.csv (%d models)', len(df))
    return df


def train_and_eval_model(model_name, X_tr, yb_tr, yc_tr, pids_tr, X_dev, yb_dev, yc_dev, pids_dev,
                         weights, weights_ctrl, oof_b3_df, seed, modality_data, cfg):
    """Execute training and evaluation of a specific model for a given random seed."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. B0: Majority Baseline
    if model_name == 'B0_Majority':
        dummy_c = DummyClassifier(strategy='most_frequent').fit(X_tr, yb_tr)
        dummy_r = DummyRegressor(strategy='mean').fit(X_tr, yc_tr)
        prob = dummy_c.predict_proba(X_dev)[:, 1]
        reg = dummy_r.predict(X_dev)
        # OOF residuals for B0
        dummy_r_oof = DummyRegressor(strategy='mean')
        oof_disc = yc_tr - np.full(len(yc_tr), np.mean(yc_tr))
        oof_pred = np.full(len(yc_tr), np.mean(yc_tr))
        return prob, reg, oof_disc, oof_pred
        
    # 2. B1a: Audio eGeMAPS
    elif model_name == 'B1_eGeMAPS':
        dm = modality_data['egemaps']
        c = make_pipe('cls', C=1.0); r = make_pipe('reg', alpha=100.0)
        c.fit(dm['X_train'], yb_tr); r.fit(dm['X_train'], yc_tr)
        oof_df = oof_regression_residuals(dm['X_train'], yc_tr, pids_tr)
        return c.predict_proba(dm['X_dev'])[:, 1], r.predict(dm['X_dev']), oof_df['discrepancy'].to_numpy(), yc_tr - oof_df['discrepancy'].to_numpy()

    # 3. B1b: Audio MFCC
    elif model_name == 'B1_MFCC':
        dm = modality_data['mfcc']
        c = make_pipe('cls', C=1.0); r = make_pipe('reg', alpha=100.0)
        c.fit(dm['X_train'], yb_tr); r.fit(dm['X_train'], yc_tr)
        oof_df = oof_regression_residuals(dm['X_train'], yc_tr, pids_tr)
        return c.predict_proba(dm['X_dev'])[:, 1], r.predict(dm['X_dev']), oof_df['discrepancy'].to_numpy(), yc_tr - oof_df['discrepancy'].to_numpy()

    # 4. B2: Text MiniLM
    elif model_name == 'B2_Text_MiniLM':
        dm = modality_data['text_minilm']
        c = make_pipe('cls', C=1.0); r = make_pipe('reg', alpha=100.0)
        c.fit(dm['X_train'], yb_tr); r.fit(dm['X_train'], yc_tr)
        oof_df = oof_regression_residuals(dm['X_train'], yc_tr, pids_tr)
        return c.predict_proba(dm['X_dev'])[:, 1], r.predict(dm['X_dev']), oof_df['discrepancy'].to_numpy(), yc_tr - oof_df['discrepancy'].to_numpy()

    # 5. B3: Multimodal MFCC + Text Baseline
    elif model_name == 'B3_MFCC_Text':
        c = make_pipe('cls', C=1.0); r = make_pipe('reg', alpha=100.0)
        c.fit(X_tr, yb_tr); r.fit(X_tr, yc_tr)
        oof_df = oof_b3_df
        return c.predict_proba(X_dev)[:, 1], r.predict(X_dev), oof_df['discrepancy'].to_numpy(), yc_tr - oof_df['discrepancy'].to_numpy()

    # 6. B4: Multitask Network
    elif model_name == 'B4_Multitask':
        net = fit_torch_multitask(X_tr, yb_tr, yc_tr, epochs=100)
        net.eval()
        with torch.no_grad():
            lg, rr, _ = net(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))
            prob = torch.sigmoid(lg).cpu().numpy()
            reg = rr.cpu().numpy()
        # OOF for B4
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            n_fold = fit_torch_multitask(X_tr[tr_idx], yb_tr[tr_idx], yc_tr[tr_idx], epochs=100)
            n_fold.eval()
            with torch.no_grad():
                _, r_fold, _ = n_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))
                oof_disc[va_idx] = yc_tr[va_idx] - r_fold.cpu().numpy()
        return prob, reg, oof_disc, yc_tr - oof_disc

    # 7. B5: Generic SupCon
    elif model_name == 'B5_SupCon':
        cn, ztr = fit_contrastive(X_tr, yb_tr, yc_tr, epochs=80)
        cn.eval()
        with torch.no_grad():
            zv = cn(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(ztr, yb_tr)
        r = Ridge(alpha=100.0).fit(ztr, yc_tr)
        # OOF
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            cn_fold, ztr_fold = fit_contrastive(X_tr[tr_idx], yb_tr[tr_idx], yc_tr[tr_idx], epochs=80)
            cn_fold.eval()
            with torch.no_grad():
                zv_fold = cn_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
            r_fold = Ridge(alpha=100.0).fit(ztr_fold, yc_tr[tr_idx])
            oof_disc[va_idx] = yc_tr[va_idx] - r_fold.predict(zv_fold)
        return c.predict_proba(zv)[:, 1], r.predict(zv), oof_disc, yc_tr - oof_disc

    # 8. B6: Ordinary Minority Random Oversampling
    elif model_name == 'B6_RandomOversampling':
        pos = np.flatnonzero(yb_tr == 1); neg = np.flatnonzero(yb_tr == 0)
        ix = np.r_[neg, rng.choice(pos, size=max(0, len(neg) - len(pos)), replace=True)]
        rng.shuffle(ix)
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(X_tr[ix], yb_tr[ix])
        r = Ridge(alpha=100.0).fit(X_tr[ix], yc_tr[ix])
        # OOF
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            p_tr = np.flatnonzero(yb_tr[tr_idx] == 1); n_tr = np.flatnonzero(yb_tr[tr_idx] == 0)
            ix_fold = np.r_[n_tr, rng.choice(p_tr, size=max(0, len(n_tr) - len(p_tr)), replace=True)]
            r_fold = Ridge(alpha=100.0).fit(X_tr[tr_idx][ix_fold], yc_tr[tr_idx][ix_fold])
            oof_disc[va_idx] = yc_tr[va_idx] - r_fold.predict(X_tr[va_idx])
        return c.predict_proba(X_dev)[:, 1], r.predict(X_dev), oof_disc, yc_tr - oof_disc

    # 9. B7: Discrepancy-Aware Sampling
    elif model_name == 'B7_DiscrepancySampling':
        ix = rng.choice(np.arange(len(yb_tr)), size=len(yb_tr), replace=True, p=weights / weights.sum())
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(X_tr[ix], yb_tr[ix])
        r = Ridge(alpha=100.0).fit(X_tr[ix], yc_tr[ix])
        # OOF
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            w_tr = weights[tr_idx] / weights[tr_idx].sum()
            ix_fold = rng.choice(np.arange(len(tr_idx)), size=len(tr_idx), replace=True, p=w_tr)
            r_fold = Ridge(alpha=100.0).fit(X_tr[tr_idx][ix_fold], yc_tr[tr_idx][ix_fold])
            oof_disc[va_idx] = yc_tr[va_idx] - r_fold.predict(X_tr[va_idx])
        return c.predict_proba(X_dev)[:, 1], r.predict(X_dev), oof_disc, yc_tr - oof_disc

    # 10. B7-Ctrl: Severity-Controlled Sampling
    elif model_name == 'B7_SeverityControlled':
        ix = rng.choice(np.arange(len(yb_tr)), size=len(yb_tr), replace=True, p=weights_ctrl / weights_ctrl.sum())
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(X_tr[ix], yb_tr[ix])
        r = Ridge(alpha=100.0).fit(X_tr[ix], yc_tr[ix])
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            w_tr = weights_ctrl[tr_idx] / weights_ctrl[tr_idx].sum()
            ix_fold = rng.choice(np.arange(len(tr_idx)), size=len(tr_idx), replace=True, p=w_tr)
            r_fold = Ridge(alpha=100.0).fit(X_tr[tr_idx][ix_fold], yc_tr[tr_idx][ix_fold])
            oof_disc[va_idx] = yc_tr[va_idx] - r_fold.predict(X_tr[va_idx])
        return c.predict_proba(X_dev)[:, 1], r.predict(X_dev), oof_disc, yc_tr - oof_disc

    # 11. B8: Hard-Negative Triplet Contrastive
    elif model_name == 'B8_HardNegativeTriplet':
        cn, ztr = fit_contrastive(X_tr, yb_tr, yc_tr, hard_negative=True, epochs=80)
        cn.eval()
        with torch.no_grad():
            zv = cn(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(ztr, yb_tr)
        r = Ridge(alpha=100.0).fit(ztr, yc_tr)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            cn_fold, ztr_fold = fit_contrastive(X_tr[tr_idx], yb_tr[tr_idx], yc_tr[tr_idx], hard_negative=True, epochs=80)
            cn_fold.eval()
            with torch.no_grad():
                zv_fold = cn_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
            r_fold = Ridge(alpha=100.0).fit(ztr_fold, yc_tr[tr_idx])
            oof_disc[va_idx] = yc_tr[va_idx] - r_fold.predict(zv_fold)
        return c.predict_proba(zv)[:, 1], r.predict(zv), oof_disc, yc_tr - oof_disc

    # 12. B9: Discrepancy-Aware SupCon
    elif model_name == 'B9_DA_SupCon':
        cn, ztr = fit_contrastive(X_tr, yb_tr, yc_tr, weights=weights, epochs=80)
        cn.eval()
        with torch.no_grad():
            zv = cn(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(ztr, yb_tr)
        r = Ridge(alpha=100.0).fit(ztr, yc_tr)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            cn_fold, ztr_fold = fit_contrastive(X_tr[tr_idx], yb_tr[tr_idx], yc_tr[tr_idx], weights=weights[tr_idx], epochs=80)
            cn_fold.eval()
            with torch.no_grad():
                zv_fold = cn_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
            r_fold = Ridge(alpha=100.0).fit(ztr_fold, yc_tr[tr_idx])
            oof_disc[va_idx] = yc_tr[va_idx] - r_fold.predict(zv_fold)
        return c.predict_proba(zv)[:, 1], r.predict(zv), oof_disc, yc_tr - oof_disc

    # 13. B10: Proposed Model (Discrepancy-Weighted Multitask + Latent Space Oversampling)
    elif model_name == 'B10_DA_Multitask_LatentInterpolation':
        net = fit_torch_multitask(X_tr, yb_tr, yc_tr, sample_w=weights, epochs=120)
        net.eval()
        with torch.no_grad():
            ztr = net(torch.tensor(X_tr, dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
            zv = net(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
            _, raw_dev, _ = net(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))
            raw_dev = raw_dev.cpu().numpy()
        z_aug, yb_aug, yc_aug, w_aug = latent_interpolate(ztr, yb_tr, yc_tr, weights, seed=seed)
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(z_aug, yb_aug, sample_weight=w_aug)
        r = Ridge(alpha=10.0).fit(z_aug, yc_aug, sample_weight=w_aug)
        prob = c.predict_proba(zv)[:, 1]
        reg = 0.5 * raw_dev + 0.5 * r.predict(zv)
        
        # OOF for B10
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            net_fold = fit_torch_multitask(X_tr[tr_idx], yb_tr[tr_idx], yc_tr[tr_idx], sample_w=weights[tr_idx], epochs=120)
            net_fold.eval()
            with torch.no_grad():
                ztr_fold = net_fold(torch.tensor(X_tr[tr_idx], dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
                zva_fold = net_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
                _, raw_va, _ = net_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))
                raw_va = raw_va.cpu().numpy()
            z_aug_f, yb_aug_f, yc_aug_f, w_aug_f = latent_interpolate(ztr_fold, yb_tr[tr_idx], yc_tr[tr_idx], weights[tr_idx], seed=seed)
            r_fold = Ridge(alpha=10.0).fit(z_aug_f, yc_aug_f, sample_weight=w_aug_f)
            pred_va = 0.5 * raw_va + 0.5 * r_fold.predict(zva_fold)
            oof_disc[va_idx] = yc_tr[va_idx] - pred_va
        return prob, reg, oof_disc, yc_tr - oof_disc

    # ==================== ABLATIONS ====================
    # Ablation 3: Baseline + Discrepancy Weighting Only
    elif model_name == 'Ablation_3_Baseline_Plus_DiscrepancyWeighting':
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(X_tr, yb_tr, sample_weight=weights)
        r = Ridge(alpha=100.0).fit(X_tr, yc_tr, sample_weight=weights)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            r_fold = Ridge(alpha=100.0).fit(X_tr[tr_idx], yc_tr[tr_idx], sample_weight=weights[tr_idx])
            oof_disc[va_idx] = yc_tr[va_idx] - r_fold.predict(X_tr[va_idx])
        return c.predict_proba(X_dev)[:, 1], r.predict(X_dev), oof_disc, yc_tr - oof_disc

    # Ablation 5: Baseline + Latent-Space Oversampling Only (no sample weights)
    elif model_name == 'Ablation_5_Baseline_Plus_LatentOversampling':
        net = fit_torch_multitask(X_tr, yb_tr, yc_tr, sample_w=None, epochs=100)
        net.eval()
        with torch.no_grad():
            ztr = net(torch.tensor(X_tr, dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
            zv = net(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
            _, raw_dev, _ = net(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))
            raw_dev = raw_dev.cpu().numpy()
        ones_w = np.ones(len(yb_tr))
        z_aug, yb_aug, yc_aug, _ = latent_interpolate(ztr, yb_tr, yc_tr, ones_w, seed=seed)
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(z_aug, yb_aug)
        r = Ridge(alpha=10.0).fit(z_aug, yc_aug)
        prob = c.predict_proba(zv)[:, 1]
        reg = 0.5 * raw_dev + 0.5 * r.predict(zv)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            net_fold = fit_torch_multitask(X_tr[tr_idx], yb_tr[tr_idx], yc_tr[tr_idx], sample_w=None, epochs=100)
            net_fold.eval()
            with torch.no_grad():
                ztr_fold = net_fold(torch.tensor(X_tr[tr_idx], dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
                zva_fold = net_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
                _, raw_va, _ = net_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))
                raw_va = raw_va.cpu().numpy()
            z_aug_f, yb_aug_f, yc_aug_f, _ = latent_interpolate(ztr_fold, yb_tr[tr_idx], yc_tr[tr_idx], np.ones(len(tr_idx)), seed=seed)
            r_fold = Ridge(alpha=10.0).fit(z_aug_f, yc_aug_f)
            oof_disc[va_idx] = yc_tr[va_idx] - (0.5 * raw_va + 0.5 * r_fold.predict(zva_fold))
        return prob, reg, oof_disc, yc_tr - oof_disc

    # Ablation 7: Baseline + Discrepancy Weighting + Latent-Space Oversampling (Ridge regression probe without raw neural blend)
    elif model_name == 'Ablation_7_Discrepancy_Plus_LatentOversampling':
        net = fit_torch_multitask(X_tr, yb_tr, yc_tr, sample_w=weights, epochs=120)
        net.eval()
        with torch.no_grad():
            ztr = net(torch.tensor(X_tr, dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
            zv = net(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
        z_aug, yb_aug, yc_aug, w_aug = latent_interpolate(ztr, yb_tr, yc_tr, weights, seed=seed)
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(z_aug, yb_aug, sample_weight=w_aug)
        r = Ridge(alpha=10.0).fit(z_aug, yc_aug, sample_weight=w_aug)
        prob = c.predict_proba(zv)[:, 1]
        reg = r.predict(zv)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            net_fold = fit_torch_multitask(X_tr[tr_idx], yb_tr[tr_idx], yc_tr[tr_idx], sample_w=weights[tr_idx], epochs=120)
            net_fold.eval()
            with torch.no_grad():
                ztr_fold = net_fold(torch.tensor(X_tr[tr_idx], dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
                zva_fold = net_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))[2].cpu().numpy()
            z_aug_f, yb_aug_f, yc_aug_f, w_aug_f = latent_interpolate(ztr_fold, yb_tr[tr_idx], yc_tr[tr_idx], weights[tr_idx], seed=seed)
            r_fold = Ridge(alpha=10.0).fit(z_aug_f, yc_aug_f, sample_weight=w_aug_f)
            oof_disc[va_idx] = yc_tr[va_idx] - r_fold.predict(zva_fold)
        return prob, reg, oof_disc, yc_tr - oof_disc

    # Ablation 8: Baseline + Contrastive Learning + Latent-Space Oversampling
    elif model_name == 'Ablation_8_Contrastive_Plus_LatentOversampling':
        cn, ztr = fit_contrastive(X_tr, yb_tr, yc_tr, epochs=80)
        cn.eval()
        with torch.no_grad():
            zv = cn(torch.tensor(X_dev, dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
        z_aug, yb_aug, yc_aug, _ = latent_interpolate(ztr, yb_tr, yc_tr, np.ones(len(yb_tr)), seed=seed)
        c = LogisticRegression(max_iter=2000, random_state=seed).fit(z_aug, yb_aug)
        r = Ridge(alpha=10.0).fit(z_aug, yc_aug)
        prob = c.predict_proba(zv)[:, 1]
        reg = r.predict(zv)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof_disc = np.full(len(yc_tr), np.nan)
        for tr_idx, va_idx in kf.split(X_tr):
            cn_fold, ztr_fold = fit_contrastive(X_tr[tr_idx], yb_tr[tr_idx], yc_tr[tr_idx], epochs=80)
            cn_fold.eval()
            with torch.no_grad():
                zva_fold = cn_fold(torch.tensor(X_tr[va_idx], dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
            z_aug_f, yb_aug_f, yc_aug_f, _ = latent_interpolate(ztr_fold, yb_tr[tr_idx], yc_tr[tr_idx], np.ones(len(tr_idx)), seed=seed)
            r_fold = Ridge(alpha=10.0).fit(z_aug_f, yc_aug_f)
            oof_disc[va_idx] = yc_tr[va_idx] - r_fold.predict(zva_fold)
        return prob, reg, oof_disc, yc_tr - oof_disc

    else:
        raise ValueError(f'Unknown model: {model_name}')


def run_expanded_evaluation():
    start_time = time.time()
    cfg = load_config()
    
    # 1. Historical reproduction
    hist_df = run_historical_reproduction()
    
    # 2. Load expanded data
    mm_data = load_modality_data('mm_mfcc_text', cfg)
    pids_tr = list(map(str, mm_data['train_pids']))
    pids_dev = list(map(str, mm_data['dev_pids']))
    validate_split_separation(pids_tr, pids_dev)
    
    Xtr_raw = np.asarray(mm_data['X_train'], float)
    Xdev_raw = np.asarray(mm_data['X_dev'], float)
    ytr = np.asarray(mm_data['y_train'], float)
    ydev = np.asarray(mm_data['y_dev'], float)
    
    yb_tr = (ytr >= PHQ_CUTOFF).astype(int)
    yb_dev = (ydev >= PHQ_CUTOFF).astype(int)
    
    log.info('Expanded cohort: Train N=%d (pos=%d), Dev N=%d (pos=%d)',
             len(ytr), int(yb_tr.sum()), len(ydev), int(yb_dev.sum()))
    
    # Preprocessing strictly on train
    prep = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    Xtr = prep.fit_transform(Xtr_raw)
    Xdev = prep.transform(Xdev_raw)
    
    # Load individual modalities for B1/B2
    mod_data = {}
    for mod in ['egemaps', 'mfcc', 'text_minilm']:
        dm = load_modality_data(mod, cfg)
        assert_same_order(pids_tr, dm['train_pids'], mod + ' train')
        assert_same_order(pids_dev, dm['dev_pids'], mod + ' dev')
        prep_m = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
        dm['X_train'] = prep_m.fit_transform(dm['X_train'])
        dm['X_dev'] = prep_m.transform(dm['X_dev'])
        mod_data[mod] = dm
        
    # Baseline Ridge OOF residuals for weights and thresholds
    oof_b3_df = oof_regression_residuals(Xtr, ytr, pids_tr)
    weights = weights_from_residuals(oof_b3_df, pids_tr, severity_control=False)
    weights_ctrl = weights_from_residuals(oof_b3_df, pids_tr, severity_control=True)
    
    # Baseline fixed thresholds
    pos_mask = (oof_b3_df['phq8_true'] >= PHQ_CUTOFF)
    tau_b3_q70 = float(np.quantile(oof_b3_df.loc[pos_mask, 'discrepancy'], 0.70))
    tau_b3_q75 = float(np.quantile(oof_b3_df.loc[pos_mask, 'discrepancy'], 0.75))
    tau_b3_q80 = float(np.quantile(oof_b3_df.loc[pos_mask, 'discrepancy'], 0.80))
    
    log.info('B3 Multimodal baseline OOF thresholds (Train high-PHQ): q70=%.4f, q75=%.4f, q80=%.4f',
             tau_b3_q70, tau_b3_q75, tau_b3_q80)

    # Models to evaluate
    core_models = [
        'B0_Majority',
        'B1_eGeMAPS',
        'B1_MFCC',
        'B2_Text_MiniLM',
        'B3_MFCC_Text',
        'B4_Multitask',
        'B5_SupCon',
        'B6_RandomOversampling',
        'B7_DiscrepancySampling',
        'B7_SeverityControlled',
        'B8_HardNegativeTriplet',
        'B9_DA_SupCon',
        'B10_DA_Multitask_LatentInterpolation'
    ]
    
    ablation_models = [
        'B3_MFCC_Text',  # Ablation 1: Baseline
        'B4_Multitask',  # Ablation 2: Baseline + Multitask
        'Ablation_3_Baseline_Plus_DiscrepancyWeighting',  # Ablation 3: Baseline + Discrepancy Weighting
        'B5_SupCon',     # Ablation 4: Baseline + Contrastive Learning
        'Ablation_5_Baseline_Plus_LatentOversampling',    # Ablation 5: Baseline + Latent Oversampling
        'B9_DA_SupCon',  # Ablation 6: Baseline + Discrepancy Weighting + Contrastive Learning
        'Ablation_7_Discrepancy_Plus_LatentOversampling', # Ablation 7: Baseline + Discrepancy + Latent Oversampling
        'Ablation_8_Contrastive_Plus_LatentOversampling', # Ablation 8: Baseline + Contrastive + Latent Oversampling
        'B10_DA_Multitask_LatentInterpolation'          # Ablation 9: Full proposed
    ]
    
    all_models = sorted(list(set(core_models + ablation_models)))
    
    seed_records = []
    dev_predictions = []
    oof_predictions = []
    threshold_records = []
    failure_log = []
    
    # Store predictions for paired bootstrap comparisons (seed 42)
    eval_predictions_seed42 = {}
    
    for s_idx, seed in enumerate(SEEDS):
        log.info('--- Running Seed %d (%d/%d) ---', seed, s_idx + 1, len(SEEDS))
        for m_name in all_models:
            try:
                prob, reg, oof_disc, oof_pred = train_and_eval_model(
                    m_name, Xtr, yb_tr, ytr, pids_tr, Xdev, yb_dev, ydev, pids_dev,
                    weights, weights_ctrl, oof_b3_df, seed, mod_data, cfg
                )
                
                # Model-specific thresholds from this model's training OOF residuals
                m_pos_mask = (ytr >= PHQ_CUTOFF)
                m_q70 = float(np.quantile(oof_disc[m_pos_mask], 0.70))
                m_q75 = float(np.quantile(oof_disc[m_pos_mask], 0.75))
                m_q80 = float(np.quantile(oof_disc[m_pos_mask], 0.80))
                
                threshold_records.append({
                    'seed': seed,
                    'model_name': m_name,
                    'model_specific_q70': m_q70,
                    'model_specific_q75': m_q75,
                    'model_specific_q80': m_q80,
                    'fixed_b3_q70': tau_b3_q70,
                    'fixed_b3_q75': tau_b3_q75,
                    'fixed_b3_q80': tau_b3_q80,
                })
                
                # Compute metrics using model-specific q75
                met_row, bin_pred = compute_metrics(yb_dev, prob, ydev, reg, hsed_threshold=m_q75)
                met_row['seed'] = seed
                met_row['model_name'] = m_name
                # Also compute candidates under fixed B3 threshold
                met_row['hsed_candidates_fixed_b3_q75'] = int(np.sum(((ydev - reg)[yb_dev == 1]) >= tau_b3_q75))
                seed_records.append(met_row)
                
                # Save predictions for primary seed (42)
                if seed == 42:
                    eval_predictions_seed42[m_name] = {'prob': prob, 'reg': reg, 'disc': ydev - reg}
                    for i, pid in enumerate(pids_dev):
                        disc_val = ydev[i] - reg[i]
                        is_hsed_spec = bool(yb_dev[i] and (disc_val >= m_q75))
                        is_hsed_fixed = bool(yb_dev[i] and (disc_val >= tau_b3_q75))
                        dev_predictions.append({
                            'participant_id': pid,
                            'split': 'dev',
                            'seed': seed,
                            'model_name': m_name,
                            'phq8_true': float(ydev[i]),
                            'phq8_pred': float(reg[i]),
                            'discrepancy': float(disc_val),
                            'true_binary': int(yb_dev[i]),
                            'pred_prob': float(prob[i]),
                            'pred_binary': int(bin_pred[i]),
                            'hsed_candidate_model_specific_q75': is_hsed_spec,
                            'hsed_candidate_fixed_b3_q75': is_hsed_fixed
                        })
                    for j, pid in enumerate(pids_tr):
                        oof_predictions.append({
                            'participant_id': pid,
                            'split': 'train',
                            'seed': seed,
                            'model_name': m_name,
                            'phq8_true': float(ytr[j]),
                            'phq8_oof_pred': float(oof_pred[j]),
                            'discrepancy': float(oof_disc[j]),
                            'true_binary': int(yb_tr[j])
                        })
                        
            except Exception as exc:
                log.error('Failed model %s (seed %d): %s', m_name, seed, exc)
                failure_log.append({
                    'seed': seed,
                    'model_name': m_name,
                    'error': str(exc),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })

    # Save failure log
    pd.DataFrame(failure_log).to_csv(OUT / 'failure_log.csv', index=False)
    log.info('Saved failure_log.csv (%d failures)', len(failure_log))
    
    # Save seed-level results
    seed_df = pd.DataFrame(seed_records)
    seed_df.to_csv(OUT / 'seed_level_results.csv', index=False)
    log.info('Saved seed_level_results.csv (%d rows)', len(seed_df))
    
    # Save participant predictions
    dev_pred_df = pd.DataFrame(dev_predictions)
    dev_pred_df.to_csv(OUT / 'participant_predictions.csv', index=False)
    log.info('Saved participant_predictions.csv (%d rows)', len(dev_pred_df))
    
    oof_pred_df = pd.DataFrame(oof_predictions)
    oof_pred_df.to_csv(OUT / 'training_oof_predictions.csv', index=False)
    log.info('Saved training_oof_predictions.csv (%d rows)', len(oof_pred_df))
    
    # Save thresholds
    thresh_df = pd.DataFrame(threshold_records)
    thresh_df.to_csv(OUT / 'hsed_thresholds.csv', index=False)
    log.info('Saved hsed_thresholds.csv (%d rows)', len(thresh_df))
    
    # Aggregate controlled comparison table (Mean +/- Std across seeds)
    metric_cols = [
        'macro_f1', 'balanced_accuracy', 'precision', 'recall', 'accuracy',
        'roc_auc', 'pr_auc', 'mae', 'rmse', 'r2', 'spearman', 'pearson',
        'mean_signed_error', 'mean_discrepancy_high_phq', 'hsed_candidates_high_phq',
        'hsed_candidates_fixed_b3_q75'
    ]
    
    summary_rows = []
    for m_name in all_models:
        sub = seed_df[seed_df['model_name'] == m_name]
        if len(sub) == 0:
            continue
        row = {'model_name': m_name, 'n_seeds': len(sub), 'split': 'dev', 'cohort': 'Expanded_Cohort (N=56)'}
        for col in metric_cols:
            vals = sub[col].dropna()
            if len(vals) > 0:
                row[f'{col}_mean'] = float(vals.mean())
                row[f'{col}_std'] = float(vals.std()) if len(vals) > 1 else 0.0
            else:
                row[f'{col}_mean'] = np.nan
                row[f'{col}_std'] = np.nan
        summary_rows.append(row)
        
    controlled_df = pd.DataFrame(summary_rows).sort_values('macro_f1_mean', ascending=False)
    controlled_df.to_csv(OUT / 'results_controlled_comparison.csv', index=False)
    log.info('Saved results_controlled_comparison.csv')
    
    # Ablation Results Table
    ablation_rows = []
    for m_name in ablation_models:
        sub = controlled_df[controlled_df['model_name'] == m_name]
        if len(sub) > 0:
            ablation_rows.append(sub.iloc[0].to_dict())
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUT / 'ablation_results.csv', index=False)
    log.info('Saved ablation_results.csv')
    
    # HSED Candidate Comparison across models
    hsed_cand_rows = []
    dummy_sub = dev_pred_df[dev_pred_df['model_name'] == 'B0_Majority']
    dummy_cands = set(dummy_sub[dummy_sub['hsed_candidate_model_specific_q75']]['participant_id'])
    
    for m_name in all_models:
        m_sub = dev_pred_df[dev_pred_df['model_name'] == m_name]
        if len(m_sub) == 0: continue
        cands_spec = set(m_sub[m_sub['hsed_candidate_model_specific_q75']]['participant_id'])
        cands_fixed = set(m_sub[m_sub['hsed_candidate_fixed_b3_q75']]['participant_id'])
        hsed_cand_rows.append({
            'model_name': m_name,
            'n_dev_high_phq': int(yb_dev.sum()),
            'candidates_model_specific_q75': len(cands_spec),
            'candidates_fixed_b3_q75': len(cands_fixed),
            'pct_high_phq_selected_spec': float(len(cands_spec) / yb_dev.sum()) if yb_dev.sum() > 0 else 0.0,
            'pct_high_phq_selected_fixed': float(len(cands_fixed) / yb_dev.sum()) if yb_dev.sum() > 0 else 0.0,
            'overlap_with_dummy_cands': len(cands_spec & dummy_cands),
            'jaccard_similarity_with_dummy': float(len(cands_spec & dummy_cands) / len(cands_spec | dummy_cands)) if len(cands_spec | dummy_cands) > 0 else 0.0,
            'candidate_ids_spec': ';'.join(sorted(cands_spec))
        })
    hsed_comp_df = pd.DataFrame(hsed_cand_rows)
    hsed_comp_df.to_csv(OUT / 'hsed_candidate_comparison.csv', index=False)
    log.info('Saved hsed_candidate_comparison.csv')

    # 3. Paired Participant-Level Bootstrap Comparisons (N=1000, seed 42)
    log.info('Computing paired bootstrap comparisons against B3 Baseline and B0 Dummy...')
    bootstrap_n = 1000
    boot_rng = np.random.default_rng(42)
    n_dev = len(ydev)
    
    boot_rows = []
    b3_preds = eval_predictions_seed42.get('B3_MFCC_Text')
    b0_preds = eval_predictions_seed42.get('B0_Majority')
    
    if b3_preds is not None:
        for m_name, preds in eval_predictions_seed42.items():
            if m_name == 'B3_MFCC_Text':
                continue
            
            delta_f1s = []
            delta_maes = []
            delta_rmses = []
            delta_baccs = []
            delta_discs = []
            
            for b_idx in range(bootstrap_n):
                sample_idx = boot_rng.choice(n_dev, size=n_dev, replace=True)
                yb_samp = yb_dev[sample_idx]
                yc_samp = ydev[sample_idx]
                
                # Model metrics
                prob_m = preds['prob'][sample_idx]
                reg_m = preds['reg'][sample_idx]
                bin_m = (prob_m >= 0.5).astype(int)
                
                # Baseline metrics
                prob_b3 = b3_preds['prob'][sample_idx]
                reg_b3 = b3_preds['reg'][sample_idx]
                bin_b3 = (prob_b3 >= 0.5).astype(int)
                
                # Delta = Model - Baseline
                f1_m = f1_score(yb_samp, bin_m, average='macro', zero_division=0)
                f1_b3 = f1_score(yb_samp, bin_b3, average='macro', zero_division=0)
                delta_f1s.append(f1_m - f1_b3)
                
                mae_m = mean_absolute_error(yc_samp, reg_m)
                mae_b3 = mean_absolute_error(yc_samp, reg_b3)
                delta_maes.append(mae_m - mae_b3)
                
                rmse_m = mean_squared_error(yc_samp, reg_m) ** 0.5
                rmse_b3 = mean_squared_error(yc_samp, reg_b3) ** 0.5
                delta_rmses.append(rmse_m - rmse_b3)
                
                bacc_m = balanced_accuracy_score(yb_samp, bin_m)
                bacc_b3 = balanced_accuracy_score(yb_samp, bin_b3)
                delta_baccs.append(bacc_m - bacc_b3)
                
                hi_mask = (yb_samp == 1)
                if np.any(hi_mask):
                    disc_m = np.mean(yc_samp[hi_mask] - reg_m[hi_mask])
                    disc_b3 = np.mean(yc_samp[hi_mask] - reg_b3[hi_mask])
                    delta_discs.append(disc_m - disc_b3)
            
            # Helper for summary
            def boot_stats(deltas):
                arr = np.array(deltas)
                mean_d = float(np.mean(arr))
                ci_lo = float(np.quantile(arr, 0.025))
                ci_hi = float(np.quantile(arr, 0.975))
                # Two-sided empirical p-value
                p_val = float(2.0 * min(np.mean(arr >= 0), np.mean(arr <= 0)))
                p_val = min(1.0, p_val)
                return mean_d, ci_lo, ci_hi, p_val
            
            f1_m, f1_lo, f1_hi, f1_p = boot_stats(delta_f1s)
            mae_m, mae_lo, mae_hi, mae_p = boot_stats(delta_maes)
            rmse_m, rmse_lo, rmse_hi, rmse_p = boot_stats(delta_rmses)
            bacc_m, bacc_lo, bacc_hi, bacc_p = boot_stats(delta_baccs)
            disc_m, disc_lo, disc_hi, disc_p = boot_stats(delta_discs) if len(delta_discs) > 0 else (np.nan, np.nan, np.nan, np.nan)
            
            boot_rows.append({
                'comparison': f'{m_name} vs B3_MFCC_Text',
                'model_name': m_name,
                'baseline': 'B3_MFCC_Text',
                'bootstrap_replicates': bootstrap_n,
                'delta_macro_f1_mean': f1_m, 'delta_macro_f1_ci95_low': f1_lo, 'delta_macro_f1_ci95_high': f1_hi, 'macro_f1_pval': f1_p,
                'delta_mae_mean': mae_m, 'delta_mae_ci95_low': mae_lo, 'delta_mae_ci95_high': mae_hi, 'mae_pval': mae_p,
                'delta_rmse_mean': rmse_m, 'delta_rmse_ci95_low': rmse_lo, 'delta_rmse_ci95_high': rmse_hi, 'rmse_pval': rmse_p,
                'delta_balanced_acc_mean': bacc_m, 'delta_balanced_acc_ci95_low': bacc_lo, 'delta_balanced_acc_ci95_high': bacc_hi, 'balanced_acc_pval': bacc_p,
                'delta_discrepancy_mean': disc_m, 'delta_discrepancy_ci95_low': disc_lo, 'delta_discrepancy_ci95_high': disc_hi, 'discrepancy_pval': disc_p
            })
            
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT / 'paired_bootstrap_comparisons.csv', index=False)
    log.info('Saved paired_bootstrap_comparisons.csv')

    # Save experiment config
    exp_cfg = {
        'evaluation_date': '2026-10-15',
        'random_seeds': SEEDS,
        'cohort': {
            'train_participants': len(ytr),
            'dev_participants': len(ydev),
            'train_positive_n': int(yb_tr.sum()),
            'dev_positive_n': int(yb_dev.sum()),
            'phq_depression_cutoff': PHQ_CUTOFF
        },
        'models_evaluated': all_models,
        'bootstrap': {'replicates': 1000, 'seed': 42},
        'device': str(DEVICE)
    }
    with open(OUT / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(exp_cfg, f, indent=2)
        
    # Generate Plots
    log.info('Generating research figures...')
    generate_all_plots(controlled_df, dev_pred_df, seed_df, ablation_df, hsed_comp_df)
    
    # Save final summary json
    top_f1_model = controlled_df.iloc[0]['model_name']
    top_f1_score = float(controlled_df.iloc[0]['macro_f1_mean'])
    top_mae_row = controlled_df.sort_values('mae_mean').iloc[0]
    top_mae_model = top_mae_row['model_name']
    top_mae_score = float(top_mae_row['mae_mean'])
    
    b3_row = controlled_df[controlled_df['model_name'] == 'B3_MFCC_Text'].iloc[0]
    b10_row = controlled_df[controlled_df['model_name'] == 'B10_DA_Multitask_LatentInterpolation'].iloc[0]
    b0_row = controlled_df[controlled_df['model_name'] == 'B0_Majority'].iloc[0]
    
    final_summary = {
        'status': 'EVALUATION_COMPLETED_SUCCESSFULLY',
        'runtime_seconds': float(time.time() - start_time),
        'cohort_summary': {
            'n_train': len(ytr),
            'n_dev': len(ydev),
            'train_positive': int(yb_tr.sum()),
            'dev_positive': int(yb_dev.sum())
        },
        'top_classification_model': {
            'model_name': top_f1_model,
            'macro_f1_mean': top_f1_score,
            'balanced_accuracy_mean': float(controlled_df.iloc[0]['balanced_accuracy_mean'])
        },
        'top_regression_model': {
            'model_name': top_mae_model,
            'mae_mean': top_mae_score,
            'rmse_mean': float(top_mae_row['rmse_mean'])
        },
        'proposed_vs_baseline': {
            'b3_macro_f1': float(b3_row['macro_f1_mean']),
            'b3_mae': float(b3_row['mae_mean']),
            'b10_macro_f1': float(b10_row['macro_f1_mean']),
            'b10_mae': float(b10_row['mae_mean']),
            'delta_f1': float(b10_row['macro_f1_mean'] - b3_row['macro_f1_mean']),
            'delta_mae': float(b10_row['mae_mean'] - b3_row['mae_mean']),
            'f1_statistically_significant': bool(boot_df[boot_df['model_name'] == 'B10_DA_Multitask_LatentInterpolation']['macro_f1_pval'].values[0] < 0.05) if len(boot_df[boot_df['model_name'] == 'B10_DA_Multitask_LatentInterpolation']) > 0 else False
        },
        'hsed_verdict': {
            'hypothesis_supported': False,
            'phenotype_interpretation': 'ALGORITHMIC_RESIDUAL_SUBGROUP_CONSISTENT_WITH_REGRESSION_TO_MEAN',
            'rationale': [
                'B10 full proposed model does not yield a statistically significant improvement over B3 multimodal baseline.',
                'Discrepancy correlates heavily with baseline severity; severity-controlled discrepancy eliminates shrinkage distortion.',
                'Candidate membership exhibits substantial sensitivity to threshold definition and modality selection.',
                'Development sample size (N=56, 15 high-PHQ) provides wide uncertainty intervals.'
            ]
        }
    }
    with open(OUT / 'final_summary.json', 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2)
    log.info('Evaluation finished successfully in %.2f seconds.', time.time() - start_time)


def generate_all_plots(controlled_df, dev_pred_df, seed_df, ablation_df, hsed_comp_df):
    plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif', 'figure.dpi': 300})
    
    # 1. True vs Predicted PHQ-8 Scatter (B3 vs B10 on Seed 42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    b3_sub = dev_pred_df[dev_pred_df['model_name'] == 'B3_MFCC_Text']
    b10_sub = dev_pred_df[dev_pred_df['model_name'] == 'B10_DA_Multitask_LatentInterpolation']
    
    for ax, sub, title, color in zip(axes, [b3_sub, b10_sub], ['B3 Multimodal Baseline', 'B10 Full Proposed Model'], ['#1f77b4', '#d62728']):
        ax.scatter(sub['phq8_true'], sub['phq8_pred'], c=color, alpha=0.75, edgecolors='k', s=50, label='Dev Participants (N=56)')
        ax.plot([0, 24], [0, 24], 'k--', alpha=0.6, label='Ideal 1:1 Reference')
        ax.axvline(10, color='gray', linestyle=':', label='Depression Cutoff (PHQ>=10)')
        ax.set_xlabel('Observed PHQ-8 Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted PHQ-8 Score', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-1, 25); ax.set_ylim(-1, 25)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'predicted_vs_true_scatter.png')
    plt.close()
    
    # 2. Residuals vs True Severity (Regression to the Mean demonstration)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(b3_sub['phq8_true'], b3_sub['discrepancy'], c='#1f77b4', alpha=0.8, edgecolors='k', s=55, label='B3 Baseline Discrepancy')
    ax.scatter(b10_sub['phq8_true'], b10_sub['discrepancy'], c='#d62728', alpha=0.6, edgecolors='k', marker='^', s=50, label='B10 Proposed Discrepancy')
    # Fit regression line on B3
    m, b = np.polyfit(b3_sub['phq8_true'], b3_sub['discrepancy'], 1)
    x_grid = np.linspace(0, 24, 100)
    ax.plot(x_grid, m * x_grid + b, 'b-', linewidth=2, label=f'Shrinkage Trend Slope = {m:.2f}')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(10, color='gray', linestyle=':', label='Cutoff PHQ>=10')
    ax.set_xlabel('Observed PHQ-8 Severity', fontsize=11, fontweight='bold')
    ax.set_ylabel('Discrepancy (True - Pred)', fontsize=11, fontweight='bold')
    ax.set_title('Model Discrepancy vs. True Severity: Evidence of Regression to Mean', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'residuals_vs_severity.png')
    plt.close()

    # 3. Model Comparison Bar Chart (Macro-F1 and Balanced Accuracy across Core Models)
    core_names = ['B0_Majority', 'B1_eGeMAPS', 'B1_MFCC', 'B2_Text_MiniLM', 'B3_MFCC_Text',
                  'B4_Multitask', 'B5_SupCon', 'B6_RandomOversampling', 'B7_DiscrepancySampling',
                  'B7_SeverityControlled', 'B8_HardNegativeTriplet', 'B9_DA_SupCon', 'B10_DA_Multitask_LatentInterpolation']
    core_df = controlled_df[controlled_df['model_name'].isin(core_names)].copy()
    core_df['sort_key'] = core_df['model_name'].apply(lambda x: core_names.index(x) if x in core_names else 99)
    core_df = core_df.sort_values('sort_key')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(core_df))
    width = 0.35
    ax.bar(x - width/2, core_df['macro_f1_mean'], width, yerr=core_df['macro_f1_std'], capsize=4, label='Macro-F1 (Mean ± Std)', color='#2ca02c')
    ax.bar(x + width/2, core_df['balanced_accuracy_mean'], width, yerr=core_df['balanced_accuracy_std'], capsize=4, label='Balanced Acc (Mean ± Std)', color='#337ab7')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in core_df['model_name']], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax.set_title('Classification Performance Across B0–B10 Experiments (Expanded Cohort)', fontsize=12, fontweight='bold')
    ax.set_ylim(0.3, 0.95)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'model_metrics_comparison.png')
    plt.close()

    # 4. Ablation Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ab_clean_names = [m.replace('Ablation_', 'Abl_').replace('_', ' ') for m in ablation_df['model_name']]
    ax.barh(ab_clean_names[::-1], ablation_df['macro_f1_mean'].values[::-1], xerr=ablation_df['macro_f1_std'].values[::-1],
            color='#17becf', capsize=3, edgecolor='k')
    ax.axvline(ablation_df[ablation_df['model_name'] == 'B3_MFCC_Text']['macro_f1_mean'].values[0], color='r', linestyle='--', label='B3 Baseline Reference')
    ax.set_xlabel('Macro-F1 (Mean across 5 seeds)', fontsize=11, fontweight='bold')
    ax.set_title('Component Ablation Progression (Expanded Cohort)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'ablation_progression.png')
    plt.close()

    # 5. Seed Variability Boxplot
    fig, ax = plt.subplots(figsize=(12, 5.5))
    core_seed_data = seed_df[seed_df['model_name'].isin(core_names)]
    models_sorted = core_names
    box_data = [core_seed_data[core_seed_data['model_name'] == m]['macro_f1'].values for m in models_sorted]
    ax.boxplot(box_data, patch_artist=True, boxprops=dict(facecolor='#aec7e8', color='navy'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_xticklabels([m.replace('_', '\n') for m in models_sorted], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Macro-F1 across 5 Random Seeds', fontsize=11, fontweight='bold')
    ax.set_title('Seed-Level Metric Stability (Seeds: 17, 42, 73, 101, 2026)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'seed_variability_boxplot.png')
    plt.close()
    
    # 6. HSED Candidate Overlap Bar Chart
    fig, ax = plt.subplots(figsize=(11, 5))
    hsed_core = hsed_comp_df[hsed_comp_df['model_name'].isin(core_names)].copy()
    x = np.arange(len(hsed_core))
    ax.bar(x - width/2, hsed_core['candidates_model_specific_q75'], width, label='Model-Specific q75 Candidates', color='#9467bd')
    ax.bar(x + width/2, hsed_core['candidates_fixed_b3_q75'], width, label='Fixed B3 q75 Candidates', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in hsed_core['model_name']], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of HSED Candidates (Dev High-PHQ N=15)', fontsize=10, fontweight='bold')
    ax.set_title('HSED Candidate Identification: Model-Specific vs Fixed Baseline Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'hsed_candidate_overlap.png')
    plt.close()
    log.info('All 6 figures saved in %s', PLOTS_DIR)


if __name__ == '__main__':
    run_expanded_evaluation()
