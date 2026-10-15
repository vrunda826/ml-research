"""Controlled Evaluation of Latent-Space Oversampling for DAIC-WOZ Depression Detection.

Executes:
- E0: Original B3 Baseline (MFCC + Text, Ridge + Logistic Regression, no oversampling)
- E1: Ordinary Random Minority Oversampling (input space)
- E2a: Linear Latent-Space Oversampling (PCA latent space, k-NN interpolation)
- E2b: Neural Latent-Space Oversampling (Autoencoder bottleneck, k-NN interpolation)
- E3: Oversampling Sensitivity Analysis (ratios alpha in {0.35, 0.50, 0.65}, k in {1, 3, 5} via training CV)
- E4: Constrained Latent-Space Variant (Centroid-bounded manifold oversampling)

Includes:
- Strict training-only preprocessing and representation fitting
- 5-fold stratified cross-validation on training partition (N=163)
- Evaluation on untouched development cohort (N=56) across seeds [17, 42, 73, 101, 2026]
- Paired participant-level bootstrap analysis (B=1000)
- Synthetic sample provenance tracking and audit
- Production of all deliverables under reports/latent_oversampling_evaluation/
"""

from __future__ import annotations
import json, logging, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
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
OUT_DIR = ROOT / 'reports' / 'latent_oversampling_evaluation'
PLOTS_DIR = OUT_DIR / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [17, 42, 73, 101, 2026]
PHQ_CUTOFF = 10.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger('latent_oversampling')

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase6_regression import load_config, load_modality_data
from src.validation import validate_split_separation


# ==============================================================================
# 1. REPRESENTATION ENCODERS
# ==============================================================================

class PyTorchAutoencoder(nn.Module):
    """Regularized non-linear autoencoder for latent manifold compression."""
    def __init__(self, in_dim=462, hidden_dim=128, latent_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


def fit_autoencoder(X_train, latent_dim=32, epochs=80, lr=0.005, weight_decay=1e-4, seed=42):
    """Train autoencoder strictly on training data."""
    torch.manual_seed(seed)
    net = PyTorchAutoencoder(in_dim=X_train.shape[1], latent_dim=latent_dim).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        _, x_rec = net(X_t)
        loss = F.mse_loss(x_rec, X_t)
        loss.backward()
        opt.step()
        
    net.eval()
    with torch.no_grad():
        z_train = net.encoder(X_t).cpu().numpy()
    return net, z_train


def encode_with_autoencoder(net, X):
    net.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        z = net.encoder(X_t).cpu().numpy()
    return z


# ==============================================================================
# 2. LATENT-SPACE INTERPOLATION FUNCTIONS
# ==============================================================================

def latent_knn_oversample(Z, yb, yc, pids=None, target_ratio=0.50, k_neighbors=3,
                          constrained=False, seed=42):
    """Perform k-NN interpolation between positive participants in latent space.
    
    Returns:
    --------
    Z_aug, yb_aug, yc_aug, audit_records
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(yb == 1)
    neg = np.flatnonzero(yb == 0)
    
    if len(pos) < 2 or len(neg) == 0:
        return Z, yb, yc, []
        
    target_pos = int(np.ceil(target_ratio * len(neg) / (1.0 - target_ratio)))
    n_new = max(0, target_pos - len(pos))
    if n_new == 0:
        return Z, yb, yc, []
        
    # Constrained distance threshold (2 * dispersion from positive centroid)
    if constrained:
        centroid = np.mean(Z[pos], axis=0)
        dists_from_centroid = np.linalg.norm(Z[pos] - centroid, axis=1)
        max_allowed_dist = 2.0 * np.std(dists_from_centroid) + np.mean(dists_from_centroid)
    else:
        centroid = None
        max_allowed_dist = np.inf
        
    rows, labs, regs, audit_records = [], [], [], []
    attempts = 0
    max_attempts = n_new * 10
    
    while len(rows) < n_new and attempts < max_attempts:
        attempts += 1
        i = int(rng.choice(pos))
        cands = pos[pos != i]
        
        # Compute distances to other positives in latent space
        dists = np.linalg.norm(Z[cands] - Z[i], axis=1)
        k = min(k_neighbors, len(cands))
        top_k_idx = cands[np.argsort(dists)[:k]]
        j = int(rng.choice(top_k_idx))
        
        lam = float(rng.uniform(0.15, 0.85))
        z_synth = Z[i] + lam * (Z[j] - Z[i])
        
        # Check constraint if enabled
        if constrained:
            if np.linalg.norm(z_synth - centroid) > max_allowed_dist:
                continue
                
        y_c_synth = float(yc[i] + lam * (yc[j] - yc[i]))
        
        rows.append(z_synth)
        labs.append(1)
        regs.append(y_c_synth)
        
        if pids is not None:
            audit_records.append({
                'synth_idx': len(rows),
                'anchor_pid': str(pids[i]),
                'neighbor_pid': str(pids[j]),
                'lambda': lam,
                'latent_distance': float(np.linalg.norm(Z[j] - Z[i])),
                'synth_phq_score': y_c_synth,
                'anchor_phq_score': float(yc[i]),
                'neighbor_phq_score': float(yc[j]),
                'constrained_filter': constrained
            })
            
    if len(rows) == 0:
        return Z, yb, yc, []
        
    Z_aug = np.vstack([Z, np.asarray(rows)])
    yb_aug = np.r_[yb, np.asarray(labs)]
    yc_aug = np.r_[yc, np.asarray(regs)]
    return Z_aug, yb_aug, yc_aug, audit_records


# ==============================================================================
# 3. METRICS HELPER
# ==============================================================================

def evaluate_predictions(yb_true, yb_prob, yc_true, yc_pred, threshold=0.50):
    yb_pred = (yb_prob >= threshold).astype(int)
    cm = confusion_matrix(yb_true, yb_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    unique_classes = len(np.unique(yb_true))
    roc_auc = float(roc_auc_score(yb_true, yb_prob)) if unique_classes == 2 else np.nan
    pr_auc = float(average_precision_score(yb_true, yb_prob)) if unique_classes == 2 else np.nan
    
    mae = float(mean_absolute_error(yc_true, yc_pred))
    rmse = float(mean_squared_error(yc_true, yc_pred) ** 0.5)
    r2 = float(r2_score(yc_true, yc_pred))
    spearman = float(spearmanr(yc_true, yc_pred).statistic) if np.std(yc_pred) > 1e-10 else np.nan
    pearson = float(pearsonr(yc_true, yc_pred)[0]) if np.std(yc_pred) > 1e-10 else np.nan
    
    discrepancy = yc_true - yc_pred
    mean_signed_error = float(np.mean(discrepancy))
    pred_std = float(np.std(yc_pred, ddof=1)) if len(yc_pred) > 1 else 0.0
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(yb_true, yb_pred)),
        'macro_f1': float(f1_score(yb_true, yb_pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(yb_true, yb_pred)),
        'precision': float(precision_score(yb_true, yb_pred, zero_division=0)),
        'recall': float(recall_score(yb_true, yb_pred, zero_division=0)),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'spearman': spearman, 'pearson': pearson,
        'mean_signed_error': mean_signed_error,
        'pred_std': pred_std
    }


def find_optimal_threshold(yb_true, yb_prob):
    """Find threshold that maximizes Macro-F1 on training OOF predictions."""
    best_f1 = -1.0
    best_tau = 0.50
    for tau in np.linspace(0.15, 0.85, 36):
        pred = (yb_prob >= tau).astype(int)
        f1 = f1_score(yb_true, pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau


# ==============================================================================
# 4. MODEL RUNNERS FOR A GIVEN FOLD OR FULL FIT
# ==============================================================================

def train_and_predict_condition(cond_name, X_tr, yb_tr, yc_tr, pids_tr, X_eval,
                                target_ratio=0.50, k_neighbors=3, seed=42):
    """Fit condition on X_tr, evaluate on X_eval. Returns (prob, reg, audit_records)."""
    rng = np.random.default_rng(seed)
    
    # E0: Original B3 Baseline
    if cond_name == 'E0_Baseline_B3':
        cls_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(X_tr, yb_tr)
        reg_model = Ridge(alpha=100.0).fit(X_tr, yc_tr)
        prob = cls_model.predict_proba(X_eval)[:, 1]
        reg = reg_model.predict(X_eval)
        return prob, reg, []
        
    # E1: Ordinary Random Oversampling (Input Space)
    elif cond_name == 'E1_Ordinary_Random_Oversampling':
        pos = np.flatnonzero(yb_tr == 1)
        neg = np.flatnonzero(yb_tr == 0)
        target_pos = int(np.ceil(target_ratio * len(neg) / (1.0 - target_ratio)))
        n_new = max(0, target_pos - len(pos))
        if n_new > 0:
            dup_idx = rng.choice(pos, size=n_new, replace=True)
            ix = np.r_[np.arange(len(yb_tr)), dup_idx]
            rng.shuffle(ix)
            X_fit = X_tr[ix]
            yb_fit = yb_tr[ix]
            yc_fit = yc_tr[ix]
        else:
            X_fit, yb_fit, yc_fit = X_tr, yb_tr, yc_tr
            
        cls_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(X_fit, yb_fit)
        reg_model = Ridge(alpha=100.0).fit(X_fit, yc_fit)
        prob = cls_model.predict_proba(X_eval)[:, 1]
        reg = reg_model.predict(X_eval)
        return prob, reg, []

    # E2a: Linear Latent-Space Oversampling (PCA Latent Space)
    elif cond_name in ['E2a_Linear_Latent_PCA', 'E3_Sensitivity']:
        pca = PCA(n_components=32, random_state=seed)
        Z_tr = pca.fit_transform(X_tr)
        Z_eval = pca.transform(X_eval)
        
        Z_aug, yb_aug, yc_aug, audit = latent_knn_oversample(
            Z_tr, yb_tr, yc_tr, pids=pids_tr, target_ratio=target_ratio,
            k_neighbors=k_neighbors, constrained=False, seed=seed
        )
        
        cls_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(Z_aug, yb_aug)
        reg_model = Ridge(alpha=10.0).fit(Z_aug, yc_aug)
        prob = cls_model.predict_proba(Z_eval)[:, 1]
        reg = reg_model.predict(Z_eval)
        return prob, reg, audit

    # E2b: Neural Latent-Space Oversampling (Autoencoder Bottleneck)
    elif cond_name == 'E2b_Neural_Latent_Autoencoder':
        ae_net, Z_tr = fit_autoencoder(X_tr, latent_dim=32, epochs=80, seed=seed)
        Z_eval = encode_with_autoencoder(ae_net, X_eval)
        
        Z_aug, yb_aug, yc_aug, audit = latent_knn_oversample(
            Z_tr, yb_tr, yc_tr, pids=pids_tr, target_ratio=target_ratio,
            k_neighbors=k_neighbors, constrained=False, seed=seed
        )
        
        cls_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(Z_aug, yb_aug)
        reg_model = Ridge(alpha=10.0).fit(Z_aug, yc_aug)
        prob = cls_model.predict_proba(Z_eval)[:, 1]
        reg = reg_model.predict(Z_eval)
        return prob, reg, audit

    # E4: Constrained Latent-Space Oversampling (PCA + Centroid Constraint)
    elif cond_name == 'E4_Constrained_Latent_Manifold':
        pca = PCA(n_components=32, random_state=seed)
        Z_tr = pca.fit_transform(X_tr)
        Z_eval = pca.transform(X_eval)
        
        Z_aug, yb_aug, yc_aug, audit = latent_knn_oversample(
            Z_tr, yb_tr, yc_tr, pids=pids_tr, target_ratio=target_ratio,
            k_neighbors=k_neighbors, constrained=True, seed=seed
        )
        
        cls_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(Z_aug, yb_aug)
        reg_model = Ridge(alpha=10.0).fit(Z_aug, yc_aug)
        prob = cls_model.predict_proba(Z_eval)[:, 1]
        reg = reg_model.predict(Z_eval)
        return prob, reg, audit

    else:
        raise ValueError(f"Unknown condition: {cond_name}")


# ==============================================================================
# 5. MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_all_latent_oversampling_experiments():
    start_time = time.time()
    cfg = load_config()
    
    # Load dataset
    data = load_modality_data('mm_mfcc_text', cfg)
    pids_tr = list(map(str, data['train_pids']))
    pids_dev = list(map(str, data['dev_pids']))
    validate_split_separation(pids_tr, pids_dev)
    
    Xtr_raw = np.asarray(data['X_train'], float)
    Xdev_raw = np.asarray(data['X_dev'], float)
    ytr = np.asarray(data['y_train'], float)
    ydev = np.asarray(data['y_dev'], float)
    
    yb_tr = (ytr >= PHQ_CUTOFF).astype(int)
    yb_dev = (ydev >= PHQ_CUTOFF).astype(int)
    
    log.info("Dataset verified: Train N=%d (pos=%d), Dev N=%d (pos=%d)",
             len(ytr), int(yb_tr.sum()), len(ydev), int(yb_dev.sum()))
             
    # Global training preprocessing (fit on train only)
    scaler = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    Xtr = scaler.fit_transform(Xtr_raw)
    Xdev = scaler.transform(Xdev_raw)
    
    # --------------------------------------------------------------------------
    # STEP 1: TRAINING 5-FOLD CV & HYPERPARAMETER GRID (E3)
    # --------------------------------------------------------------------------
    log.info("Executing 5-Fold Stratified Cross-Validation on Training Set...")
    
    cv_records = []
    sensitivity_grid = [
        (0.35, 1), (0.35, 3), (0.35, 5),
        (0.50, 1), (0.50, 3), (0.50, 5),
        (0.65, 1), (0.65, 3), (0.65, 5)
    ]
    
    # Primary conditions to evaluate in CV
    core_conditions = [
        'E0_Baseline_B3',
        'E1_Ordinary_Random_Oversampling',
        'E2a_Linear_Latent_PCA',
        'E2b_Neural_Latent_Autoencoder',
        'E4_Constrained_Latent_Manifold'
    ]
    
    # Pre-compute CV folds across seeds
    cv_oof_predictions = {seed: {} for seed in SEEDS}
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        # 1. Run core conditions in CV
        for cond in core_conditions:
            oof_prob = np.full(len(yb_tr), np.nan)
            oof_reg = np.full(len(ytr), np.nan)
            
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(Xtr_raw, yb_tr)):
                # Preprocessing fitted inside fold
                f_scaler = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
                X_tr_f = f_scaler.fit_transform(Xtr_raw[tr_idx])
                X_val_f = f_scaler.transform(Xtr_raw[val_idx])
                
                prob_val, reg_val, _ = train_and_predict_condition(
                    cond, X_tr_f, yb_tr[tr_idx], ytr[tr_idx], [pids_tr[i] for i in tr_idx],
                    X_val_f, target_ratio=0.50, k_neighbors=3, seed=seed
                )
                oof_prob[val_idx] = prob_val
                oof_reg[val_idx] = reg_val
                
            opt_tau = find_optimal_threshold(yb_tr, oof_prob)
            cv_oof_predictions[seed][cond] = {'prob': oof_prob, 'reg': oof_reg, 'opt_tau': opt_tau}
            
            # Evaluate CV metrics at default 0.50 and optimal tau
            m_fixed = evaluate_predictions(yb_tr, oof_prob, ytr, oof_reg, threshold=0.50)
            m_opt = evaluate_predictions(yb_tr, oof_prob, ytr, oof_reg, threshold=opt_tau)
            
            cv_records.append({
                'seed': seed,
                'condition': cond,
                'eval_mode': 'train_5fold_cv',
                'param_ratio': 0.50,
                'param_k': 3,
                'macro_f1_fixed_05': m_fixed['macro_f1'],
                'balanced_acc_fixed_05': m_fixed['balanced_accuracy'],
                'optimal_threshold': opt_tau,
                'macro_f1_optimal_tau': m_opt['macro_f1'],
                'balanced_acc_optimal_tau': m_opt['balanced_accuracy'],
                'precision': m_fixed['precision'],
                'recall': m_fixed['recall'],
                'mae': m_fixed['mae'],
                'rmse': m_fixed['rmse'],
                'r2': m_fixed['r2'],
                'spearman': m_fixed['spearman']
            })
            
        # 2. Run E3 Sensitivity Grid in CV (Linear Latent PCA)
        for (ratio, k) in sensitivity_grid:
            cond_grid_name = f"E3_PCA_ratio_{ratio:.2f}_k_{k}"
            oof_prob = np.full(len(yb_tr), np.nan)
            oof_reg = np.full(len(ytr), np.nan)
            
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(Xtr_raw, yb_tr)):
                f_scaler = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
                X_tr_f = f_scaler.fit_transform(Xtr_raw[tr_idx])
                X_val_f = f_scaler.transform(Xtr_raw[val_idx])
                
                prob_val, reg_val, _ = train_and_predict_condition(
                    'E3_Sensitivity', X_tr_f, yb_tr[tr_idx], ytr[tr_idx], [pids_tr[i] for i in tr_idx],
                    X_val_f, target_ratio=ratio, k_neighbors=k, seed=seed
                )
                oof_prob[val_idx] = prob_val
                oof_reg[val_idx] = reg_val
                
            opt_tau = find_optimal_threshold(yb_tr, oof_prob)
            m_fixed = evaluate_predictions(yb_tr, oof_prob, ytr, oof_reg, threshold=0.50)
            
            cv_records.append({
                'seed': seed,
                'condition': cond_grid_name,
                'eval_mode': 'train_5fold_cv',
                'param_ratio': ratio,
                'param_k': k,
                'macro_f1_fixed_05': m_fixed['macro_f1'],
                'balanced_acc_fixed_05': m_fixed['balanced_accuracy'],
                'optimal_threshold': opt_tau,
                'macro_f1_optimal_tau': evaluate_predictions(yb_tr, oof_prob, ytr, oof_reg, threshold=opt_tau)['macro_f1'],
                'balanced_acc_optimal_tau': evaluate_predictions(yb_tr, oof_prob, ytr, oof_reg, threshold=opt_tau)['balanced_accuracy'],
                'precision': m_fixed['precision'],
                'recall': m_fixed['recall'],
                'mae': m_fixed['mae'],
                'rmse': m_fixed['rmse'],
                'r2': m_fixed['r2'],
                'spearman': m_fixed['spearman']
            })

    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(OUT_DIR / 'cross_validation_results.csv', index=False)
    log.info("Saved cross_validation_results.csv (%d rows)", len(cv_df))
    
    # Determine best E3 setting based purely on CV Macro-F1
    e3_cv_summary = cv_df[cv_df['condition'].str.startswith('E3_PCA')].groupby(['param_ratio', 'param_k'])['macro_f1_fixed_05'].mean().reset_index()
    best_e3_row = e3_cv_summary.sort_values('macro_f1_fixed_05', ascending=False).iloc[0]
    best_ratio = float(best_e3_row['param_ratio'])
    best_k = int(best_e3_row['param_k'])
    log.info("Optimal E3 configuration from training CV: ratio=%.2f, k=%d (CV Macro-F1 = %.4f)",
             best_ratio, best_k, float(best_e3_row['macro_f1_fixed_05']))

    # --------------------------------------------------------------------------
    # STEP 2: EVALUATION ON UNTOUCHED DEVELOPMENT COHORT (N=56)
    # --------------------------------------------------------------------------
    log.info("Evaluating Frozen Models on Official Development Partition...")
    
    dev_conditions = [
        'E0_Baseline_B3',
        'E1_Ordinary_Random_Oversampling',
        'E2a_Linear_Latent_PCA',
        'E2b_Neural_Latent_Autoencoder',
        'E3_Selected_Optimal',
        'E4_Constrained_Latent_Manifold'
    ]
    
    dev_records = []
    dev_predictions_seed42 = {}
    synthetic_sample_audit = []
    failure_log = []
    
    for seed in SEEDS:
        for cond in dev_conditions:
            try:
                # Set parameters
                if cond == 'E3_Selected_Optimal':
                    actual_cond = 'E2a_Linear_Latent_PCA'
                    curr_ratio, curr_k = best_ratio, best_k
                elif cond == 'E1_Ordinary_Random_Oversampling':
                    actual_cond = cond
                    curr_ratio, curr_k = 0.50, 1
                else:
                    actual_cond = cond
                    curr_ratio, curr_k = 0.50, 3
                    
                prob_dev, reg_dev, audit = train_and_predict_condition(
                    actual_cond, Xtr, yb_tr, ytr, pids_tr, Xdev,
                    target_ratio=curr_ratio, k_neighbors=curr_k, seed=seed
                )
                
                # Retrieve optimal threshold tuned on training OOF for this seed
                base_key = cond if cond in cv_oof_predictions[seed] else ('E2a_Linear_Latent_PCA' if cond == 'E3_Selected_Optimal' else 'E0_Baseline_B3')
                opt_tau = cv_oof_predictions[seed][base_key]['opt_tau']
                
                m_fixed = evaluate_predictions(yb_dev, prob_dev, ydev, reg_dev, threshold=0.50)
                m_opt = evaluate_predictions(yb_dev, prob_dev, ydev, reg_dev, threshold=opt_tau)
                
                dev_records.append({
                    'seed': seed,
                    'condition': cond,
                    'param_ratio': curr_ratio,
                    'param_k': curr_k,
                    'threshold_fixed': 0.50,
                    'macro_f1_fixed': m_fixed['macro_f1'],
                    'balanced_acc_fixed': m_fixed['balanced_accuracy'],
                    'precision_fixed': m_fixed['precision'],
                    'recall_fixed': m_fixed['recall'],
                    'threshold_opt_tau': opt_tau,
                    'macro_f1_opt': m_opt['macro_f1'],
                    'balanced_acc_opt': m_opt['balanced_accuracy'],
                    'precision_opt': m_opt['precision'],
                    'recall_opt': m_opt['recall'],
                    'accuracy': m_fixed['accuracy'],
                    'roc_auc': m_fixed['roc_auc'],
                    'pr_auc': m_fixed['pr_auc'],
                    'mae': m_fixed['mae'],
                    'rmse': m_fixed['rmse'],
                    'r2': m_fixed['r2'],
                    'spearman': m_fixed['spearman'],
                    'pearson': m_fixed['pearson'],
                    'mean_signed_error': m_fixed['mean_signed_error'],
                    'pred_std': m_fixed['pred_std'],
                    'tn': m_fixed['tn'], 'fp': m_fixed['fp'],
                    'fn': m_fixed['fn'], 'tp': m_fixed['tp']
                })
                
                # Save predictions for seed 42 (primary seed)
                if seed == 42:
                    dev_predictions_seed42[cond] = {'prob': prob_dev, 'reg': reg_dev}
                    if audit and len(synthetic_sample_audit) == 0:
                        synthetic_sample_audit.extend(audit)
                        
            except Exception as exc:
                log.error("Failure in condition %s (seed %d): %s", cond, seed, exc)
                failure_log.append({'seed': seed, 'condition': cond, 'error': str(exc)})
                
    pd.DataFrame(failure_log).to_csv(OUT_DIR / 'failure_log.csv', index=False)
    log.info("Saved failure_log.csv (%d failures)", len(failure_log))
    
    # Save results by seed
    seed_df = pd.DataFrame(dev_records)
    seed_df.to_csv(OUT_DIR / 'results_by_seed.csv', index=False)
    log.info("Saved results_by_seed.csv (%d rows)", len(seed_df))
    
    # Save synthetic sample audit
    if len(synthetic_sample_audit) == 0:
        # Generate representative sample audit for record
        _, _, _, audit_sample = train_and_predict_condition(
            'E2a_Linear_Latent_PCA', Xtr, yb_tr, ytr, pids_tr, Xdev,
            target_ratio=0.50, k_neighbors=3, seed=42
        )
        synthetic_sample_audit = audit_sample
    pd.DataFrame(synthetic_sample_audit).to_csv(OUT_DIR / 'synthetic_sample_audit.csv', index=False)
    log.info("Saved synthetic_sample_audit.csv (%d samples)", len(synthetic_sample_audit))
    
    # Aggregate development comparison table
    summary_rows = []
    metrics_to_agg = [
        'macro_f1_fixed', 'balanced_acc_fixed', 'precision_fixed', 'recall_fixed',
        'macro_f1_opt', 'balanced_acc_opt', 'precision_opt', 'recall_opt',
        'accuracy', 'roc_auc', 'pr_auc', 'mae', 'rmse', 'r2', 'spearman', 'pearson'
    ]
    for cond in dev_conditions:
        sub = seed_df[seed_df['condition'] == cond]
        row = {'condition': cond, 'n_seeds': len(sub), 'param_ratio': sub['param_ratio'].iloc[0], 'param_k': sub['param_k'].iloc[0]}
        for m in metrics_to_agg:
            row[f'{m}_mean'] = float(sub[m].mean())
            row[f'{m}_std'] = float(sub[m].std()) if len(sub) > 1 else 0.0
        summary_rows.append(row)
        
    dev_summary_df = pd.DataFrame(summary_rows).sort_values('macro_f1_fixed_mean', ascending=False)
    dev_summary_df.to_csv(OUT_DIR / 'development_results.csv', index=False)
    log.info("Saved development_results.csv")

    # --------------------------------------------------------------------------
    # STEP 3: PAIRED PARTICIPANT-LEVEL BOOTSTRAP ANALYSIS (B=1000, SEED 42)
    # --------------------------------------------------------------------------
    log.info("Computing Paired Bootstrap Comparisons (B=1000)...")
    boot_n = 1000
    boot_rng = np.random.default_rng(42)
    n_dev = len(ydev)
    
    comparisons = [
        ('E2a_Linear_Latent_PCA', 'E0_Baseline_B3'),
        ('E2a_Linear_Latent_PCA', 'E1_Ordinary_Random_Oversampling'),
        ('E2b_Neural_Latent_Autoencoder', 'E0_Baseline_B3'),
        ('E2b_Neural_Latent_Autoencoder', 'E1_Ordinary_Random_Oversampling'),
        ('E3_Selected_Optimal', 'E0_Baseline_B3'),
        ('E4_Constrained_Latent_Manifold', 'E2a_Linear_Latent_PCA')
    ]
    
    boot_records = []
    for (m_test, m_base) in comparisons:
        pred_test = dev_predictions_seed42[m_test]
        pred_base = dev_predictions_seed42[m_base]
        
        delta_f1s, delta_baccs, delta_precs, delta_recs = [], [], [], []
        delta_maes, delta_rmses = [], []
        
        for _ in range(boot_n):
            idx = boot_rng.choice(n_dev, size=n_dev, replace=True)
            yb_s = yb_dev[idx]
            yc_s = ydev[idx]
            
            p_test = pred_test['prob'][idx]
            r_test = pred_test['reg'][idx]
            bin_test = (p_test >= 0.50).astype(int)
            
            p_base = pred_base['prob'][idx]
            r_base = pred_base['reg'][idx]
            bin_base = (p_base >= 0.50).astype(int)
            
            delta_f1s.append(f1_score(yb_s, bin_test, average='macro', zero_division=0) - f1_score(yb_s, bin_base, average='macro', zero_division=0))
            delta_baccs.append(balanced_accuracy_score(yb_s, bin_test) - balanced_accuracy_score(yb_s, bin_base))
            delta_precs.append(precision_score(yb_s, bin_test, zero_division=0) - precision_score(yb_s, bin_base, zero_division=0))
            delta_recs.append(recall_score(yb_s, bin_test, zero_division=0) - recall_score(yb_s, bin_base, zero_division=0))
            delta_maes.append(mean_absolute_error(yc_s, r_test) - mean_absolute_error(yc_s, r_base))
            delta_rmses.append(mean_squared_error(yc_s, r_test)**0.5 - mean_squared_error(yc_s, r_base)**0.5)
            
        def get_ci_p(deltas):
            arr = np.array(deltas)
            mean_d = float(np.mean(arr))
            ci_lo = float(np.quantile(arr, 0.025))
            ci_hi = float(np.quantile(arr, 0.975))
            pval = float(2.0 * min(np.mean(arr >= 0), np.mean(arr <= 0)))
            return mean_d, ci_lo, ci_hi, min(1.0, pval)
            
        f1_m, f1_lo, f1_hi, f1_p = get_ci_p(delta_f1s)
        bacc_m, bacc_lo, bacc_hi, bacc_p = get_ci_p(delta_baccs)
        prec_m, prec_lo, prec_hi, prec_p = get_ci_p(delta_precs)
        rec_m, rec_lo, rec_hi, rec_p = get_ci_p(delta_recs)
        mae_m, mae_lo, mae_hi, mae_p = get_ci_p(delta_maes)
        rmse_m, rmse_lo, rmse_hi, rmse_p = get_ci_p(delta_rmses)
        
        boot_records.append({
            'comparison': f"{m_test} vs {m_base}",
            'test_condition': m_test,
            'baseline_condition': m_base,
            'delta_macro_f1_mean': f1_m, 'delta_macro_f1_ci95_low': f1_lo, 'delta_macro_f1_ci95_high': f1_hi, 'macro_f1_pval': f1_p,
            'delta_balanced_acc_mean': bacc_m, 'delta_balanced_acc_ci95_low': bacc_lo, 'delta_balanced_acc_ci95_high': bacc_hi, 'balanced_acc_pval': bacc_p,
            'delta_precision_mean': prec_m, 'delta_precision_ci95_low': prec_lo, 'delta_precision_ci95_high': prec_hi, 'precision_pval': prec_p,
            'delta_recall_mean': rec_m, 'delta_recall_ci95_low': rec_lo, 'delta_recall_ci95_high': rec_hi, 'recall_pval': rec_p,
            'delta_mae_mean': mae_m, 'delta_mae_ci95_low': mae_lo, 'delta_mae_ci95_high': mae_hi, 'mae_pval': mae_p,
            'delta_rmse_mean': rmse_m, 'delta_rmse_ci95_low': rmse_lo, 'delta_rmse_ci95_high': rmse_hi, 'rmse_pval': rmse_p,
        })
        
    boot_df = pd.DataFrame(boot_records)
    boot_df.to_csv(OUT_DIR / 'paired_bootstrap_comparisons.csv', index=False)
    log.info("Saved paired_bootstrap_comparisons.csv")

    # --------------------------------------------------------------------------
    # STEP 4: CONFIGURATION & METADATA OUTPUTS
    # --------------------------------------------------------------------------
    exp_cfg = {
        'study_title': 'Controlled Evaluation of Latent-Space Oversampling on DAIC-WOZ',
        'date': '2026-10-15',
        'seeds': SEEDS,
        'cohort': {
            'train_participants': len(ytr),
            'dev_participants': len(ydev),
            'train_positive': int(yb_tr.sum()),
            'dev_positive': int(yb_dev.sum())
        },
        'experimental_conditions': dev_conditions,
        'optimal_e3_cv_params': {'target_ratio': best_ratio, 'k_neighbors': best_k},
        'bootstrap_replicates': boot_n
    }
    with open(OUT_DIR / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(exp_cfg, f, indent=2)

    # --------------------------------------------------------------------------
    # STEP 5: GENERATE COMPARISON PLOTS
    # --------------------------------------------------------------------------
    log.info("Generating research figures...")
    generate_latent_evaluation_plots(dev_summary_df, seed_df, boot_df, cv_df, Xtr, yb_tr, ytr)
    
    log.info("Latent oversampling evaluation completed in %.2f seconds.", time.time() - start_time)


def generate_latent_evaluation_plots(dev_summary_df, seed_df, boot_df, cv_df, Xtr, yb_tr, ytr):
    plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif', 'figure.dpi': 300})
    
    cond_order = [
        'E0_Baseline_B3',
        'E1_Ordinary_Random_Oversampling',
        'E2a_Linear_Latent_PCA',
        'E2b_Neural_Latent_Autoencoder',
        'E3_Selected_Optimal',
        'E4_Constrained_Latent_Manifold'
    ]
    labels_clean = [
        'E0: B3 Baseline\n(No Oversampling)',
        'E1: Random\nOversampling',
        'E2a: Linear Latent\n(PCA k-NN)',
        'E2b: Neural Latent\n(Autoencoder)',
        'E3: CV-Optimal\nLatent Oversampling',
        'E4: Constrained\nLatent Manifold'
    ]
    
    plot_df = dev_summary_df.set_index('condition').loc[cond_order].reset_index()
    
    # 1. Classification Metrics Comparison Bar Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width/2, plot_df['macro_f1_fixed_mean'], width, yerr=plot_df['macro_f1_fixed_std'],
           capsize=4, label='Macro-F1 (Mean ± Std)', color='#2ca02c', edgecolor='black')
    ax.bar(x + width/2, plot_df['balanced_acc_fixed_mean'], width, yerr=plot_df['balanced_acc_fixed_std'],
           capsize=4, label='Balanced Acc (Mean ± Std)', color='#1f77b4', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_clean, fontsize=9)
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Classification Performance across Oversampling Conditions (Dev N=56)', fontsize=12, fontweight='bold')
    ax.set_ylim(0.5, 0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'classification_metrics_comparison.png')
    plt.close()
    
    # 2. Continuous Prediction Metrics (MAE & RMSE) Bar Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width/2, plot_df['mae_mean'], width, yerr=plot_df['mae_std'],
           capsize=4, label='MAE (Mean ± Std)', color='#ff7f0e', edgecolor='black')
    ax.bar(x + width/2, plot_df['rmse_mean'], width, yerr=plot_df['rmse_std'],
           capsize=4, label='RMSE (Mean ± Std)', color='#d62728', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_clean, fontsize=9)
    ax.set_ylabel('Error (PHQ-8 points)', fontsize=11, fontweight='bold')
    ax.set_title('Continuous Severity Prediction Error across Conditions (Dev N=56)', fontsize=12, fontweight='bold')
    ax.set_ylim(2.5, 6.0)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'continuous_metrics_comparison.png')
    plt.close()

    # 3. Seed Variability Boxplot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    box_data = [seed_df[seed_df['condition'] == c]['macro_f1_fixed'].values for c in cond_order]
    bp = ax.boxplot(box_data, patch_artist=True, boxprops=dict(facecolor='#aec7e8', color='navy'),
                    medianprops=dict(color='red', linewidth=2))
    ax.set_xticklabels(labels_clean, fontsize=9)
    ax.set_ylabel('Macro-F1 across 5 Random Seeds', fontsize=11, fontweight='bold')
    ax.set_title('Stability of Classification Performance Across Random Seeds', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'seed_variability_boxplot.png')
    plt.close()

    # 4. Paired Bootstrap Delta Distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    key_comp = boot_df.set_index('comparison')
    
    comparisons_to_plot = [
        ('E2a_Linear_Latent_PCA vs E0_Baseline_B3', 'E2a (Linear Latent) vs E0 Baseline'),
        ('E2a_Linear_Latent_PCA vs E1_Ordinary_Random_Oversampling', 'E2a (Linear Latent) vs E1 Random Oversampling')
    ]
    
    for ax, (comp_id, comp_title) in zip(axes, comparisons_to_plot):
        row = key_comp.loc[comp_id]
        delta_m = row['delta_macro_f1_mean']
        ci_lo = row['delta_macro_f1_ci95_low']
        ci_hi = row['delta_macro_f1_ci95_high']
        pval = row['macro_f1_pval']
        
        ax.errorbar([0], [delta_m], yerr=[[delta_m - ci_lo], [ci_hi - delta_m]],
                    fmt='o', color='purple', ecolor='darkviolet', elinewidth=3, capsize=8, ms=8)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Null Effect (Δ = 0)')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.25, 0.25)
        ax.set_xticks([])
        ax.set_ylabel('Δ Macro-F1 (95% Bootstrap CI)', fontsize=11, fontweight='bold')
        ax.set_title(f"{comp_title}\nΔ = {delta_m:+.4f} [{ci_lo:+.3f}, {ci_hi:+.3f}], p = {pval:.3f}", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'bootstrap_delta_distributions.png')
    plt.close()

    # 5. Sensitivity Grid Heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    grid_sub = cv_df[cv_df['condition'].str.startswith('E3_PCA')].groupby(['param_ratio', 'param_k'])['macro_f1_fixed_05'].mean().unstack()
    im = ax.imshow(grid_sub.values, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(np.arange(len(grid_sub.columns)))
    ax.set_yticks(np.arange(len(grid_sub.index)))
    ax.set_xticklabels([f"k = {col}" for col in grid_sub.columns], fontsize=10)
    ax.set_yticklabels([f"Ratio = {idx:.2f}" for idx in grid_sub.index], fontsize=10)
    ax.set_xlabel('Number of Neighbors (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Target Positive Ratio (α)', fontsize=11, fontweight='bold')
    ax.set_title('Training CV Macro-F1 across Oversampling Grid (E3)', fontsize=12, fontweight='bold')
    for i in range(len(grid_sub.index)):
        for j in range(len(grid_sub.columns)):
            val = grid_sub.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha='center', va='center', color='black' if val < 0.72 else 'white', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Training 5-Fold CV Macro-F1')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'sensitivity_grid_heatmap.png')
    plt.close()

    # 6. Latent Space 2D PCA Geometry Visualization
    pca_vis = PCA(n_components=2, random_state=42)
    Z_vis = pca_vis.fit_transform(Xtr)
    _, _, _, audit_vis = latent_knn_oversample(Z_vis, yb_tr, ytr, target_ratio=0.50, k_neighbors=3, seed=42)
    # Generate synthetic points for plotting
    Z_aug_vis, yb_aug_vis, _, _ = latent_knn_oversample(Z_vis, yb_tr, ytr, target_ratio=0.50, k_neighbors=3, seed=42)
    Z_synth_vis = Z_aug_vis[len(yb_tr):]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Z_vis[yb_tr == 0, 0], Z_vis[yb_tr == 0, 1], c='#1f77b4', alpha=0.6, s=45, label='Real Negative (PHQ<10, N=113)')
    ax.scatter(Z_vis[yb_tr == 1, 0], Z_vis[yb_tr == 1, 1], c='#2ca02c', edgecolors='k', s=65, label='Real Positive (PHQ>=10, N=50)')
    if len(Z_synth_vis) > 0:
        ax.scatter(Z_synth_vis[:, 0], Z_synth_vis[:, 1], c='#d62728', marker='^', edgecolors='k', s=55, label=f'Synthetic Latent Points (N={len(Z_synth_vis)})')
    ax.set_xlabel('Principal Component 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Principal Component 2', fontsize=11, fontweight='bold')
    ax.set_title('Latent Manifold Geometry: Real vs. Synthetic Samples (PCA 2D)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'latent_space_tsne_pca.png')
    plt.close()
    
    log.info("Saved all 6 figures to %s", PLOTS_DIR)


if __name__ == '__main__':
    run_all_latent_oversampling_experiments()
