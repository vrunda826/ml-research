# Dataset and Baseline Audit: Latent-Space Oversampling Evaluation

**Date of Audit:** October 15, 2026  
**Auditor:** Antigravity Research Engineering  
**Project Root:** `D:\ml-research`  
**Evaluation Scope:** Controlled Investigation of Latent-Space Oversampling on the Expanded DAIC-WOZ Cohort ($N=219$ labeled participants)  
**Target Namespace:** `reports/latent_oversampling_evaluation/`

---

## 1. Dataset Partitioning & Cohort Verification

The dataset has been re-audited and verified row-by-row against the official metadata splits and extracted lightweight feature directories:

| Partition | Total In Metadata | Lightweight Feature Archives | Missing Feature Archives | Labeled Participants | PHQ-8 $\ge$ 10 (Positive) | PHQ-8 < 10 (Negative) | Prevalence ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Split** | 163 | 163 | 0 | 163 | 50 | 113 | 30.67% |
| **Development Split** | 56 | 56 | 0 | 56 | 15 | 41 | 26.79% |
| **Total Labeled Cohort** | **219** | **219** | **0** | **219** | **65** | **154** | **29.68%** |
| **Test Split** | 56 | 51 | 5 (`615`, `618`, `622`, `650`, `651`) | 56 (Holdout) | 21 (Holdout) | 30 (Holdout) | 41.18% (of 51) |
| **Total Full Cohort** | **275** | **270** | **5** | **275** | **86** | **184** | **31.85%** |

### Split Integrity & Isolation Rules:
1. **Zero Partition Alteration:** Train ($N=163$) and Development ($N=56$) partitions strictly follow official AVEC allocations without random merging, pooling, or repartitioning.
2. **Zero Test Data Influence:** The test set ($N=51$ feature-extracted) is held out completely. Test labels are strictly never used for model selection, threshold selection, or representation fitting.
3. **ID Alignment:** Feature matrices and target vectors are aligned strictly using normalized participant ID strings (`pid_series`), asserting exact equality row-by-row before fitting or evaluation.

---

## 2. Audit of the Primary Baseline: Model B3 (Multimodal MFCC + Text)

- **Source Code:** `src/phase6_regression.py` (`load_modality_data`), `src/models/multimodal_baseline.py`, and `scripts/phase7_hsed_experiments.py`.
- **Modality Key:** `mm_mfcc_text` (`data/processed/features/multimodal/multimodal_mfcc_text.csv`).
- **Feature Dimensionality ($D=462$):**
  - Audio ($D=78$): OpenSMILE 2.3.0 MFCC summary statistics (mean and standard deviation computed over active conversational frames).
  - Text ($D=384$): Mean-pooled sentence embeddings extracted from conversation transcripts via frozen `sentence-transformers/all-MiniLM-L6-v2`.
- **Learned Preprocessing Pipeline:**
  - `SimpleImputer(strategy='median')`
  - `StandardScaler()`
  - Preprocessing parameters ($\mu, \sigma$, median) are fitted strictly on the training partition ($N=163$). In cross-validation, preprocessing is refitted separately inside each training fold without access to validation folds.
- **Classification Head:** `LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)`.
  - Primary decision rule: Standard probability cutoff $\hat{y} = \mathbb{I}(P(y=1|x) \ge 0.50)$ as well as training-OOF calibrated threshold.
- **Regression Head:** `Ridge(alpha=100.0)`.
- **Verified Baseline Performance on Expanded Dev Set ($N=56$, 15 high-PHQ):**
  - **Macro-F1:** **$0.7804 \pm 0.000$**
  - **Balanced Accuracy:** **$0.7634 \pm 0.000$**
  - **Precision:** $0.750$, **Recall:** $0.600$ (9 / 15 high-PHQ detected)
  - **Accuracy:** $83.93\%$ ($47 / 56$)
  - **ROC-AUC:** $0.7415$, **PR-AUC:** $0.5784$
  - **MAE:** **$3.791 \pm 0.000$**, **RMSE:** **$4.759 \pm 0.000$**, **$R^2$:** **$0.294 \pm 0.000$**
  - **Spearman $\rho$:** **$0.528 \pm 0.000$** ($p = 0.0001$)
- **Status:** **VERIFIED & FROZEN.**

---

## 3. Audit of Ordinary Random Oversampling (Model B6 / E1)

- **Source Code:** `scripts/phase7_hsed_experiments.py` (lines 246–249) and `scripts/run_expanded_evaluation_suite.py` (lines 142–156).
- **Mechanism:** Direct sample duplication with replacement in the raw 462-dimensional input feature space.
  - Identifies training positive indices `pos` ($N=50$) and negative indices `neg` ($N=113$).
  - Duplicates $N_{\text{new}} = |neg| - |pos| = 63$ positive samples with replacement: `ix = np.r_[neg, rng.choice(pos, size=63, replace=True)]`.
  - Re-fits standard `LogisticRegression` and `Ridge` on the balanced training set ($N=226$).
- **Verified Performance on Expanded Dev Set ($N=56$):**
  - **Macro-F1:** $0.6897 \pm 0.044$ (drop of $-0.0907$ vs B3)
  - **Balanced Accuracy:** $0.6743 \pm 0.042$ (drop of $-0.0891$ vs B3)
  - **Precision:** $0.667 \pm 0.023$, **Recall:** $0.427 \pm 0.101$
  - **MAE:** $3.918 \pm 0.162$, **RMSE:** $4.905 \pm 0.172$
- **Diagnosis:** In 462-dimensional feature space with small sample size, exact sample replication causes significant overfitting to duplicated training outliers, reducing generalization performance on the held-out development set.

---

## 4. Audit of Existing Latent-Space Oversampling Implementations

### 4.1 B10 Full Proposed Model (MDA-Net)
- **Source Code:** `scripts/phase7_hsed_experiments.py` (lines 262–273).
- **Latent Representation:** 64-dimensional learned representation $z$ extracted from the second hidden layer of `MultiTaskNet` (462 $\to$ 64 $\to$ ReLU $\to$ Dropout(0.25) $\to$ 64 $\to$ ReLU), trained with AdamW ($\text{lr}=0.002, \text{weight\_decay}=0.01$) on a weighted combination of BCE and MSE losses ($w_i \cdot \text{BCE} + 0.05 \cdot w_i \cdot \text{MSE}$).
- **Interpolation Scheme:**
  - For each synthetic sample, selects an anchor positive $i \in \text{pos}$ with probability proportional to OOF discrepancy weights $w_i$.
  - Finds the single nearest positive neighbor $j = \arg\min_{k \in \text{pos}, k \ne i} \|z_k - z_i\|_2$ in the 64-dimensional latent space.
  - Samples $\lambda \sim \text{Uniform}(0.15, 0.85)$.
  - Constructs $z_{\text{synth}} = z_i + \lambda (z_j - z_i)$, $y_{\text{synth}} = y_i + \lambda (y_j - y_i)$, label $= 1$, weight $= (w_i + w_j) / 2$.
  - Fits downstream `LogisticRegression` and `Ridge(alpha=10.0)` on augmented latent representations.
  - Continuous regression prediction is a fixed 50/50 blend between the raw neural regression head and the latent Ridge head: $\hat{y} = 0.5 \hat{y}_{\text{raw}} + 0.5 \hat{y}_{\text{ridge}}$.
- **Verified Performance on Expanded Dev Set ($N=56$):**
  - **Macro-F1:** $0.7555 \pm 0.011$
  - **Balanced Accuracy:** $0.7446 \pm 0.018$
  - **MAE:** $3.836 \pm 0.056$, **RMSE:** $5.054 \pm 0.074$
- **Paired Bootstrap Comparison vs B3 ($B=1,000$):**
  - $\Delta\text{Macro-F1} = -0.0321$ (95% CI: $[-0.1098, 0.0000]$, $p = 0.718$).
  - $\Delta\text{MAE} = +0.0152$ (95% CI: $[-0.5125, +0.5109]$, $p = 0.902$).
  - Not statistically significant; underperforms the baseline.

### 4.2 Existing Standalone Latent Oversampling Ablations (Ablation 5 & Ablation 8)
- **Ablation 5 (Baseline Multitask Encoder + Latent Oversampling without Discrepancy Weights):**
  - Macro-F1 = $0.7581 \pm 0.017$, MAE = $3.866 \pm 0.066$.
- **Ablation 8 (Contrastive SupCon Encoder + Latent Oversampling):**
  - Macro-F1 = $0.6946 \pm 0.034$, MAE = $3.938 \pm 0.228$.

---

## 5. Methodological Defects and Missing Elements in Previous Oversampling Studies

Our audit identified the following critical methodological gaps that prevent a conclusive assessment of pure latent-space oversampling:

1. **Confounded Implementation in B10:** In B10, latent oversampling was bundled simultaneously with discrepancy sample weights ($w_i$), multitask loss balancing, and regression prediction blending. The effect of oversampling was never cleanly isolated from loss weighting.
2. **Absence of a Non-Neural / Linear Latent-Space Baseline:** All prior latent oversampling relied on deep neural MLP encoders (trained for 100–120 epochs). With only 163 training participants, a 462 $\to$ 64 $\to$ 64 MLP has thousands of parameters and easily overfits. No simple, well-regularized linear manifold representation (e.g. PCA bottleneck or regularized linear autoencoder) was tested as an encoder for latent oversampling.
3. **No Training-Only Cross-Validation Tuning for Oversampling Hyperparameters:** Prior scripts used arbitrary fixed parameters (nearest neighbor $k=1$, target positive ratio $\alpha = 0.50$, uniform $\lambda \in [0.15, 0.85]$) without systematic cross-validation tuning.
4. **Lack of Synthetic Sample Inspection:** No audit table was generated to verify whether synthetic latent representations remain within plausible training bounds or drift away from real data density.

---

## 6. Audit Verdict & Experimental Plan

To resolve these defects and deliver a definitive verdict, we establish:
- **E0:** Frozen B3 Baseline (MFCC + Text with Ridge and Logistic Regression).
- **E1:** Ordinary Random Minority Oversampling in input feature space.
- **E2:** Clean, standalone Latent-Space Oversampling:
  - Evaluating both a well-regularized linear manifold representation (PCA / SVD bottleneck, $D=32$ or $48$) and a regularized neural encoder (MLP Autoencoder).
  - Explicitly preventing data leakage by fitting representation models strictly inside training folds.
- **E3:** Training-only 5-fold cross-validation sensitivity analysis across oversampling ratios ($\alpha \in \{0.35, 0.50, 0.65\}$) and neighborhood settings ($k \in \{1, 3, 5\}$).
- **E4:** Constrained latent-space variant (e.g., class-conditional autoencoder with reconstruction fidelity constraint).
- **Paired Bootstrap:** $B=1,000$ paired participant-level resamples on development data for all primary comparisons.
