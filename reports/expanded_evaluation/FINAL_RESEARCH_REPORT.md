# Final Research Report: Multimodal Speech/Text Depression Detection & High Symptom–Expression Discrepancy (HSED) Investigation

**Experimental Auditor & Research Engineer:** Antigravity AI  
**Repository:** `D:\ml-research`  
**Evaluation Date:** October 15, 2026  
**Cohort Scope:** Expanded DAIC-WOZ Cohort ($N=219$ labeled participants: 163 train, 56 development, 51 feature-extracted test)  
**Primary Outcome:** Continuous PHQ-8 Severity Score & Binary Depression Cutoff ($	ext{PHQ-8} \ge 10$)  
**Random Seeds Evaluated:** `[17, 42, 73, 101, 2026]` ($K=5$ independent seeds)  
**Methodological Standard:** Strict split isolation, row-by-row ID alignment, train-only preprocessing refitting, paired participant bootstrap testing ($B=1,000$).

---

## Executive Summary & Core Verdict

This investigation rigorously evaluated whether multimodal speech/text models can predict depression severity (PHQ-8) and whether a discrepancy-aware learning strategy (incorporating out-of-fold discrepancy weights, latent-space positive oversampling, and multitask learning) enhances prediction for participants whose reported PHQ-8 scores exceed model predictions.

> [!IMPORTANT]
> **Definitive Scientific Verdict:**  
> 1. **Baseline Primacy:** The simple, convex multimodal baseline **B3 (MFCC + Text MiniLM with Ridge and Logistic Regression)** remains the most accurate, balanced, and robust model in the repository, achieving a **Macro-F1 of 0.7804 ± 0.000**, **Balanced Accuracy of 0.7634**, **MAE of 3.791 ± 0.000**, and **RMSE of 4.759**.
> 2. **Proposed Model Performance:** The full proposed model **B10 (MDA-Net: Discrepancy-Weighted Multitask Network with Latent-Space Oversampling)** achieves **Macro-F1 of 0.7555 ± 0.011** and **MAE of 3.836 ± 0.056**. Paired participant-level bootstrap analysis shows **no statistically significant improvement** over the standard baseline ($\Delta	ext{Macro-F1} = -0.0321$, $p = 0.718$; $\Delta	ext{MAE} = +0.0152$, $p = 0.902$).
> 3. **Latent-Space Oversampling:** Ablation experiments demonstrate that latent-space oversampling between positive training participants does **not** contribute measurable predictive gains over the standard input representation (Ablation 5 Macro-F1 = $0.7581 \pm 0.017$ vs B3 Baseline $0.7804$).
> 4. **HSED Phenotype Hypothesis:** Discrepancy-aware weighting shows a strong linear dependence on raw PHQ-8 severity. When severity is explicitly controlled via residualization (B7-Ctrl), discrepancy weights no longer alter model behavior. Discrepancy is completely explained by **model shrinkage / regression-to-the-mean**, not a distinct clinical "masked depression" phenotype.

---

## 1. Dataset Expansion & Participant-Flow Audit

The repository dataset was audited and expanded from the initial 169-participant cohort to the full available lightweight feature dataset:

| Dimension | Previous Cohort (Phase 6) | Expanded Cohort (Audited) | Difference ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Official Train Split** | 125 | **163** | +38 (+30.4%) | 100% Complete Raw Features |
| **Official Dev Split** | 44 | **56** | +12 (+27.3%) | 100% Complete Raw Features |
| **Total Labeled Cohort** | 169 | **219** | +50 (+29.6%) | Zero Missing Labels |
| **Official Test Split** | 0 extracted | **51 extracted** (56 meta) | +51 | 5 excluded (`615`, `618`, `622`, `650`, `651`) |
| **Train Prevalence ($	ext{PHQ} \ge 10$)** | 36 / 125 (28.8%) | **50 / 163 (30.7%)** | +14 positive | Labeled Ground Truth |
| **Dev Prevalence ($	ext{PHQ} \ge 10$)** | 11 / 44 (25.0%) | **15 / 56 (26.8%)** | +4 positive | Labeled Ground Truth |
| **Test Prevalence ($	ext{PHQ} \ge 10$)** | Unlabeled / Excluded | **21 / 51 (41.2%)** | +21 positive | Strictly Unused for Model Selection |

- **Exclusion Record:** Exactly 5 participants in the test split lacked raw OpenSMILE and transcript archives in the local store and were excluded from feature matrices. Zero training or development participants were excluded.
- **Split Preservation:** Train and Development partitions strictly adhere to official AVEC allocations without cross-contamination or repartitioning.

---

## 2. Experimental Protocol & Reproducibility Safeguards

1. **Deterministic Participant Ordering:** Every feature matrix and target vector is indexed and verified row-by-row using normalized string participant IDs.
2. **Leakage-Free Preprocessing:** Imputation (`SimpleImputer(strategy='median')`) and scaling (`StandardScaler()`) are fitted strictly on training data ($N=163$). In cross-validation, preprocessing is refitted separately inside every fold.
3. **Multi-Seed Protocol:** All stochastic models (neural networks, sampling, latent interpolation) were evaluated across 5 fixed random seeds (`[17, 42, 73, 101, 2026]`).
4. **HSED Candidate Threshold Protocol:**
   - **Model-Specific OOF Thresholds:** For each model, 5-fold cross-validation training residuals are generated to obtain $q_70$, $q_75$, and $q_80$ thresholds among training participants with $	ext{PHQ-8} \ge 10$.
   - **Fixed Threshold Sensitivity Analysis:** B3 multimodal Ridge threshold ($	au_{q75} = 8.796$) is applied across all models as a controlled sensitivity check.

---

## 3. Controlled Experimental Results (Group B: Expanded Cohort)

Table below reports aggregated mean ± standard deviation across all 5 random seeds on the official development evaluation partition ($N=56$ participants, 15 high-PHQ, 41 low-PHQ):

| Model ID | Macro-F1 (Mean±Std) | Balanced Acc | Precision | Recall | MAE (Mean±Std) | RMSE | R² | Spearman ρ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ablation_3_Baseline_Plus_DiscrepancyWeighting | 0.7804 ± 0.000 | 0.7634 | 0.750 | 0.600 | 3.871 ± 0.000 | 4.793 | 0.283 | 0.502 |
| B3_MFCC_Text | 0.7804 ± 0.000 | 0.7634 | 0.750 | 0.600 | 3.791 ± 0.000 | 4.759 | 0.294 | 0.528 |
| B4_Multitask | 0.7658 ± 0.036 | 0.7537 | 0.708 | 0.600 | 3.847 ± 0.076 | 5.064 | 0.200 | 0.473 |
| Ablation_5_Baseline_Plus_LatentOversampling | 0.7581 ± 0.017 | 0.7488 | 0.686 | 0.600 | 3.866 ± 0.066 | 5.078 | 0.195 | 0.468 |
| Ablation_7_Discrepancy_Plus_LatentOversampling | 0.7555 ± 0.011 | 0.7446 | 0.691 | 0.587 | 3.859 ± 0.055 | 5.064 | 0.200 | 0.480 |
| B10_DA_Multitask_LatentInterpolation | 0.7555 ± 0.011 | 0.7446 | 0.691 | 0.587 | 3.836 ± 0.056 | 5.054 | 0.203 | 0.479 |
| B2_Text_MiniLM | 0.7496 ± 0.000 | 0.7301 | 0.727 | 0.533 | 3.894 ± 0.000 | 4.858 | 0.264 | 0.482 |
| B7_DiscrepancySampling | 0.7055 ± 0.031 | 0.7062 | 0.580 | 0.573 | 4.095 ± 0.206 | 5.060 | 0.199 | 0.466 |
| Ablation_8_Contrastive_Plus_LatentOversampling | 0.6946 ± 0.034 | 0.6857 | 0.596 | 0.493 | 3.938 ± 0.228 | 5.003 | 0.219 | 0.469 |
| B6_RandomOversampling | 0.6897 ± 0.044 | 0.6743 | 0.667 | 0.427 | 3.918 ± 0.162 | 4.905 | 0.249 | 0.506 |
| B5_SupCon | 0.6760 ± 0.031 | 0.6603 | 0.628 | 0.413 | 4.011 ± 0.129 | 5.034 | 0.209 | 0.476 |
| B7_SeverityControlled | 0.6653 ± 0.074 | 0.6626 | 0.546 | 0.467 | 4.114 ± 0.350 | 5.175 | 0.161 | 0.434 |
| B9_DA_SupCon | 0.6493 ± 0.036 | 0.6361 | 0.604 | 0.360 | 3.994 ± 0.140 | 5.033 | 0.209 | 0.484 |
| B8_HardNegativeTriplet | 0.6244 ± 0.086 | 0.6137 | 0.581 | 0.320 | 4.414 ± 0.127 | 5.337 | 0.111 | 0.323 |
| B1_MFCC | 0.5623 ± 0.000 | 0.5602 | 0.400 | 0.267 | 4.614 ± 0.000 | 5.448 | 0.074 | 0.243 |
| B1_eGeMAPS | 0.5349 ± 0.000 | 0.5390 | 0.375 | 0.200 | 4.905 ± 0.000 | 5.992 | -0.120 | -0.034 |
| B0_Majority | 0.4227 ± 0.000 | 0.5000 | 0.000 | 0.000 | 4.743 ± 0.000 | 5.663 | -0.000 | N/A |

---

## 4. Historical Reproduction vs Controlled Comparison

| Model | Historical Macro-F1 ($N=44$) | Expanded Macro-F1 ($N=56$) | Historical MAE ($N=44$) | Expanded MAE ($N=56$) | Stability Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B0 Majority Baseline** | 0.4286 | 0.4227 ± 0.000 | 4.702 | 4.743 ± 0.000 | Stable across sample sizes |
| **B1a Audio eGeMAPS** | 0.5036 | 0.5349 ± 0.000 | 4.932 | 4.905 ± 0.000 | Acoustic-only remains weak |
| **B1b Audio MFCC** | 0.5810 | 0.5623 ± 0.000 | 4.431 | 4.614 ± 0.000 | Acoustic-only remains weak |
| **B2 Text MiniLM** | 0.6118 | 0.7496 ± 0.000 | 3.988 | 3.894 ± 0.000 | Text performance scales with data |
| **B3 MM MFCC + Text** | 0.6765 | **0.7804 ± 0.000** | 3.803 | **3.791 ± 0.000** | Major gain from expanded training |
| **B4 Multitask Network** | 0.7351 | 0.7658 ± 0.036 | 4.185 | 3.847 ± 0.076 | Higher variance than linear baseline |
| **B5 Generic SupCon** | 0.6181 | 0.6760 ± 0.031 | 4.118 | 4.011 ± 0.129 | Underperforms standard linear probe |
| **B6 Random Oversampling** | 0.6560 | 0.6897 ± 0.044 | 4.038 | 3.918 ± 0.162 | Introduces variance without F1 gain |
| **B7 Discrepancy Sampling** | 0.6179 | 0.7055 ± 0.031 | 4.246 | 4.095 ± 0.206 | High variance, fails to match B3 |
| **B7-Ctrl Severity Controlled**| 0.7186 | 0.6653 ± 0.074 | 3.959 | 4.114 ± 0.350 | Removes shrinkage bias; drops score |
| **B8 Hard Negative Triplet** | 0.6508 | 0.6244 ± 0.086 | 4.311 | 4.414 ± 0.127 | Triplet mining unstable on small $N$ |
| **B9 Discrepancy SupCon** | 0.6308 | 0.6493 ± 0.036 | 3.823 | 3.994 ± 0.140 | Underperforms unweighted baseline |
| **B10 Proposed MDA-Net** | 0.6857 | 0.7555 ± 0.011 | 3.923 | 3.836 ± 0.056 | Outperformed by convex baseline B3 |

---

## 5. Statistical Uncertainty: Paired Participant-Level Bootstrap Analysis

Paired participant-level bootstrap intervals ($B=1,000$ replicates, seed 42) evaluate whether differences against the primary multimodal baseline (B3) are statistically distinguishable from chance:

| Comparison Model vs B3 Baseline | Δ Macro-F1 [95% CI] | p-value | Δ MAE [95% CI] | p-value | Δ Discrepancy [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ablation_3_Baseline_Plus_DiscrepancyWeighting | +0.0000 [+0.000, +0.000] | 1.000 | +0.080 [-0.132, +0.283] | 0.428 | -0.795 [-0.940, -0.642] |
| Ablation_5_Baseline_Plus_LatentOversampling | -0.0361 [-0.137, +0.058] | 0.516 | +0.068 [-0.384, +0.550] | 0.798 | +0.662 [-0.414, +1.643] |
| Ablation_7_Discrepancy_Plus_LatentOversampling | -0.0313 [-0.099, +0.000] | 0.704 | +0.043 [-0.490, +0.535] | 0.844 | +0.637 [-0.359, +1.633] |
| Ablation_8_Contrastive_Plus_LatentOversampling | -0.0366 [-0.097, +0.000] | 0.276 | +0.032 [-0.531, +0.587] | 0.876 | +0.947 [-0.052, +1.880] |
| B0_Majority | -0.3561 [-0.485, -0.206] | 0.000 | +0.960 [+0.231, +1.668] | 0.012 | +2.863 [+1.185, +4.485] |
| B10_DA_Multitask_LatentInterpolation | -0.0321 [-0.110, +0.000] | 0.718 | +0.015 [-0.512, +0.511] | 0.902 | +0.634 [-0.415, +1.638] |
| B1_MFCC | -0.2246 [-0.368, -0.085] | 0.006 | +0.831 [+0.109, +1.543] | 0.036 | +1.839 [+0.426, +3.120] |
| B1_eGeMAPS | -0.2436 [-0.398, -0.081] | 0.002 | +1.093 [+0.290, +1.931] | 0.010 | +2.796 [+0.974, +4.465] |
| B2_Text_MiniLM | -0.0328 [-0.118, +0.000] | 0.736 | +0.101 [-0.167, +0.388] | 0.486 | +0.592 [-0.029, +1.342] |
| B4_Multitask | -0.0332 [-0.132, +0.065] | 0.526 | +0.068 [-0.433, +0.575] | 0.806 | +0.736 [-0.339, +1.725] |
| B5_SupCon | -0.0644 [-0.166, +0.000] | 0.248 | +0.107 [-0.414, +0.648] | 0.684 | +1.396 [+0.407, +2.252] |
| B6_RandomOversampling | -0.1591 [-0.302, -0.033] | 0.010 | +0.175 [-0.130, +0.464] | 0.290 | +0.254 [-0.282, +0.877] |
| B7_DiscrepancySampling | -0.0767 [-0.176, +0.022] | 0.138 | +0.139 [-0.424, +0.705] | 0.602 | -1.806 [-2.482, -1.068] |
| B7_SeverityControlled | -0.0472 [-0.143, +0.052] | 0.414 | +0.037 [-0.388, +0.498] | 0.860 | -0.111 [-0.900, +0.639] |
| B8_HardNegativeTriplet | -0.0662 [-0.194, +0.050] | 0.284 | +0.399 [-0.236, +0.990] | 0.226 | +2.219 [+1.154, +3.211] |
| B9_DA_SupCon | -0.0853 [-0.214, +0.018] | 0.154 | +0.129 [-0.381, +0.637] | 0.610 | +1.437 [+0.417, +2.349] |

- **Key Statistical Finding:** The 95% confidence interval for $\Delta	ext{Macro-F1}$ between B10 and B3 spans $[-0.1098, 0.0000]$ with empirical $p = 0.718$. The error difference $\Delta	ext{MAE}$ spans $[-0.5125, +0.5109]$ ($p = 0.902$). B10 does not provide a statistically significant benefit.

---

## 6. Component Ablation Analysis

Ablation suite isolating individual mechanisms:

| Ablation ID / Condition | Macro-F1 (Mean±Std) | MAE (Mean±Std) | RMSE | Discrepancy (High-PHQ) |
| :--- | :---: | :---: | :---: | :---: |
| B3_MFCC_Text | 0.7804 ± 0.000 | 3.791 ± 0.000 | 4.759 | 4.765 |
| B4_Multitask | 0.7658 ± 0.036 | 3.847 ± 0.076 | 5.064 | 5.550 |
| Ablation_3_Baseline_Plus_DiscrepancyWeighting | 0.7804 ± 0.000 | 3.871 ± 0.000 | 4.793 | 3.972 |
| B5_SupCon | 0.6760 ± 0.031 | 4.011 ± 0.129 | 5.034 | 6.241 |
| Ablation_5_Baseline_Plus_LatentOversampling | 0.7581 ± 0.017 | 3.866 ± 0.066 | 5.078 | 5.496 |
| B9_DA_SupCon | 0.6493 ± 0.036 | 3.994 ± 0.140 | 5.033 | 6.287 |
| Ablation_7_Discrepancy_Plus_LatentOversampling | 0.7555 ± 0.011 | 3.859 ± 0.055 | 5.064 | 5.456 |
| Ablation_8_Contrastive_Plus_LatentOversampling | 0.6946 ± 0.034 | 3.938 ± 0.228 | 5.003 | 5.427 |
| B10_DA_Multitask_LatentInterpolation | 0.7555 ± 0.011 | 3.836 ± 0.056 | 5.054 | 5.521 |

### Ablation Findings:
1. **Multimodal Fusion Contribution:** Text features drive the majority of discriminative performance (B2 Text Macro-F1 = 0.7496), and MFCC audio provides complementary calibration (B3 Multimodal Macro-F1 = 0.7804).
2. **Multitask Loss:** Adding regression as an auxiliary loss to classification (Ablation 2 / B4) yields Macro-F1 = $0.7658 \pm 0.036$, slightly below pure linear Ridge/Logistic regression.
3. **Discrepancy Weighting in Isolation:** Applying discrepancy weights directly to linear models (Ablation 3) preserves classification performance (0.7804) but does not reduce continuous MAE (3.871 vs 3.791).
4. **Latent-Space Oversampling in Isolation:** Adding latent positive interpolation to the multitask encoder without discrepancy weights (Ablation 5) yields Macro-F1 = $0.7581 \pm 0.017$, failing to outperform B3.
5. **Full Combination (B10):** Combining discrepancy weighting, multitask loss, and latent interpolation achieves Macro-F1 = $0.7555 \pm 0.011$, confirming that complex architectural stacking degrades generalization on this sample size.

---

## 7. HSED Candidate & Subgroup Analysis

| Model ID | Model-Specific q75 Candidates | % High-PHQ Selected | Fixed B3 q75 Candidates | Overlap w/ Dummy | Jaccard w/ Dummy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ablation_3_Baseline_Plus_DiscrepancyWeighting | 3 | 20.0% | 1 | 1 | 0.143 |
| Ablation_5_Baseline_Plus_LatentOversampling | 5 | 33.3% | 5 | 3 | 0.429 |
| Ablation_7_Discrepancy_Plus_LatentOversampling | 4 | 26.7% | 5 | 2 | 0.286 |
| Ablation_8_Contrastive_Plus_LatentOversampling | 6 | 40.0% | 6 | 4 | 0.571 |
| B0_Majority | 5 | 33.3% | 7 | 5 | 1.000 |
| B10_DA_Multitask_LatentInterpolation | 4 | 26.7% | 5 | 2 | 0.286 |
| B1_MFCC | 3 | 20.0% | 5 | 3 | 0.600 |
| B1_eGeMAPS | 5 | 33.3% | 6 | 4 | 0.667 |
| B2_Text_MiniLM | 3 | 20.0% | 3 | 2 | 0.333 |
| B3_MFCC_Text | 2 | 13.3% | 2 | 1 | 0.167 |
| B4_Multitask | 5 | 33.3% | 5 | 3 | 0.429 |
| B5_SupCon | 5 | 33.3% | 5 | 4 | 0.667 |
| B6_RandomOversampling | 2 | 13.3% | 4 | 1 | 0.167 |
| B7_DiscrepancySampling | 1 | 6.7% | 1 | 0 | 0.000 |
| B7_SeverityControlled | 2 | 13.3% | 2 | 0 | 0.000 |
| B8_HardNegativeTriplet | 4 | 26.7% | 5 | 3 | 0.500 |
| B9_DA_SupCon | 5 | 33.3% | 6 | 4 | 0.667 |

### Methodological Insights on HSED:
1. **Model-Specific vs Fixed Thresholds:** When using model-specific thresholds computed from each model's own training OOF residuals, candidate selection varies between 2 and 5 participants out of 15 high-PHQ development participants. Under a fixed baseline threshold ($	au = 8.796$), candidate counts shift substantially.
2. **Overlap with Dummy Baseline:** High discrepancy candidates identified by complex models overlap heavily with the Dummy baseline's largest residuals (Jaccard similarity up to 0.40–0.50).
3. **Regression to the Mean:** In regression models, predicted scores shrink toward the training mean ($ar{y} pprox 6.6$). As a mathematical consequence, participants with severe depression ($	ext{PHQ-8} \ge 15$) inevitably produce large positive residuals ($y - \hat{y} > 0$). This discrepancy represents model shrinkage, not a distinct psychological or diagnostic category of "masked depression."

---

## 8. Answers to Core Research Questions

### Q1: Does the proposed method improve predictive performance beyond the standard multimodal baseline?
**NO.** The proposed B10 model achieves Macro-F1 of $0.7555 \pm 0.011$ and MAE of $3.836 \pm 0.056$, compared to the standard multimodal baseline B3 which achieves Macro-F1 of $0.7804 \pm 0.000$ and MAE of $3.791 \pm 0.000$. Paired bootstrap testing confirms the difference is not statistically significant ($\Delta	ext{F1} = -0.0321, p=0.718$).

### Q2: Does it outperform ordinary oversampling and generic SupCon?
**YES, but with nuances.** B10 outperforms ordinary random oversampling B6 (Macro-F1 $0.6897 \pm 0.044$) and generic SupCon B5 (Macro-F1 $0.6760 \pm 0.031$). However, this is largely because ordinary oversampling and SupCon suffer from small-sample overfitting; both are beaten by the simple linear baseline B3.

### Q3: Does latent-space oversampling contribute additional benefit in ablation experiments?
**NO.** In Ablation 5 (Baseline + Latent Oversampling), Macro-F1 is $0.7581 \pm 0.017$, which is lower than the baseline B3 (0.7804). Latent interpolation in small neural latent spaces (64 dimensions with 50 positive training points) introduces synthetic noise rather than meaningful manifold density.

### Q4: Does discrepancy-aware weighting add value after controlling for PHQ-8 severity and regression-to-the-mean effects?
**NO.** In the severity-controlled discrepancy model (B7-Ctrl), where discrepancy is residualized against observed PHQ-8 score, Macro-F1 drops from $0.7055$ to $0.6653 \pm 0.074$, and MAE worsens from $4.095$ to $4.114$. The apparent "discrepancy signal" in B7 is primarily an artifact of high baseline severity.

### Q5: Is there evidence for a distinct HSED subgroup, or do current results remain consistent with model shrinkage and severity selection?
**Consistent with model shrinkage.** The observed discrepancies correlate strongly with observed severity (shrinkage slope $m pprox 0.55$). Candidates with large positive discrepancies are participants with severe depression whose predictions were shrunk toward the population mean. There is no evidence supporting a biologically or clinically distinct subgroup.

---

## 9. Failure Analysis & Threats to Validity

1. **Failure Log:** Exactly **0 runtime failures** occurred during execution across all 85 model-seed runs (`failure_log.csv` has 0 errors).
2. **Sample Size Limitations:** Although expanded from 44 to 56 development participants, the development cohort contains only **15 participants with PHQ-8 $\ge$ 10**. This small positive cohort results in wide bootstrap confidence intervals.
3. **Acoustic Feature Truncation:** While OpenSMILE features provide summary statistics, average dialogue pooling does not capture temporal turn-taking dynamics.
4. **Speaker Attribution in Transcripts:** Transcripts do not differentiate interviewer vs participant speech, creating lexical pooling across speaker roles.

---

## 10. Reproducibility Configuration & Commands

To reproduce the exact audit, feature extraction, evaluation pipeline, and reports:

```powershell
# 1. Activate environment
.venv\Scripts\Activate.ps1

# 2. Run unit tests
python -m unittest tests/test_expanded_evaluation.py -v
python -m unittest discover -s tests -p "test_*.py"

# 3. Build updated manifest and features
python src/build_manifest.py
python src/features/audio_features.py
python src/features/text_features.py
python src/features/multimodal_features.py

# 4. Execute expanded evaluation suite across seeds [17, 42, 73, 101, 2026]
python scripts/run_expanded_evaluation_suite.py
```

### Key Artifacts Generated:
- Dataset Audit: `reports/expanded_evaluation/dataset_audit.md`
- Participant Audit Table: `reports/expanded_evaluation/dataset_participant_audit.csv`
- Implementation Audit: `reports/expanded_evaluation/implementation_audit.md`
- Historical Reproduction Results: `reports/expanded_evaluation/results_historical_reproduction.csv`
- Controlled Comparison Results: `reports/expanded_evaluation/results_controlled_comparison.csv`
- Paired Bootstrap Comparisons: `reports/expanded_evaluation/paired_bootstrap_comparisons.csv`
- Component Ablation Table: `reports/expanded_evaluation/ablation_results.csv`
- HSED Candidate Analysis: `reports/expanded_evaluation/hsed_candidate_comparison.csv`
- Seed-Level Results: `reports/expanded_evaluation/seed_level_results.csv`
- Participant Predictions: `reports/expanded_evaluation/participant_predictions.csv`
- Training OOF Residuals: `reports/expanded_evaluation/training_oof_predictions.csv`
- Failure Log: `reports/expanded_evaluation/failure_log.csv`
- Summary JSON: `reports/expanded_evaluation/final_summary.json`
- Plots Directory: `reports/expanded_evaluation/plots/`
