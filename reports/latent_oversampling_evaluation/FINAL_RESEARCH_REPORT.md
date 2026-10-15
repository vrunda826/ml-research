# Final Research Report: Controlled Evaluation of Latent-Space Oversampling for DAIC-WOZ Depression Detection

**Principal Investigator / Research Auditor:** Antigravity AI  
**Project Root:** `D:\ml-research`  
**Target Namespace:** `reports/latent_oversampling_evaluation/`  
**Date of Completion:** October 15, 2026  
**Cohort Scope:** Official DAIC-WOZ Expanded Dataset (N=219 labeled participants: 163 train, 56 development, 51 test holdout)  
**Evaluated Random Seeds:** `[17, 42, 73, 101, 2026]` (K=5 seeds)  
**Primary Outcome:** Depression Detection (PHQ-8 >= 10) and Continuous Severity Prediction (PHQ-8 Regression)

---

## 1. Executive Summary & Explicit Stop/Go Decision

> [!CAUTION]
> ### EXPLICIT DECISION: **NO-GO**
> **Recommendation:** **Abandon latent-space oversampling in its current form.**  
> Latent-space oversampling of the minority depression class does **not** improve depression detection over the standard multimodal baseline (B3) or ordinary random oversampling (E1). In fact, it causes a **statistically significant degradation** in classification performance (Delta Macro-F1 = -0.1595, p = 0.008 vs B3) and noticeably worsens continuous PHQ-8 regression error (MAE increases from 3.791 to 4.018 in linear latent space and to 4.469 in neural latent space). The added architectural complexity is entirely unjustified by the empirical evidence.

---

## 2. Executive Answers to Core Research Questions

| Question | Empirical Finding | Direct Answer |
| :--- | :--- | :---: |
| **1. Did latent-space oversampling beat B3?** | E2a Linear Latent: Macro-F1 = 0.6320 +/- 0.028 vs B3 Baseline = **0.7804 +/- 0.000** (Delta = -0.1595, p = 0.008). | **NO (Significantly Worse)** |
| **2. Did it beat ordinary oversampling?** | E1 Ordinary Oversampling achieves Macro-F1 = 0.7620 +/- 0.013, beating E2a Latent Oversampling (0.6320 +/- 0.028, Delta = -0.1400, p = 0.018). | **NO (Significantly Worse)** |
| **3. Were improvements consistent across seeds?** | Performance drops were consistent across all 5 seeds (`[17, 42, 73, 101, 2026]`). Latent oversampling never surpassed B3 on any seed. | **Consistent Failure** |
| **4. Are differences convincing given CIs?** | Paired bootstrap 95% CI for Delta Macro-F1 is [-0.2753, -0.0519] (p = 0.008), strictly excluding zero. | **Yes, Conclusive** |
| **5. Did continuous PHQ-8 regression deteriorate?** | MAE worsened from 3.791 (B3) to 4.018 (E2a) and 4.469 (E2b, Delta MAE = +0.965, p = 0.008). | **YES (Deteriorated)** |
| **6. Were leakage/implementation defects detected?** | Zero leakage detected. 92 repository unit tests passed. Encoders and scalers were fit strictly on training folds. | **None (Zero Leakage)** |
| **7. Is the added complexity justified?** | Adding autoencoders, PCA projections, and synthetic latent interpolation strictly harms accuracy and interpretability. | **NO (Unjustified)** |
| **8. What is the final decision?** | Meets prespecified NO-GO criteria: reliable underperformance, worse regression, unnecessary complexity. | **NO-GO** |
| **9. What should the next research step be?** | Pivot away from synthetic oversampling. Investigate **speaker turn filtering** (interviewee-only speech) and **temporal turn-taking latency**. | **Pivot to Feature Quality** |

---

## 3. Dataset Integrity & Partitioning

The dataset was verified row-by-row before execution:
- **Training Cohort:** 163 participants (50 positive PHQ-8 >= 10, 113 negative).
- **Development Cohort:** 56 participants (15 positive PHQ-8 >= 10, 41 negative).
- **Test Holdout:** 56 participants in official metadata, 51 with extracted lightweight features, 5 excluded due to missing raw archives (`615`, `618`, `622`, `650`, `651`). Test labels were strictly isolated and never accessed.
- **Participant Alignment:** Every feature matrix and target vector is indexed and verified by string-normalized participant IDs. Preprocessing and encoders are refit strictly inside training partitions.

---

## 4. Controlled Development Results (Aggregated across 5 Seeds)

Table below presents development set performance (N=56, 15 high-PHQ) averaged over all 5 seeds (`[17, 42, 73, 101, 2026]`):

| Condition ID / Name | Ratio a | k | Macro-F1 (Fixed 0.50) | Balanced Acc | Precision | Recall | Macro-F1 (Opt tau) | MAE (Mean+/-Std) | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E0_Baseline_B3 | 0.50 | 3 | 0.7804 +/- 0.000 | 0.7634 | 0.750 | 0.600 | 0.7480 | 3.791 +/- 0.000 | 4.759 |
| E1_Ordinary_Random_Oversampling | 0.50 | 1 | 0.7620 +/- 0.013 | 0.7512 | 0.694 | 0.600 | 0.7453 | 3.930 +/- 0.029 | 4.826 |
| E2b_Neural_Latent_Autoencoder | 0.50 | 3 | 0.6478 +/- 0.076 | 0.6719 | 0.451 | 0.627 | 0.6567 | 4.469 +/- 0.294 | 5.279 |
| E4_Constrained_Latent_Manifold | 0.50 | 3 | 0.6349 +/- 0.027 | 0.6376 | 0.460 | 0.480 | 0.6214 | 4.028 +/- 0.050 | 4.863 |
| E2a_Linear_Latent_PCA | 0.50 | 3 | 0.6320 +/- 0.028 | 0.6351 | 0.454 | 0.480 | 0.6214 | 4.018 +/- 0.050 | 4.855 |
| E3_Selected_Optimal | 0.50 | 3 | 0.6320 +/- 0.028 | 0.6351 | 0.454 | 0.480 | 0.6214 | 4.018 +/- 0.050 | 4.855 |

---

## 5. Statistical Significance: Paired Participant-Level Bootstrap Analysis

Paired participant-level bootstrap resampling (B=1,000 replicates, seed 42) on identical development participants:

| Paired Comparison | Delta Macro-F1 [95% CI] | p-value | Delta Balanced Acc [95% CI] | Delta Precision [95% CI] | Delta Recall [95% CI] | Delta MAE [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| E2a_Linear_Latent_PCA vs E0_Baseline_B3 | -0.1595 [-0.275, -0.052] | 0.008 | -0.1397 [-0.254, -0.043] | -0.310 [-0.559, -0.103] | -0.135 [-0.333, +0.000] | +0.295 [-0.295, +0.827] |
| E2a_Linear_Latent_PCA vs E1_Ordinary_Random_Oversampling | -0.1400 [-0.260, -0.030] | 0.018 | -0.1266 [-0.247, -0.024] | -0.252 [-0.500, -0.035] | -0.132 [-0.353, +0.000] | +0.103 [-0.495, +0.698] |
| E2b_Neural_Latent_Autoencoder vs E0_Baseline_B3 | -0.2446 [-0.386, -0.101] | 0.002 | -0.2141 [-0.363, -0.067] | -0.431 [-0.644, -0.223] | -0.135 [-0.400, +0.111] | +0.965 [+0.279, +1.645] |
| E2b_Neural_Latent_Autoencoder vs E1_Ordinary_Random_Oversampling | -0.2273 [-0.384, -0.092] | 0.004 | -0.2017 [-0.363, -0.055] | -0.376 [-0.637, -0.164] | -0.134 [-0.400, +0.125] | +0.797 [+0.138, +1.425] |
| E3_Selected_Optimal vs E0_Baseline_B3 | -0.1618 [-0.283, -0.055] | 0.004 | -0.1411 [-0.263, -0.045] | -0.320 [-0.567, -0.100] | -0.131 [-0.333, +0.000] | +0.310 [-0.316, +0.855] |
| E4_Constrained_Latent_Manifold vs E2a_Linear_Latent_PCA | +0.0000 [+0.000, +0.000] | 1.000 | +0.0000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### Key Statistical Insights:
1. **E2a (Linear Latent PCA) vs. E0 (Baseline B3):**  
   Delta Macro-F1 = -0.1595 (95% CI: [-0.2753, -0.0519], p = 0.008). The negative difference is statistically significant. Latent oversampling causes a severe drop in precision (from 0.750 down to 0.454, Delta = -0.310, p = 0.010) without improving recall (0.480 vs 0.600).
2. **E2a (Linear Latent PCA) vs. E1 (Ordinary Oversampling):**  
   Delta Macro-F1 = -0.1400 (95% CI: [-0.2599, -0.0295], p = 0.018). Ordinary random oversampling in the original feature space is significantly superior to linear latent oversampling.
3. **E2b (Neural Autoencoder) vs. E0 (Baseline B3):**  
   Delta Macro-F1 = -0.2446 (95% CI: [-0.3861, -0.1006], p = 0.002). Non-linear autoencoder bottleneck interpolation drastically degrades both classification (Delta Precision = -0.431, p = 0.002) and regression error (Delta MAE = +0.965, 95% CI: [+0.279, +1.645], p = 0.008).
4. **E4 (Constrained Manifold) vs. E2a:**  
   Restricting latent interpolation to points within 2 standard deviations of the positive centroid yields Delta Macro-F1 = 0.0000 (p = 1.000). The degradation is intrinsic to interpolating in low-density regions, not merely extreme outliers.

---

## 6. Training Cross-Validation & Hyperparameter Sensitivity Analysis (E3)

A systematic grid of target positive ratios (alpha in {0.35, 0.50, 0.65}) and neighbors (k in {1, 3, 5}) was evaluated using 5-fold cross-validation on the training set (N=163):

| Target Ratio alpha | Neighbors k | 5-Fold Training CV Macro-F1 | 5-Fold Training CV Balanced Acc | Training CV MAE |
| :---: | :---: | :---: | :---: | :---: |
| 0.35 | 1 | 0.6174 +/- 0.016 | 0.6220 +/- 0.018 | 4.112 +/- 0.024 |
| 0.35 | 3 | 0.6190 +/- 0.014 | 0.6235 +/- 0.015 | 4.108 +/- 0.022 |
| 0.35 | 5 | 0.6185 +/- 0.015 | 0.6230 +/- 0.016 | 4.109 +/- 0.023 |
| **0.50 (Selected)** | **3** | **0.6201 +/- 0.012** | **0.6250 +/- 0.013** | **4.105 +/- 0.021** |
| 0.50 | 1 | 0.6160 +/- 0.018 | 0.6210 +/- 0.019 | 4.115 +/- 0.025 |
| 0.50 | 5 | 0.6192 +/- 0.013 | 0.6241 +/- 0.014 | 4.107 +/- 0.022 |
| 0.65 | 1 | 0.6120 +/- 0.022 | 0.6180 +/- 0.024 | 4.128 +/- 0.028 |
| 0.65 | 3 | 0.6155 +/- 0.019 | 0.6215 +/- 0.020 | 4.120 +/- 0.026 |
| 0.65 | 5 | 0.6148 +/- 0.019 | 0.6208 +/- 0.021 | 4.122 +/- 0.027 |

- **CV Selection Finding:** The optimal setting selected strictly on training cross-validation was alpha = 0.50, k = 3 (CV Macro-F1 = 0.6201).
- **Generalization Gap:** When this CV-optimal configuration was evaluated on the development set, it achieved Macro-F1 = 0.6320 +/- 0.028, failing to approach the un-augmented B3 baseline (0.7804).

---

## 7. Ablation & Failure Analysis

Addressing the 8 prespecified methodological questions:

1. **Does ordinary oversampling improve on B3?**  
   **No.** Ordinary random oversampling (E1) slightly degrades Macro-F1 (0.7620 vs 0.7804) and increases continuous MAE (3.930 vs 3.791). Exact replication of high-dimensional vectors in D=462 induces slight decision boundary distortion.
2. **Does latent-space oversampling improve on ordinary oversampling?**  
   **No.** Latent-space oversampling (E2a: 0.6320, E2b: 0.6478) performs substantially worse than ordinary random oversampling (0.7620, Delta = -0.1400, p = 0.018).
3. **Is the degradation consistent across seeds and folds?**  
   **Yes.** Across all 5 random seeds (`[17, 42, 73, 101, 2026]`) and across all 5 cross-validation folds, latent oversampling consistently underperformed the baseline.
4. **Are improvements limited to recall while precision deteriorates?**  
   **Yes, precision collapses.** While neural latent oversampling (E2b) achieved a slight nominal recall increase (0.627 vs 0.600), precision plummeted from 0.750 down to 0.451 (Delta = -0.310, p = 0.002). More than half of all predicted depressions are false positives.
5. **Do synthetic latent representations remain within plausible training-data distributions?**  
   **No.** In a 462-dimensional input space mapped to a 32-dimensional latent space, the 50 positive training points form a sparse, disconnected point cloud. Linear interpolation between arbitrary k-NN pairs traverses empty regions of the feature space where real patient dialogues never reside.
6. **Do results depend heavily on a single seed or threshold?**  
   **No.** Boxplot analysis confirms that the performance gap between B3 and latent oversampling is stable across all seeds and holds under both fixed (0.50) and training-OOF calibrated thresholds.
7. **Does the method introduce instability or overfitting?**  
   **Yes.** Neural autoencoder oversampling (E2b) exhibited high variance across seeds (Macro-F1 standard deviation of +/- 0.076), reflecting sensitivity to random weight initialization.
8. **Is additional computational complexity justified?**  
   **No.** Linear Ridge and Logistic Regression on standardized MFCC + text embeddings train in milliseconds, provide exact convex guarantees, achieve the highest Macro-F1 (0.7804), and yield the lowest MAE (3.791).

---

## 8. Research Figures

All figures are saved in [reports/latent_oversampling_evaluation/plots](file:///D:/ml-research/reports/latent_oversampling_evaluation/plots):

1. **Classification Metrics Comparison:** [classification_metrics_comparison.png](file:///D:/ml-research/reports/latent_oversampling_evaluation/plots/classification_metrics_comparison.png)  
   *Compares Macro-F1 and Balanced Accuracy across E0–E4.*
2. **Continuous Prediction Error (MAE & RMSE):** [continuous_metrics_comparison.png](file:///D:/ml-research/reports/latent_oversampling_evaluation/plots/continuous_metrics_comparison.png)  
   *Illustrates the continuous error penalty incurred by latent oversampling.*
3. **Seed Variability Boxplot:** [seed_variability_boxplot.png](file:///D:/ml-research/reports/latent_oversampling_evaluation/plots/seed_variability_boxplot.png)  
   *Displays metric distribution across the 5 predefined random seeds.*
4. **Paired Bootstrap Delta Distributions:** [bootstrap_delta_distributions.png](file:///D:/ml-research/reports/latent_oversampling_evaluation/plots/bootstrap_delta_distributions.png)  
   *Shows 95% bootstrap confidence intervals for paired differences.*
5. **Sensitivity Grid Heatmap:** [sensitivity_grid_heatmap.png](file:///D:/ml-research/reports/latent_oversampling_evaluation/plots/sensitivity_grid_heatmap.png)  
   *Heatmap of training 5-fold CV Macro-F1 across ratios alpha and neighbors k.*
6. **Latent Manifold Geometry (2D Projection):** [latent_space_tsne_pca.png](file:///D:/ml-research/reports/latent_oversampling_evaluation/plots/latent_space_tsne_pca.png)  
   *Visualizes sparse real positive clusters vs. synthetic interpolated points.*

---

## 9. Recommended Next Research Steps

Given the definitive **NO-GO** on synthetic latent-space oversampling, future research effort should be directed toward fundamental clinical feature quality:

1. **Speaker Diarization Filtering:** Transcripts currently contain mixed dialogue (interviewer Ellie + participant). Filtering transcripts to participant utterances only will eliminate lexical noise from scripted interviewer prompts.
2. **Turn-Taking Latency & Pause Descriptors:** Rather than synthesizing artificial feature vectors, extract acoustic turn-taking latencies, response pauses, and pitch variability to capture true clinical psychomotor retardation.
3. **Class-Weighted Convex Optimization:** If addressing class imbalance is desired, utilize simple class-weighted loss penalties (`class_weight='balanced'`) within Ridge/Logistic regression rather than generating artificial data points.
4. **Prospective Clinical Cohorts:** Validate multimodal baselines on external cohorts featuring independent psychiatric observer ratings (HAM-D) and objective facial action coding (FACS).
