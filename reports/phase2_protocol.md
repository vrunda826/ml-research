# Phase 2: Experimental Protocol Specification & Freeze

## 1. Executive Summary

This protocol formalizes the experimental design, participant splits, label definitions, leakage safeguards, and evaluation metrics for the DAIC-WOZ depression detection research project. The protocol is frozen prior to model fitting to guarantee research integrity, transparency, and reproducibility.

---

## 2. Participant Cohort & Official Split Topology

### 2.1 Context: AVEC 2019 Metadata vs. Classic DAIC-WOZ
Historical DAIC-WOZ papers (e.g., Gratch et al., 2014; DeVault et al., 2014) report a 189-participant cohort (107 Training, 35 Development, 47 Test). The official dataset package available in this project originates from the **AVEC 2019 Depression Sub-Challenge**, which extended the cohort to **275 participants** (163 Training, 56 Development, 56 Test).

Our Phase 1 structural audit of `data/processed/lightweight/` confirmed that **169 participants** have complete local lightweight features (Audio MFCC, Audio eGeMAPS, and Transcripts). All 169 participants map unambiguously into the official AVEC 2019 splits:

| Split | Official Metadata Total | Locally Extracted & Complete | Missing Local Archives | Ground-Truth Labeled |
| :--- | :---: | :---: | :---: | :---: |
| **Training (`train`)** | 163 | **125** | 38 | 125 |
| **Development (`dev`)** | 56 | **44** | 12 | 44 |
| **Test (`test`)** | 56 | **0** | 56 | 0 (No local features) |
| **Total** | **275** | **169** | **106** | **169** |

**Rule on Test Set**: No local audio or transcript features exist for the 56 test participants. Furthermore, even if test labels exist in metadata archives, **test labels are strictly barred from model selection, architecture tuning, thresholding, or evaluation during development**. The labeled development set ($N=44$) serves as the primary benchmark for model selection.

---

## 3. Label Definition & Class Distribution

### 3.1 Standardized Target Definition
The primary binary depression label is derived using the standard clinical threshold for moderate-to-severe depression on the PHQ-8:

$$\text{depression\_label} = \begin{cases} 1, & \text{if } \text{PHQ-8} \ge 10 \\ 0, & \text{if } \text{PHQ-8} < 10 \end{cases}$$

### 3.2 Audit of Official AVEC Binary Discrepancies
In official AVEC metadata (`dev_split.csv`, `train_split.csv`), the column `PHQ_Binary` differs from $\text{PHQ-8} \ge 10$ for 20 participants (13 in train, 3 in dev, 4 in test) who scored between 10 and 16, but were marked `0` due to strict clinical DSM criteria requiring specific symptom combinations. In this project, our primary target is explicitly documented and evaluated using the research standard **$\text{PHQ-8} \ge 10$** (`phq_binary_cutoff10`).

### 3.3 Class Distribution for Complete Experimental Cohort ($N=169$)
| Cohort Subset | Total ($N$) | Non-Depressed ($<10$) | Depressed ($\ge 10$) | Depression Prevalence |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 125 | 87 | 38 | **30.40%** |
| **Development Set** | 44 | 33 | 11 | **25.00%** |
| **Combined Available** | 169 | 120 | 49 | **28.99%** |

---

## 4. Leakage Prevention Rules

To ensure scientific validity, the following constraints are programmatically enforced via `src/validation.py`:

1. **Strict Participant Partitioning**:
   - Splitting is performed strictly by participant ID before feature aggregation and model fitting.
   - All utterances, acoustic frames, transcripts, and aggregated vectors for a participant remain exclusively within that participant's assigned split.
   - Train and Development sets are verified to have **zero overlap** ($\text{Train} \cap \text{Dev} = \emptyset$).
2. **Preprocessing Fitting**:
   - All learned preprocessing steps (`StandardScaler`, missing value imputers, vocabulary counters) must be fitted **strictly on training participants**.
   - The fitted transformation is applied to development participants without refitting.
   - No scaler or imputer may be fitted on the combined dataset.
3. **Target Isolation**:
   - Neither PHQ-8 scores nor derived discrepancy values ($D_i$) may ever be used as input features for classification or regression inference.
4. **Model Selection Independence**:
   - The test set is completely excluded from hyperparameter tuning, model selection, loss weighting, and feature selection.

---

## 5. Model Selection Criteria & Evaluation Metrics

### 5.1 Primary Selection Metric
- **Development Set Macro-F1**:
  $$\text{Macro-F1} = \frac{\text{F1}_{\text{depressed}} + \text{F1}_{\text{non-depressed}}}{2}$$
  Selected because the dataset exhibits class imbalance (approx. 25% depressed in dev), ensuring balanced sensitivity across both clinical classes without favoring the majority non-depressed class.

### 5.2 Complementary Metrics
- **Depression-Class Recall**: Missing individuals with elevated depression symptoms carries substantial clinical and research risk.
- **Precision-Recall AUC (PR-AUC)**: Evaluates ranking quality across decision thresholds under class imbalance using average precision.
- **Accuracy & Depression-Class F1**: Reported for contextual comparison with published literature.
- **Confusion Matrix**: Explicit counts of TN, FP, FN, and TP.

### 5.3 Metric Robustness & Edge Cases
If an evaluation subset contains only a single class, ROC-AUC and PR-AUC are undefined. In such cases, `src/metrics.py` reports `None` with an explicit reason rather than inventing a metric value.

---

## 6. High Symptom–Expression Discrepancy (HSED) Metric Design

### 6.1 Conceptual Definition
For a participant $i$ with observed $\text{PHQ8}_i$ and model-predicted $\widehat{\text{PHQ8}}_i$:

$$D_i = \text{PHQ8}_i - \widehat{\text{PHQ8}}_i$$

A positive discrepancy ($D_i > 0$) indicates that a participant reports more severe depression symptoms than their observable acoustic/textual expression predicts.

### 6.2 Safeguards and Status
1. **Out-of-Fold Estimation**: For training participants, $D_i$ must be generated via out-of-fold cross-validation, never from models trained on the same participant.
2. **Restricted Subgroup**: Discrepancy analysis is restricted to participants with $\text{PHQ8}_i \ge 10$.
3. **Threshold Definition**: Thresholds $\tau$ (e.g. 70th, 75th, 80th percentiles of training discrepancies) must be computed exclusively from training data.
4. **Current Phase Status**: In Phases 3–5, models are binary classifiers predicting depression class rather than continuous PHQ-8 scores. Therefore, continuous regression discrepancy metrics are marked as **PENDING** in this phase until dedicated regression baselines are executed.

---

## 7. Approval & Protocol Freeze

This protocol is frozen and active as of Phase 2 completion. All subsequent modeling in Phase 3, 4, and 5 adheres strictly to these rules.
