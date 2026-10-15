# Phase 3: Conventional Audio Baseline (B1) Evaluation Report

## 1. Experimental Overview
- **Training Participants**: 163 participants (official train split)
- **Development Participants**: 56 participants (official dev split)
- **Evaluation Metric**: Primary selection performed using **Development Macro-F1**.
- **Preprocessing Safeguards**: Imputation and StandardScaler fitted strictly on training data.
- **Feature Sets Evaluated**: OpenSMILE eGeMAPS (46 features) and MFCC (78 features).

## 2. Audio Baselines Comparison Table

| Model ID | Model Name | Features | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Majority_Class | egemaps | 0.7321 | **0.4227** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2679 |
| **B1a** | eGeMAPS_LogisticRegression | egemaps | 0.6964 | **0.5349** | 0.2000 | 0.3750 | 0.2609 | 0.5593 | 0.3910 |
| **B1a_bal** | eGeMAPS_LogisticRegression_Balanced | egemaps | 0.6429 | **0.5625** | 0.4000 | 0.3529 | 0.3750 | 0.5496 | 0.3943 |
| **B1b_lin** | eGeMAPS_LinearSVM | egemaps | 0.6786 | **0.5227** | 0.2000 | 0.3333 | 0.2500 | 0.5317 | 0.3717 |
| **B1b_rbf** | eGeMAPS_RBFSVM | egemaps | 0.7321 | **0.4227** | 0.0000 | 0.0000 | 0.0000 | 0.4016 | 0.2299 |
| **B1c_lr** | MFCC_LogisticRegression | mfcc | 0.6964 | **0.5623** | 0.2667 | 0.4000 | 0.3200 | 0.5577 | 0.3426 |
| **B1c_svm** | MFCC_LinearSVM | mfcc | 0.6964 | **0.5623** | 0.2667 | 0.4000 | 0.3200 | 0.5642 | 0.3546 |

## 3. Selected Best Audio Baseline
- **Selected Configuration**: **eGeMAPS_LogisticRegression_Balanced** (B1a_bal)
- **Primary Criterion (Dev Macro-F1)**: **0.5625**
- **Depression Recall**: 0.4000 (TP=6, FN=9)
- **Depression Precision**: 0.3529 (FP=11)
- **Confusion Matrix**: TN=30, FP=11, FN=9, TP=6
- **Artifact Saved To**: `artifacts/audio/eGeMAPS_LogisticRegression_Balanced.joblib` and `artifacts/audio/best_audio_model.joblib`

## 4. Observations and Limitations
- The majority class baseline (B0) establishes the lower bound (Macro-F1 ~0.4286, depression recall 0.0).
- Acoustic features demonstrate predictive signal above majority chance, but unimodal depression classification remains challenging due to acoustic heterogeneity and subtle clinical manifestation.
- Class-weight adjustments significantly improve depression recall at the expense of lower precision.
