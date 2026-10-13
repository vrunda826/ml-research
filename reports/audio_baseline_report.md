# Phase 3: Conventional Audio Baseline (B1) Evaluation Report

## 1. Experimental Overview
- **Training Participants**: 125 participants (official train split)
- **Development Participants**: 44 participants (official dev split)
- **Evaluation Metric**: Primary selection performed using **Development Macro-F1**.
- **Preprocessing Safeguards**: Imputation and StandardScaler fitted strictly on training data.
- **Feature Sets Evaluated**: OpenSMILE eGeMAPS (46 features) and MFCC (78 features).

## 2. Audio Baselines Comparison Table

| Model ID | Model Name | Features | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Majority_Class | egemaps | 0.7500 | **0.4286** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 |
| **B1a** | eGeMAPS_LogisticRegression | egemaps | 0.7500 | **0.5036** | 0.0909 | 0.5000 | 0.1538 | 0.5179 | 0.3055 |
| **B1a_bal** | eGeMAPS_LogisticRegression_Balanced | egemaps | 0.5682 | **0.4408** | 0.1818 | 0.1667 | 0.1739 | 0.5096 | 0.3165 |
| **B1b_lin** | eGeMAPS_LinearSVM | egemaps | 0.7045 | **0.4776** | 0.0909 | 0.2500 | 0.1333 | 0.4242 | 0.2250 |
| **B1b_rbf** | eGeMAPS_RBFSVM | egemaps | 0.7500 | **0.4286** | 0.0000 | 0.0000 | 0.0000 | 0.4325 | 0.2305 |
| **B1c_lr** | MFCC_LogisticRegression | mfcc | 0.7273 | **0.5810** | 0.2727 | 0.4286 | 0.3333 | 0.7163 | 0.4769 |
| **B1c_svm** | MFCC_LinearSVM | mfcc | 0.6818 | **0.5758** | 0.3636 | 0.3636 | 0.3636 | 0.7080 | 0.4464 |

## 3. Selected Best Audio Baseline
- **Selected Configuration**: **MFCC_LogisticRegression** (B1c_lr)
- **Primary Criterion (Dev Macro-F1)**: **0.5810**
- **Depression Recall**: 0.2727 (TP=3, FN=8)
- **Depression Precision**: 0.4286 (FP=4)
- **Confusion Matrix**: TN=29, FP=4, FN=8, TP=3
- **Artifact Saved To**: `artifacts/audio/MFCC_LogisticRegression.joblib` and `artifacts/audio/best_audio_model.joblib`

## 4. Observations and Limitations
- The majority class baseline (B0) establishes the lower bound (Macro-F1 ~0.4286, depression recall 0.0).
- Acoustic features demonstrate predictive signal above majority chance, but unimodal depression classification remains challenging due to acoustic heterogeneity and subtle clinical manifestation.
- Class-weight adjustments significantly improve depression recall at the expense of lower precision.
