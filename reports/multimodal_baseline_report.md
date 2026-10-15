# Phase 5: Multimodal Baseline (B3) and Cross-Modal Comparison Report

## 1. Experimental Overview
- **Evaluation Cohort**: Exactly 44 common Development participants (11 depressed, 33 non-depressed).
- **Training Cohort**: Exactly 125 common Training participants (38 depressed, 87 non-depressed).
- **Fusion Strategy**: Late feature concatenation (`audio_features` + `text_embeddings`) with training-fitted StandardScaler.
- **Primary Selection Metric**: Development Set Macro-F1.

## 2. Multimodal Baselines Comparison (B3 Sub-experiments)

| Model ID | Model Name | Features | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B3a_lr** | MM_eGeMAPS_Text_LogisticRegression | egemaps_text | 0.8214 | **0.7496** | 0.5333 | 0.7273 | 0.6154 | 0.7089 | 0.5523 |
| **B3a_bal** | MM_eGeMAPS_Text_LogisticRegression_Balanced | egemaps_text | 0.8036 | **0.7316** | 0.5333 | 0.6667 | 0.5926 | 0.7089 | 0.5523 |
| **B3a_svm** | MM_eGeMAPS_Text_LinearSVM | egemaps_text | 0.7857 | **0.6995** | 0.4667 | 0.6364 | 0.5385 | 0.7089 | 0.5615 |
| **B3b_lr** | MM_MFCC_Text_LogisticRegression | mfcc_text | 0.8393 | **0.7804** | 0.6000 | 0.7500 | 0.6667 | 0.7415 | 0.5784 |
| **B3b_bal** | MM_MFCC_Text_LogisticRegression_Balanced | mfcc_text | 0.8214 | **0.7619** | 0.6000 | 0.6923 | 0.6429 | 0.7447 | 0.5796 |
| **B3b_svm** | MM_MFCC_Text_LinearSVM | mfcc_text | 0.8036 | **0.7316** | 0.5333 | 0.6667 | 0.5926 | 0.7350 | 0.5608 |

## 3. Cross-Modal Baseline Comparison Table (B0 vs B1 vs B2 vs B3)

| Tier | Baseline Name | Modality | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Majority)** | Majority_Class | egemaps | 0.7321 | **0.4227** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2679 |
| **B1 (Audio Best)** | eGeMAPS_LogisticRegression_Balanced | egemaps | 0.6429 | **0.5625** | 0.4000 | 0.3529 | 0.3750 | 0.5496 | 0.3943 |
| **B2 (Text Best)** | Text_MiniLM_LogisticRegression | nan | 0.8214 | **0.7496** | 0.5333 | 0.7273 | 0.6154 | 0.7317 | 0.5651 |
| **B3 (Multimodal Best)** | MM_MFCC_Text_LogisticRegression | mfcc_text | 0.8393 | **0.7804** | 0.6000 | 0.7500 | 0.6667 | 0.7415 | 0.5784 |

## 4. Key Findings & Research Insights
- **Best Unimodal Baseline**: eGeMAPS_LogisticRegression_Balanced (Audio, Macro-F1: 0.5625) and Text_MiniLM_LogisticRegression (Text, Macro-F1: 0.7496).
- **Best Multimodal Baseline**: MM_MFCC_Text_LogisticRegression (Macro-F1: 0.7804).
- **Multimodal Complementarity**: Fusing acoustic summaries with contextual dialogue embeddings combines phonetic prosodic dynamics with semantic affect, providing a more balanced predictive representation across participants.
- **Leakage Controls**: Zero test labels accessed, zero cross-split participant contamination, and all scalers fitted strictly on training data.
