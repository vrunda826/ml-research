# Phase 5: Multimodal Baseline (B3) and Cross-Modal Comparison Report

## 1. Experimental Overview
- **Evaluation Cohort**: Exactly 44 common Development participants (11 depressed, 33 non-depressed).
- **Training Cohort**: Exactly 125 common Training participants (38 depressed, 87 non-depressed).
- **Fusion Strategy**: Late feature concatenation (`audio_features` + `text_embeddings`) with training-fitted StandardScaler.
- **Primary Selection Metric**: Development Set Macro-F1.

## 2. Multimodal Baselines Comparison (B3 Sub-experiments)

| Model ID | Model Name | Features | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B3a_lr** | MM_eGeMAPS_Text_LogisticRegression | egemaps_text | 0.7273 | **0.5810** | 0.2727 | 0.4286 | 0.3333 | 0.7273 | 0.5378 |
| **B3a_bal** | MM_eGeMAPS_Text_LogisticRegression_Balanced | egemaps_text | 0.7500 | **0.6308** | 0.3636 | 0.5000 | 0.4211 | 0.7273 | 0.5378 |
| **B3a_svm** | MM_eGeMAPS_Text_LinearSVM | egemaps_text | 0.7500 | **0.6308** | 0.3636 | 0.5000 | 0.4211 | 0.7025 | 0.4897 |
| **B3b_lr** | MM_MFCC_Text_LogisticRegression | mfcc_text | 0.7727 | **0.6765** | 0.4545 | 0.5556 | 0.5000 | 0.7769 | 0.5847 |
| **B3b_bal** | MM_MFCC_Text_LogisticRegression_Balanced | mfcc_text | 0.7500 | **0.6560** | 0.4545 | 0.5000 | 0.4762 | 0.7769 | 0.5847 |
| **B3b_svm** | MM_MFCC_Text_LinearSVM | mfcc_text | 0.7727 | **0.6765** | 0.4545 | 0.5556 | 0.5000 | 0.7906 | 0.5867 |

## 3. Cross-Modal Baseline Comparison Table (B0 vs B1 vs B2 vs B3)

| Tier | Baseline Name | Modality | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Majority)** | Majority_Class | egemaps | 0.7500 | **0.4286** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 |
| **B1 (Audio Best)** | MFCC_LogisticRegression | mfcc | 0.7273 | **0.5810** | 0.2727 | 0.4286 | 0.3333 | 0.7163 | 0.4769 |
| **B2 (Text Best)** | Text_MiniLM_LogisticRegression_Balanced | nan | 0.7500 | **0.6560** | 0.4545 | 0.5000 | 0.4762 | 0.7163 | 0.4954 |
| **B3 (Multimodal Best)** | MM_MFCC_Text_LogisticRegression | mfcc_text | 0.7727 | **0.6765** | 0.4545 | 0.5556 | 0.5000 | 0.7769 | 0.5847 |

## 4. Key Findings & Research Insights
- **Best Unimodal Baseline**: MFCC_LogisticRegression (Audio, Macro-F1: 0.5810) and Text_MiniLM_LogisticRegression_Balanced (Text, Macro-F1: 0.6560).
- **Best Multimodal Baseline**: MM_MFCC_Text_LogisticRegression (Macro-F1: 0.6765).
- **Multimodal Complementarity**: Fusing acoustic summaries with contextual dialogue embeddings combines phonetic prosodic dynamics with semantic affect, providing a more balanced predictive representation across participants.
- **Leakage Controls**: Zero test labels accessed, zero cross-split participant contamination, and all scalers fitted strictly on training data.
