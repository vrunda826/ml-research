# Phase 4: Conventional Text Baseline (B2) Evaluation Report

## 1. Experimental Overview
- **Training Participants**: 125 participants (official train split)
- **Development Participants**: 44 participants (official dev split)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, frozen).
- **Aggregation**: Utterance embeddings pooled via mean pooling to form one vector per participant.
- **Primary Selection Metric**: Development Set Macro-F1.

## 2. Text Baselines Comparison Table

| Model ID | Model Name | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Majority_Class | 0.7500 | **0.4286** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 |
| **B2a** | Text_MiniLM_LogisticRegression | 0.7273 | **0.6118** | 0.3636 | 0.4444 | 0.4000 | 0.7163 | 0.4954 |
| **B2a_bal** | Text_MiniLM_LogisticRegression_Balanced | 0.7500 | **0.6560** | 0.4545 | 0.5000 | 0.4762 | 0.7163 | 0.4954 |
| **B2b_lin** | Text_MiniLM_LinearSVM | 0.6818 | **0.5471** | 0.2727 | 0.3333 | 0.3000 | 0.6556 | 0.3889 |
| **B2b_rbf** | Text_MiniLM_RBFSVM | 0.7500 | **0.4286** | 0.0000 | 0.0000 | 0.0000 | 0.7906 | 0.5011 |

## 3. Selected Best Text Baseline
- **Selected Configuration**: **Text_MiniLM_LogisticRegression_Balanced** (B2a_bal)
- **Primary Criterion (Dev Macro-F1)**: **0.6560**
- **Depression Recall**: 0.4545 (TP=5, FN=6)
- **Depression Precision**: 0.5000 (FP=5)
- **Confusion Matrix**: TN=28, FP=5, FN=6, TP=5
- **Artifact Saved To**: `artifacts/text/best_text_model.joblib`

## 4. Linguistic Observations and Known Limitations
- **Absence of Speaker Annotations**: In the lightweight transcript schema, utterances are not explicitly tagged with speaker identities (Ellie vs. Participant). Embeddings pool dialogue across the session. The model captures overall session semantic content rather than isolated participant speech.
- **Preserved Clinical Vocabulary**: Text cleaning conservatively retained all negations, symptom keywords, and affective terms without stopword deletion.
- **Model Comparison**: Contextual sentence embeddings provide strong semantic representations for depression detection, establishing a solid unimodal benchmark for multimodal fusion.
