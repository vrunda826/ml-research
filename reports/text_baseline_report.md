# Phase 4: Conventional Text Baseline (B2) Evaluation Report

## 1. Experimental Overview
- **Training Participants**: 163 participants (official train split)
- **Development Participants**: 56 participants (official dev split)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, frozen).
- **Aggregation**: Utterance embeddings pooled via mean pooling to form one vector per participant.
- **Primary Selection Metric**: Development Set Macro-F1.

## 2. Text Baselines Comparison Table

| Model ID | Model Name | Accuracy | Macro-F1 | Dep. Recall | Dep. Prec | Dep. F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Majority_Class | 0.7321 | **0.4227** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2679 |
| **B2a** | Text_MiniLM_LogisticRegression | 0.8214 | **0.7496** | 0.5333 | 0.7273 | 0.6154 | 0.7317 | 0.5651 |
| **B2a_bal** | Text_MiniLM_LogisticRegression_Balanced | 0.8036 | **0.7316** | 0.5333 | 0.6667 | 0.5926 | 0.7317 | 0.5651 |
| **B2b_lin** | Text_MiniLM_LinearSVM | 0.7857 | **0.7143** | 0.5333 | 0.6154 | 0.5714 | 0.7187 | 0.6006 |
| **B2b_rbf** | Text_MiniLM_RBFSVM | 0.7143 | **0.4167** | 0.0000 | 0.0000 | 0.0000 | 0.7805 | 0.5630 |

## 3. Selected Best Text Baseline
- **Selected Configuration**: **Text_MiniLM_LogisticRegression** (B2a)
- **Primary Criterion (Dev Macro-F1)**: **0.7496**
- **Depression Recall**: 0.5333 (TP=8, FN=7)
- **Depression Precision**: 0.7273 (FP=3)
- **Confusion Matrix**: TN=38, FP=3, FN=7, TP=8
- **Artifact Saved To**: `artifacts/text/best_text_model.joblib`

## 4. Linguistic Observations and Known Limitations
- **Absence of Speaker Annotations**: In the lightweight transcript schema, utterances are not explicitly tagged with speaker identities (Ellie vs. Participant). Embeddings pool dialogue across the session. The model captures overall session semantic content rather than isolated participant speech.
- **Preserved Clinical Vocabulary**: Text cleaning conservatively retained all negations, symptom keywords, and affective terms without stopword deletion.
- **Model Comparison**: Contextual sentence embeddings provide strong semantic representations for depression detection, establishing a solid unimodal benchmark for multimodal fusion.
