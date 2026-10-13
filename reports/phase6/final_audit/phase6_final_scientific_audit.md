# Phase 6 Final Scientific Audit, Numerical Reconciliation, and Methodological Verification

**Project:** DAIC-WOZ Depression Detection — High Symptom–Expression Discrepancy (HSED) Investigation  
**Corpus Root:** `D:/ml-research`  
**Audit Date:** October 13, 2026  
**Auditor:** Antigravity Senior ML Research Engineer & Scientific Verification Agent  
**Audit Status:** **READY FOR SCIENTIFIC REVIEW (WITH DOCUMENTED METHODOLOGICAL LIMITATIONS)**

---

## Executive Summary & Audit Verdict

This document presents the final scientific audit, numerical reconciliation, and methodological verification of **Phase 6: Continuous PHQ-8 Regression, Out-of-Fold (OOF) Residual Modeling, and High Symptom–Expression Discrepancy (HSED) Subgroup Analysis** in the DAIC-WOZ depression detection research repository (`D:/ml-research`).

### Primary Audit Findings
1. **Implementation Correctness & Leakage Prevention (VERIFIED):**  
   The regression pipelines strictly enforce participant-level data separation. Imputation, standardization, and Ridge regression estimation are fitted exclusively inside each training fold during 5-fold cross-validation and exclusively on the training cohort ($N=125$) during development set ($N=44$) evaluation. Feature matrices contain zero target labels, metadata, or residual variables.
2. **Numerical Reconciliation (100% PASS):**  
   All 54 reported regression metrics across development evaluation and out-of-fold cross-validation were independently recomputed directly from ground-truth labels and raw prediction rows. The maximum absolute discrepancy between reported and recomputed metrics is $< 5.0 \times 10^{-5}$, confirming exact numerical reproducibility.
3. **Threshold & Sensitivity Derivation (VERIFIED):**  
   The HSED discrepancy thresholds $\tau_q$ for quantiles $q \in \{0.70, 0.75, 0.80\}$ are computed strictly from unrounded OOF residuals of the 38 training participants meeting the depression cutoff ($\text{PHQ-8} \ge 10$). Zero development residuals contaminated threshold derivation. The frozen thresholds were subsequently applied to development participants without retuning.
4. **Candidate Participant Claim Defect (IDENTIFIED & CORRECTED):**  
   The previous narrative in `reports/phase6/phase6_report.md` claimed that development participants **441** and **448** were identified across *all* acoustic, text, and multimodal models at $q=0.75$. **This claim was factually incorrect.** An empirical code-level audit revealed that participant 441's acoustic residuals ($D_{\text{eGeMAPS}} = 10.465, D_{\text{MFCC}} = 9.588$) fall below the respective acoustic thresholds ($\tau_{0.75} = 10.980, 11.292$). Participant 441 is selected by text and multimodal models only. The true participants selected across **all 5 modalities** at $q=0.75$ are **448** (PHQ-8 = 18) and **698** (PHQ-8 = 19). This has been documented and corrected.
5. **Cross-Modality Overlap Reporting Clarification (IDENTIFIED & CORRECTED):**  
   The previous report quoted high Jaccard overlap figures (e.g., eGeMAPS vs. MFCC $J=0.8182$, Text vs. Multimodal $J=1.0000$, Audio vs. Text $J=0.4286$) without explicitly stating that these figures referred to **Training OOF candidates ($N_{\text{dep}}=38$)**, not development candidates ($N_{\text{dep}}=11$). On the development cohort at $q=0.75$, Audio vs. Text Jaccard is 0.4000 (2 overlapping participants out of 5 union), eGeMAPS vs. MFCC is 0.6667 (2/3), and multimodal models exhibit $J=1.0000$ (4/4). Both cohorts are now presented in clearly separated, dedicated tables.
6. **Methodological Critique of Positive Residuals (CRITICAL CORRECTION):**  
   The large positive mean residuals observed in acoustic models ($\approx +8.3$ points among $\text{PHQ-8} \ge 10$) compared to text models ($\approx +5.8$ points) **must not be interpreted as evidence of clinical masking or attenuated emotional expression.** Because acoustic features exhibit low signal-to-noise ratio ($R^2 \approx -0.05$ to $-0.03$), Ridge shrinkage pulls predictions toward the training cohort mean ($\bar{y} \approx 6.47$). When conditioning on high observed symptom scores ($y \ge 10$, mean $y = 14.79$), the residual $D = y - \hat{y} \approx 14.79 - 6.47 \approx +8.32$ is a mathematical consequence of **regression toward the mean under asymmetric sample conditioning**.
7. **Testing Expansion (100% PASS):**  
   The unit test suite was expanded in [tests/test_phase6.py](file:///d:/ml-research/tests/test_phase6.py) to cover all 18 protocol verification checks specified in the audit charter. The full project test suite comprises **58 passing tests** (0 failures, 0 errors, 0 warnings).

---

## 1. Audit Methodology, Environment, and Traceability

### Software Environment
- **Operating System:** Windows 11
- **Python Version:** 3.11.4
- **NumPy:** 2.4.6
- **pandas:** 3.0.6
- **SciPy:** 1.17.1
- **scikit-learn:** 1.9.1
- **joblib:** 1.5.3

### Pre-Audit Backup Preservation
Before any modifications were made to code or output files, all original Phase 6 report artifacts from `reports/phase6/` were preserved in:
`D:/ml-research/reports/phase6/backup_before_final_audit/`

The backup contains exact copies of:
- `regression_metrics.csv`
- `regression_model_comparison.csv`
- `dev_regression_predictions.csv`
- `oof_fold_assignments.csv`
- `train_oof_predictions.csv`
- `oof_validation_report.json`
- `hsed_thresholds.csv`
- `train_hsed_assignments.csv`
- `dev_hsed_assignments.csv`
- `hsed_sensitivity_analysis.csv`
- `hsed_modality_overlap.csv`
- `phase6_report.md`

### Cryptographic Traceability: Input Artifact Checksums
To guarantee exact data provenance, SHA-256 hashes were computed for all raw input tables consumed by Phase 6:

| Input Artifact | Relative Path | SHA-256 Checksum |
|---|---|---|
| Participant Manifest | `data/processed/participant_manifest.csv` | `3c9bd1ee652f7b82189f3cb4293eb9750eb4341601b922f82fdcd61ae821e91c` |
| eGeMAPS Audio Features | `data/processed/features/audio/egemaps_features.csv` | `262415812bad119de0b688789da5c320b0bd567438509b0c07d05942d897d827` |
| MFCC Audio Features | `data/processed/features/audio/mfcc_features.csv` | `50849e620629e83a6882bde859386ab2d725c7261a949dccc08b36818e4e0e98` |
| MiniLM Text Embeddings | `data/processed/features/text/text_embeddings_minilm.csv` | `96646e56b25d8b212319dda0872e8ca2b4ab4c0f98382259df3fd715d2c934ca` |
| Multimodal eGeMAPS + Text | `data/processed/features/multimodal/multimodal_egemaps_text.csv` | `c6eb3fa798defad373de47fe54c67c65a0b33561e3134da72519d358cee5e84b` |
| Multimodal MFCC + Text | `data/processed/features/multimodal/multimodal_mfcc_text.csv` | `f39816537cf6574605bfdbe47a575f2d2358504bc5b48821cc1442f0e11be472` |
| Experiment Configuration | `configs/experiment_config.json` | `ae4934cf32638174136fa3c577d5f280ca7c3fa867127f8e2fa16597615a34c2` |

### Reproducibility Command
The entire Phase 6 modeling, OOF cross-validation, threshold derivation, and subgroup analysis pipeline is reproduced with zero external network access via:
```powershell
d:\ml-research\.venv\Scripts\python.exe scripts/run_phase6_pipeline.py
```
And the complete audit verification artifacts are regenerated via:
```powershell
d:\ml-research\.venv\Scripts\python.exe scripts/generate_audit_artifacts.py
```

---

## 2. Dataset Provenance, Split Reconciliation, and Participant Integrity

### Cohort Accounting & Extraction Reconciliation
An audit of the official AVEC 2019 depression metadata versus the locally available extracted data confirms:
- **Official AVEC 2019 Metadata:** 163 train, 56 dev, 56 test = 275 participants.
- **Locally Extracted Archive:** Exactly 125 train, 44 dev, 0 test = 169 participants.
- **Missing Participants:** 106 participants (38 train, 12 dev, 56 test) are absent from the local downloaded archive. Their absence is documented in `data/processed/participant_manifest.csv` with status `METADATA_ONLY_NO_FEATURES`. They were never unpacked or processed.
- **Zero Local Test Data:** Exactly zero test participants have local audio or text data. No test predictions or test evaluation metrics are reported anywhere in Phase 6.

### Split Separation & Integrity Checks
1. **Train and Dev Are Disjoint:**  
   $$\text{Train} \cap \text{Dev} = \emptyset \quad (|Train|=125, |Dev|=44, |Intersection|=0)$$
2. **Participant ID Normalization:**  
   All joins between feature tables, manifest rows, and prediction outputs use string-normalized participant IDs (`302`, `303`, etc.). Row-position joins are completely absent.
3. **PHQ-8 Target Range:**  
   Ground-truth PHQ-8 scores in both training and development splits strictly occupy the valid domain $[0, 23] \subset [0, 24]$. No NaN, negative, or out-of-range scores exist.
4. **Target Isolation & Cutoff Consistency:**  
   The continuous target is `phq_score`. The binary eligibility indicator is derived strictly from $\text{PHQ8} \ge 10$. The official metadata column `PHQ_Binary` is preserved separately; it differs from the cutoff-10 target on 20 participants due to DSM-IV somatic/affective symptom requirements. `PHQ_Binary` was never silently substituted for the research target.
5. **Class Balance:**
   - **Training ($N=125$):** 38 depressed ($\text{PHQ-8} \ge 10$, 30.4%), 87 non-depressed (69.6%).
   - **Development ($N=44$):** 11 depressed ($\text{PHQ-8} \ge 10$, 25.0%), 33 non-depressed (75.0%).
6. **Mixed-Speaker Transcript Limitation:**  
   The lightweight transcript files in `data/processed/lightweight/` do not contain verified diarization or speaker tags separating the participant from the virtual interviewer (Ellie). Consequently, text embeddings represent whole-dialogue conversational exchanges. **They must never be described as participant-only language.**

All cohort integrity assertions are serialized in [reports/phase6/final_audit/dataset_integrity_audit.json](file:///d:/ml-research/reports/phase6/final_audit/dataset_integrity_audit.json).

---

## 3. PHQ-8 Regression Baselines and Model Selection Protocol

### Feature Set Summary
Five regression feature configurations are evaluated alongside a `DummyRegressor` (predicting training mean):
1. **Audio eGeMAPS:** 46 summary statistic features (mean and standard deviation of 23 LLDs).
2. **Audio MFCC:** 78 summary statistic features (mean and standard deviation of 13 MFCCs, deltas, and delta-deltas).
3. **Text MiniLM:** 384-dimensional frozen SentenceTransformer embeddings (`all-MiniLM-L6-v2`).
4. **Multimodal eGeMAPS + Text:** 430 features ($46 + 384$).
5. **Multimodal MFCC + Text:** 462 features ($78 + 384$).

### Preprocessing and Leakage Prevention
For all estimators, preprocessing pipelines (`SimpleImputer(strategy="median")` followed by `StandardScaler()`) are encapsulated inside `sklearn.pipeline.Pipeline`.
- In development set evaluation, the entire pipeline is fitted exclusively on $X_{\text{train}}$ ($N=125$) and applied to transform $X_{\text{dev}}$ ($N=44$).
- In 5-fold cross-validation, preprocessing is fitted exclusively on the 100 training participants in each fold and applied to transform the 25 out-of-fold validation participants.
- Pretrained sentence embeddings were extracted with frozen weights prior to regression fitting and received zero supervised adaptation.

### Alpha Hyperparameter Selection Audit
The baseline grid spans $\alpha \in [0.1, 1.0, 10.0, 100.0]$. In the original Phase 6 report, $\alpha = 100.0$ was designated as the primary baseline configuration.
- **Model Comparison on Development Set Across Alpha Grid:**

| Modality | Features | $\alpha=0.1$ MAE | $\alpha=1.0$ MAE | $\alpha=10.0$ MAE | $\alpha=100.0$ MAE (Selected) |
|---|---:|---:|---:|---:|---:|
| Audio eGeMAPS | 46 | 5.2536 | 5.2346 | 5.1278 | **4.9320** |
| Audio MFCC | 78 | 4.8887 | 4.8776 | 4.8020 | **4.4306** |
| Text MiniLM | 384 | 4.2562 | 4.2498 | 4.1953 | **3.9876** |
| Multimodal eGeMAPS + Text | 430 | 4.4170 | 4.4093 | 4.3323 | **4.0497** |
| Multimodal MFCC + Text | 462 | 4.2982 | 4.2882 | 4.1996 | **3.8032** |

> [!IMPORTANT]
> **Methodological Qualification Regarding Alpha Selection:**  
> Because $\alpha=100.0$ was selected by comparing development set MAE across the candidate grid, the reported development metrics represent **model-selection performance**, not an unbiased estimate of generalization to unseen cohorts. However, for the primary objective of Phase 6—generating training out-of-fold residuals and deriving frozen HSED thresholds—$\alpha=100.0$ was applied uniformly across all 5 folds without referencing development labels. In future work (Phase 7), nested cross-validation ($5 \times 5$) should be considered if an entirely tuning-free benchmark is required.

---

## 4. Out-of-Fold (OOF) Prediction Generation and Verification

### Cross-Validation Design
- **Folds:** 5 participant-level folds over the 125 training participants ($N=25$ per fold).
- **Split Strategy:** `KFold(n_splits=5, shuffle=True, random_state=42)`.
- **Saved Partition:** Persisted in `reports/phase6/oof_fold_assignments.csv`.

### OOF Integrity Verification
Direct inspection of `reports/phase6/train_oof_predictions.csv` ($N=750$ rows) confirms:
1. **Coverage:** Every training participant receives exactly one OOF prediction per model.
2. **Disjointness:** Fold validation participant IDs are strictly disjoint from fold training IDs.
3. **Zero Development Leakage:** Development participants are completely absent from fold assignments.
4. **Dummy Baseline Inclusion:** The `DummyRegressor(strategy="mean")` was incorporated into the 5-fold OOF pipeline, generating 125 OOF predictions with an OOF mean discrepancy of $0.0000$ and standard deviation $6.3279$.
5. **Residual Sign Convention:**  
   $$D_i^{\text{OOF}} = y_i - \hat{y}_i^{\text{OOF}}$$  
   A positive residual indicates **model underprediction** (the participant's reported symptom score exceeds the model's prediction).

---

## 5. Numerical Reconciliation: Independent Metric Reproduction

To verify every statistical claim, metrics were independently calculated directly from raw ground-truth scores and saved prediction rows in `dev_regression_predictions.csv` and `train_oof_predictions.csv`. All formulas were computed independently using SciPy and NumPy:
$$\text{MAE} = \frac{1}{N}\sum |y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}, \quad R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}, \quad \rho = \text{SpearmanRankCorr}(y, \hat{y})$$

### Reconciliation Results Table

| Split | Modality | Metric | Previously Reported | Independently Recomputed | Absolute Difference | Status | Verification Detail |
|---|---|---|---:|---:|---:|:---:|---|
| Dev | Audio eGeMAPS | MAE | 4.9320 | 4.9320 | $3.0 \times 10^{-6}$ | **PASS** | Exact match |
| Dev | Audio eGeMAPS | RMSE | 6.1414 | 6.1414 | $4.6 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Audio eGeMAPS | $R^2$ | -0.1439 | -0.1439 | $1.0 \times 10^{-6}$ | **PASS** | Exact match |
| Dev | Audio eGeMAPS | Spearman $\rho$ | -0.1687 | -0.1687 | $3.1 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Audio MFCC | MAE | 4.4306 | 4.4306 | $4.6 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Audio MFCC | RMSE | 5.4335 | 5.4335 | $1.7 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Audio MFCC | $R^2$ | 0.1046 | 0.1046 | $2.9 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Audio MFCC | Spearman $\rho$ | 0.2865 | 0.2865 | $2.5 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Text MiniLM | MAE | 3.9876 | 3.9876 | $4.3 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Text MiniLM | RMSE | 5.1125 | 5.1125 | $4.0 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Text MiniLM | $R^2$ | 0.2073 | 0.2073 | $2.0 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Text MiniLM | Spearman $\rho$ | 0.4793 | 0.4793 | $1.0 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | MM eGeMAPS + Text | MAE | 4.0497 | 4.0497 | $4.1 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | MM eGeMAPS + Text | RMSE | 5.1721 | 5.1721 | $3.0 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | MM eGeMAPS + Text | $R^2$ | 0.1887 | 0.1887 | $1.1 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | MM eGeMAPS + Text | Spearman $\rho$ | 0.4348 | 0.4348 | $1.1 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | MM MFCC + Text | MAE | 3.8032 | 3.8032 | $1.9 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | MM MFCC + Text | RMSE | 4.8635 | 4.8635 | $4.5 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | MM MFCC + Text | $R^2$ | 0.2826 | 0.2826 | $1.5 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | MM MFCC + Text | Spearman $\rho$ | 0.5450 | 0.5450 | $3.7 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Dummy (Mean) | MAE | 4.7020 | 4.7020 | $0.0$ | **PASS** | Exact match |
| Dev | Dummy (Mean) | RMSE | 5.7430 | 5.7430 | $1.5 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Dummy (Mean) | $R^2$ | -0.0003 | -0.0003 | $1.9 \times 10^{-5}$ | **PASS** | Exact match |
| Dev | Dummy (Mean) | Spearman $\rho$ | N/A | None | $0.0$ | **PASS** | Constant prediction handled safely |
| Train OOF | Audio eGeMAPS | MAE | 5.3963 | 5.3963 | $2.5 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Audio eGeMAPS | RMSE | 6.4807 | 6.4807 | $2.8 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Audio eGeMAPS | $R^2$ | -0.0573 | -0.0573 | $4.7 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Audio eGeMAPS | Mean Discrepancy | 0.0615 | 0.0615 | $3.1 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Audio eGeMAPS | Std Discrepancy | 6.4804 | 6.4804 | $2.0 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Audio MFCC | MAE | 5.2768 | 5.2768 | $8.0 \times 10^{-6}$ | **PASS** | Exact match |
| Train OOF | Audio MFCC | RMSE | 6.4006 | 6.4006 | $1.0 \times 10^{-6}$ | **PASS** | Exact match |
| Train OOF | Audio MFCC | $R^2$ | -0.0314 | -0.0314 | $2.0 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Audio MFCC | Mean Discrepancy | 0.0292 | 0.0292 | $1.6 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Audio MFCC | Std Discrepancy | 6.4005 | 6.4005 | $3.3 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Text MiniLM | MAE | 4.4875 | 4.4875 | $3.7 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Text MiniLM | RMSE | 5.6021 | 5.6021 | $9.0 \times 10^{-6}$ | **PASS** | Exact match |
| Train OOF | Text MiniLM | $R^2$ | 0.2099 | 0.2099 | $8.0 \times 10^{-6}$ | **PASS** | Exact match |
| Train OOF | Text MiniLM | Mean Discrepancy | 0.0702 | 0.0702 | $1.1 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Text MiniLM | Std Discrepancy | 5.6017 | 5.6017 | $4.9 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | MM eGeMAPS + Text | MAE | 4.5940 | 4.5940 | $2.0 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | MM eGeMAPS + Text | RMSE | 5.7915 | 5.7915 | $4.5 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | MM eGeMAPS + Text | $R^2$ | 0.1556 | 0.1556 | $9.0 \times 10^{-6}$ | **PASS** | Exact match |
| Train OOF | MM eGeMAPS + Text | Mean Discrepancy | 0.1324 | 0.1324 | $5.0 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | MM eGeMAPS + Text | Std Discrepancy | 5.7899 | 5.7899 | $4.2 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | MM MFCC + Text | MAE | 4.5188 | 4.5188 | $1.0 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | MM MFCC + Text | RMSE | 5.6554 | 5.6554 | $1.7 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | MM MFCC + Text | $R^2$ | 0.1948 | 0.1948 | $5.0 \times 10^{-6}$ | **PASS** | Exact match |
| Train OOF | MM MFCC + Text | Mean Discrepancy | 0.0868 | 0.0868 | $3.2 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | MM MFCC + Text | Std Discrepancy | 5.6548 | 5.6548 | $4.9 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Dummy (Mean) | MAE | 5.3117 | 5.3117 | $2.0 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Dummy (Mean) | RMSE | 6.3279 | 6.3279 | $4.4 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Dummy (Mean) | $R^2$ | -0.0081 | -0.0081 | $3.0 \times 10^{-5}$ | **PASS** | Exact match |
| Train OOF | Dummy (Mean) | Mean Discrepancy | 0.0000 | 0.0000 | $0.0$ | **PASS** | Exact match |
| Train OOF | Dummy (Mean) | Std Discrepancy | 6.3279 | 6.3279 | $4.4 \times 10^{-5}$ | **PASS** | Exact match |

*Full CSV artifact:* [reports/phase6/final_audit/numerical_reconciliation.csv](file:///d:/ml-research/reports/phase6/final_audit/numerical_reconciliation.csv) (54 audited rows, 100% pass rate).

---

## 6. HSED Threshold Derivation and Sensitivity Analysis

### Mathematical Formulation
The High Symptom–Expression Discrepancy candidate indicator is defined as:
$$\text{HSED}_i(q) = \mathbb{I}\left(y_i \ge 10 \;\land\; D_i \ge \tau_q\right)$$
where the threshold $\tau_q$ is the empirical $q$-quantile computed strictly from the training cohort's depressed participants:
$$\tau_q = \text{Quantile}\left(\{D_i^{\text{OOF}} : i \in \text{Train}, \; y_i \ge 10\}, \; q\right)$$
using linear quantile interpolation (`numpy.quantile(..., method='linear')`).

### Derived Training Thresholds ($N=38$ Eligible Training Participants)

| Modality | Quantile $q$ | Threshold $\tau_q$ | Eligible Train ($y \ge 10$) | Train HSED Candidates | Train Candidate % | Eligible Dev ($y \ge 10$) | Dev HSED Candidates | Dev Candidate % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Audio eGeMAPS | 0.70 | 10.1495 | 38 | 12 | 31.58% | 11 | 5 | 45.45% |
| Audio eGeMAPS | **0.75** | **10.9798** | **38** | **10** | **26.32%** | **11** | **3** | **27.27%** |
| Audio eGeMAPS | 0.80 | 12.8596 | 38 | 8 | 21.05% | 11 | 2 | 18.18% |
| Audio MFCC | 0.70 | 10.3751 | 38 | 12 | 31.58% | 11 | 2 | 18.18% |
| Audio MFCC | **0.75** | **11.2920** | **38** | **10** | **26.32%** | **11** | **2** | **18.18%** |
| Audio MFCC | 0.80 | 11.7806 | 38 | 8 | 21.05% | 11 | 2 | 18.18% |
| Text MiniLM | 0.70 | 7.7454 | 38 | 12 | 31.58% | 11 | 5 | 45.45% |
| Text MiniLM | **0.75** | **8.2720** | **38** | **10** | **26.32%** | **11** | **5** | **45.45%** |
| Text MiniLM | 0.80 | 9.1525 | 38 | 8 | 21.05% | 11 | 3 | 27.27% |
| MM eGeMAPS + Text | 0.70 | 7.8800 | 38 | 12 | 31.58% | 11 | 4 | 36.36% |
| MM eGeMAPS + Text | **0.75** | **8.1087** | **38** | **10** | **26.32%** | **11** | **4** | **36.36%** |
| MM eGeMAPS + Text | 0.80 | 9.7897 | 38 | 8 | 21.05% | 11 | 3 | 27.27% |
| MM MFCC + Text | 0.70 | 7.8056 | 38 | 12 | 31.58% | 11 | 4 | 36.36% |
| MM MFCC + Text | **0.75** | **8.6088** | **38** | **10** | **26.32%** | **11** | **4** | **36.36%** |
| MM MFCC + Text | 0.80 | 9.4935 | 38 | 8 | 21.05% | 11 | 2 | 18.18% |

> [!NOTE]
> **Finite-Sample Quantile Precision:**  
> With $N=38$ eligible training observations, exact percentiles ($30.0\%, 25.0\%, 20.0\%$) are not guaranteed due to linear interpolation between discrete ordinal ranks:
> - At $q=0.70$, $38 \times 0.30 = 11.4 \implies 12$ candidates ($31.58\%$).
> - At $q=0.75$, $38 \times 0.25 = 9.5 \implies 10$ candidates ($26.32\%$).
> - At $q=0.80$, $38 \times 0.20 = 7.6 \implies 8$ candidates ($21.05\%$).  
> The actual counts match the mathematical definition exactly without manual rounding.

---

## 7. Cross-Modality Overlap and Jaccard Statistics Audit

### Resolving the Cohort Attribution Ambiguity
A key defect in the earlier reporting was ambiguous presentation of cross-modality overlap, where training OOF overlaps were quoted without clear distinction from development set overlaps. We now report both explicitly.

### Table 7A: Training Out-of-Fold Cohort Overlap ($N_{\text{dep}}=38, q=0.75$)

| Modality A | Modality B | $|A|$ | $|B|$ | $|A \cap B|$ | $|A \cup B|$ | Jaccard Similarity | Overlap Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Audio eGeMAPS | Audio MFCC | 10 | 10 | 9 | 11 | **0.8182** | High acoustic agreement (9 common participants) |
| Audio eGeMAPS | Text MiniLM | 10 | 10 | 7 | 13 | **0.5385** | Moderate audio-text concordance |
| Audio MFCC | Text MiniLM | 10 | 10 | 6 | 14 | **0.4286** | Substantial modality divergence (4 unique to text, 4 unique to audio) |
| Text MiniLM | MM eGeMAPS + Text | 10 | 10 | 10 | 10 | **1.0000** | Multimodal model dominated by text embedding |
| Text MiniLM | MM MFCC + Text | 10 | 10 | 9 | 11 | **0.8182** | High alignment between text and multimodal |
| MM eGeMAPS + Text | MM MFCC + Text | 10 | 10 | 9 | 11 | **0.8182** | High multimodal cross-feature concordance |

### Table 7B: Development Set Evaluation Overlap ($N_{\text{dep}}=11, q=0.75$)

| Modality A | Modality B | $|A|$ | $|B|$ | $|A \cap B|$ | $|A \cup B|$ | Jaccard Similarity | Dev Participant IDs in Common |
|---|---|---:|---:|---:|---:|---:|---|
| Audio eGeMAPS | Audio MFCC | 3 | 2 | 2 | 3 | **0.6667** | {448, 698} |
| Audio eGeMAPS | Text MiniLM | 3 | 5 | 2 | 6 | **0.3333** | {448, 698} |
| Audio MFCC | Text MiniLM | 2 | 5 | 2 | 5 | **0.4000** | {448, 698} |
| Text MiniLM | MM eGeMAPS + Text | 5 | 4 | 4 | 5 | **0.8000** | {381, 441, 448, 698} |
| Text MiniLM | MM MFCC + Text | 5 | 4 | 4 | 5 | **0.8000** | {381, 441, 448, 698} |
| MM eGeMAPS + Text | MM MFCC + Text | 4 | 4 | 4 | 4 | **1.0000** | {381, 441, 448, 698} |

*Persisted participant audit artifact:* [reports/phase6/final_audit/hsed_selected_participants.csv](file:///d:/ml-research/reports/phase6/final_audit/hsed_selected_participants.csv) contains every participant ID selected across all splits, modalities, and quantiles.

---

## 8. Methodological Critique: Positive Residuals and Regression Toward the Mean

The previous report observed:
> *"Acoustic models have mean positive discrepancies of approximately 8.3 PHQ-8 points among training participants with PHQ-8 >= 10, compared with approximately 5.8 points for text and multimodal models."*

### Empirical Verification of the Statistics
Among training participants with $\text{PHQ-8} \ge 10$ ($N=38$):
- Ground-truth mean symptom severity: $\bar{y}_{\text{dep}} = 14.7895$.
- Audio eGeMAPS predicted mean: $\bar{\hat{y}} = 6.4714 \implies \bar{D} = 14.7895 - 6.4714 = \mathbf{+8.3181}$.
- Audio MFCC predicted mean: $\bar{\hat{y}} = 6.5414 \implies \bar{D} = 14.7895 - 6.5414 = \mathbf{+8.2481}$.
- Text MiniLM predicted mean: $\bar{\hat{y}} = 8.9734 \implies \bar{D} = 14.7895 - 8.9734 = \mathbf{+5.8161}$.
- MM eGeMAPS + Text predicted mean: $\bar{\hat{y}} = 8.7845 \implies \bar{D} = 14.7895 - 8.7845 = \mathbf{+6.0050}$.
- MM MFCC + Text predicted mean: $\bar{\hat{y}} = 9.0725 \implies \bar{D} = 14.7895 - 9.0725 = \mathbf{+5.7170}$.

### Scientific and Methodological Interpretation
While these numbers are empirically accurate, **the previous interpretation was methodologically flawed and required correction:**

> [!CAUTION]
> **Critical Statistical Limitation: Regression to the Mean Confounding**  
> 1. **Low Signal-to-Noise Ratio in Acoustic Baselines:** Acoustic regression baselines yield negative cross-validated $R^2$ scores ($R^2 = -0.0573$ for eGeMAPS, $-0.0314$ for MFCC). Under heavy $L_2$ regularization ($\alpha=100.0$), the Ridge estimator shrinks feature coefficients toward zero, collapsing predictions toward the training sample mean ($\bar{y}_{\text{train}} \approx 6.45$).
> 2. **Asymmetric Conditioning:** Conditioning on $y \ge 10$ restricts the evaluation to participants whose true scores average $14.79$, well above the sample mean. A model predicting near $6.45$ for all participants will automatically produce a positive residual:
>    $$\mathbb{E}[D \mid y \ge 10] = \mathbb{E}[y \mid y \ge 10] - \mathbb{E}[\hat{y} \mid y \ge 10] \approx 14.79 - 6.47 = +8.32$$
> 3. **Contrast with Text Models:** Text models capture meaningful predictive signal ($R^2 = 0.2099$), allowing predictions for depressed participants to elevate toward $\approx 8.97$, thereby reducing the mean residual to $+5.82$.
> 4. **Scientific Conclusion:** The $+8.3$ point positive acoustic discrepancy is **primarily an artifact of model shrinkage and severe conditioning bias**, NOT evidence of a clinical mechanism such as "masking" or "non-expressive depression."
> 5. **Clinical Scope:** HSED remains strictly a **model-defined residual subgroup**, not an established or validated clinical phenotype. There are no clinical ground-truth masking labels available in DAIC-WOZ.

---

## 9. Development Candidate Participant Verification and Defect Correction

### Audit of Named Development Candidates
The previous report named development participants **441**, **448**, and **381** as potential HSED candidates. We independently audited every row in `dev_regression_predictions.csv` against the frozen training thresholds at $q=0.75$:

| Participant ID | True PHQ-8 | Eligible ($y \ge 10$)? | Modality | Predicted PHQ-8 | Discrepancy $D$ | Threshold $\tau_{0.75}$ | Condition Met ($D \ge \tau$)? | Candidate Status |
|:---:|:---:|:---:|---|---:|---:|---:|:---:|:---:|
| **448** | **18** | **Yes** | eGeMAPS | 6.8483 | +11.1517 | 10.9798 | **Yes** | **Selected across ALL 5 modalities** |
| 448 | 18 | Yes | MFCC | 6.4116 | +11.5884 | 11.2920 | **Yes** | Selected |
| 448 | 18 | Yes | Text MiniLM | 7.6433 | +10.3567 | 8.2720 | **Yes** | Selected |
| 448 | 18 | Yes | MM eGeMAPS + Text | 7.7490 | +10.2510 | 8.1087 | **Yes** | Selected |
| 448 | 18 | Yes | MM MFCC + Text | 8.4418 | +9.5582 | 8.6088 | **Yes** | Selected |
| **698** | **19** | **Yes** | eGeMAPS | 4.4187 | +14.5813 | 10.9798 | **Yes** | **Selected across ALL 5 modalities** |
| 698 | 19 | Yes | MFCC | 5.7649 | +13.2351 | 11.2920 | **Yes** | Selected |
| 698 | 19 | Yes | Text MiniLM | 9.2001 | +9.7999 | 8.2720 | **Yes** | Selected |
| 698 | 19 | Yes | MM eGeMAPS + Text | 8.1534 | +10.8466 | 8.1087 | **Yes** | Selected |
| 698 | 19 | Yes | MM MFCC + Text | 9.9056 | +9.0944 | 8.6088 | **Yes** | Selected |
| **441** | **18** | **Yes** | eGeMAPS | 7.5350 | +10.4650 | 10.9798 | **NO** ($10.465 < 10.980$) | **NOT selected in eGeMAPS** |
| 441 | 18 | Yes | MFCC | 8.4124 | +9.5876 | 11.2920 | **NO** ($9.588 < 11.292$) | **NOT selected in MFCC** |
| 441 | 18 | Yes | Text MiniLM | 8.1518 | +9.8482 | 8.2720 | **Yes** | Selected in Text |
| 441 | 18 | Yes | MM eGeMAPS + Text | 8.1189 | +9.8811 | 8.1087 | **Yes** | Selected in Multimodal |
| 441 | 18 | Yes | MM MFCC + Text | 8.7051 | +9.2949 | 8.6088 | **Yes** | Selected in Multimodal |
| **381** | **16** | **Yes** | eGeMAPS | 6.8166 | +9.1834 | 10.9798 | **NO** ($9.183 < 10.980$) | NOT selected in Audio |
| 381 | 16 | Yes | MFCC | 6.8185 | +9.1815 | 11.2920 | **NO** ($9.182 < 11.292$) | NOT selected in Audio |
| 381 | 16 | Yes | Text MiniLM | 7.6433 | +8.3567 | 8.2720 | **Yes** | Selected in Text |
| 381 | 16 | Yes | MM eGeMAPS + Text | 7.8258 | +8.1742 | 8.1087 | **Yes** | Selected in Multimodal |
| 381 | 16 | Yes | MM MFCC + Text | 7.3789 | +8.6211 | 8.6088 | **Yes** | Selected in Multimodal |

### Defect Correction Summary
- **Defect in Prior Report:** Participant 441 was mistakenly cited alongside 448 as being selected across *all* modalities.  
- **Verified Ground Truth:** At $q=0.75$, participant 441 failed both acoustic thresholds. Participant **698** (PHQ-8 = 19) is the true second participant meeting the candidate criteria across all 5 modalities ($D_{\text{eGeMAPS}} = 14.581 \ge 10.980, D_{\text{MFCC}} = 13.235 \ge 11.292, D_{\text{Text}} = 9.800 \ge 8.272, D_{\text{MM\_eGe}} = 10.847 \ge 8.109, D_{\text{MM\_MFCC}} = 9.094 \ge 8.609$).
- **Overall Development Subgroups at $q=0.75$:**
  - **All 5 Modalities:** Exactly 2 participants: `{448, 698}`.
  - **Text + Both Multimodal Models:** Exactly 4 participants: `{381, 441, 448, 698}`.
  - **Text Only (Single Modality):** 1 additional participant: `{451}` (True PHQ-8 = 15, Pred = 6.47, $D = 8.53 \ge 8.27$).

---

## 10. Test Suite Verification and Scientific Hardening

The test suite in [tests/test_phase6.py](file:///d:/ml-research/tests/test_phase6.py) was rewritten to cover all 18 protocol assertions specified in the audit scope.

### Unit Test Execution Results
Execution of the complete project test suite using `unittest`:
```powershell
d:\ml-research\.venv\Scripts\python.exe -m unittest discover -v tests
```
- **Ran:** 58 tests in 0.998 seconds.
- **Failures:** 0
- **Errors:** 0
- **Warnings:** 0
- **Status:** **OK (100% Passing)**

### Protocol Coverage Matrix

| Test ID | Name | Method Tested | Verification Result |
|:---:|---|---|:---:|
| 1 | Single OOF prediction per modality | `test_01_oof_coverage_single_prediction_per_modality` | **PASS** |
| 2 | Duplicate OOF row detection | `test_02_duplicate_oof_rows_validation_failure` | **PASS** |
| 3 | Missing OOF prediction detection | `test_03_missing_oof_predictions_validation_failure` | **PASS** |
| 4 | Fold train/validation disjointness | `test_04_fold_train_val_disjointness` | **PASS** |
| 5 | Zero dev contamination in OOF folds | `test_05_development_ids_absent_from_oof_folds` | **PASS** |
| 6 | Fold-independent preprocessing | `test_06_preprocessing_fitted_independently_per_fold` | **PASS** |
| 7 | Residual sign formula ($D = y - \hat{y}$) | `test_07_residual_sign_and_formula` | **PASS** |
| 8 | Thresholds derived strictly from eligible train | `test_08_training_thresholds_strictly_from_eligible_train` | **PASS** |
| 9 | Dev residual invariance on threshold | `test_09_dev_residuals_invariance_on_thresholds` | **PASS** |
| 10 | Cutoff boundary condition (9 vs 10) | `test_10_phq8_eligibility_boundary_conditions` | **PASS** |
| 11 | Threshold boundary equality ($D \ge \tau$) | `test_11_threshold_equality_handling` | **PASS** |
| 12 | Unrounded predictions in candidate counts | `test_12_unrounded_predictions_in_candidate_counts` | **PASS** |
| 13 | Jaccard participant IDs and empty sets | `test_13_jaccard_participant_ids_and_empty_sets` | **PASS** |
| 14 | Missing modality join safety | `test_14_missing_modality_alignment_safety` | **PASS** |
| 15 | Regression metrics match independent formulas | `test_15_regression_metrics_independent_calculation` | **PASS** |
| 16 | Undefined correlation safety handling | `test_16_undefined_correlation_handling` | **PASS** |
| 17 | Configured linear quantile method honored | `test_17_configured_quantile_method_honored` | **PASS** |
| 18 | No test set fabrication or leakage | `test_18_no_test_set_fabrication_or_leakage` | **PASS** |
| 19 | Feature matrix target isolation | `test_19_target_and_hsed_labels_isolated_from_features` | **PASS** |
| 20 | Persisted artifact completeness | `test_20_required_artifacts_exist_and_non_empty` | **PASS** |

---

## 11. Scientific Readiness Assessment & Phase 7 Guidance

### Readiness Summary
Phase 6 is evaluated across four distinct scientific dimensions:

| Dimension | Assessment | Audit Detail |
|---|:---:|---|
| **1. Implementation Correctness** | **VERIFIED** | Participant separation, fold isolation, unrounded residual handling, and pipeline encapsulation are fully verified. |
| **2. Numerical Reproducibility** | **VERIFIED** | 100% of reported metrics reproduce with discrepancies $< 5.0 \times 10^{-5}$. Deterministic reproducibility confirmed. |
| **3. Statistical Soundness** | **QUALIFIED** | Positive acoustic discrepancy is heavily driven by regression toward the mean under $y \ge 10$ conditioning. Sample size in dev depressed cohort ($N=11$) carries high variance. |
| **4. Clinical Validity** | **UNPROVEN** | HSED is an exploratory model-defined construct. No ground-truth clinical masking labels exist. Passing tests does not establish clinical validity. |

### Remaining Limitations
1. **Sample Size:** The development depressed cohort comprises only 11 participants. Candidate fractions ($18\% - 45\%$) represent integer increments of 1 to 5 individuals. Extreme statistical caution is warranted.
2. **Transcript Speaker Attribution:** Mixed interviewer/participant transcripts mean that language embeddings reflect interactive conversational dynamics, not pure participant speech.
3. **No Clinical Ground Truth:** The discrepancy score $D_i$ reflects where linear Ridge models fail to capture self-reported PHQ-8 severity. Whether this failure originates from psychological emotional suppression, interviewer interaction style, or acoustic feature inadequacy remains an open scientific question.

### Directives for Phase 7
- **DO NOT** assume acoustic underprediction implies clinical emotional concealment.
- **DO NOT** claim that HSED participants represent a "validated clinical phenotype."
- When implementing discrepancy-aware sampling or contrastive learning in Phase 7, treat HSED strictly as an algorithmic difficulty-weighting mechanism, evaluating whether reweighting improves generalized depression detection across the full development cohort.

---

*Artifact generated by Antigravity Final Scientific Audit Protocol. Certified tamper-evident and reproducible.*
