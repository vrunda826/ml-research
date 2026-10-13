# High Symptom–Expression Discrepancy (HSED) Investigation in Depression Detection

[![Tests](https://img.shields.io/badge/Unit%20Tests-58%20Passed%20(100%25)-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-DAIC--WOZ-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![Audit Status](https://img.shields.io/badge/Phase%206%20Audit-Verified-success.svg)]()
[![License](https://img.shields.io/badge/License-Research%20Only-orange.svg)]()

This repository contains the complete experimental framework, feature extraction pipelines, baseline models, out-of-fold discrepancy modeling, and rigorous scientific audit code for investigating **High Symptom–Expression Discrepancy (HSED)** in multimodal depression detection using the **DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)** dataset.

---

## 1. Scientific Context & Research Objective

Clinical depression screening typically relies on standardized self-report instruments such as the **PHQ-8 (Patient Health Questionnaire-8)**. Computational depression detection models attempt to predict either binary depression status ($\text{PHQ-8} \ge 10$) or continuous symptom severity from nonverbal acoustic signals and conversational speech.

### Research Hypothesis
Certain individuals exhibiting clinically elevated self-reported depression ($\text{PHQ-8} \ge 10$) display comparatively weak or attenuated observable acoustic or linguistic indicators during clinical interviews, leading predictive models to systematically underestimate symptom severity.

### Crucial Methodological Boundary
> [!IMPORTANT]
> **HSED is an exploratory, model-defined residual subgroup, NOT an established clinical diagnosis.**  
> Observed model underpredictions ($D_i = y_i - \hat{y}_i > 0$) reflect instances where statistical estimators fail to capture self-reported severity. These discrepancies do not, by themselves, prove psychological masking, emotional suppression, or deliberate symptom concealment. No ground-truth clinical masking labels exist in DAIC-WOZ.

---

## 2. Project Architecture & Phase Breakdown

The project follows a phased, leak-free research engineering pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DAIC-WOZ Research Pipeline                      │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Data Audit & Manifest Construction                            │
│ • Deterministic participant-level alignment across metadata and splits │
│ • Validated PHQ-8 range [0, 24] and cutoff-10 binary research target   │
│ • Strict isolation: Train (125) ∩ Dev (44) = ∅; Test (56) unextracted │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASES 2–5: Multimodal Classification Baselines                        │
│ • Phase 2: Experimental protocol, imputer/scaler fold isolation        │
│ • Phase 3: Acoustic baselines (OpenSMILE eGeMAPS: 46, MFCC: 78 feats)  │
│ • Phase 4: Linguistic baselines (Frozen MiniLM-L6-v2: 384 embeddings)  │
│ • Phase 5: Early feature fusion (eGeMAPS+Text: 430, MFCC+Text: 462)   │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: Continuous Regression & HSED Modeling                         │
│ • L2 Ridge regression (alpha=100.0) across all 5 modality configs      │
│ • 5-fold cross-validation on Train (125) -> 750 unrounded OOF residuals│
│ • Derivation of HSED thresholds tau_q (q in {0.70, 0.75, 0.80}) on     │
│   eligible depressed training participants (S_train: PHQ-8 >= 10, N=38)│
│ • Frozen application of tau_q to Development depressed cohort (N=11)   │
│ • Cross-modality candidate overlap & Jaccard concordance analysis      │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 6 SCIENTIFIC AUDIT & NUMERICAL RECONCILIATION                    │
│ • 100% numerical reproduction of 54 metrics (|diff| < 5e-5)            │
│ • Corrected dev candidate identification (448 and 698 across all mods) │
│ • Methodological critique of positive residuals & regression to mean   │
│ • 58 unit tests passing (100% pass rate)                               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset Provenance & Cohort Accounting

| Cohort | Official AVEC 2019 Metadata | Local Extracted Archive | $\text{PHQ-8} \ge 10$ (Depressed) | $\text{PHQ-8} < 10$ (Non-Depressed) | Depression Prevalence |
|---|---:|---:|---:|---:|---:|
| **Training** | 163 | **125** | 38 | 87 | 30.4% |
| **Development** | 56 | **44** | 11 | 33 | 25.0% |
| **Test** | 56 | **0** | — | — | — |
| **Total** | **275** | **169** | **49** | **120** | **29.0%** |

- **Extraction Reconciliation:** Exactly 106 participants (38 train, 12 dev, 56 test) are missing from the raw download archive and are tracked as `METADATA_ONLY_NO_FEATURES` in `participant_manifest.csv`.
- **Zero Test Contamination:** Zero test audio or transcripts exist locally. No test labels or test evaluation scores were fabricated or evaluated.
- **Transcript Limitation:** The lightweight interview transcripts represent whole-dialogue interactions without verified speaker diarization. Text embeddings reflect combined participant and interviewer (Ellie) conversational dynamics.

---

## 4. Key Experimental Results

### Continuous PHQ-8 Regression Baselines (Development Set, $N=44$)
Evaluated using primary baseline configuration ($\text{Ridge}, \alpha=100.0$):

| Model / Modality | Feature Dim | MAE | RMSE | $R^2$ | Spearman $\rho$ | Spearman $p$-value |
|---|---:|---:|---:|---:|---:|---:|
| **Dummy (Mean Baseline)** | — | 4.7020 | 5.7430 | -0.0003 | *Undefined* | *Undefined* |
| **Audio eGeMAPS** | 46 | 4.9320 | 6.1414 | -0.1439 | -0.1687 | 0.2737 |
| **Audio MFCC** | 78 | 4.4306 | 5.4335 | 0.1046 | 0.2865 | 0.0594 |
| **Text MiniLM** | 384 | 3.9876 | 5.1125 | 0.2073 | 0.4793 | 0.0010 |
| **MM eGeMAPS + Text** | 430 | 4.0497 | 5.1721 | 0.1887 | 0.4348 | 0.0032 |
| **MM MFCC + Text** | 462 | **3.8032** | **4.8635** | **0.2826** | **0.5450** | **0.0001** |

### Training Out-of-Fold (OOF) Cross-Validation ($N=125$, 5 Folds)

| Modality | OOF MAE | OOF RMSE | OOF $R^2$ | Mean Residual $\bar{D}$ | Std Residual $\sigma_D$ |
|---|---:|---:|---:|---:|---:|
| **Dummy Baseline** | 5.3117 | 6.3279 | -0.0081 | 0.0000 | 6.3279 |
| **Audio eGeMAPS** | 5.3963 | 6.4807 | -0.0573 | +0.0615 | 6.4804 |
| **Audio MFCC** | 5.2768 | 6.4006 | -0.0314 | +0.0292 | 6.4005 |
| **Text MiniLM** | 4.4875 | 5.6021 | 0.2099 | +0.0702 | 5.6017 |
| **MM eGeMAPS + Text** | 4.5940 | 5.7915 | 0.1556 | +0.1324 | 5.7899 |
| **MM MFCC + Text** | **4.5188** | **5.6554** | **0.1948** | **+0.0868** | **5.6548** |

### HSED Thresholds & Subgroup Sensitivity Analysis
Thresholds derived strictly from eligible depressed training participants ($S_{\text{train}}: \text{PHQ-8} \ge 10, N=38$):

| Modality | $\tau_{0.70}$ | $\tau_{0.75}$ | $\tau_{0.80}$ | Train Cand ($q=0.75$) | Dev Cand ($q=0.75, N=11$) | Dev Candidate % |
|---|---:|---:|---:|---:|---:|---:|
| **Audio eGeMAPS** | 10.1495 | **10.9798** | 12.8596 | 10 (26.3%) | **3** | 27.3% |
| **Audio MFCC** | 10.3751 | **11.2920** | 11.7806 | 10 (26.3%) | **2** | 18.2% |
| **Text MiniLM** | 7.7454 | **8.2720** | 9.1525 | 10 (26.3%) | **5** | 45.5% |
| **MM eGeMAPS + Text** | 7.8800 | **8.1087** | 9.7897 | 10 (26.3%) | **4** | 36.4% |
| **MM MFCC + Text** | 7.8056 | **8.6088** | 9.4935 | 10 (26.3%) | **4** | 36.4% |

### Cross-Modality Overlap & Jaccard Concordance ($q=0.75$)
- **Training OOF Cohort ($N_{\text{dep}}=38$):**
  - Audio eGeMAPS vs Audio MFCC: $J = 0.8182$ (9 common candidates out of 11 union).
  - Audio MFCC vs Text MiniLM: $J = 0.4286$ (6 common candidates out of 14 union).
  - Text MiniLM vs MM eGeMAPS+Text: $J = 1.0000$ (10 common candidates out of 10 union).
- **Development Cohort ($N_{\text{dep}}=11$):**
  - Candidates selected across **all 5 modalities**: Exactly 2 participants: `{448, 698}`.
  - Candidates selected across **Text and both Multimodal models**: Exactly 4 participants: `{381, 441, 448, 698}`.
  - Text MiniLM vs Multimodal models: $J = 0.8000$ (4/5).
  - Audio eGeMAPS vs Audio MFCC: $J = 0.6667$ (2/3).
  - Audio vs Text concordance: $J = 0.4000$ (2/5).

---

## 5. Key Audit Discoveries & Methodological Hardening

1. **Correction of Candidate Participant Claims:**  
   Prior documentation claimed development participants 441 and 448 met candidate criteria across all models. The empirical audit revealed that **participant 441 failed both acoustic thresholds** ($D_{\text{eGeMAPS}}=10.465 < 10.980; D_{\text{MFCC}}=9.588 < 11.292$). Participant **698** (PHQ-8 = 19) and **448** (PHQ-8 = 18) are the only two development participants meeting candidate criteria across all 5 modalities.
2. **Critique of Positive Residuals & Regression Toward the Mean:**  
   Acoustic models exhibit large mean positive discrepancies among depressed participants ($\bar{D} \approx +8.3$ points). This occurs because Ridge models shrink toward the training sample mean ($\approx 6.47$) due to low acoustic predictive signal ($R^2 \approx -0.04$). Conditioning on $y \ge 10$ ($\bar{y} = 14.79$) mathematically guarantees a large positive residual ($14.79 - 6.47 \approx +8.32$). **This is regression toward the mean under asymmetric sample conditioning, not clinical masking.**
3. **Model Selection vs. Independent Generalization:**  
   Hyperparameter $\alpha=100.0$ was selected after comparing development set MAE across the candidate grid $[0.1, 1.0, 10.0, 100.0]$. Development results reflect model-selection performance, not an independent generalization benchmark.

---

## 6. Repository Layout

```
ml-research/
├── configs/
│   └── experiment_config.json                 # Global configuration & baseline hyperparameters
├── data/
│   ├── metadata/                              # Official AVEC 2019 metadata splits & PHQ-8 labels
│   │   ├── Detailed_PHQ8_Labels.csv
│   │   ├── train_split.csv
│   │   ├── dev_split.csv
│   │   └── test_split.csv
│   └── processed/
│       ├── participant_manifest.csv           # Master participant manifest (275 tracked, 169 local)
│       ├── dataset_audit.csv                  # Integrity audit summary
│       └── features/                          # Aggregated tabular feature matrices
│           ├── audio/
│           │   ├── egemaps_features.csv       # 46 summary features (mean/std of 23 LLDs)
│           │   └── mfcc_features.csv          # 78 summary features (mean/std of 13 MFCCs+deltas)
│           ├── text/
│           │   └── text_embeddings_minilm.csv # 384-dim frozen sentence embeddings
│           └── multimodal/
│               ├── multimodal_egemaps_text.csv# 430-dim concatenated features
│               └── multimodal_mfcc_text.csv   # 462-dim concatenated features
├── artifacts/                                 # Serialized model pipelines & metadata (.joblib)
├── reports/
│   ├── phase2_protocol.md
│   ├── phase3_audio.md
│   ├── phase4_text.md
│   ├── phase5_multimodal.md
│   ├── pre_phase6_audit.md
│   └── phase6/                                # Phase 6 experiment artifacts & tables
│       ├── regression_metrics.csv
│       ├── regression_model_comparison.csv
│       ├── dev_regression_predictions.csv
│       ├── oof_fold_assignments.csv
│       ├── train_oof_predictions.csv
│       ├── oof_validation_report.json
│       ├── hsed_thresholds.csv
│       ├── train_hsed_assignments.csv
│       ├── dev_hsed_assignments.csv
│       ├── hsed_sensitivity_analysis.csv
│       ├── hsed_modality_overlap.csv
│       ├── phase6_report.md
│       ├── backup_before_final_audit/         # Pre-audit immutable backup
│       └── final_audit/                       # Audit verification deliverables
│           ├── phase6_final_scientific_audit.md
│           ├── audit_summary.json
│           ├── dataset_integrity_audit.json
│           ├── numerical_reconciliation.csv   # 54 audited metrics (100% pass)
│           ├── hsed_selected_participants.csv # Exact participant IDs per subgroup
│           └── hsed_selected_participants.json
├── src/                                       # Core source library
│   ├── validation.py                          # Participant split isolation & leakage prevention
│   ├── metrics.py                             # Classification & continuous regression metrics
│   ├── pipeline.py                            # Leak-free scikit-learn pipeline builder
│   ├── phase6_regression.py                   # Phase 6 regression, OOF, threshold engine
│   ├── features/                              # Feature aggregation modules
│   └── models/                                # Model definitions & wrappers
├── scripts/                                   # Standalone execution & audit scripts
│   ├── run_phase6_pipeline.py                 # Pipeline reproducibility runner
│   └── generate_audit_artifacts.py            # Audit reconciliation generator
├── tests/                                     # Comprehensive automated test suite
│   ├── test_phase1.py                         # Phase 1 metadata & manifest tests
│   ├── test_phase2.py                         # Phase 2 metrics & validation tests
│   ├── test_baselines.py                      # Classification baseline tests
│   ├── test_audit_hardening.py                # Pre-Phase-6 audit hardening tests
│   └── test_phase6.py                         # Complete Phase 6 protocol audit suite (20 tests)
├── .gitignore                                 # Excludes raw archives, 12.5GB frame data, venvs
├── requirements.txt                           # Core dependencies
└── README.md                                  # Repository documentation
```

---

## 7. Installation & Reproducibility Guide

### Environment Setup
Clone the repository and initialize a virtual environment:
```powershell
git clone https://github.com/vrunda826/ml-research.git
cd ml-research

# Create and activate Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Reproducing Phase 6 Pipeline
Run the complete regression, out-of-fold discrepancy generation, and HSED subgroup analysis:
```powershell
python scripts/run_phase6_pipeline.py
```

### Regenerating Audit Verification Artifacts
Independently verify all 54 numerical metrics and regenerate reconciliation tables:
```powershell
python scripts/generate_audit_artifacts.py
```

### Running the Test Suite
Execute the entire project test suite:
```powershell
python -m unittest discover -v tests
```
*Expected output: 58 tests passed in < 1.0s.*

---

## 8. Data Use & Ethical Disclosure

- **DAIC-WOZ Data License:** Access to raw DAIC-WOZ interview recordings and transcripts is restricted under a Data Use Agreement (DUA) granted by the University of Southern California (USC) Institute for Creative Technologies (ICT). In compliance with DUA regulations, raw participant archives (`data/raw/`) and frame-by-frame acoustic features (`data/processed/lightweight/`) are excluded from version control via `.gitignore`.
- **Anonymization:** Participant identifiers are numeric research tokens. All identifiable personal health information was redacted by the original corpus creators.
- **Intended Use:** This software is for academic research into affective computing and multimodal machine learning. It is **not a medical device** and is not intended for diagnostic or clinical decision-making.

---

## 9. Citation

If you use this codebase or audit methodology in your research, please cite:

```bibtex
@misc{hsed_daic_woz_2026,
  author = {Vrunda Patel},
  title = {High Symptom--Expression Discrepancy (HSED) Investigation in Multimodal Depression Detection: Phase 6 Implementation and Final Scientific Audit},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/vrunda826/ml-research}}
}
```
