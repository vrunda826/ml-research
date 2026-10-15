# Expanded Dataset Audit Report

**Date of Audit:** October 15, 2026  
**Project:** DAIC-WOZ Multimodal Depression Research (Expanded Cohort Evaluation)  
**Root Directory:** D:\ml-research  
**Audit Status:** COMPLETE & VERIFIED

---

## 1. Executive Summary & Participant Counts

| Metric / Dimension | Train Split | Development Split | Test Split | Total Cohort |
| :--- | :---: | :---: | :---: | :---: |
| **Official Metadata Count** | 163 | 56 | 56 | 275 |
| **Previous Dataset Count (Phase 6)** | 125 | 44 | 0 | 169 |
| **Newly Added Participants** | +38 | +12 | +51 (with features) | +101 |
| **Lightweight Raw Features Available** | 163 | 56 | 51 | 270 |
| **Missing Feature Files** | 0 | 0 | 5 (615, 618, 622, 650, 651) | 5 |
| **Eligible for Multimodal Evaluation** | 163 | 56 | 51 (Holdout / Unlabeled Selection) | 270 |
| **Predefined Cutoff (PHQ-8 >= 10)** | 50 (30.67%) | 15 (26.79%) | 21 (41.18% of 51) | 86 (31.85%) |
| **Predefined Cutoff (PHQ-8 < 10)** | 113 (69.33%) | 41 (73.21%) | 30 (58.82% of 51) | 184 (68.15%) |

> [!IMPORTANT]
> The expanded dataset successfully increases the labeled training cohort from 125 to **163 participants** (+30.4%) and the development evaluation cohort from 44 to **56 participants** (+27.3%). The total labeled evaluation cohort is exactly **219 participants** (up from 169). In accordance with the evaluation protocol, the official split boundaries are strictly frozen: zero cross-split repartitioning or pooling was performed.

---

## 2. Participant ID Integrity, Uniqueness, and Split Disjointness

- **Total Unique Participant IDs:** 275 unique IDs across all official splits.
- **Duplicate IDs:** 0 duplicates detected.
- **Split Overlaps:**
  - Train $\cap$ Development = $\emptyset$ (0 overlap)
  - Train $\cap$ Test = $\emptyset$ (0 overlap)
  - Development $\cap$ Test = $\emptyset$ (0 overlap)
- **ID Normalization:** All participant IDs have been string-normalized (removing whitespace, floating-point string conversions like '300.0', and archive suffix patterns like '_P').

---

## 3. Modality Completeness & Feature Matrix Finiteness

Every extracted participant has complete, non-zero raw files (_Transcript.csv, _OpenSMILE2.3.0_mfcc.csv, _OpenSMILE2.3.0_egemaps.csv) without truncation.

| Feature Table | Total Rows | Total Columns | NaN Count | Inf Count | Finiteness Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **eGeMAPS Summary** (egemaps_features.csv) | 270 | 51 | 0 | 0 | PASS (100% Finite) |
| **MFCC Summary** (mfcc_features.csv) | 270 | 83 | 0 | 0 | PASS (100% Finite) |
| **Text MiniLM Embeddings** (	ext_embeddings_minilm.csv) | 270 | 390 | 0 | 0 | PASS (100% Finite) |
| **MM eGeMAPS + Text** (multimodal_egemaps_text.csv) | 270 | 435 | 0 | 0 | PASS (100% Finite) |
| **MM MFCC + Text** (multimodal_mfcc_text.csv) | 270 | 467 | 0 | 0 | PASS (100% Finite) |

- **Zero Non-Finite Values:** There are 0 NaN and 0 infinite values across all 270 participants in all feature matrices.
- **Missing Raw Features:** Exactly 5 participants in the official test split (615, 618, 622, 650, 651) have metadata records in 	est_split.csv but lack raw audio/transcript archives in the lightweight directory.

---

## 4. Metadata and Label Discrepancy Reconciliation

### 4.1 Detailed PHQ-8 Questionnaire vs Split Summary Scores
Cross-referencing 	rain_split.csv and dev_split.csv against itemized sums in Detailed_PHQ8_Labels.csv reveals two historical AVEC data discrepancies:
1. **Participant 485 (Train):** PHQ_Score = 2 in 	rain_split.csv, but PHQ_8Total = 4 in Detailed_PHQ8_Labels.csv (difference = +2). Both values are < 10 (PHQ_Binary = 0).
2. **Participant 486 (Dev):** PHQ_Score = 4 in dev_split.csv, but PHQ_8Total = 2 in Detailed_PHQ8_Labels.csv (difference = -2). Both values are < 10 (PHQ_Binary = 0).

Neither discrepancy alters the binary classification status of either participant. The official split file values are retained for backward consistency with AVEC benchmarks.

### 4.2 Predefined Cutoff (PHQ-8 >= 10) vs Official AVEC PHQ_Binary
In the official AVEC challenge metadata, PHQ_Binary was partially constructed using DSM-IV criteria and symptom combinations rather than a pure threshold. Across the 275 participants, **20 participants** show a divergence between PHQ_Score >= 10 and PHQ_Binary:
- **Train Split:** 13 participants with PHQ-8 >= 10 were labeled 0 in official AVEC binary.
- **Development Split:** 3 participants with PHQ-8 >= 10 were labeled 0 in official AVEC binary.
- **Test Split:** 4 participants with PHQ-8 >= 10 were labeled 0 in official AVEC binary.

**Protocol Enforcement:** In accordance with clinical guidelines (Kroenke et al.) and the experimental design, the primary binary target is strictly defined as PHQ-8 >= 10.

---

## 5. Exclusions and Audit Log

| Participant ID | Split | Raw Features Present | Exclusion Reason | Disposition |
| :---: | :---: | :---: | :---: | :---: |
| **615** | Test | No | Missing audio & transcript archives in lightweight store | Excluded from test feature matrix |
| **618** | Test | No | Missing audio & transcript archives in lightweight store | Excluded from test feature matrix |
| **622** | Test | No | Missing audio & transcript archives in lightweight store | Excluded from test feature matrix |
| **650** | Test | No | Missing audio & transcript archives in lightweight store | Excluded from test feature matrix |
| **651** | Test | No | Missing audio & transcript archives in lightweight store | Excluded from test feature matrix |

**Total Exclusions:** Exactly 5 participants (all in official test split). Zero training or development participants are excluded.

---

## 6. Official Split Isolation and Test Set Protocol

- **Zero Test Data Influence:** As strictly required by methodological auditing, test split labels are NEVER used for hyperparameter tuning, model selection, early stopping, threshold setting, or candidate identification.
- **Split Preservation:** Train (N=163) and Development (N=56) splits are strictly preserved as defined by the official DAIC-WOZ consortium. No random merging or repartitioning was conducted.
