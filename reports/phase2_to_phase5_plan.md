# Implementation Plan: Phase 2 to Phase 5 (DAIC-WOZ Depression Detection)

## 1. Project Overview & Context

This document outlines the design, execution plan, and real-time status of Phase 2 (Experimental Protocol), Phase 3 (Audio Baseline B1), Phase 4 (Text Baseline B2), and Phase 5 (Multimodal Baseline B3) for the DAIC-WOZ depression detection research project at `D:\ml-research`.

### Primary Goal
Build a strictly leakage-free, reproducible, and verifiable baseline benchmark comparing unimodal (audio, text) and multimodal representations on the official DAIC-WOZ participant splits.

### Key Constraints & Principles
- **Participant-level separation**: Never split frames, utterances, or audio across splits.
- **Strict leakage prevention**: Preprocessing (`StandardScaler`, imputers, feature selectors, vocabularies) fitted strictly on training participants only; applied to dev/test without refitting.
- **Standardized label definition**: Binary depression target defined as `PHQ8_Score >= 10`.
- **Honest evaluation**: Model selection performed exclusively on the labeled development set using **Macro-F1** as primary criterion, supplemented by depression-class recall and PR-AUC. Official test set labels (even if present in archive files) are never used for model tuning or selection.
- **Resource awareness**: Modest 16 GB RAM + 6 GB VRAM Windows environment. Prefer CPU-based scikit-learn models, avoid simultaneous in-memory loading of massive frame-level CSVs.

---

## 2. Phase-by-Phase Plan & Status Tracker

| Phase | Component | Key Deliverables | Status |
| :--- | :--- | :--- | :---: |
| **Phase 2** | Experimental Protocol | `src/validation.py`<br>`src/metrics.py`<br>`configs/experiment_config.json`<br>`reports/phase2_protocol.md`<br>Automated protocol unit tests | ✅ Completed & Verified |
| **Phase 3** | Audio Baselines (B0, B1) | `src/features/audio_features.py`<br>`src/models/audio_baseline.py`<br>`src/train_audio_baseline.py`<br>`reports/audio_baseline_report.md`<br>`reports/audio_baseline_results.csv`<br>`artifacts/audio/` | ✅ Completed & Verified |
| **Phase 4** | Text Baseline (B2) | `src/features/text_features.py`<br>`src/models/text_baseline.py`<br>`src/train_text_baseline.py`<br>`reports/text_baseline_report.md`<br>`reports/text_baseline_results.csv`<br>`artifacts/text/` | ✅ Completed & Verified |
| **Phase 5** | Multimodal Baseline (B3) | `src/features/multimodal_features.py`<br>`src/models/multimodal_baseline.py`<br>`src/train_multimodal_baseline.py`<br>`reports/multimodal_baseline_report.md`<br>`reports/baseline_comparison.csv`<br>`artifacts/multimodal/` | ✅ Completed & Verified |

---

## 3. Discovered Dataset Topology

Based on Phase 1 verification:
- **Available Extracted Participants**: 169 complete participants in `data/processed/lightweight/` (125 in official Training split, 44 in official Development split).
- **Official Metadata Totals**: 275 participants in AVEC 2019 metadata (163 Train, 56 Dev, 56 Test).
  - Train: 125 extracted & complete, 38 metadata-only.
  - Dev: 44 extracted & complete, 12 metadata-only.
  - Test: 0 extracted, 56 metadata-only.
- **Primary Experimental Set**: The 169 participants with full lightweight features (125 training, 44 development) form the exact closed set for training and model selection.
- **Discrepancy Note**: Classic DAIC-WOZ reports (e.g. Gratch et al., 2014) described ~189 participants (107 train / 35 dev / 47 test). AVEC 2019 extended the dataset to 275 participants. All available 169 local participants map deterministically to the official AVEC 2019 train (125) and dev (44) splits.

---

## 4. Next Action Items
1. Complete package installation (`scikit-learn`, `joblib`, `sentence-transformers`).
2. Write and test `src/validation.py` and `src/metrics.py`.
3. Freeze protocol in `configs/experiment_config.json` and `reports/phase2_protocol.md`.
4. Proceed to Phase 3 audio aggregation and modeling.
