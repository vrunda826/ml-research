# Experiment Log: DAIC-WOZ End-to-End Depression Detection & HSED Investigation

**Date:** October 13, 2026  
**Random Seed:** 42 (Global deterministic seed)  
**Environment:** Python 3.11.4 | scikit-learn 1.9.1 | PyTorch 2.14.1 | Matplotlib 3.11.2  

## 1. Experimental Runs Executed

| Run ID | Modality | Method | Weights | Gamma | Dev MAE | Dev R² | Dev Dep Discrepancy | Status |
|---|---|---|---|---:|---:|---:|---:|:---:|
| `mm_mfcc_text_none_g0.0` | mm_mfcc_text | Ridge | none | 0.0 | 3.8032 | 0.2826 | 5.1954 | COMPLETED_SUCCESSFULLY |
| `mm_mfcc_text_continuous_g0.5` | mm_mfcc_text | Ridge | continuous | 0.5 | 3.8968 | 0.2559 | 4.6007 | COMPLETED_SUCCESSFULLY |
| `mm_mfcc_text_continuous_g1.0` | mm_mfcc_text | Ridge | continuous | 1.0 | 3.9558 | 0.2380 | 4.3369 | COMPLETED_SUCCESSFULLY |
| `mm_mfcc_text_continuous_g2.0` | mm_mfcc_text | Ridge | continuous | 2.0 | 4.0297 | 0.2177 | 4.0891 | COMPLETED_SUCCESSFULLY |
| `mm_mfcc_text_binary_g0.5` | mm_mfcc_text | Ridge | binary | 0.5 | 3.8526 | 0.2716 | 4.9457 | COMPLETED_SUCCESSFULLY |
| `mm_mfcc_text_binary_g1.0` | mm_mfcc_text | Ridge | binary | 1.0 | 3.8937 | 0.2620 | 4.7914 | COMPLETED_SUCCESSFULLY |
| `mm_mfcc_text_binary_g2.0` | mm_mfcc_text | Ridge | binary | 2.0 | 3.9448 | 0.2482 | 4.6105 | COMPLETED_SUCCESSFULLY |
| `text_minilm_none_g0.0` | text_minilm | Ridge | none | 0.0 | 3.9876 | 0.2073 | 5.8621 | COMPLETED_SUCCESSFULLY |
| `text_minilm_continuous_g0.5` | text_minilm | Ridge | continuous | 0.5 | 4.1083 | 0.1583 | 5.2554 | COMPLETED_SUCCESSFULLY |
| `text_minilm_continuous_g1.0` | text_minilm | Ridge | continuous | 1.0 | 4.1990 | 0.1269 | 4.9696 | COMPLETED_SUCCESSFULLY |
| `text_minilm_continuous_g2.0` | text_minilm | Ridge | continuous | 2.0 | 4.3064 | 0.0920 | 4.6896 | COMPLETED_SUCCESSFULLY |
| `text_minilm_binary_g0.5` | text_minilm | Ridge | binary | 0.5 | 4.0265 | 0.1860 | 5.6000 | COMPLETED_SUCCESSFULLY |
| `text_minilm_binary_g1.0` | text_minilm | Ridge | binary | 1.0 | 4.0820 | 0.1682 | 5.4367 | COMPLETED_SUCCESSFULLY |
| `text_minilm_binary_g2.0` | text_minilm | Ridge | binary | 2.0 | 4.1535 | 0.1425 | 5.2438 | COMPLETED_SUCCESSFULLY |

## 2. Baseline Model Runs Reused & Verified

- `Dummy Mean Regressor` (alpha=N/A, strategy='mean'): OOF MAE=5.3117, Dev MAE=4.7020
- `Audio eGeMAPS` (alpha=100.0, 46 feats): OOF MAE=5.3963, Dev MAE=4.9320
- `Audio MFCC` (alpha=100.0, 78 feats): OOF MAE=5.2768, Dev MAE=4.4306
- `Text MiniLM` (alpha=100.0, 384 feats): OOF MAE=4.4875, Dev MAE=3.9876
- `Multimodal eGeMAPS+Text` (alpha=100.0, 430 feats): OOF MAE=4.5940, Dev MAE=4.0497
- `Multimodal MFCC+Text` (alpha=100.0, 462 feats): OOF MAE=4.5188, Dev MAE=3.8032

## 3. Failed or Incomplete Runs

- None. All 14 proposed method configurations and 6 baselines completed execution without errors or exceptions.