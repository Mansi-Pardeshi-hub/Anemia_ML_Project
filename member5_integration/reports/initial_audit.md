# Initial Audit Report — Member 5 Integration

**Date:** 2026-11-12  
**Project:** Hybrid AI-Based Non-Invasive Hemoglobin Estimation: A Feasibility Study Using Image-Derived Features  
**Scope:** Academic research prototype only. Not a clinical diagnostic system.

---

## 1. Discovered Project Structure & Files

Inspection of `Anemia_ML_Project/` confirmed the following key directories and files:
- `Run_Project.bat` — Guarded root execution script verifying existing Member 4 artifacts without overwriting them.
- `datasets/` — Original `.zip` archives (`dataset_edi_eyes.zip`, `Figshare_Hb_Preprocessed_nail_skin.zip`, `PalmDataset.zip`, `member3_model_training.zip`) plus extracted subdirectories (`eye_extracted/`, `figshare_extracted/`, `palm_extracted/`, `member3_extracted/`).
- `cnn_env/` — Isolated Python 3.10.0 virtual environment with `torch==2.14.1+cpu` and `torchvision==0.29.1+cpu`.
- `member4_ml_modelling/` — Complete Member 4 implementation (`src/`, `notebooks/`, `outputs/`, `reports/`, `requirements.txt`, `README.md`).

---

## 2. Available Models & Input Contracts

Inspection of all `.pkl` and `.pth` files under `member4_ml_modelling/outputs/models/` and `datasets/member3_extracted/models/` verified the following schemas and contracts:

| Model File | Modality & Task | Bundle Keys | Feature / Embedding Dimension | Notes |
|---|---|---|---|---|
| `eye_xgb_regression.pkl` | Eye Hb Regression (g/dL) | `['scaler', 'xgb', 'feat_cols']` | 130 handcrafted features | Ordered by `feat_cols` |
| `nail_xgb_regression.pkl` | Nail Hb Regression (g/dL) | `['scaler', 'xgb', 'feat_cols']` | 130 handcrafted features | Ordered by `feat_cols` |
| `full_hand_xgb_regression.pkl` | Full-hand Hb Regression (g/dL) | `['scaler', 'xgb', 'feat_cols']` | 130 handcrafted features | Ordered by `feat_cols` |
| `eye_handcrafted_regression.pkl` | Eye Hb Regression (RF/ET) | `['scaler', 'rf', 'et', 'feature_cols']` | 130 handcrafted features | Uses `feature_cols` key |
| `nail_handcrafted_regression.pkl` | Nail Hb Regression (RF/ET) | `['scaler', 'rf', 'et', 'feature_cols']` | 130 handcrafted features | Uses `feature_cols` key |
| `full_hand_handcrafted_regression.pkl` | Full-hand Hb Regression (RF/ET) | `['scaler', 'rf', 'et', 'feature_cols']` | 130 handcrafted features | Uses `feature_cols` key |
| `eye_handcrafted_classification.pkl` | Eye Screening Classification | `['scaler', 'rf', 'et', 'feat_cols']` | 130 handcrafted features | Inherited WHO sex-specific cutoffs |
| `nail_handcrafted_classification.pkl` | Nail Screening Classification | `['scaler', 'rf', 'et', 'feat_cols']` | 130 handcrafted features | Inherited `<11 g/dL` cutoff (unreliable, F1=0.10) |
| `palm_handcrafted_classification.pkl` | Palm Folder-Label Classification | `['scaler', 'rf', 'et', 'feat_cols']` | 130 handcrafted features | Folder labels only; no measured Hb |
| `cnn_embedding/m3_eye_cnn_embedding_xgb.pkl` | Eye CNN-Embedding Hb Regression | `['scaler', 'xgb', 'embedding_columns', 'checkpoint', 'checkpoint_sha256', 'dataset_id']` | 1024-D (`embedding_000`..`1023`) | Extracted from Member 3 Eye regression CNN |
| `cnn_embedding/m3_nail_cnn_embedding_xgb.pkl` | Nail CNN-Embedding Hb Regression | `['scaler', 'xgb', 'embedding_columns', 'checkpoint', 'checkpoint_sha256', 'dataset_id']` | 1024-D (`embedding_000`..`1023`) | Extracted from Member 3 Nail regression CNN |
| `fullhand_cnn_scratch/fullhand_scratch_cnn_embedding_xgb.pkl` | Full-hand Scratch-CNN Embedding XGBoost | `['scaler', 'xgb', 'embedding_columns', 'cnn_checkpoint', 'cnn_checkpoint_sha256', 'dataset_id', 'selection_metric', 'selected_parameters']` | 64-D (`embedding_000`..`063`) | Paired with `fullhand_scratch_cnn_regressor.pth` |

---

## 3. Image Modalities & Verified Split Counts

1. **Eye / Conjunctiva (`eye_splits.csv` after exact-duplicate filtering):**
   - **Total usable images:** 610 across 202 participants (21 exact byte-identical variant duplicates removed by Member 4).
   - **Split counts (images):** Train = 432, Validation = 89, Test = 89.
   - **Targets:** Numeric laboratory Hb in g/dL; inherited sex-specific screening labels (`<13 g/dL` male, `<12 g/dL` female).
2. **Figshare Nail Crops (`nail_skin_nail_splits.csv`):**
   - **Total usable images:** 719 nail crops across 250 participants (1 to 3 crops per participant; mean 2.88).
   - **Split counts (crops):** Train = 503, Validation = 107, Test = 109.
   - **Split counts (participants):** Train = 175, Validation = 37, Test = 38.
   - **Targets:** Numeric Hb in g/L and g/dL (`÷10`). Exploratory class label uses universal `<11 g/dL` cutoff (no age/sex metadata).
3. **Figshare Full-hand (`nail_skin_full_splits.csv`):**
   - **Total usable images:** 250 full-hand photographs across 250 participants (exactly 1 image per participant).
   - **Split counts (participants = images):** Train = 175, Validation = 37, Test = 38.
   - **Pairing with Nail:** 100% matched across all 250 participants (`participant_id`, `source_image`, `hb_g_per_dl`, and `split` match with 0 discrepancies).
4. **Palm (`palm_splits.csv` after exact & cross-split near-duplicate exclusions):**
   - **Total usable images:** 2,056 unique images across 514 inferred groups (Train = 1,452, Validation = 312, Test = 292).
   - **Targets:** Folder-derived class labels only; **no numerical Hb measurements exist**.

---

## 4. Existing Validator Results

Executed `python member4_ml_modelling/src/validate_member4_results.py`:
- **Status:** `Validation PASSED (0 mismatched checks)`
- **Verified checks:**
  - All 9 handcrafted models and 4 CNN-embedding models reload and reproduce saved metrics.
  - `eye_cnn_embedding_xgboost`: n=89, CNN MAE = 1.7452 vs handcrafted MAE = 1.4177 g/dL.
  - `nail_cnn_embedding_xgboost`: n=109 crops, CNN MAE = 2.0425 vs handcrafted MAE = 1.7171 g/dL.
  - `full_hand_palm_classifier_transfer_xgboost`: n=38, transferred-feature MAE = 2.0970 vs handcrafted MAE = 1.5819 g/dL.
  - `fullhand_scratch_cnn_embedding_xgboost`: n=38, scratch-CNN embedding MAE = 1.5599 vs handcrafted MAE = 1.5819 g/dL.

---

## 5. Environment & Dependency Status

- **System Python 3.10 (`py -3.10` / `python`):**
  - `numpy==2.2.4`, `pandas==2.2.3`, `Pillow==11.2.1`, `matplotlib==3.10.1`, `scipy==1.15.3`, `scikit-learn==1.7.2`, `xgboost==3.2.0`, `scikit-image==0.25.2`, `pyarrow==19.0.1`
  - `streamlit==1.44.1`, `torch==2.14.1+cpu`, `shap==0.49.1` (installed and available).
- **Known Risk — XGBoost 3.2.0 + SHAP `TreeExplainer` Format Bug:**
  - `xgboost==3.2.0` serializes `base_score` in its JSON config as a bracketed string `"[1.2562639E1]"`, which causes `shap.TreeExplainer` to raise `ValueError: could not convert string to float: '[1.2562639E1]'` unless handled or via `xgboost.Booster.predict(pred_contribs=True)` (exact TreeSHAP algorithm built natively into XGBoost). We will inspect whether native XGBoost `pred_contribs=True` (exact TreeSHAP) or model config sanitization allows exact SHAP values to be computed and plotted cleanly.

---

## 6. Known Scientific & Technical Risks

1. **Crop vs. Participant Granularity:** Nail has 109 test crops across 38 test participants, whereas Full-hand has 38 test images across the same 38 test participants. Any fusion or direct Nail-vs-Full-hand comparison must aggregate Nail crops to the participant level first.
2. **Same-Photo Views vs. True Multimodal Sensing:** Figshare Nail crops are bounding-box crops extracted from the same source photographs (`photo/<id>.jpg`) as the Full-hand images. Combining them is **multi-view feature fusion from a single photograph**, not independent clinical multi-sensor fusion.
3. **Small Test Sample Size:** The held-out Figshare test set has only 38 participants. Small differences in MAE (< 0.10 g/dL) are within sampling noise.
4. **Incomplete Demographic Context:** Figshare lacks age and sex metadata; Eye lacks pregnancy and altitude/smoking context and includes 19 participants aged > 65.
5. **No Clinical Validity:** Models are exploratory academic prototypes and must never be presented as diagnostic tools or calibrated medical risk scores.

---

## 7. Planned Member 5 Implementation Path

1. **Package Structure (`member5_integration/`):** Implement `paths.py`, `image_quality.py`, `model_adapters.py`, `pairing.py`, `fusion.py`, `explainability.py`, `evaluation.py`, and `report_utils.py`.
2. **Model Adapters:** Support Handcrafted XGBoost (Full-hand, Nail, Eye), Full-hand Scratch-CNN Embedding XGBoost, and exploratory Eye/Nail screening adapters with strict feature alignment (`feat_cols` / `feature_cols`) and no on-the-fly training.
3. **Pairing Verification (`pairing.py`):** Verify all 250 Nail–Full-hand participant pairs, source images, Hb values, and 175/37/38 split assignments.
4. **Participant-Level Fusion (`fusion.py`):** Aggregate Nail crops by participant mean, concatenate with Full-hand features, tune on validation (n=37), and evaluate once on the 38 held-out test participants against Full-hand-only and Nail-only participant-level baselines.
5. **Explainability (`explainability.py`):** Compute exact TreeSHAP explanations on the trusted handcrafted XGBoost model and generate global summary plot, local explanation plot, local CSV, and `explainability.md`.
6. **Streamlit Demo (`app.py`):** Build cached inference app with prominent academic disclaimer, engineering image quality checks, single-image estimation, paired view fusion option, and strict guardrails.
7. **Automated Testing & Reports:** Implement `test_model_adapters.py`, `test_image_quality.py`, `test_pairing.py`, and `test_fusion.py`, execute all tests, and generate all required reports in `member5_integration/reports/`.

