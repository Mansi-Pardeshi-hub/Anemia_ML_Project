# Member 5 Integration Report

**Date:** 2026-11-12  
**Project:** Hybrid AI-Based Non-Invasive Hemoglobin Estimation: A Feasibility Study Using Image-Derived Features  
**Scope:** Academic research prototype only. Not a clinical diagnostic system.

---

## 1. Executive Summary

Member 5 completed the system integration, paired Nail–Full-hand verification and multi-view fusion, TreeSHAP explainability, Streamlit demonstration interface, automated test suite, and project-wide documentation without altering or overwriting any Member 1–4 artifacts.

All existing Member 4 models and predictions were verified via `validate_member4_results.py` (`0 mismatched checks`, `13/13 model reload checks passed`).

---

## 2. Integrated Components

1. **Path & Root Resolver (`member5_integration/src/paths.py`):**
   - Resolves `Anemia_ML_Project/` dynamically via `pathlib.Path` so all modules run identically across machines and working directories.
2. **Model Adapters (`member5_integration/src/model_adapters.py`):**
   - Implements the unified `ModelAdapter` base class and concrete adapters:
     - `FullHandXGBAdapter` (`full_hand_xgb_regression.pkl`, 130 handcrafted features)
     - `NailXGBAdapter` (`nail_xgb_regression.pkl`, 130 handcrafted features)
     - `EyeXGBAdapter` (`eye_xgb_regression.pkl`, 130 handcrafted features)
     - `FullHandScratchCNNXGBAdapter` (`fullhand_scratch_cnn_regressor.pth` + `fullhand_scratch_cnn_embedding_xgb.pkl`, 64-D embedding)
     - `FusedHandNailXGBAdapter` (`fused_model.pkl`, 260-D paired Full-hand + mean Nail features)
     - `EyeClassificationAdapter` (`eye_handcrafted_classification.pkl`, exploratory WHO sex-specific cutoffs)
     - `NailClassificationAdapter` (`nail_handcrafted_classification.pkl`, exploratory `<11 g/dL` cutoff)
   - Enforces exact feature-column ordering from `bundle.get("feat_cols", bundle.get("feature_cols"))` and verifies exact agreement (`abs_diff < 1e-6`) with saved Member 4 test predictions.
3. **Engineering Image Quality Checks (`member5_integration/src/image_quality.py`):**
   - Checks file readability, non-zero byte size, truncation (`Image.verify()`), RGB conversion, minimum dimensions (`>=64x64`), aspect ratio (`<=10:1`), mean brightness (`[30, 220]`), greyscale contrast (`std >= 15`), Laplacian variance blur (`var >= 50`), HSV saturation (`<240`), and central-region framing.
4. **Paired Data Verification (`member5_integration/src/pairing.py`):**
   - Verified all 250 Full-hand participants against 719 Nail crop rows (250 participants): 0 Hb mismatches, 0 source-image mismatches, and exact 175 / 37 / 38 train/validation/test participant splits.
5. **Participant-Level Fusion (`member5_integration/src/fusion.py`):**
   - Aggregated Nail crops per participant via deterministic mean (`(250, 130)`), concatenated with Full-hand (`(250, 260)`), tuned XGBoost on validation MAE (n=37), and evaluated once on the 38 held-out test participants.
6. **TreeSHAP Explainability (`member5_integration/src/explainability.py`):**
   - Resolved the XGBoost 3.2.0 UBJ bracketed `base_score` string format (`"[1.2745714E1]"`) in memory so `shap.TreeExplainer` runs natively on `full_hand_xgb_regression.pkl` using the 175 training participants as background.
7. **Streamlit Demo (`member5_integration/app.py`):**
   - Loads models once via `@st.cache_resource`, displays the mandatory academic disclaimer prominently, runs quality checks on uploads, supports single-image and paired multi-view inference, and never retrains models.

