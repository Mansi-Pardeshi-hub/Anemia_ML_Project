# Final Submission Checklist — Member 5 Integration

**Project:** Hybrid AI-Based Non-Invasive Hemoglobin Estimation: A Feasibility Study Using Image-Derived Features  
**Date:** 2026-11-12

---

## Verification Checklist

- [x] **Existing project files were preserved.** (`datasets/`, `cnn_env/`, `member4_ml_modelling/`, and `Run_Project.bat` remain intact.)
- [x] **Existing model artifacts were not overwritten.** (Verified via `validate_member4_results.py` with 0 mismatches.)
- [x] **Model bundles were inspected rather than assumed.** (Inspected keys across all 13 `.pkl` bundles: `feat_cols` vs `feature_cols` vs `embedding_columns`.)
- [x] **Feature order is taken from saved model metadata.** (`_get_feat_cols(bundle)` strictly orders features according to `bundle.get("feat_cols", bundle.get("feature_cols"))`.)
- [x] **Participant-level splitting is preserved.** (Verified 0 cross-split participant leakage across Eye, Nail, Full-hand, and Palm.)
- [x] **Nail crops are aggregated before participant-level evaluation.** (Deterministic per-participant mean across 1–3 Nail crops reduces 719 crops to 250 participants.)
- [x] **The 250 Nail–Full-hand pairs are independently verified.** (`pairing.py` confirmed 250/250 matched participant IDs, 0 Hb mismatches, 0 source-image mismatches, and 0 split mismatches.)
- [x] **Fusion is implemented only if pairing and features are valid.** (`fusion.py` verified pairing and concatenated 130 Full-hand + 130 mean-aggregated Nail handcrafted features = 260-D.)
- [x] **Fusion uses 175/37/38 participant splits.** (Train = 175, Validation = 37, Test = 38 participants.)
- [x] **Test participants were not used for tuning.** (Imputation means, `StandardScaler`, and XGBoost hyperparameter selection used training and validation participants only; test participants were evaluated once.)
- [x] **MAE, RMSE, and R² are reported.** (Reported in `g/dL` across all regression models and in `member5_consolidated_comparison.csv`.)
- [x] **Crop counts and participant counts are clearly distinguished.** (Nail has 719 crops across 250 participants — 503/107/109 crops vs 175/37/38 participants.)
- [x] **SHAP is used only if it runs successfully.** (`shap.TreeExplainer` ran successfully on `full_hand_xgb_regression.pkl` with the 175-participant training background after normalizing XGBoost 3.2.0's bracketed `base_score` string in memory.)
- [x] **Fallback feature importance is labelled correctly.** (The fallback code path in `explainability.py` explicitly labels global XGBoost gain importance as non-local and non-clinical.)
- [x] **Streamlit loads models without retraining.** (`@st.cache_resource` loads pickled bundles once at startup; static AST test confirms zero `.fit()` calls.)
- [x] **Streamlit includes quality checks.** (Displays dimensions, aspect ratio, brightness, contrast, Laplacian blur variance, and quality warnings.)
- [x] **Streamlit handles invalid uploads.** (Gracefully rejects missing, empty, or corrupted files without crashing or invoking model inference.)
- [x] **Streamlit displays the academic disclaimer.** (Prominent top banner: *"For academic research only. This experimental result is not a diagnosis and must not replace laboratory complete blood count (CBC) testing or qualified medical advice."*)
- [x] **Palm folder-label prediction is not presented as Hb estimation.** (Palm is excluded from Hb estimation in `app.py` and documented as folder-label prediction only.)
- [x] **App output is checked against a saved prediction.** (`evaluation.py` and `test_model_adapters.py` verified exact agreement `< 1e-6` between adapters and saved CSV predictions.)
- [x] **Existing Member 4 validation is run.** (`validate_member4_results.py` passed with 0 mismatches.)
- [x] **Member 5 tests are run.** (All 31 automated checks across `test_model_adapters.py`, `test_image_quality.py`, `test_pairing.py`, `test_fusion.py`, and `AppTest` passed.)
- [x] **README is complete.** (`member5_integration/README.md` created with platform-independent and Windows commands.)
- [x] **Final report is complete.** (`member5_integration/reports/final_report.md` created with all 26 required sections.)
- [x] **Equal contribution summary is included.** (Section 25 of `final_report.md` documents the equal contributions of Members 1–5.)
- [x] **Licensing status is clearly marked as verified or pending.** (Marked as *Pending Upstream Verification* in Section 6 of `final_report.md` and `README.md`.)
- [x] **No clinical claims are made.** (All outputs use cautious academic prototype terminology.)

