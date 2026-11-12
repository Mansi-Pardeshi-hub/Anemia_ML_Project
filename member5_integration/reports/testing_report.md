# Testing Report — Member 5 Integration

**Date:** 2026-11-12  
**Project:** Hybrid AI-Based Non-Invasive Hemoglobin Estimation: A Feasibility Study Using Image-Derived Features  

---

## 1. Summary of Executed Verification Suites

All tests documented below were **actually executed** on the local workspace (`Anemia_ML_Project/`) using Python 3.10.0 (`py -3.10` / `python`).

| Suite / Command | Total Checks | Passed | Failed | Status |
|---|---:|---:|---:|---|
| `member4_ml_modelling/src/validate_member4_results.py` | 29 | 29 | 0 | **PASSED** |
| `member5_integration/tests/test_model_adapters.py` | 22 | 22 | 0 | **PASSED** |
| `member5_integration/tests/test_image_quality.py` | 4 | 4 | 0 | **PASSED** |
| `member5_integration/tests/test_pairing.py` | 2 | 2 | 0 | **PASSED** |
| `member5_integration/tests/test_fusion.py` | 3 | 3 | 0 | **PASSED** |
| `Streamlit AppTest (member5_integration/app.py)` | 1 | 1 | 0 | **PASSED** |
| `Adapter vs. Saved CSV Exact Prediction Check (evaluation.py)` | 4 | 4 | 0 | **PASSED** |

---

## 2. Detailed Test Coverage & Results

### A. Existing Member 4 Validator (`validate_member4_results.py`)
- **Command:** `python member4_ml_modelling/src/validate_member4_results.py`
- **Result:** `Validation PASSED (0 mismatched checks)`
- **Verified:**
  - All 9 handcrafted regression/classification JSON metric files match recomputed predictions.
  - All 4 Member 3 saved prediction tables match summary table records.
  - Eye CNN-embedding XGBoost (`n=89`, MAE `1.7452` vs handcrafted `1.4177`), Nail CNN-embedding XGBoost (`n=109`, MAE `2.0425` vs handcrafted `1.7171`), Full-hand Palm-transfer XGBoost (`n=38`, MAE `2.0970`), and Full-hand Scratch-CNN XGBoost (`n=38`, MAE `1.5599` vs handcrafted `1.5819`) match saved metrics within `< 1e-6`.
  - All 13 `.pkl` model bundles reload and execute inference cleanly.

### B. Model Adapter & Pipeline Tests (`test_model_adapters.py` — 22/22 Passed)
1. `test_project_root`: Dynamic resolution of `Anemia_ML_Project/` via `pathlib.Path`.
2. `test_eye_xgb_bundle`, `test_nail_xgb_bundle`, `test_fullhand_xgb_bundle`, `test_eye_cls_bundle`: Bundle dictionary schemas and 130-feature contracts verified.
3. `test_feature_col_ordering`: Verifies unique, non-degenerate feature column ordering from `bundle.get("feat_cols", bundle.get("feature_cols"))`.
4. `test_quality_unreadable`, `test_quality_very_small`, `test_quality_very_dark`, `test_quality_very_bright`, `test_quality_ok_image`: Verifies quality flags on synthetic edge-case images.
5. `test_fullhand_adapter_prediction`, `test_eye_adapter_prediction`: Verifies float Hb predictions in `g/dL` within `[3.0, 20.0]`.
6. `test_adapter_unreadable_image`: Confirms unreadable files return `hb_g_dl: None` with clear error warnings and never invoke the estimator.
7. `test_pairing_passes`, `test_pairing_one_to_one`, `test_nail_crop_aggregation`, `test_split_integrity`: Confirms 250 paired participants, 1-to-1 Full-hand mapping, Nail crop mean aggregation, and zero cross-split leakage.
8. `test_fusion_input_dimensions`: Confirms `130 + 130 = 260` fused feature dimension.
9. `test_prediction_consistency`: Confirms adapter output matches direct `scaler` + `xgb.predict` call.
10. `test_app_inference_matches_saved_csv_predictions`: Confirms `FullHandXGBAdapter`, `NailXGBAdapter`, `EyeXGBAdapter`, and `FullHandScratchCNNXGBAdapter` match the saved Member 4 test prediction CSVs (`abs_diff <= 4.4e-7`).
11. `test_no_training_code_in_app_or_adapters`: Static AST verification confirming neither `app.py` nor `model_adapters.py` contains `.fit()` or `.fit_transform()` calls.

### C. Image Quality Tests (`test_image_quality.py` — 4/4 Passed)
- `test_unreadable_and_corrupt_files`: Tested missing file path, 0-byte empty file, and corrupt non-image byte payload (`is_readable == False`).
- `test_very_small_and_extreme_aspect_ratio`: Tested `16×16` small image and `800×64` (`12.5:1`) aspect ratio warning.
- `test_very_dark_and_very_bright`: Tested nearly black (`mean=5.0 < 30.0`) and nearly white (`mean=250.0 > 220.0`) exposure warnings.
- `test_blur_and_sharp_images`: Tested smooth gradient image (`Laplacian variance < 50.0` triggers blur warning) vs. high-frequency random texture (`no blur warning`).

### D. Pairing & Split Integrity Tests (`test_pairing.py` — 2/2 Passed)
- `test_pairing_audit_counts_and_alignment`: Verified 719 Nail crop rows, 250 Full-hand rows, 250 paired participants, 0 Hb mismatches, 0 split mismatches, 0 source-image mismatches, and `175 / 37 / 38` participant split counts (`503 / 107 / 109` Nail crops).
- `test_one_to_one_participant_joins_and_no_split_leakage`: Verified unique participant IDs in Full-hand and `pairing_audit.csv`, and confirmed `split.nunique() == 1` per participant across all splits.

### E. Fusion Tests (`test_fusion.py` — 3/3 Passed)
- `test_nail_crop_aggregation_to_participant_level`: Confirmed 719 Nail crops aggregate via deterministic mean to `(250, 130)` aligned with Full-hand participants.
- `test_fusion_bundle_and_dimensions`: Confirmed `fused_model.pkl` has 130 Full-hand + 130 Nail = 260 fused features and `175 / 37 / 38` splits.
- `test_fused_adapter_paired_prediction_matches_saved`: Confirmed `FusedHandNailXGBAdapter.predict_paired()` on held-out test participant `6` (`0006_full.jpg` + `0006_nail_1..3.jpg`) reproduces the exact saved prediction in `fusion_predictions.csv` (`abs_diff < 1e-4`).

### F. Streamlit Startup, Valid Prediction, and Invalid Upload Tests
- **Headless Startup (`streamlit.testing.v1.AppTest`):**
  - Executed `AppTest.from_file('member5_integration/app.py').run()`.
  - Confirmed `0` unhandled exceptions, verified all 6 model adapters loaded via `@st.cache_resource`, and confirmed the mandatory academic disclaimer rendered at the top of the page.
- **Exact Match on Held-Out Test Images (`outputs/predictions/sample_adapter_verification.csv`):**
  - `full_hand_handcrafted_xgb` on `0006_full.jpg` (True Hb `13.8` g/dL): Saved = `15.072118`, Adapter = `15.072118` (`abs_diff = 1.95e-7`).
  - `nail_handcrafted_xgb` on `0006_nail_1.jpg` (True Hb `13.8` g/dL): Saved = `12.229047`, Adapter = `12.229047` (`abs_diff = 1.78e-7`).
  - `eye_handcrafted_xgb` on `India/13/20200203_190841.jpg` (True Hb `9.1` g/dL): Saved = `11.613071`, Adapter = `11.613071` (`abs_diff = 4.42e-7`).
  - `full_hand_scratch_cnn_xgb` on `0006_full.jpg` (True Hb `13.8` g/dL): Saved = `15.043043`, Adapter = `15.043043` (`abs_diff = 3.32e-12`).

