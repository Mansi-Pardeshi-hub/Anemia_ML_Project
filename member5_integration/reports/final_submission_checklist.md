# Final Submission Checklist — Member 5 Integration + Project Polish

**Project:** Hybrid AI-Based Non-Invasive Hemoglobin Estimation: A Feasibility Study Using Image-Derived Features  
**Updated:** 2026-11-12  
**Status key:** ✅ Complete · ⚠ Partial/Caveat · 🔴 Blocked · N/A Not Applicable

---

## A. Artifact Preservation

| Item | Status | Evidence |
|------|--------|---------|
| Datasets preserved (original ZIPs and extracted images unchanged) | ✅ | Verified — no source image modification performed |
| Member 4 baseline model artifacts not overwritten | ✅ | `validate_member4_results.py` → 13/13 passing, 0 mismatches |
| Member 4 metrics and predictions not overwritten | ✅ | Launcher checks baseline artifacts before any stage |
| Member 3 DenseNet-121 checkpoints preserved | ✅ | `.pt` files in `datasets/member3_extracted/models/` intact |
| `cnn_env` isolated CNN runtime preserved | ✅ | `cnn_env/Scripts/python.exe` checked by launcher |

---

## B. Data Integrity

| Item | Status | Evidence |
|------|--------|---------|
| Eye: 146 exact duplicate images removed (610 final) | ✅ | `eye_exact_duplicate_audit.csv`; participant count 202 unchanged |
| Palm: 2,157 exact + 4 near-duplicate test images excluded | ✅ | `palm_near_duplicate_candidates.csv`; `palm_near_duplicate_test_exclusions.csv` |
| Figshare: 1 corrupted 0-byte nail file excluded | ✅ | `0236_nail_1.jpg` removed; participant 236 retains valid crops |
| Hb unit standardisation (Figshare g/L → g/dL) | ✅ | Verified in `dataset_audit_report.md`; range 6.5–17.7 g/dL |
| Zero participant leakage across all split boundaries | ✅ | `validate_member4_results.py` and `pairing.py` both confirm 0 cross-split participants |
| 250/250 Nail–Full-hand participant ID match | ✅ | `pairing_audit.csv`: 0 ID mismatches, 0 Hb mismatches, 0 split mismatches |

---

## C. Model Training & Evaluation

| Item | Status | Evidence |
|------|--------|---------|
| Handcrafted XGBoost regression: Eye, Nail, Full-hand | ✅ | Metrics in `all_metrics_summary.csv`; MAE 1.4177/1.7171/1.5819 g/dL |
| RF/ExtraTrees regression baselines | ✅ | Saved in `*_handcrafted_regression.pkl` |
| ExtraTrees/RF classification: Eye, Nail, Palm | ✅ | Saved in `*_handcrafted_classification.pkl` |
| M3 DenseNet-121 1024-D XGBoost: Eye, Nail | ✅ | `cnn_embedding_experiment/` metrics; underperforms handcrafted |
| Full-hand Scratch-CNN 64-D + XGBoost | ✅ | MAE 1.5599, RMSE 1.8946, R² 0.4072 (n=38) |
| Full-hand Palm-checkpoint transfer XGBoost | ✅ | MAE 2.0970, R² −0.1305; clearly labelled exploratory |
| Fusion 260-D XGBoost (Member 5) | ✅ | `fused_model.pkl`; MAE 1.6792, RMSE 2.2338, R² 0.1759 (n=38) |
| Test participants not used for tuning | ✅ | Imputer/scaler/hyperparameters tuned on train+val only; test evaluated once |
| Validation-selected hyperparameters documented | ✅ | `fusion_report.md` and `member4_final_completion_audit.md` |

---

## D. Explainability

| Item | Status | Evidence |
|------|--------|---------|
| TreeSHAP global summary plot generated | ✅ | `outputs/explainability/shap_global_summary.png` |
| TreeSHAP local example plot generated | ✅ | `outputs/explainability/shap_local_example.png` |
| Local attribution CSV saved | ✅ | `outputs/explainability/shap_local_example.csv` |
| XGBoost 3.2.0 base_score compatibility handled in-memory | ✅ | `explainability.py` normalises bracketed string without disk modification |
| SHAP described as model attribution, not causal/clinical | ✅ | Documented in `explainability.md` and `final_report.md` §20 |
| No Grad-CAM claimed (not implemented) | ✅ | Absent from all reports |

---

## E. Streamlit Application

| Item | Status | Evidence |
|------|--------|---------|
| Prominent academic disclaimer on every page | ✅ | `st.error()` banner at top; footer; sidebar |
| Models loaded once at startup, never retrained | ✅ | `@st.cache_resource`; static AST check in test suite confirms 0 `.fit()` calls |
| All six model modes available (5 single + 1 paired) | ✅ | Sidebar selectbox; loads only models present on disk |
| Engineering quality checks displayed before prediction | ✅ | `check_image_quality()` called on every upload |
| Invalid uploads handled gracefully | ✅ | `is_readable` guard; `try/finally` temp file cleanup |
| Fusion mode requires paired images from same photograph | ✅ | UI enforces description and warning |
| Palm excluded from Hb estimation | ✅ | Not in modality options; documented in sidebar |
| Classifier output labelled as uncalibrated exploratory score | ✅ | Screening caveat shown after every classification result |
| Tab 1: Demo upload & predict | ✅ | Single-image and paired fusion modes |
| Tab 2: Benchmark results from saved CSVs | ✅ | Loads `member5_consolidated_comparison.csv` if present; falls back to static verified table |
| Tab 3: SHAP explainability viewer | ✅ | Displays saved plots + top-10 static table |
| Tab 4: Project info, limitations, architecture, run commands | ✅ | 10 limitations, team table, reproducibility code blocks |
| App tested with Streamlit AppTest | ✅ | `test_model_adapters.py` includes AppTest smoke check |

---

## F. Testing

| Item | Status | Evidence |
|------|--------|---------|
| `test_model_adapters.py` — adapter load + predict checks | ✅ | All adapter tests passing |
| `test_image_quality.py` — quality check edge cases | ✅ | Corrupt, tiny, dark, blank images all handled |
| `test_pairing.py` — participant pairing verification | ✅ | 250/250 match confirmed |
| `test_fusion.py` — fusion pipeline checks | ✅ | Dimension, imputer, scaler, predict output verified |
| Total automated checks | ✅ | **31/31 passing** |

---

## G. Documentation

| Item | Status | Evidence |
|------|--------|---------|
| Root `README.md` with setup, run commands, artifacts, troubleshooting | ✅ | `Anemia_ML_Project/README.md` created |
| `member5_integration/README.md` with all commands | ✅ | Complete |
| `member4_ml_modelling/README.md` with pipeline and audit info | ✅ | Complete |
| `final_report.md` — 26-section full project report | ✅ | `member5_integration/reports/final_report.md` |
| `results_report.md` — verified Member 4 metrics narrative | ✅ | `member4_ml_modelling/reports/results_report.md` |
| `member4_final_completion_audit.md` | ✅ | `member4_ml_modelling/reports/` |
| `fusion_report.md` | ✅ | `member5_integration/reports/fusion_report.md` |
| `explainability.md` | ✅ | `member5_integration/reports/explainability.md` |
| `limitations.md` | ✅ | `member5_integration/reports/limitations.md` |
| `pairing_audit.md` | ✅ | `member5_integration/reports/pairing_audit.md` |
| Presentation slide deck | ✅ | `Anemia_ML_Project/presentation/slides.html` (14 slides, keyboard-navigable) |
| HTML results dashboard | ✅ | `member4_ml_modelling/reports/results_dashboard.html` |

---

## H. Reproducibility

| Item | Status | Evidence |
|------|--------|---------|
| `Run_Project.bat` completes with exit code 0 | ✅ | Launcher final addendum 2026-10-30 |
| Notebook plain-Python fallback executes 9/9 cells | ✅ | `execute_notebook.py` output saved to `.ipynb` |
| Standard Jupyter kernel execution | 🔴 **BLOCKED** | `jupyter`/`nbconvert`/`nbclient`/`ipykernel` not installed in any Python runtime. Install with: `py -3.10 -m pip install notebook nbconvert ipykernel` then `py -3.10 -m nbconvert --to notebook --execute --inplace member4_ml_modelling\notebooks\member4_analysis.ipynb` |
| Clean virtual environment install | ⚠ **NOW UNBLOCKED** | Python 3.10.11 installed on this machine. Install: `py -3.10 -m pip install -r member5_integration\requirements.txt` (torch requires `--index-url https://download.pytorch.org/whl/cpu`) |
| All seeds deterministic (seed=42) | ✅ | `random_state=42` in all models; `torch.manual_seed(42)` in Scratch-CNN |

---

## I. Scientific Accuracy & Ethics

| Item | Status | Evidence |
|------|--------|---------|
| No clinical validity claims made | ✅ | All outputs use "exploratory", "experimental", "not clinically validated" |
| Fusion correctly described as paired views from one photo | ✅ | All reports and UI include this disclaimer |
| Palm excluded from Hb regression | ✅ | No Palm regression results anywhere; app excludes Palm from Hb estimation |
| Classification thresholds described as exploratory proxies | ✅ | Multiple caveats in `final_report.md` §14, `limitations.md`, app UI |
| No invented metrics, pairings, or evidence | ✅ | All numbers traceable to saved CSV/JSON artifacts |
| Dataset licensing marked as pending | ✅ | `final_report.md` §6; README; app Tab 4 |
| Medical disclaimer prominent in app and all reports | ✅ | Consistent wording throughout |

---

## J. Deviations from Original Plan

| Deviation | Description | Impact |
|-----------|-------------|--------|
| Pretrained MobileNetV3/EfficientNet-B0 not used | Member 3 implemented `LightweightCNN` (scratch-trained, MobileNet-inspired) rather than frozen pretrained ImageNet weights. No Full-hand M3 checkpoint exists. Member 4 addressed this with a task-specific Scratch-CNN. | Planned pretrained feature extractor comparison not completed. The existing scratch-trained vs. handcrafted comparison is documented and valid. |
| Eye labels described as "assumed" in M3 config | M3 config says `assumed` but source study reports laboratory Hb in g/dL. Labels are recomputed from source study values. | Classification caveat stands; regression targets are verified as lab-measured. |
| PyPI DNS failure on original development machine | Clean virtual environment reproducibility blocked during Member 4 development. | Unblocked now that Python 3.10.11 is installed. |

---

## K. Open Items Requiring External Action

| Item | Action Needed | Owner |
|------|--------------|-------|
| Dataset licensing | Confirm redistribution terms for EDI Eye, Figshare Nail/Skin, Mendeley Palm with original publishers | Member 1 / Project supervisor |
| WHO-stratified anemia thresholds | Obtain complete sex, age, pregnancy status for Figshare cohort | Requires new data collection |
| Jupyter kernel execution | `py -3.10 -m pip install notebook nbconvert ipykernel` then run nbconvert | Any team member |
| Pretrained MobileNetV3/EfficientNet comparison | Requires internet access, pretrained weights, and new training run on 175 participants | Optional future work |
| Member 3 Full-hand regression checkpoint | No Full-hand trained M3 checkpoint was delivered | Member 3 to confirm or provide checkpoint |

