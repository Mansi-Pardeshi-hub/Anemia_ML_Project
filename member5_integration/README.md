# Member 5 — Integration, Explainability, Fusion, and Streamlit Demo

**Project Title:** Hybrid AI-Based Non-Invasive Hemoglobin Estimation: A Feasibility Study Using Image-Derived Features  
**Role:** Member 5 (System Integration, Paired Multi-View Fusion, Explainability, Streamlit Demo, Testing, and Final Report)

> [!CAUTION]
> **MEDICAL DISCLAIMER:** For academic research only. This experimental result is not a diagnosis and must not replace laboratory complete blood count (CBC) testing or qualified medical advice.

---

## 1. Environment Requirements

- **Operating System:** Platform-independent (`pathlib.Path` project-root resolution; tested on Windows 10/11 with Python 3.10.0).
- **Python Interpreter:** Python 3.10+ (`py -3.10` on Windows or `python3` on Linux/macOS).
- **Core Packages (`member5_integration/requirements.txt`):**
  - `numpy==2.2.4`, `pandas==2.2.3`, `Pillow==11.2.1`, `matplotlib==3.10.1`, `scipy==1.15.3`
  - `scikit-learn==1.7.2`, `xgboost==3.2.0`, `scikit-image==0.25.2`, `pyarrow==19.0.1`
  - `shap==0.49.1`, `streamlit==1.44.1`, `torch==2.14.1+cpu`

---

## 2. Installation Commands

### Platform-Independent (Linux / macOS / Windows)
```bash
python -m pip install -r member5_integration/requirements.txt
```

### Windows-Specific (Using Python 3.10 Launcher)
```powershell
py -3.10 -m pip install -r member5_integration\requirements.txt
```

---

## 3. Running Existing Member 4 Validation

Before running Member 5 scripts, verify that all Member 4 artifacts, checkpoints, and predictions are intact:

```bash
# Platform-independent
python member4_ml_modelling/src/validate_member4_results.py
```

```powershell
# Windows
py -3.10 member4_ml_modelling\src\validate_member4_results.py
```

---

## 4. Running the Nail–Full-hand Pairing Audit

Verifies that all 250 Figshare Full-hand participants match the 719 Nail crop rows (250 unique participants), with 0 Hb target mismatches, 0 source-image mismatches, and exact 175 / 37 / 38 train/validation/test participant splits:

```bash
python member5_integration/src/pairing.py
```

Outputs:
- `member5_integration/outputs/fusion/pairing_audit.csv`
- `member5_integration/reports/pairing_audit.md`

---

## 5. Training and Evaluating Paired Nail + Full-hand Fusion

Aggregates Nail crop features to the participant level (deterministic mean of 1–3 crops per participant), concatenates `[Full-hand (130-D), Nail-mean (130-D)] -> 260-D`, tunes XGBoost on the 37 validation participants, and evaluates once on the 38 held-out test participants alongside Full-hand-only and Nail-only participant-level baselines:

```bash
python member5_integration/src/fusion.py
```

Outputs:
- `member5_integration/outputs/fusion/fused_model.pkl`
- `member5_integration/outputs/fusion/fusion_predictions.csv`
- `member5_integration/outputs/fusion/fusion_metrics.json`
- `member5_integration/outputs/fusion/fusion_comparison.csv`
- `member5_integration/outputs/fusion/fusion_comparison.png`
- `member5_integration/reports/fusion_report.md`

---

## 6. Generating Explainability Outputs (TreeSHAP)

Runs `shap.TreeExplainer` (using training features only as background, n=175) on the trusted Full-hand Handcrafted XGBoost bundle (`full_hand_xgb_regression.pkl`), preserving the exact `feat_cols` order:

```bash
python member5_integration/src/explainability.py
```

Outputs:
- `member5_integration/outputs/explainability/shap_global_summary.png`
- `member5_integration/outputs/explainability/shap_local_example.png`
- `member5_integration/outputs/explainability/shap_local_example.csv`
- `member5_integration/reports/explainability.md`

---

## 7. Consolidating Metrics & Running Automated Tests

```bash
# Build consolidated comparison table and verify adapter vs saved CSV predictions
python member5_integration/src/evaluation.py
python member5_integration/src/report_utils.py

# Run all Member 5 automated test suites
python member5_integration/tests/test_model_adapters.py
python member5_integration/tests/test_image_quality.py
python member5_integration/tests/test_pairing.py
python member5_integration/tests/test_fusion.py
```

---

## 8. Launching the Streamlit Application

```bash
# Platform-independent (from project root Anemia_ML_Project/)
python -m streamlit run member5_integration/app.py
```

```powershell
# Windows PowerShell
py -3.10 -m streamlit run member5_integration\app.py
```

The Streamlit application:
- Displays the mandatory academic research disclaimer prominently at the top.
- Loads and caches trained models once at startup (`@st.cache_resource`) and **never retrains** models on user uploads.
- Performs engineering image-quality checks (readability, RGB conversion, dimensions, aspect ratio, dark/bright exposure, contrast, Laplacian variance blur score, saturation, and central framing).
- Supports Full-hand Handcrafted XGBoost, Full-hand Scratch-CNN (64-D) + XGBoost, Nail Handcrafted XGBoost, Eye Handcrafted XGBoost, Exploratory Eye Screening, and Paired Full-hand + Nail Multi-View Fusion.
- Excludes Palm from Hb estimation (Palm has folder-derived labels only and no measured laboratory Hb).

---

## 9. Expected Output Directory Structure

```text
member5_integration/
├── __init__.py
├── requirements.txt
├── README.md
├── app.py
├── src/
│   ├── __init__.py
│   ├── paths.py
│   ├── model_adapters.py
│   ├── image_quality.py
│   ├── explainability.py
│   ├── pairing.py
│   ├── fusion.py
│   ├── evaluation.py
│   └── report_utils.py
├── tests/
│   ├── test_model_adapters.py
│   ├── test_image_quality.py
│   ├── test_pairing.py
│   └── test_fusion.py
├── outputs/
│   ├── fusion/
│   ├── explainability/
│   ├── quality/
│   ├── metrics/
│   └── predictions/
└── reports/
    ├── initial_audit.md
    ├── pairing_audit.md
    ├── explainability.md
    ├── integration_report.md
    ├── fusion_report.md
    ├── testing_report.md
    ├── limitations.md
    ├── final_submission_checklist.md
    └── final_report.md
```

---

## 10. Known Limitations & Dataset Licensing Caveats

1. **Small Held-Out Test Cohorts:** The Figshare test set comprises only 38 participants (38 Full-hand images, 109 Nail crops), and the deduplicated Eye test set comprises 89 images (31 participants). Metric differences of `< 0.10 g/dL` MAE fall within small-sample uncertainty.
2. **Paired Views vs. Independent Multimodal Sensors:** Nail and Full-hand images in the Figshare dataset are bounding-box regions cropped from the **same source hand photograph** (`photo/<id>.jpg`). They are not independent clinical sensors, and cross-dataset fusion (e.g., Eye + Nail) is impossible because datasets come from unrelated studies with no shared participants.
3. **Opaque CNN Embeddings:** Individual coordinates of the 1024-D Member 3 CNN embeddings and 64-D Scratch-CNN embeddings are latent neural activations and must not be assigned biological interpretations.
4. **Incomplete Demographic Context:** Figshare lacks age and sex metadata (its `<11 g/dL` cutoff is exploratory low-Hb flagging only). The Eye dataset includes 19 participants aged > 65 and lacks pregnancy, altitude, and smoking records needed for strict WHO 2024 clinical anemia diagnosis.
5. **Dataset Licensing & Provenance Status:** Local archives contain preprocessed images and split manifests; formal upstream dataset licensing and redistribution terms for the Eye EDI dataset (India/Italy), Figshare Nail/Skin dataset, and Mendeley Palm dataset (Ghana) remain **pending verification against original publisher records** before any external redistribution.

