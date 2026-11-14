# Hybrid AI-Based Non-Invasive Hemoglobin Estimation
## A Feasibility Study Using Image-Derived Features

> **Academic Research Prototype — NOT a Diagnostic Tool.**  
> This is a third-year engineering capstone project. All outputs are experimental.  
> This system has not been clinically validated and must not replace laboratory CBC testing or medical advice.

---

## Quick Start

```powershell
# 1 — Install dependencies (Python 3.10 required)
py -3.10 -m pip install -r member5_integration\requirements.txt

# 2 — Verify all Member 4 saved artifacts
py -3.10 member4_ml_modelling\src\validate_member4_results.py

# 3 — Launch the interactive demo
py -3.10 -m streamlit run member5_integration\app.py
```

The Streamlit app opens at **http://localhost:8501** in your browser.

---

## Project Overview

This project investigates whether non-invasive digital photographs of peripheral
anatomical sites can support numerical hemoglobin (Hb) estimation and exploratory
anemia screening using machine-learning models under strict participant-level data
partitioning.

**Three datasets · Four anatomical views · Nine regression models · Leakage-free splits**

| Modality | Dataset | Participants | Test MAE (g/dL) | Best Model |
|----------|---------|:---:|:---:|:---|
| Eye / Conjunctiva | EDI Eye (India + Italy) | 202 | **1.4177** | XGBoost on 130-D handcrafted features |
| Full-hand | Figshare Nail/Skin | 250 | **1.5599** | Scratch-CNN 64-D + XGBoost |
| Nail crops | Figshare Nail/Skin | 250 | 1.7171 | XGBoost on 130-D handcrafted features |
| Palm | Mendeley (Ghana) | 526 groups | — | Folder-label classifier only (no Hb) |

Paired Full-hand + Nail fusion (260-D) achieves MAE = 1.6792 g/dL — beating Nail-only
but not Full-hand-only. Both views originate from the **same source photograph**.

---

## Repository Structure

```
Anemia_ML_Project/
├── README.md                          ← This file
├── Run_Project.bat                    ← Member 4 pipeline launcher (double-click)
├── datasets/
│   ├── dataset_edi_eyes.zip           ← Eye dataset archive (preserved, do not modify)
│   ├── eye_extracted/                 ← Extracted Eye images
│   ├── figshare_extracted/            ← Extracted Figshare Full-hand + Nail images
│   ├── palm_extracted/                ← Extracted Palm images
│   └── member3_extracted/             ← Member 3 DenseNet-121 checkpoints + configs
│
├── member4_ml_modelling/
│   ├── README.md                      ← Member 4 detailed documentation
│   ├── requirements.txt               ← Member 4 Python dependencies
│   ├── src/                           ← Pipeline scripts
│   ├── notebooks/member4_analysis.ipynb
│   ├── outputs/
│   │   ├── models/                    ← Saved model .pkl and .pth files (DO NOT OVERWRITE)
│   │   ├── metrics/                   ← JSON metrics + prediction CSVs
│   │   ├── plots/                     ← Evaluation figures
│   │   ├── features/                  ← Cached feature .parquet files
│   │   └── embeddings/                ← CNN embedding CSVs
│   └── reports/
│       ├── results_dashboard.html     ← Self-contained results page (open in browser)
│       ├── results_report.md          ← Verified metrics narrative
│       └── member4_final_completion_audit.md
│
└── member5_integration/
    ├── README.md                      ← Member 5 detailed documentation
    ├── requirements.txt               ← Full dependencies (includes torch, shap, streamlit)
    ├── app.py                         ← Streamlit demo application ← START HERE
    ├── src/                           ← Adapter, fusion, explainability, quality modules
    ├── tests/                         ← 31 automated pytest checks
    ├── outputs/
    │   ├── fusion/                    ← fused_model.pkl, fusion_metrics.json, comparison plots
    │   ├── explainability/            ← SHAP global/local plots + CSV
    │   ├── metrics/                   ← Consolidated comparison CSV
    │   └── predictions/               ← Sample adapter predictions
    └── reports/
        ├── final_report.md            ← Full 26-section project report
        ├── final_submission_checklist.md
        ├── fusion_report.md
        ├── explainability.md
        └── limitations.md
```

---

## Detailed Run Guide

### Step 1 — Install Python 3.10

Download from https://www.python.org/downloads/release/python-31011/  
Install with "Add to PATH" and "py launcher" checked.

Verify:
```powershell
py -3.10 --version    # should print: Python 3.10.11
```

### Step 2 — Install dependencies

```powershell
# From the project root (Anemia_ML_Project/)
py -3.10 -m pip install -r member5_integration\requirements.txt
```

This installs all packages for both Member 4 and Member 5.
Approximate download size: ~500 MB (PyTorch CPU is the largest component).

### Step 3 — Verify existing artifacts

```powershell
py -3.10 member4_ml_modelling\src\validate_member4_results.py
```

Expected output: `13/13 model reload/inference checks passing, 0 mismatches`.

### Step 4 — Launch the app

```powershell
py -3.10 -m streamlit run member5_integration\app.py
```

Open http://localhost:8501 in your browser.

---

## Running the Full Member 4 Pipeline

The `Run_Project.bat` launcher:
- Checks all existing baseline and CNN artifacts
- Reuses verified saved artifacts without retraining
- Runs CNN stages only if artifacts are missing or fail validation
- Executes the notebook (Jupyter if available, plain-Python fallback otherwise)
- Regenerates the HTML results dashboard

```powershell
# From project root PowerShell
cmd.exe /c Run_Project.bat
# OR double-click Run_Project.bat in Explorer
```

Expected completion: `Open member4_ml_modelling\reports\results_dashboard.html`

> **Requires:** `cnn_env\Scripts\python.exe` (isolated CNN runtime with PyTorch).
> If `cnn_env` is missing, the CNN stages will be skipped — handcrafted baselines still work.

---

## Running Member 5 Scripts Individually

All commands from project root (`Anemia_ML_Project/`):

```powershell
# Pairing audit: verify 250/250 Nail–Full-hand participant match
py -3.10 member5_integration\src\pairing.py

# Fusion: re-evaluate 260-D paired model on saved features
py -3.10 member5_integration\src\fusion.py

# SHAP explainability: regenerate global/local plots
py -3.10 member5_integration\src\explainability.py

# Consolidated metrics table
py -3.10 member5_integration\src\evaluation.py
py -3.10 member5_integration\src\report_utils.py

# Automated test suite (31 checks)
py -3.10 -m pytest member5_integration\tests -v
```

---

## What to Expect on Screen

### Streamlit App (http://localhost:8501)

| Tab | Contents |
|-----|---------|
| **Demo — Upload & Predict** | Upload an image → quality checks → experimental Hb estimate or screening score |
| **Benchmark Results** | Verified regression and classification metrics tables + fusion comparison plot |
| **SHAP Explainability** | Global importance chart + local attribution for test participant 6 |
| **Project Info & Limitations** | Architecture, team roles, 10 limitations, dataset licensing, reproducibility commands |

### HTML Results Dashboard

Open `member4_ml_modelling/reports/results_dashboard.html` in any browser.
Regenerate with:
```powershell
py -3.10 member4_ml_modelling\src\generate_results_dashboard.py
```

---

## Key Artifact Locations

| Artifact | Path |
|----------|------|
| All trained models (.pkl) | `member4_ml_modelling/outputs/models/` |
| Full-hand Scratch-CNN (.pth) | `member4_ml_modelling/outputs/models/fullhand_cnn_scratch/` |
| Fusion model (.pkl) | `member5_integration/outputs/fusion/fused_model.pkl` |
| Test predictions (CSV) | `member4_ml_modelling/outputs/metrics/*_predictions.csv` |
| Consolidated metrics (CSV) | `member5_integration/outputs/metrics/member5_consolidated_comparison.csv` |
| SHAP plots | `member5_integration/outputs/explainability/` |
| Evaluation plots | `member4_ml_modelling/outputs/plots/` |
| HTML dashboard | `member4_ml_modelling/reports/results_dashboard.html` |
| Final report | `member5_integration/reports/final_report.md` |

---

## Verified Results (from saved artifacts)

### Hb Regression — Best Models per View (held-out test set)

| View | Model | Test n | MAE (g/dL) | RMSE (g/dL) | R² |
|------|-------|:---:|:---:|:---:|:---:|
| Eye | Handcrafted XGBoost | 89 images / 31 participants | 1.4177 | 1.9744 | 0.4912 |
| Full-hand | Scratch-CNN 64-D + XGBoost | 38 participants | 1.5599 | 1.8946 | 0.4072 |
| Full-hand | Handcrafted XGBoost | 38 participants | 1.5819 | 2.0355 | 0.3157 |
| Nail (crop-level) | Handcrafted XGBoost | 109 crops / 38 participants | 1.7171 | 2.3079 | 0.0548 |
| Fused Full+Nail | 260-D XGBoost | 38 participants | 1.6792 | 2.2338 | 0.1759 |

### Exploratory Classification — Best Models per View

| View | Model | Test n | Accuracy | F1 | ROC-AUC |
|------|-------|:---:|:---:|:---:|:---:|
| Eye | ExtraTrees | 89 images | 0.8876 | 0.8810 | 0.9702 |
| Nail | ExtraTrees | 109 crops | 0.7798 | 0.1429 | 0.6099 |
| Palm | ExtraTrees | 292 images | 0.7500 | 0.7821 | 0.7788 |

---

## Known Blockers and Open Items

| Item | Status | Action Needed |
|------|--------|--------------|
| Clean venv from PyPI | Blocked (DNS failed on original machine) | Run `py -3.10 -m pip install -r member5_integration\requirements.txt` on this machine ✓ |
| Jupyter notebook kernel execution | Blocked (no nbconvert installed) | `py -3.10 -m pip install notebook nbconvert ipykernel` then `py -3.10 -m nbconvert --to notebook --execute --inplace member4_ml_modelling\notebooks\member4_analysis.ipynb` |
| Dataset license verification | Pending | Confirm redistribution terms for EDI Eye, Figshare, and Mendeley Palm with original publishers before any external release |
| Member 3 Full-hand CNN checkpoint | Missing | No Full-hand trained M3 checkpoint was delivered; the separate Palm-checkpoint transfer is exploratory only |
| WHO-stratified classification thresholds | Not feasible | Sex + pregnancy + age metadata absent from Figshare; Eye cohort lacks complete demographics |

---

## Medical Disclaimer

> This software and its underlying machine-learning models are developed strictly for
> educational and research purposes as part of an engineering capstone project.  
> It is **not** a clinical diagnostic device, has **not** been clinically validated,
> must **not** be used to replace laboratory Complete Blood Count (CBC) testing,
> and must **not** be used to provide medical advice or treatment recommendations.  
> Test errors of MAE ≈ 1.42–1.83 g/dL and RMSE ≈ 1.89–2.41 g/dL are substantial
> relative to clinical decision thresholds.

---

## Team

| Member | Role |
|--------|------|
| Member 1 | Dataset & data audit lead |
| Member 2 | Image preprocessing & quality |
| Member 3 | Deep learning feature extraction |
| Member 4 | ML modelling & evaluation |
| Member 5 | Fusion, explainability & demo/report |

---

*Last updated: 2026-11-12*
