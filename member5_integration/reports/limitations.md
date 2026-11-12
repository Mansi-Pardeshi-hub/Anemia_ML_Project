# Limitations and Scientific Caveats

**Project:** Hybrid AI-Based Non-Invasive Hemoglobin Estimation: A Feasibility Study Using Image-Derived Features  
**Status:** Academic Research Prototype Only

---

## 1. Clinical and Ethical Boundaries

1. **Not a Diagnostic Device:** This software is an undergraduate machine-learning feasibility study. It must **never** be used to diagnose anemia, guide therapy, or replace a laboratory Complete Blood Count (CBC).
2. **Uncalibrated Screening Scores:** Classification probabilities produced by Random Forest / Extra Trees models represent raw tree-vote fractions on imbalanced research cohorts. They are **not** calibrated clinical risk probabilities.
3. **Environmental & Hardware Confounding:** Handcrafted colour and texture features (RGB/HSV/LAB statistics, colour histograms, pallor/redness ratios, GLCM) and uncalibrated RGB CNN inputs depend heavily on ambient illumination, camera sensor response, white balance, exposure, and background surfaces.

---

## 2. Dataset and Cohort Limitations

1. **Small Held-Out Test Sets:**
   - **Figshare Nail / Full-hand:** 250 total participants (175 train / 37 validation / **38 test participants**). On a 38-participant test set, differences of `0.02–0.15 g/dL` in MAE fall well within sampling variability.
   - **Eye / Conjunctiva (EDI):** 202 participants, 610 deduplicated images (432 train / 89 validation / **89 test images across 31 test participants**).
2. **Crop Count vs. Participant Count:**
   - The Figshare Nail split contains **719 crops** across **250 participants** (1 to 3 crops per person; 109 crops in the 38-participant test split). Treating nail crops as independent participants inflates sample size; therefore, participant-level aggregation (mean across crops) is required when comparing against Full-hand or performing fusion.
3. **Same-Photograph Views vs. True Multimodal Sensors:**
   - Figshare Full-hand images (`0001_full.jpg`) and Nail crops (`0001_nail_1.jpg`) are extracted from the **same source photograph** (`photo/1.jpg`). Combining them represents **paired multi-view feature fusion from a single photograph**, not independent clinical multi-sensor fusion.
   - Cross-dataset fusion across Eye (India/Italy), Figshare Hand/Nail, and Mendeley Palm (Ghana) is scientifically invalid and impossible because the cohorts come from completely separate studies with no shared participants.
4. **Incomplete Demographic & Clinical Context for Thresholds:**
   - **Figshare (`<11 g/dL` rule):** Lacks age, sex, and pregnancy metadata. The universal `<11 g/dL` cutoff is an arbitrary low-Hb flag, and severe class imbalance (83% non-anemic) causes poor sensitivity (10.5% for Extra Trees, F1 = 0.1429).
   - **Eye (`<13 g/dL` male, `<12 g/dL` female rule):** Joins with source tables show ages 19–88 (including 19 participants aged > 65) and no pregnancy, altitude, or smoking records. Thus, labels are exploratory threshold derivations rather than confirmed clinical diagnoses.
   - **Palm (Folder Labels Only):** The local Mendeley Palm archive contains 4,260 augmented images (2,056 unique after exact and cross-split near-duplicate exclusions across 514 inferred groups) with folder labels (`anemic` / `non_anemic`) and **zero numerical Hb measurements**. Palm models predict folder labels only and must never be presented as Hb estimators.
5. **Licensing and Redistribution Status:**
   - While public academic repositories (Figshare, Mendeley, EDI) were used to prepare the local archives, explicit license files are absent from the local extracted folders. Licensing status is marked as **Pending Upstream Verification** prior to any public redistribution.

---

## 3. Modelling, Explainability, and Runtime Limitations

1. **Opaque CNN Embedding Dimensions:**
   - Member 3's 1024-D RegressionCNN embeddings and Member 4's 64-D Full-hand Scratch-CNN embeddings are abstract latent representations. Individual coordinates (`embedding_000`, etc.) cannot be assigned biological or anatomical meaning.
2. **SHAP Attribution Boundaries:**
   - TreeSHAP explanations on the 130 handcrafted features quantify feature contributions to the XGBoost model's output relative to the 175-participant training background. They reflect statistical correlations in the training images, not causal physiological mechanisms.
3. **Engineering Image Quality Checks Only:**
   - The quality checks in `image_quality.py` (resolution, aspect ratio, brightness, contrast, Laplacian blur variance, HSV saturation, central framing) catch corrupted or extreme uploads but cannot verify anatomical framing or lighting standardization.
4. **Offline / Clean-Install Constraints:**
   - The local Windows environment has all required packages installed (`numpy`, `pandas`, `scikit-learn`, `xgboost`, `scikit-image`, `pyarrow`, `shap`, `streamlit`, `torch`), and `Run_Project.bat` executes cleanly. However, isolated clean-venv recreation from PyPI requires internet DNS access, and `jupyter`/`nbconvert` are not installed locally (notebook execution uses the plain-Python runner `execute_notebook.py`).

