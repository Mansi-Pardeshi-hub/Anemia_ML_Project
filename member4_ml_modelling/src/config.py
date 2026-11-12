"""
config.py — Centralised path and experiment configuration for Member 4 pipeline.
All paths are relative to the project root (Anemia_ML_Project/).
"""

import os
from pathlib import Path

# ── Root locations ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # Anemia_ML_Project/
DATASETS_DIR = PROJECT_ROOT / "datasets"
M4_DIR       = PROJECT_ROOT / "member4_ml_modelling"

# ── Dataset paths (post-extraction) ───────────────────────────────────────────
EYE_PREPROCESSED_DIR  = DATASETS_DIR / "eye_extracted" / "dataset_edi" / "Eye_Preprocessed"
EYE_SPLITS_CSV        = DATASETS_DIR / "member3_extracted" / "configs" / "eye_splits.csv"

FIGSHARE_DIR          = DATASETS_DIR / "figshare_extracted" / "Figshare_Hb_Preprocessed"
NAIL_SPLITS_CSV       = DATASETS_DIR / "member3_extracted" / "configs" / "nail_skin_nail_splits.csv"
FULL_SPLITS_CSV       = DATASETS_DIR / "member3_extracted" / "configs" / "nail_skin_full_splits.csv"

PALM_DIR              = DATASETS_DIR / "palm_extracted" / "Palm_Dataset_Preprocessed"
PALM_SPLITS_CSV       = DATASETS_DIR / "member3_extracted" / "configs" / "palm_splits.csv"
PALM_NEAR_DUPLICATE_EXCLUSIONS_CSV = M4_DIR / "outputs" / "metrics" / "palm_near_duplicate_test_exclusions.csv"

# ── Member 3 model artefacts ───────────────────────────────────────────────────
M3_DIR          = DATASETS_DIR / "member3_extracted"
M3_REPORTS_DIR  = M3_DIR / "reports"
M3_RESULTS_DIR  = M3_DIR / "results" / "metrics"
M3_MODELS_DIR   = M3_DIR / "models"

EYE_REG_PREDICTIONS_CSV  = M3_RESULTS_DIR / "eye_reg_test_predictions.csv"
NAIL_REG_PREDICTIONS_CSV = M3_RESULTS_DIR / "nail_reg_test_predictions.csv"
EYE_CLS_PREDICTIONS_CSV  = M3_RESULTS_DIR / "eye_cls_test_predictions.csv"
NAIL_CLS_PREDICTIONS_CSV = M3_RESULTS_DIR / "nail_cls_test_predictions.csv"
PALM_CLS_PREDICTIONS_CSV = M3_RESULTS_DIR / "palm_test_predictions.csv"

# ── Output directories ─────────────────────────────────────────────────────────
OUT_METRICS  = M4_DIR / "outputs" / "metrics"
OUT_PLOTS    = M4_DIR / "outputs" / "plots"
OUT_MODELS   = M4_DIR / "outputs" / "models"
OUT_FEATURES = M4_DIR / "outputs" / "features"
REPORTS_DIR  = M4_DIR / "reports"

for _d in [OUT_METRICS, OUT_PLOTS, OUT_MODELS, OUT_FEATURES, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Clinical thresholds (WHO) ─────────────────────────────────────────────────
# Applied to eye dataset which has gender info
ANEMIA_THRESHOLD_MALE   = 13.0   # g/dL
ANEMIA_THRESHOLD_FEMALE = 12.0   # g/dL
ANEMIA_THRESHOLD_UNK    = 11.0   # g/dL — used when gender unavailable (Figshare)

