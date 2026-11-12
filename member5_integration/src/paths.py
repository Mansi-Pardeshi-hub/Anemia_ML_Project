"""
paths.py — Project-root resolver and all canonical paths for Member 5.
Works from any subdirectory; no hardcoded machine paths.
"""
from pathlib import Path

def _find_project_root() -> Path:
    """Walk up from this file until we find Anemia_ML_Project/."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if parent.name == "Anemia_ML_Project":
            return parent
        if (parent / "member4_ml_modelling").is_dir():
            return parent
    raise RuntimeError(
        "Cannot locate Anemia_ML_Project/ root. "
        "Run from inside the project directory."
    )

PROJECT_ROOT = _find_project_root()

# ── Member 4 artefacts ────────────────────────────────────────────────────────
M4_DIR        = PROJECT_ROOT / "member4_ml_modelling"
M4_MODELS_DIR = M4_DIR / "outputs" / "models"
M4_FEAT_DIR   = M4_DIR / "outputs" / "features"
M4_METRICS    = M4_DIR / "outputs" / "metrics"
M4_SRC        = M4_DIR / "src"

# Handcrafted XGBoost regression bundles
EYE_XGB_REG_PKL      = M4_MODELS_DIR / "eye_xgb_regression.pkl"
NAIL_XGB_REG_PKL     = M4_MODELS_DIR / "nail_xgb_regression.pkl"
FULLHAND_XGB_REG_PKL = M4_MODELS_DIR / "full_hand_xgb_regression.pkl"

# RF/ET regression bundles
EYE_RF_REG_PKL      = M4_MODELS_DIR / "eye_handcrafted_regression.pkl"
NAIL_RF_REG_PKL     = M4_MODELS_DIR / "nail_handcrafted_regression.pkl"
FULLHAND_RF_REG_PKL = M4_MODELS_DIR / "full_hand_handcrafted_regression.pkl"

# Classification bundles
EYE_CLS_PKL  = M4_MODELS_DIR / "eye_handcrafted_classification.pkl"
NAIL_CLS_PKL = M4_MODELS_DIR / "nail_handcrafted_classification.pkl"
PALM_CLS_PKL = M4_MODELS_DIR / "palm_handcrafted_classification.pkl"

# CNN embedding bundles
CNN_EMB_DIR           = M4_MODELS_DIR / "cnn_embedding"
EYE_CNN_EMB_XGB_PKL  = CNN_EMB_DIR / "m3_eye_cnn_embedding_xgb.pkl"
NAIL_CNN_EMB_XGB_PKL = CNN_EMB_DIR / "m3_nail_cnn_embedding_xgb.pkl"

# Full-hand scratch CNN
SCRATCH_CNN_DIR = M4_MODELS_DIR / "fullhand_cnn_scratch"
FULLHAND_SCRATCH_XGB_PKL = SCRATCH_CNN_DIR / "fullhand_scratch_cnn_embedding_xgb.pkl"
FULLHAND_SCRATCH_PTH     = SCRATCH_CNN_DIR / "fullhand_scratch_cnn_regressor.pth"

# Datasets
DATASETS_DIR     = PROJECT_ROOT / "datasets"
FIGSHARE_DIR     = DATASETS_DIR / "figshare_extracted" / "Figshare_Hb_Preprocessed"
NAIL_SPLITS_CSV  = DATASETS_DIR / "member3_extracted" / "configs" / "nail_skin_nail_splits.csv"
FULL_SPLITS_CSV  = DATASETS_DIR / "member3_extracted" / "configs" / "nail_skin_full_splits.csv"
EYE_SPLITS_CSV   = DATASETS_DIR / "member3_extracted" / "configs" / "eye_splits.csv"
PALM_SPLITS_CSV  = DATASETS_DIR / "member3_extracted" / "configs" / "palm_splits.csv"

# Feature caches
EYE_FEAT_PARQUET      = M4_FEAT_DIR / "eye_handcrafted.parquet"
NAIL_FEAT_PARQUET     = M4_FEAT_DIR / "nail_handcrafted.parquet"
FULLHAND_FEAT_PARQUET = M4_FEAT_DIR / "full_handcrafted.parquet"
PALM_FEAT_PARQUET     = M4_FEAT_DIR / "palm_handcrafted.parquet"

# ── Member 5 outputs ──────────────────────────────────────────────────────────
M5_DIR          = PROJECT_ROOT / "member5_integration"
M5_SRC          = M5_DIR / "src"
M5_OUT          = M5_DIR / "outputs"
M5_REPORTS      = M5_DIR / "reports"
M5_FUSION       = M5_OUT / "fusion"
M5_EXPL         = M5_OUT / "explainability"
M5_QUALITY      = M5_OUT / "quality"
M5_METRICS      = M5_OUT / "metrics"
M5_PREDICTIONS  = M5_OUT / "predictions"

for _d in [M5_FUSION, M5_EXPL, M5_QUALITY, M5_METRICS, M5_PREDICTIONS, M5_REPORTS]:
    _d.mkdir(parents=True, exist_ok=True)

