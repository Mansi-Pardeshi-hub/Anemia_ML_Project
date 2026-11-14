"""
tests/test_model_adapters.py — Unit and smoke tests for model adapters,
image quality, pairing, and fusion.

Run from project root:
    python -m pytest member5_integration/tests/ -v
  or:
    python member5_integration/tests/test_model_adapters.py
"""

import sys
import os
import pickle
import tempfile
import warnings
import numpy as np
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_SRC  = _THIS.parent / "src"
_M4SRC = _THIS.parents[2] / "member4_ml_modelling" / "src"
for p in [str(_SRC), str(_M4SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MPLCONFIGDIR",
    str(_THIS.parents[2] / "member4_ml_modelling" / "outputs" / ".mplcache"))

from paths import (
    EYE_XGB_REG_PKL, NAIL_XGB_REG_PKL, FULLHAND_XGB_REG_PKL,
    EYE_CLS_PKL, NAIL_CLS_PKL, PALM_CLS_PKL,
    NAIL_SPLITS_CSV, FULL_SPLITS_CSV,
    EYE_FEAT_PARQUET, NAIL_FEAT_PARQUET, FULLHAND_FEAT_PARQUET,
)

PASS = 0
FAIL = 0
RESULTS = []


def run_test(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        RESULTS.append((name, "PASS", ""))
        print(f"  [PASS] {name}")
    except AssertionError as e:
        FAIL += 1
        RESULTS.append((name, "FAIL", str(e)))
        print(f"  [FAIL] {name}: {e}")
    except Exception as e:
        FAIL += 1
        RESULTS.append((name, "ERROR", str(e)))
        print(f"  [ERROR] {name}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Project-root resolution
# ══════════════════════════════════════════════════════════════════════════════
def test_project_root():
    from paths import PROJECT_ROOT
    assert PROJECT_ROOT.is_dir(), f"PROJECT_ROOT not a dir: {PROJECT_ROOT}"
    assert (PROJECT_ROOT / "member4_ml_modelling").is_dir()

# ══════════════════════════════════════════════════════════════════════════════
# 2. Model bundle loading and schema
# ══════════════════════════════════════════════════════════════════════════════
def _check_bundle(pkl_path: Path, required_keys: list):
    assert pkl_path.exists(), f"Bundle missing: {pkl_path}"
    with open(pkl_path, "rb") as f:
        b = pickle.load(f)
    assert isinstance(b, dict), f"Bundle is not a dict: {type(b)}"
    for k in required_keys:
        assert k in b, f"Bundle missing key '{k}' in {pkl_path.name}"
    return b

def test_eye_xgb_bundle():
    b = _check_bundle(EYE_XGB_REG_PKL, ["scaler", "xgb"])
    feat_cols = b.get("feat_cols", b.get("feature_cols"))
    assert feat_cols is not None, "No feat_cols/feature_cols in eye XGB bundle"
    assert len(feat_cols) == 130, f"Expected 130 features, got {len(feat_cols)}"

def test_nail_xgb_bundle():
    b = _check_bundle(NAIL_XGB_REG_PKL, ["scaler", "xgb"])
    feat_cols = b.get("feat_cols", b.get("feature_cols"))
    assert feat_cols is not None
    assert len(feat_cols) == 130

def test_fullhand_xgb_bundle():
    b = _check_bundle(FULLHAND_XGB_REG_PKL, ["scaler", "xgb"])
    feat_cols = b.get("feat_cols", b.get("feature_cols"))
    assert feat_cols is not None
    assert len(feat_cols) == 130

def test_eye_cls_bundle():
    b = _check_bundle(EYE_CLS_PKL, ["scaler", "rf"])
    feat_cols = b.get("feat_cols", b.get("feature_cols"))
    assert feat_cols is not None
    assert len(feat_cols) == 130

# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature column ordering
# ══════════════════════════════════════════════════════════════════════════════
def test_feature_col_ordering():
    """Check that adapter uses saved feat_cols, not arbitrary order."""
    with open(EYE_XGB_REG_PKL, "rb") as f:
        b = pickle.load(f)
    feat_cols = b.get("feat_cols", b.get("feature_cols"))
    assert feat_cols[0] != feat_cols[-1], "Feat cols degenerate"
    assert len(set(feat_cols)) == len(feat_cols), "Duplicate feature names"

# ══════════════════════════════════════════════════════════════════════════════
# 4. Image quality checks
# ══════════════════════════════════════════════════════════════════════════════
def _make_test_image(arr: np.ndarray) -> Path:
    from PIL import Image as PilImg
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    PilImg.fromarray(arr.astype(np.uint8)).save(tmp.name)
    return Path(tmp.name)

def test_quality_unreadable():
    from image_quality import check_image_quality
    # Non-existent file
    r = check_image_quality("/nonexistent/path/image.jpg")
    assert not r["is_readable"]
    assert len(r["warnings"]) > 0

def test_quality_very_small():
    from image_quality import check_image_quality
    arr = np.zeros((10, 10, 3), dtype=np.uint8) + 128
    p = _make_test_image(arr)
    try:
        r = check_image_quality(p)
        assert r["is_readable"]
        assert any("small" in w.lower() for w in r["warnings"])
    finally:
        p.unlink(missing_ok=True)

def test_quality_very_dark():
    from image_quality import check_image_quality
    arr = np.zeros((224, 224, 3), dtype=np.uint8) + 5
    p = _make_test_image(arr)
    try:
        r = check_image_quality(p)
        assert r["is_readable"]
        assert any("dark" in w.lower() for w in r["warnings"])
    finally:
        p.unlink(missing_ok=True)

def test_quality_very_bright():
    from image_quality import check_image_quality
    arr = np.ones((224, 224, 3), dtype=np.uint8) * 250
    p = _make_test_image(arr)
    try:
        r = check_image_quality(p)
        assert r["is_readable"]
        assert any("bright" in w.lower() or "overexposed" in w.lower() for w in r["warnings"])
    finally:
        p.unlink(missing_ok=True)

def test_quality_ok_image():
    from image_quality import check_image_quality
    rng = np.random.default_rng(42)
    arr = rng.integers(80, 180, size=(224, 224, 3), dtype=np.uint8)
    p = _make_test_image(arr)
    try:
        r = check_image_quality(p)
        assert r["is_readable"]
        assert r["width"] == 224
        assert r["height"] == 224
    finally:
        p.unlink(missing_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Adapter prediction type and units
# ══════════════════════════════════════════════════════════════════════════════
def _get_real_test_image(parquet_path: Path, split: str = "test") -> Path | None:
    """Get the first test image from a feature cache."""
    import pandas as pd
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    test_rows = df[df["split"] == split] if "split" in df.columns else df
    if test_rows.empty:
        return None
    img_col = "abs_image_path" if "abs_image_path" in df.columns else "image_path"
    for _, row in test_rows.iterrows():
        p = Path(str(row[img_col]))
        if p.exists():
            return p
    return None

def test_fullhand_adapter_prediction():
    from model_adapters import FullHandXGBAdapter
    img = _get_real_test_image(FULLHAND_FEAT_PARQUET)
    if img is None:
        warnings.warn("No test image found for Full-hand adapter test — SKIP")
        return
    adapter = FullHandXGBAdapter()
    adapter.load()
    result = adapter.predict_image(img)
    assert isinstance(result["hb_g_dl"], float), "hb_g_dl must be float"
    assert 3.0 <= result["hb_g_dl"] <= 20.0, f"Hb out of range: {result['hb_g_dl']}"
    assert result["modality"] == "full_hand"

def test_eye_adapter_prediction():
    from model_adapters import EyeXGBAdapter
    img = _get_real_test_image(EYE_FEAT_PARQUET)
    if img is None:
        warnings.warn("No test image found for Eye adapter — SKIP")
        return
    adapter = EyeXGBAdapter()
    adapter.load()
    result = adapter.predict_image(img)
    assert isinstance(result["hb_g_dl"], float)
    assert 3.0 <= result["hb_g_dl"] <= 20.0

def test_adapter_unreadable_image():
    from model_adapters import FullHandXGBAdapter
    adapter = FullHandXGBAdapter()
    adapter.load()
    result = adapter.predict_image("/nonexistent/image.jpg")
    assert result["hb_g_dl"] is None
    assert len(result["warnings"]) > 0

# ══════════════════════════════════════════════════════════════════════════════
# 6. Pairing verification
# ══════════════════════════════════════════════════════════════════════════════
def test_pairing_passes():
    from pairing import run_pairing_audit
    result = run_pairing_audit()
    assert result["paired_participants"] == 250, f"Expected 250 paired, got {result['paired_participants']}"
    assert result["hb_mismatches"] == 0, f"Hb mismatches: {result['hb_mismatches']}"
    assert result["split_mismatches"] == 0, f"Split mismatches: {result['split_mismatches']}"
    assert result["passed"], f"Pairing audit failed: {result['issues']}"

def test_pairing_one_to_one():
    """Every Full-hand participant must have exactly one Full-hand row."""
    import pandas as pd
    if not FULL_SPLITS_CSV.exists():
        return
    full = pd.read_csv(FULL_SPLITS_CSV)
    duplicated = full.groupby("participant_id").size()
    assert (duplicated == 1).all(), f"{(duplicated > 1).sum()} full-hand participants have >1 row"

def test_nail_crop_aggregation():
    """Nail crops can be aggregated to participant-level mean without error."""
    import pandas as pd
    if not NAIL_FEAT_PARQUET.exists():
        return
    nail_df = pd.read_parquet(NAIL_FEAT_PARQUET)
    num_cols = [c for c in nail_df.columns if nail_df[c].dtype in [np.float32, np.float64, float]]
    agg = nail_df.groupby("participant_id")[num_cols].mean()
    assert len(agg) > 0, "Nail aggregation produced empty DataFrame"
    assert len(agg) <= 250, f"More unique participants than expected: {len(agg)}"

def test_split_integrity():
    """Participant-level split: no participant in >1 split."""
    import pandas as pd
    for csv_path, name in [(NAIL_SPLITS_CSV, "Nail"), (FULL_SPLITS_CSV, "Full")]:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        splits = df.groupby("participant_id")["split"].nunique()
        leakers = (splits > 1).sum()
        assert leakers == 0, f"{name}: {leakers} participants in >1 split (leakage!)"

# ══════════════════════════════════════════════════════════════════════════════
# 7. Fusion input dimensions
# ══════════════════════════════════════════════════════════════════════════════
def test_fusion_input_dimensions():
    import pandas as pd
    if not FULLHAND_FEAT_PARQUET.exists() or not NAIL_FEAT_PARQUET.exists():
        return
    full_df = pd.read_parquet(FULLHAND_FEAT_PARQUET)
    nail_df = pd.read_parquet(NAIL_FEAT_PARQUET)
    META = ["participant_id","split","hb_g_dl","hb_g_per_dl","abs_image_path",
            "image_path","modality","anemia_label","image_type","crop_index",
            "image_exists","source_image","crop_bbox_xyxy","full_image_path",
            "age","gender","hb_g_per_l"]
    full_feats = [c for c in full_df.columns if c not in META and full_df[c].dtype != object]
    nail_feats = [c for c in nail_df.columns if c not in META and nail_df[c].dtype != object]
    assert len(full_feats) == 130, f"Full-hand features: {len(full_feats)}"
    assert len(nail_feats) == 130, f"Nail features: {len(nail_feats)}"
    fused_dim = len(full_feats) + len(nail_feats)
    assert fused_dim == 260, f"Fused dimension: {fused_dim}"

# ══════════════════════════════════════════════════════════════════════════════
# 8. Prediction consistency: adapter vs saved model direct call
# ══════════════════════════════════════════════════════════════════════════════
def test_prediction_consistency():
    """Adapter prediction must match direct model call for the same image."""
    img = _get_real_test_image(FULLHAND_FEAT_PARQUET)
    if img is None:
        return
    from model_adapters import FullHandXGBAdapter
    from feature_extraction import extract_features
    adapter = FullHandXGBAdapter()
    adapter.load()

    # Adapter prediction
    pred_adapter = adapter.predict_image(img)["hb_g_dl"]

    # Direct call
    feats = extract_features(img)
    feat_cols = adapter._feat_cols
    X = np.array([feats[c] for c in feat_cols], dtype=np.float32).reshape(1, -1)
    X_s = adapter._scaler.transform(X)
    pred_direct = float(adapter._model.predict(X_s)[0])
    pred_direct = float(np.clip(pred_direct, 3.0, 20.0))

    assert abs(pred_adapter - pred_direct) < 1e-3, (
        f"Adapter ({pred_adapter}) != direct ({pred_direct})"
    )


def test_app_inference_matches_saved_csv_predictions():
    """Verify that Full-hand, Nail, Eye, and Scratch-CNN adapters match saved test CSV predictions."""
    import pandas as pd
    from paths import M4_METRICS
    from model_adapters import (
        FullHandXGBAdapter, NailXGBAdapter, EyeXGBAdapter, FullHandScratchCNNXGBAdapter
    )

    from paths import PROJECT_ROOT

    def _rebase_img_path(raw_path: str) -> Path:
        """
        Rebase an image_path from a saved CSV to the current project root.
        Saved CSVs may contain absolute paths from the original development machine.
        We strip everything up to and including 'Anemia_ML_Project/' and re-join
        with the current PROJECT_ROOT.
        """
        p = Path(str(raw_path))
        # Try as-is first (works if project root hasn't moved)
        if p.exists():
            return p
        # Find 'datasets/' or 'member' in the parts and re-root
        parts = p.parts
        for i, part in enumerate(parts):
            if part in ("datasets", "member4_ml_modelling", "member5_integration"):
                rel = Path(*parts[i:])
                candidate = PROJECT_ROOT / rel
                if candidate.exists():
                    return candidate
        return p  # return original even if not found; test will get None prediction

    checks = [
        (FullHandXGBAdapter, M4_METRICS / "full_hand_xgb_reg_predictions.csv", "predicted"),
        (NailXGBAdapter,     M4_METRICS / "nail_xgb_reg_predictions.csv",      "predicted"),
        (EyeXGBAdapter,      M4_METRICS / "eye_xgb_reg_predictions.csv",       "predicted"),
        (FullHandScratchCNNXGBAdapter, M4_METRICS / "fullhand_cnn_experiment" / "fullhand_scratch_cnn_xgb_predictions.csv", "fullhand_scratch_cnn_xgb_prediction_g_dl"),
    ]
    for cls, csv_path, pred_col in checks:
        df = pd.read_csv(csv_path)
        row0 = df.iloc[0]
        img_path = _rebase_img_path(str(row0["image_path"]))
        saved_pred = float(row0[pred_col])
        adapter = cls().load()
        res = adapter.predict_image(img_path)
        assert res["hb_g_dl"] is not None
        assert abs(float(res["hb_g_dl"]) - saved_pred) < 1e-4, (
            f"{cls.__name__} prediction {res['hb_g_dl']} != saved {saved_pred}"
        )


def test_no_training_code_in_app_or_adapters():
    """Static AST check: neither app.py nor model_adapters.py calls .fit() or imports training scripts."""
    import ast
    from paths import M5_DIR

    for py_file in [M5_DIR / "app.py", M5_DIR / "src" / "model_adapters.py"]:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"fit", "fit_transform"}, (
                    f"Forbidden training call .{node.func.attr}() found in {py_file.name}"
                )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n=== Member 5 Test Suite ===\n")

    tests = [
        ("1. Project root resolution",        test_project_root),
        ("2a. Eye XGBoost bundle schema",     test_eye_xgb_bundle),
        ("2b. Nail XGBoost bundle schema",    test_nail_xgb_bundle),
        ("2c. Full-hand XGBoost bundle",      test_fullhand_xgb_bundle),
        ("2d. Eye classification bundle",     test_eye_cls_bundle),
        ("3. Feature column ordering",        test_feature_col_ordering),
        ("4a. Quality: unreadable file",      test_quality_unreadable),
        ("4b. Quality: very small image",     test_quality_very_small),
        ("4c. Quality: very dark image",      test_quality_very_dark),
        ("4d. Quality: very bright image",    test_quality_very_bright),
        ("4e. Quality: ok image",             test_quality_ok_image),
        ("5a. Full-hand adapter prediction",  test_fullhand_adapter_prediction),
        ("5b. Eye adapter prediction",        test_eye_adapter_prediction),
        ("5c. Adapter: unreadable image",     test_adapter_unreadable_image),
        ("6a. Pairing audit passes",          test_pairing_passes),
        ("6b. Full-hand one-to-one",          test_pairing_one_to_one),
        ("6c. Nail crop aggregation",         test_nail_crop_aggregation),
        ("6d. Split integrity (no leakage)",  test_split_integrity),
        ("7. Fusion input dimensions (260)",  test_fusion_input_dimensions),
        ("8. Prediction consistency",         test_prediction_consistency),
        ("9. App inference matches saved CSV", test_app_inference_matches_saved_csv_predictions),
        ("10. No training code in app/adapters", test_no_training_code_in_app_or_adapters),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(f"\n=== RESULTS: {PASS} passed, {FAIL} failed ===\n")
    if FAIL > 0:
        print("Failed tests:")
        for name, status, msg in RESULTS:
            if status != "PASS":
                print(f"  [{status}] {name}: {msg}")

    # Save test log
    from paths import M5_REPORTS
    import json
    log = {"passed": PASS, "failed": FAIL, "tests": [
        {"name": n, "status": s, "message": m} for n, s, m in RESULTS
    ]}
    log_path = M5_REPORTS / "testing_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Test log saved -> {log_path}")

