"""
tests/test_fusion.py — Automated tests for Nail crop aggregation, fusion input
dimensions, split integrity, and fused model inference.
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
_SRC  = _THIS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paths import M5_FUSION, NAIL_FEAT_PARQUET, FULLHAND_FEAT_PARQUET
from model_adapters import FusedHandNailXGBAdapter


def test_nail_crop_aggregation_to_participant_level():
    nail_df = pd.read_parquet(NAIL_FEAT_PARQUET)
    full_df = pd.read_parquet(FULLHAND_FEAT_PARQUET)

    # 719 nail crops must aggregate to exactly 250 participants
    with open(M5_FUSION / "fused_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    raw_nail_cols = [c.removeprefix("nail_") for c in bundle["nail_feat_cols"]]
    agg = nail_df.groupby("participant_id")[raw_nail_cols].mean()
    assert agg.shape == (250, 130)
    assert set(agg.index.astype(str)) == set(full_df["participant_id"].astype(str))


def test_fusion_bundle_and_dimensions():
    pkl_path = M5_FUSION / "fused_model.pkl"
    assert pkl_path.exists(), "fused_model.pkl must exist"
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    assert len(bundle["full_feat_cols"]) == 130
    assert len(bundle["nail_feat_cols"]) == 130
    assert len(bundle["fused_feat_cols"]) == 260
    assert bundle["n_train"] == 175
    assert bundle["n_val"] == 37
    assert bundle["n_test"] == 38

    preds_df = pd.read_csv(M5_FUSION / "fusion_predictions.csv")
    assert len(preds_df) == 38
    assert preds_df["participant_id"].is_unique


def _rebase_img_path(raw_path: str) -> Path:
    """
    Rebase an image path from a saved parquet/CSV to the current project root.
    Saved artefacts may contain absolute paths from the original development machine.
    Strips everything before the first dataset or member directory and re-roots.
    """
    from paths import PROJECT_ROOT
    p = Path(str(raw_path))
    if p.exists():
        return p
    parts = p.parts
    for i, part in enumerate(parts):
        if part in ("datasets", "member4_ml_modelling", "member5_integration"):
            candidate = PROJECT_ROOT / Path(*parts[i:])
            if candidate.exists():
                return candidate
    return p  # return original; predict_image will handle the unreadable file gracefully


def test_fused_adapter_paired_prediction_matches_saved():
    full_df = pd.read_parquet(FULLHAND_FEAT_PARQUET)
    nail_df = pd.read_parquet(NAIL_FEAT_PARQUET)
    preds_df = pd.read_csv(M5_FUSION / "fusion_predictions.csv")

    row0 = preds_df.iloc[0]
    pid = int(float(row0["participant_id"]))
    saved_fused_pred = float(row0["fused_pred"])

    full_row = full_df[full_df["participant_id"].astype(int) == pid].iloc[0]
    nail_rows = nail_df[nail_df["participant_id"].astype(int) == pid]

    full_img = _rebase_img_path(str(full_row["abs_image_path"]))
    nail_imgs = [_rebase_img_path(str(p)) for p in nail_rows["abs_image_path"].tolist()]

    adapter = FusedHandNailXGBAdapter().load()
    res = adapter.predict_paired(full_img, nail_imgs)
    assert res["hb_g_dl"] is not None
    assert abs(res["hb_g_dl"] - saved_fused_pred) < 1e-4


if __name__ == "__main__":
    for fn in [
        test_nail_crop_aggregation_to_participant_level,
        test_fusion_bundle_and_dimensions,
        test_fused_adapter_paired_prediction_matches_saved,
    ]:
        fn()
        print(f"  [PASS] {fn.__name__}")
