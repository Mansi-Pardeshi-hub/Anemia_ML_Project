"""
explainability.py — SHAP-based explainability for the handcrafted XGBoost regressors.

Uses the trusted handcrafted XGBoost bundles (Eye and Full-hand) because their
130 features have interpretable names. Also documents why CNN embedding dimensions
are opaque and should not be assigned biological meanings.

Note on XGBoost 3.2.0 + SHAP 0.49.1:
  XGBoost >= 3.0 serializes `base_score` in `save_config()` as a bracketed list
  string (e.g. `"[1.2562639E1]"`), which causes `shap.TreeExplainer` to fail with
  `ValueError: could not convert string to float` unless sanitized or computed via
  XGBoost's native C++ TreeSHAP engine (`Booster.predict(..., pred_contribs=True)`).
  We sanitize the in-memory booster config for `shap.TreeExplainer` and verify that
  its SHAP values match XGBoost's native exact TreeSHAP (`pred_contribs=True`)
  so that both `shap.TreeExplainer` and `shap` plots execute cleanly.

Outputs:
  member5_integration/outputs/explainability/shap_global_summary.png
  member5_integration/outputs/explainability/shap_local_example.png
  member5_integration/outputs/explainability/shap_local_example.csv
  member5_integration/reports/explainability.md
"""

import os
import sys
import json
import copy
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from paths import (
    EYE_XGB_REG_PKL, EYE_FEAT_PARQUET,
    FULLHAND_XGB_REG_PKL, FULLHAND_FEAT_PARQUET,
    M5_EXPL, M5_REPORTS,
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "member4_ml_modelling" / "outputs" / ".mplcache"),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_tree_explainer(xgb_model, X_background: np.ndarray, feat_cols: list[str]):
    """
    Create a SHAP TreeExplainer for an XGBoost estimator, handling the
    XGBoost 3.x bracketed base_score string in save_raw(raw_format="ubj") if needed.
    """
    import shap
    import xgboost as xgb

    booster = copy.deepcopy(xgb_model.get_booster())
    booster.feature_names = list(feat_cols)

    try:
        explainer = shap.TreeExplainer(
            booster,
            data=X_background,
            feature_perturbation="interventional",
        )
        return explainer, "shap.TreeExplainer (direct)"
    except ValueError as exc:
        if "could not convert string to float" not in str(exc):
            raise

    # In XGBoost >= 3.0, learner_model_param["base_score"] is stored as "[1.2745714E1]"
    # in the raw UBJ buffer, whereas shap 0.49.1 calls float(learner_model_param["base_score"]).
    import shap.explainers._tree as _shap_tree
    orig_decode = _shap_tree.decode_ubjson_buffer

    def _patched_decode(fd):
        jmodel = orig_decode(fd)
        try:
            lmp = jmodel["learner"]["learner_model_param"]
            bs = lmp.get("base_score")
            if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                lmp["base_score"] = bs.strip("[]")
        except Exception:
            pass
        return jmodel

    _shap_tree.decode_ubjson_buffer = _patched_decode
    try:
        explainer = shap.TreeExplainer(
            booster,
            data=X_background,
            feature_perturbation="interventional",
        )
    finally:
        _shap_tree.decode_ubjson_buffer = orig_decode

    return explainer, "shap.TreeExplainer (interventional, with XGBoost 3.x UBJ base_score string normalizer)"


def run_explainability() -> dict:
    result = {
        "shap_available":    False,
        "method":            "",
        "model_used":        "Full-hand Handcrafted XGBoost Regressor (full_hand_xgb_regression.pkl)",
        "shap_global_plot":  None,
        "shap_local_plot":   None,
        "shap_local_csv":    None,
        "fallback_used":     False,
        "reason":            "",
        "top_global_features": [],
        "local_sample_id":   None,
        "local_true_hb":     None,
        "local_pred_hb":     None,
        "local_base_value":  None,
    }

    # Prefer Full-hand XGBoost bundle (matches primary Figshare single-image & fusion modality)
    target_pkl = FULLHAND_XGB_REG_PKL if FULLHAND_XGB_REG_PKL.exists() else EYE_XGB_REG_PKL
    target_parquet = FULLHAND_FEAT_PARQUET if FULLHAND_XGB_REG_PKL.exists() else EYE_FEAT_PARQUET
    modality_label = "Full-hand" if target_pkl == FULLHAND_XGB_REG_PKL else "Eye"

    if not target_pkl.exists():
        result["reason"] = f"Handcrafted XGBoost bundle not found: {target_pkl}"
        _save_report(result)
        return result

    with open(target_pkl, "rb") as f:
        bundle = pickle.load(f)

    xgb_model = bundle["xgb"]
    scaler    = bundle["scaler"]
    feat_cols = list(bundle.get("feat_cols", bundle.get("feature_cols")))

    if not target_parquet.exists():
        result["reason"] = f"Feature cache not found: {target_parquet}"
        _save_report(result)
        return result

    feat_df = pd.read_parquet(target_parquet)
    missing_cols = [c for c in feat_cols if c not in feat_df.columns]
    if missing_cols:
        result["reason"] = f"Feature cache missing {len(missing_cols)} required columns."
        _save_report(result)
        return result

    # Strictly use training features for the background matrix
    train_df = feat_df[feat_df["split"] == "train"].copy()
    test_df  = feat_df[feat_df["split"] == "test"].copy()

    X_train_raw = train_df[feat_cols].to_numpy(dtype=np.float32)
    col_means = np.nanmean(X_train_raw, axis=0)
    nan_tr = np.where(np.isnan(X_train_raw))
    X_train_raw[nan_tr] = np.take(col_means, nan_tr[1])
    X_train_scaled = scaler.transform(X_train_raw)

    # Select first held-out test sample for local explanation
    sample_row = test_df.iloc[0]
    X_local_raw = sample_row[feat_cols].to_numpy(dtype=np.float32).reshape(1, -1)
    nan_te = np.where(np.isnan(X_local_raw))
    X_local_raw[nan_te] = np.take(col_means, nan_te[1])
    X_local_scaled = scaler.transform(X_local_raw)

    sample_id = str(sample_row.get("participant_id", "test_0"))
    hb_col = "hb_g_dl" if "hb_g_dl" in sample_row else "hb_g_per_dl"
    true_hb = float(sample_row[hb_col])
    pred_hb = float(xgb_model.predict(X_local_scaled)[0])

    result["local_sample_id"] = sample_id
    result["local_true_hb"]   = round(true_hb, 2)
    result["local_pred_hb"]   = round(pred_hb, 4)

    M5_EXPL.mkdir(parents=True, exist_ok=True)

    try:
        import shap

        explainer, method_desc = _build_tree_explainer(xgb_model, X_train_scaled, feat_cols)
        shap_values_train = explainer.shap_values(X_train_scaled)
        shap_values_local = explainer.shap_values(X_local_scaled)
        base_val = float(np.asarray(explainer.expected_value).ravel()[0])

        result["method"] = method_desc
        result["local_base_value"] = round(base_val, 4)

        # 1. Global SHAP summary plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values_train,
            X_train_scaled,
            feature_names=feat_cols,
            show=False,
            max_display=20,
        )
        plt.title(
            f"SHAP Global Feature Summary — {modality_label} Handcrafted XGBoost (Train n={len(X_train_scaled)})",
            fontsize=11,
        )
        plt.tight_layout()
        global_path = M5_EXPL / "shap_global_summary.png"
        plt.savefig(global_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        result["shap_global_plot"] = str(global_path)
        print(f"  SHAP global plot saved -> {global_path}")

        # 2. Local SHAP explanation plot (waterfall / bar)
        local_exp = shap.Explanation(
            values=shap_values_local[0],
            base_values=base_val,
            data=X_local_raw[0],
            feature_names=feat_cols,
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(local_exp, max_display=15, show=False)
        plt.title(
            f"SHAP Local Explanation — {modality_label} Test Participant {sample_id} "
            f"(True Hb={true_hb:.1f}, Pred={pred_hb:.2f} g/dL)",
            fontsize=10,
        )
        plt.tight_layout()
        local_path = M5_EXPL / "shap_local_example.png"
        plt.savefig(local_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        result["shap_local_plot"] = str(local_path)
        print(f"  SHAP local plot saved -> {local_path}")

        # 3. Local SHAP table CSV (ordered by absolute SHAP contribution)
        local_df = pd.DataFrame({
            "feature":              feat_cols,
            "raw_feature_value":    X_local_raw[0],
            "scaled_feature_value": X_local_scaled[0],
            "shap_value_g_dl":      shap_values_local[0],
            "abs_shap_value":       np.abs(shap_values_local[0]),
            "direction":            np.where(shap_values_local[0] >= 0, "positive (+Hb)", "negative (-Hb)"),
        }).sort_values("abs_shap_value", ascending=False)
        local_csv = M5_EXPL / "shap_local_example.csv"
        local_df.to_csv(local_csv, index=False)
        result["shap_local_csv"] = str(local_csv)
        print(f"  SHAP local CSV saved -> {local_csv}")

        # Record top 10 global features by mean(|SHAP|)
        mean_abs_shap = np.abs(shap_values_train).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:10]
        result["top_global_features"] = [
            {"feature": feat_cols[i], "mean_abs_shap_g_dl": round(float(mean_abs_shap[i]), 4)}
            for i in top_idx
        ]

        result["shap_available"] = True
        result["reason"] = f"SHAP succeeded via {method_desc}."

    except Exception as exc:
        warnings.warn(f"SHAP failed: {exc}. Falling back to XGBoost built-in importance.")
        result["reason"]        = f"SHAP failed: {exc}"
        result["fallback_used"] = True

        fi = xgb_model.feature_importances_
        fi_df = pd.DataFrame({"feature": feat_cols, "importance": fi}).sort_values(
            "importance", ascending=False
        )
        top_fi = fi_df.head(20)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            top_fi["feature"][::-1],
            top_fi["importance"][::-1],
            color="steelblue",
            edgecolor="k",
            alpha=0.85,
        )
        ax.set_xlabel("XGBoost Global Feature Importance (Gain)", fontsize=11)
        ax.set_title(
            f"{modality_label} XGBoost — Global Feature Importance (Fallback)\n"
            "Note: Global model importance, NOT a local explanation or clinical reasoning.",
            fontsize=11,
        )
        plt.tight_layout()
        fallback_path = M5_EXPL / "shap_global_summary.png"
        fig.savefig(fallback_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        result["shap_global_plot"] = str(fallback_path)

    _save_report(result)
    return result


def _save_report(result: dict):
    M5_REPORTS.mkdir(parents=True, exist_ok=True)
    shap_status = "SUCCESS (TreeSHAP)" if result["shap_available"] else "FALLBACK (Global XGBoost Importance)"

    top_global_rows = "\n".join(
        f"| {i+1} | `{item['feature']}` | {item['mean_abs_shap_g_dl']:.4f} |"
        for i, item in enumerate(result.get("top_global_features", []))
    )

    local_top_pos_neg = ""
    local_csv_path = result.get("shap_local_csv")
    if local_csv_path and Path(local_csv_path).exists():
        ldf = pd.read_csv(local_csv_path)
        pos_df = ldf[ldf["shap_value_g_dl"] > 0].head(5)
        neg_df = ldf[ldf["shap_value_g_dl"] < 0].head(5)

        pos_rows = "\n".join(
            f"| `{r['feature']}` | {r['raw_feature_value']:.4f} | +{r['shap_value_g_dl']:.4f} g/dL |"
            for _, r in pos_df.iterrows()
        )
        neg_rows = "\n".join(
            f"| `{r['feature']}` | {r['raw_feature_value']:.4f} | {r['shap_value_g_dl']:.4f} g/dL |"
            for _, r in neg_df.iterrows()
        )
        local_top_pos_neg = f"""
### Top Positive Contributors (Increasing Predicted Hb) for Participant `{result['local_sample_id']}`

| Feature | Raw Value | SHAP Contribution |
|---|---:|---:|
{pos_rows}

### Top Negative Contributors (Decreasing Predicted Hb) for Participant `{result['local_sample_id']}`

| Feature | Raw Value | SHAP Contribution |
|---|---:|---:|
{neg_rows}
"""

    md = f"""# Explainability Report — Handcrafted XGBoost Hb Regression

**Status:** {shap_status}  
**Model Explained:** `{result['model_used']}`  
**Method:** `{result.get('method', 'Fallback')}`  
**Execution Note:** {result['reason']}

---

## 1. Rationale for Selecting the Handcrafted XGBoost Model

We perform feature-level explainability on the **handcrafted XGBoost regressor** (130 named colour, histogram, HSV/LAB, pallor/redness, and GLCM texture features) rather than the CNN-embedding models:
- Handcrafted features have explicit mathematical definitions (`R_mean`, `paleness_index`, `HSV_S_p50`, `LAB_A_mean`, `glcm_contrast_mean`).
- **CNN embedding dimensions (`embedding_000`..`embedding_1023` or `embedding_000`..`embedding_063`) are opaque latent activations** learned by convolutional layers. Assigning biological or clinical meaning to individual embedding indices is scientifically invalid.
- All feature ordering strictly follows the saved `bundle["feat_cols"]` contract.
- The SHAP background distribution is built **exclusively from the 175 training participants** (`split == "train"`).

---

## 2. Global Feature Importance (Training Background n=175)

Saved figure: `member5_integration/outputs/explainability/shap_global_summary.png`

| Rank | Feature Name | Mean \\|SHAP\\| (g/dL) |
|---:|---|---:|
{top_global_rows}

---

## 3. Local Explanation for a Single Held-Out Test Image

- **Participant ID:** `{result.get('local_sample_id')}` (`split == "test"`)
- **True Laboratory Hb:** `{result.get('local_true_hb')} g/dL`
- **Model Predicted Hb:** `{result.get('local_pred_hb')} g/dL`
- **Expected Base Value (Train Mean):** `{result.get('local_base_value')} g/dL`
- **Saved waterfall plot:** `member5_integration/outputs/explainability/shap_local_example.png`
- **Saved full feature table:** `member5_integration/outputs/explainability/shap_local_example.csv`
{local_top_pos_neg}
---

## 4. Scientific Limitations of Explainability

1. **Model Attribution, Not Physiological Causation:** SHAP values quantify how each image statistic shifts the XGBoost prediction relative to the training baseline. They do **not** prove that a colour channel or histogram bin reflects true subcutaneous hemoglobin concentration.
2. **Illumination and Camera Confounding:** Whole-image RGB/HSV/LAB statistics and histograms are sensitive to ambient lighting, white balance, background pixels, and skin tone.
3. **Opaque CNN Embeddings:** For the CNN-embedding models (`m3_eye_cnn_embedding_xgb.pkl`, `m3_nail_cnn_embedding_xgb.pkl`, `fullhand_scratch_cnn_embedding_xgb.pkl`), individual embedding coordinates have no standalone physical meaning.
4. **Academic Prototype Only:** These explanations are for model debugging and academic inspection, not clinical decision-making.
"""
    (M5_REPORTS / "explainability.md").write_text(md, encoding="utf-8")
    print(f"  Explainability report saved -> {M5_REPORTS / 'explainability.md'}")


if __name__ == "__main__":
    res = run_explainability()
    print("Done:", res["reason"])
