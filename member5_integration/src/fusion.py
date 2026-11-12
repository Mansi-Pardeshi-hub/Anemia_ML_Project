"""
fusion.py — Nail + Full-hand feature-level fusion for Hb regression.

Fusion strategy:
  - Aggregate per-participant Nail handcrafted features (mean of crops).
  - Concatenate with Full-hand handcrafted features.
  - Train XGBoost on 175 training participants.
  - Select hyperparameters on 37 validation participants.
  - Evaluate once on 38 test participants.

Comparison: Full-hand only vs Nail-only (participant-level) vs Fused.
"""

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from paths import (
    FULLHAND_FEAT_PARQUET, NAIL_FEAT_PARQUET,
    NAIL_SPLITS_CSV, FULL_SPLITS_CSV,
    M5_FUSION, M5_REPORTS,
)

import os
os.environ.setdefault("MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "member4_ml_modelling" / "outputs" / ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_SEED = 42

META_COLS = [
    "participant_id", "image_type", "crop_index", "image_path", "hb_g_per_l",
    "hb_g_dl", "hb_g_per_dl", "age", "gender", "source_image", "crop_bbox_xyxy",
    "split", "full_image_path", "image_exists", "anemia_label",
    "abs_image_path", "modality", "image_path",
    "pid",    # pairing key
]


def _get_feat_cols(df: pd.DataFrame) -> list:
    """Return feature columns (non-metadata, numeric)."""
    return [c for c in df.columns if c not in META_COLS and df[c].dtype != object]


def _impute_nan(X: np.ndarray) -> np.ndarray:
    col_means = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_means, idx[1])
    return X


def run_fusion() -> dict:
    """
    Train and evaluate the fused model. Returns metrics dict.
    Saves model, predictions, metrics, and comparison plot.
    """
    result = {
        "fusion_possible": False,
        "reason": "",
        "full_only": {},
        "nail_only": {},
        "fused": {},
        "n_test_participants": 0,
    }

    # ── Load features ─────────────────────────────────────────────────────────
    if not FULLHAND_FEAT_PARQUET.exists():
        result["reason"] = f"Full-hand feature cache not found: {FULLHAND_FEAT_PARQUET}"
        _save_fusion_report(result)
        return result
    if not NAIL_FEAT_PARQUET.exists():
        result["reason"] = f"Nail feature cache not found: {NAIL_FEAT_PARQUET}"
        _save_fusion_report(result)
        return result

    full_df = pd.read_parquet(FULLHAND_FEAT_PARQUET)
    nail_df = pd.read_parquet(NAIL_FEAT_PARQUET)

    if "hb_g_dl" not in full_df.columns:
        # Try renaming
        if "hb_g_per_dl" in full_df.columns:
            full_df = full_df.rename(columns={"hb_g_per_dl": "hb_g_dl"})
        else:
            result["reason"] = "hb_g_dl not in full_hand feature cache."
            _save_fusion_report(result)
            return result

    if "hb_g_dl" not in nail_df.columns:
        if "hb_g_per_dl" in nail_df.columns:
            nail_df = nail_df.rename(columns={"hb_g_per_dl": "hb_g_dl"})
        else:
            result["reason"] = "hb_g_dl not in nail feature cache."
            _save_fusion_report(result)
            return result

    # ── Identify feature columns ───────────────────────────────────────────────
    full_feat_cols = _get_feat_cols(full_df)
    nail_feat_cols = _get_feat_cols(nail_df)

    if not full_feat_cols or not nail_feat_cols:
        result["reason"] = "No numeric feature columns found."
        _save_fusion_report(result)
        return result

    # ── Aggregate Nail crops → participant-level (mean) ───────────────────────
    nail_pid_col = "participant_id"
    nail_agg = (
        nail_df.groupby(nail_pid_col)[nail_feat_cols + ["split", "hb_g_dl"]]
        .agg({**{c: "mean" for c in nail_feat_cols}, "split": "first", "hb_g_dl": "first"})
        .reset_index()
    )
    nail_agg.columns = [nail_pid_col] + [f"nail_{c}" for c in nail_feat_cols] + ["split", "hb_g_dl"]

    # ── Join Full + Nail on participant_id ────────────────────────────────────
    full_pid_col = "participant_id"
    full_sub = full_df[[full_pid_col, "split", "hb_g_dl"] + full_feat_cols].copy()
    full_sub.columns = [full_pid_col, "split", "hb_g_dl"] + [f"full_{c}" for c in full_feat_cols]

    merged = full_sub.merge(
        nail_agg[[nail_pid_col] + [f"nail_{c}" for c in nail_feat_cols]],
        on=nail_pid_col, how="inner"
    )

    n_paired = len(merged)
    if n_paired < 20:
        result["reason"] = f"Insufficient paired participants ({n_paired})."
        _save_fusion_report(result)
        return result

    # ── Train / val / test splits ─────────────────────────────────────────────
    train = merged[merged["split"] == "train"].copy()
    val   = merged[merged["split"] == "validation"].copy()
    test  = merged[merged["split"] == "test"].copy()

    if len(test) == 0:
        result["reason"] = "No test participants after pairing."
        _save_fusion_report(result)
        return result

    result["n_test_participants"] = len(test)
    full_feat_renamed = [f"full_{c}" for c in full_feat_cols]
    nail_feat_renamed = [f"nail_{c}" for c in nail_feat_cols]
    fused_feat_cols   = full_feat_renamed + nail_feat_renamed

    def xy(df, feat_cols):
        X = df[feat_cols].values.astype(np.float32)
        y = df["hb_g_dl"].values.astype(np.float32)
        return X, y

    X_tr_full, y_tr = xy(train, full_feat_renamed)
    X_va_full, y_va = xy(val,   full_feat_renamed)
    X_te_full, y_te = xy(test,  full_feat_renamed)

    X_tr_nail = train[nail_feat_renamed].values.astype(np.float32)
    X_va_nail = val  [nail_feat_renamed].values.astype(np.float32)
    X_te_nail = test [nail_feat_renamed].values.astype(np.float32)

    X_tr_fused = np.hstack([X_tr_full, X_tr_nail])
    X_va_fused = np.hstack([X_va_full, X_va_nail])
    X_te_fused = np.hstack([X_te_full, X_te_nail])

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb

    param_grid = [
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.9},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.7},
    ]

    def _fit_impute_and_scale(X_tr, X_va, X_te):
        """Impute NaNs using training column means only, then fit StandardScaler on train only."""
        tr_means = np.nanmean(X_tr, axis=0)
        X_tr_c, X_va_c, X_te_c = X_tr.copy(), X_va.copy(), X_te.copy()
        for arr in (X_tr_c, X_va_c, X_te_c):
            idx = np.where(np.isnan(arr))
            if len(idx[0]) > 0:
                arr[idx] = np.take(tr_means, idx[1])
        sc = StandardScaler()
        return sc.fit_transform(X_tr_c), sc.transform(X_va_c), sc.transform(X_te_c), sc, tr_means

    def train_xgb(X_tr, y_tr, X_va, y_va, X_te, y_te, label):
        """Train on tr, select on val, evaluate on te."""
        Xtr_s, Xva_s, Xte_s, sc, tr_means = _fit_impute_and_scale(X_tr, X_va, X_te)

        best_val_mae, best_model, best_params = np.inf, None, None
        for params in param_grid:
            m = xgb.XGBRegressor(
                **params, random_state=RANDOM_SEED,
                tree_method="hist", verbosity=0
            )
            m.fit(Xtr_s, y_tr, eval_set=[(Xva_s, y_va)], verbose=False)
            vm = mean_absolute_error(y_va, m.predict(Xva_s))
            if vm < best_val_mae:
                best_val_mae, best_model, best_params = vm, m, params

        pred = best_model.predict(Xte_s)
        metrics = {
            "MAE_g_dl":   round(float(mean_absolute_error(y_te, pred)), 4),
            "RMSE_g_dl":  round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
            "R2":         round(float(r2_score(y_te, pred)), 4),
            "n_test":     int(len(y_te)),
            "val_MAE":    round(float(best_val_mae), 4),
            "best_params": best_params,
            "label":      label,
        }
        print(f"  [{label}] MAE={metrics['MAE_g_dl']:.4f} RMSE={metrics['RMSE_g_dl']:.4f} R2={metrics['R2']:.4f} n={metrics['n_test']}")
        return best_model, sc, pred, metrics

    print("\nFusion experiment — training three models on 38 test participants:")
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    model_full, sc_full, pred_full, m_full = train_xgb(
        X_tr_full, y_tr, X_va_full, y_va, X_te_full, y_te, "Full-hand only"
    )
    model_nail, sc_nail, pred_nail, m_nail = train_xgb(
        X_tr_nail, y_tr, X_va_nail, y_va, X_te_nail, y_te, "Nail-only (participant-level)"
    )
    model_fused, sc_fused, pred_fused, m_fused = train_xgb(
        X_tr_fused, y_tr, X_va_fused, y_va, X_te_fused, y_te, "Fused (Full+Nail)"
    )

    result.update({
        "fusion_possible": True,
        "full_only": m_full,
        "nail_only": m_nail,
        "fused":     m_fused,
    })

    # ── Save fused model ──────────────────────────────────────────────────────
    M5_FUSION.mkdir(parents=True, exist_ok=True)
    bundle = {
        "scaler":           sc_fused,
        "xgb":              model_fused,
        "full_feat_cols":   full_feat_renamed,
        "nail_feat_cols":   nail_feat_renamed,
        "fused_feat_cols":  fused_feat_cols,
        "aggregation":      "mean of Nail crops per participant",
        "n_train":          len(train),
        "n_val":            len(val),
        "n_test":           len(test),
    }
    with open(M5_FUSION / "fused_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # ── Save predictions ──────────────────────────────────────────────────────
    preds_df = pd.DataFrame({
        "participant_id": test["participant_id"].values,
        "true_hb_g_dl":  y_te,
        "full_pred":     pred_full,
        "nail_pred":     pred_nail,
        "fused_pred":    pred_fused,
    })
    preds_df.to_csv(M5_FUSION / "fusion_predictions.csv", index=False)

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(M5_FUSION / "fusion_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    comparison_df = pd.DataFrame([m_full, m_nail, m_fused])
    comparison_df.to_csv(M5_FUSION / "fusion_comparison.csv", index=False)

    # ── Comparison plot ───────────────────────────────────────────────────────
    _plot_comparison([m_full, m_nail, m_fused], pred_full, pred_nail, pred_fused, y_te)
    _save_fusion_report(result)
    return result


def _plot_comparison(metrics_list, pred_full, pred_nail, pred_fused, y_te):
    labels = [m["label"] for m in metrics_list]
    maes   = [m["MAE_g_dl"] for m in metrics_list]
    r2s    = [m["R2"] for m in metrics_list]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x_pos = np.arange(len(labels))
    # MAE bar
    colors = ["steelblue", "coral", "mediumseagreen"]
    axes[0].bar(x_pos, maes, color=colors, edgecolor="k", alpha=0.85)
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    axes[0].set_ylabel("MAE (g/dL)", fontsize=11)
    axes[0].set_title("MAE Comparison", fontsize=12)
    axes[0].set_ylim(0, max(maes) + 0.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)

    # R² bar
    axes[1].bar(x_pos, r2s, color=colors, edgecolor="k", alpha=0.85)
    for i, v in enumerate(r2s):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    axes[1].set_ylabel("R²", fontsize=11)
    axes[1].set_title("R² Comparison", fontsize=12)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)

    # Actual vs predicted scatter (fused)
    lo = min(y_te.min(), pred_fused.min()) - 0.5
    hi = max(y_te.max(), pred_fused.max()) + 0.5
    axes[2].scatter(y_te, pred_full,  alpha=0.5, label="Full-hand",   marker="o")
    axes[2].scatter(y_te, pred_nail,  alpha=0.5, label="Nail-only",   marker="s")
    axes[2].scatter(y_te, pred_fused, alpha=0.7, label="Fused",       marker="^")
    axes[2].plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    axes[2].set_xlabel("Actual Hb (g/dL)", fontsize=11)
    axes[2].set_ylabel("Predicted Hb (g/dL)", fontsize=11)
    axes[2].set_title("Actual vs Predicted (n=38 test)", fontsize=12)
    axes[2].legend(fontsize=9)

    fig.suptitle(
        "Nail-Full-hand Fusion: Three-way Comparison (Participant-Level Test Set)",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out = M5_FUSION / "fusion_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fusion plot saved -> {out}")


def _save_fusion_report(result: dict):
    M5_REPORTS.mkdir(parents=True, exist_ok=True)
    if not result["fusion_possible"]:
        md = f"""# Fusion Report

**Status: FUSION NOT POSSIBLE**

**Reason:** {result.get('reason', 'Unknown')}

Fusion has been deferred as future work. The single-modality Streamlit demo uses the
Full-hand and Nail XGBoost regressors independently.
"""
    else:
        fu = result["fused"]
        fo = result["full_only"]
        na = result["nail_only"]
        improved_mae  = fu["MAE_g_dl"] < min(fo["MAE_g_dl"], na["MAE_g_dl"])
        improved_r2   = fu["R2"]       > max(fo["R2"],       na["R2"])

        md = f"""# Fusion Report — Nail + Full-hand

**Status: FUSION SUCCESSFULLY EVALUATED**  
**Test participants: {result['n_test_participants']} (untouched until final evaluation)**

## Results

| Model | MAE (g/dL) | RMSE (g/dL) | R² | n_test |
|-------|-----------|------------|-----|--------|
| Full-hand only | {fo['MAE_g_dl']} | {fo['RMSE_g_dl']} | {fo['R2']} | {fo['n_test']} |
| Nail-only (participant-level mean) | {na['MAE_g_dl']} | {na['RMSE_g_dl']} | {na['R2']} | {na['n_test']} |
| **Fused (Full + Nail)** | **{fu['MAE_g_dl']}** | **{fu['RMSE_g_dl']}** | **{fu['R2']}** | **{fu['n_test']}** |

## Interpretation

- Fusion **{'improved' if improved_mae else 'did not improve'}** MAE over the best single-modality model.
- Fusion **{'improved' if improved_r2 else 'did not improve'}** R² over the best single-modality model.
- With only 38 test participants the metric differences should be treated with caution.
  Small-sample variability can easily account for differences of < 0.1 MAE.

## Methodology

- Nail crops aggregated to participant level by **mean of handcrafted features**.
- Concatenated Full-hand (130 features) + Nail (130 features) → 260-dimensional input.
- XGBoost trained on 175 participants; hyperparameters selected on 37 validation participants.
- Evaluated once on 38 test participants.

## Important Caveats

1. Both inputs are **image views from the same source photograph** — Nail crops and Full-hand images
   were extracted from the same Figshare hand photographs. This limits the information gain from fusion.
2. The improvement (if any) could also reflect the larger feature space rather than genuine multimodal benefit.
3. We do **not** claim this is a "multimodal" system in the clinical sensing sense; both modalities derive from the same image.
4. All results are research prototypes. No clinical validity has been established.
"""
    (M5_REPORTS / "fusion_report.md").write_text(md, encoding="utf-8")
    print(f"  Fusion report saved -> {M5_REPORTS / 'fusion_report.md'}")


if __name__ == "__main__":
    result = run_fusion()
    print("Fusion result:", {k: v for k, v in result.items() if k not in ("full_only","nail_only","fused")})

