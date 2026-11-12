"""
run_pipeline.py — Member 4 end-to-end ML modelling and evaluation pipeline.

Run from the project root:
    python member4_ml_modelling/src/run_pipeline.py

Tasks:
  A — Handcrafted feature baseline (eye, nail, full-hand images)
  B — XGBoost Hb regression on handcrafted features (CNN embeddings are a separate script)
  C — Anemia classification (eye + nail where defensible labels exist)
  D — Model comparison (handcrafted vs CNN results, modality comparison)
  E — Evaluation plots
"""

import os
import sys
import json
import warnings
import time
import pickle

# Fix matplotlib config dir on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "..", "outputs", ".mplcache"))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
import xgboost as xgb

# ── Resolve paths ──────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    PROJECT_ROOT, DATASETS_DIR, M4_DIR, RANDOM_SEED,
    EYE_PREPROCESSED_DIR, EYE_SPLITS_CSV,
    FIGSHARE_DIR, NAIL_SPLITS_CSV, FULL_SPLITS_CSV,
    PALM_DIR, PALM_SPLITS_CSV,
    PALM_NEAR_DUPLICATE_EXCLUSIONS_CSV,
    EYE_REG_PREDICTIONS_CSV, NAIL_REG_PREDICTIONS_CSV,
    EYE_CLS_PREDICTIONS_CSV, NAIL_CLS_PREDICTIONS_CSV, PALM_CLS_PREDICTIONS_CSV,
    OUT_METRICS, OUT_PLOTS, OUT_MODELS, OUT_FEATURES, REPORTS_DIR,
    ANEMIA_THRESHOLD_MALE, ANEMIA_THRESHOLD_FEMALE, ANEMIA_THRESHOLD_UNK,
)
from data_loading import (
    load_eye_splits, load_nail_splits, load_full_splits, load_palm_splits,
    verify_no_participant_leakage,
)
from feature_extraction import extract_feature_matrix, get_feature_columns
from evaluation import (
    regression_metrics, print_regression_metrics,
    classification_metrics, print_classification_metrics,
    save_metrics, save_predictions,
)
from plotting import (
    plot_actual_vs_predicted, plot_residuals, plot_regression_comparison,
    plot_feature_importance, plot_confusion_matrix, plot_roc_curve,
    plot_classification_comparison,
)

np.random.seed(RANDOM_SEED)

ALL_METRICS: list[dict] = []   # collected at end for summary CSV


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def split_df(df: pd.DataFrame):
    """Return (train, val, test) DataFrames using the 'split' column."""
    return (
        df[df["split"] == "train"].copy(),
        df[df["split"] == "validation"].copy(),
        df[df["split"] == "test"].copy(),
    )


def load_or_extract_features(manifest, cache_path, dataset_name):
    """Reuse a feature cache only when its image paths match the current manifest."""
    expected_paths = set(manifest["abs_image_path"].astype(str))
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached_paths = set(cached.get("image_path", pd.Series(dtype=str)).astype(str))
        if len(cached) == len(manifest) and cached_paths == expected_paths:
            print(f"  Loading cached {dataset_name} features...")
            return cached
        print(f"  Cached {dataset_name} features are stale; rebuilding after data validation...")

    t0 = time.time()
    features = extract_feature_matrix(
        manifest, image_col="abs_image_path", id_col="participant_id"
    )
    print(f"  Extracted {dataset_name} features in {time.time() - t0:.1f}s")
    features.to_parquet(cache_path, index=False)
    return features


def prepare_regression_arrays(feat_df: pd.DataFrame, meta_cols: list[str]):
    """Return X (feature matrix), y (Hb), and a participant-id array."""
    feat_cols = get_feature_columns(feat_df, meta_cols)
    X = feat_df[feat_cols].values.astype(np.float32)
    y = feat_df["hb_g_dl"].values.astype(np.float32)
    ids = feat_df["participant_id"].values
    # Impute any NaN columns with column means from this slice
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X, y, ids, feat_cols


def prepare_classification_arrays(feat_df: pd.DataFrame, meta_cols: list[str]):
    """Return X, y (label), ids."""
    feat_cols = get_feature_columns(feat_df, meta_cols)
    X = feat_df[feat_cols].values.astype(np.float32)
    y = feat_df["anemia_label"].values.astype(int)
    ids = feat_df["participant_id"].values
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X, y, ids, feat_cols


def record_metric(model_name, task, dataset, split_desc,
                  n_train, n_test, metrics_dict, caveats="", n_validation=None):
    row = {
        "model": model_name,
        "task": task,
        "dataset": dataset,
        "split": split_desc,
        "n_train": n_train,
        "n_validation": n_validation,
        "n_test": n_test,
    }
    row.update(metrics_dict)
    row["caveats"] = caveats
    ALL_METRICS.append(row)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK A — HANDCRAFTED FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def task_a_extract_features():
    print("\n" + "="*70)
    print("TASK A — Handcrafted Feature Extraction")
    print("="*70)

    # ── Eye features ──────────────────────────────────────────────────────────
    print("\n--- Eye / Conjunctiva ---")
    eye_df = load_eye_splits(
        EYE_SPLITS_CSV, EYE_PREPROCESSED_DIR,
        duplicate_audit_path=OUT_METRICS / "eye_exact_duplicate_audit.csv",
    )
    verify_no_participant_leakage(eye_df, "Eye")

    eye_cache = OUT_FEATURES / "eye_handcrafted.parquet"
    eye_feat = load_or_extract_features(eye_df, eye_cache, "eye")
    print(f"  Eye feature matrix: {eye_feat.shape}")

    # ── Nail features ─────────────────────────────────────────────────────────
    print("\n--- Nail (Figshare) ---")
    nail_df = load_nail_splits(NAIL_SPLITS_CSV, FIGSHARE_DIR)
    verify_no_participant_leakage(nail_df, "Nail")

    nail_cache = OUT_FEATURES / "nail_handcrafted.parquet"
    nail_feat = load_or_extract_features(nail_df, nail_cache, "nail")
    print(f"  Nail feature matrix: {nail_feat.shape}")

    # ── Full-hand features ────────────────────────────────────────────────────
    print("\n--- Full-hand images (Figshare) ---")
    full_df = load_full_splits(FULL_SPLITS_CSV, FIGSHARE_DIR)
    verify_no_participant_leakage(full_df, "Full-image")

    full_cache = OUT_FEATURES / "full_handcrafted.parquet"
    full_feat = load_or_extract_features(full_df, full_cache, "full-hand")
    print(f"  Full-image feature matrix: {full_feat.shape}")

    # ── Palm features ─────────────────────────────────────────────────────────
    print("\n--- Palm ---")
    palm_df = load_palm_splits(
        PALM_SPLITS_CSV, PALM_DIR,
        duplicate_audit_path=OUT_METRICS / "palm_exact_duplicate_exclusions.csv",
        near_duplicate_exclusions_path=PALM_NEAR_DUPLICATE_EXCLUSIONS_CSV,
    )
    verify_no_participant_leakage(palm_df, "Palm")

    palm_cache = OUT_FEATURES / "palm_handcrafted.parquet"
    palm_feat = load_or_extract_features(palm_df, palm_cache, "palm")
    print(f"  Palm feature matrix: {palm_feat.shape}")

    return eye_feat, nail_feat, full_feat, palm_feat


# ══════════════════════════════════════════════════════════════════════════════
#  TASK A PART 2 — BASELINE REGRESSION (HANDCRAFTED FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

META_COLS_EYE  = ["pid", "country", "participant_id", "processed_relative_path",
                   "full_image_path", "image_variant", "split", "hb_g_dl",
                   "anemia_label", "gender", "image_exists", "abs_image_path",
                   "image_path", "modality"]
META_COLS_NAIL = ["participant_id", "image_type", "crop_index", "image_path",
                   "hb_g_per_l", "hb_g_dl", "age", "gender", "source_image",
                   "crop_bbox_xyxy", "split", "full_image_path", "image_exists",
                   "anemia_label", "abs_image_path", "modality", "hb_g_per_dl"]
META_COLS_PALM = ["participant_id", "split", "anemia_label", "hb_g_dl",
                   "abs_image_path", "image_exists", "modality",
                   "processed_path", "class_label_from_filename",
                   "inferred_group_id", "label", "image_path", "group", "filename"]


def train_regression_models(feat_df, meta_cols, dataset_name, modality_label):
    """Train DummyRegressor + RandomForest + ExtraTrees for Hb regression."""
    print(f"\n  Regression — {dataset_name}")

    # Check Hb availability
    hb_col = "hb_g_dl"
    if hb_col not in feat_df.columns or feat_df[hb_col].isna().all():
        print(f"  ⚠ No Hb values for {dataset_name} — skipping regression.")
        return

    feat_df = feat_df[feat_df[hb_col].notna()].copy()
    train_df, val_df, test_df = split_df(feat_df)

    if len(train_df) < 10 or len(test_df) < 5:
        print(f"  ⚠ Insufficient samples (train={len(train_df)}, test={len(test_df)}) — skipping.")
        return

    # Resolve meta_cols to ones actually in df
    actual_meta = [c for c in meta_cols if c in feat_df.columns]
    feat_cols = get_feature_columns(feat_df, actual_meta)
    print(f"  Feature count: {len(feat_cols)}  |  "
          f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    def make_arrays(df):
        X = df[feat_cols].values.astype(np.float32)
        y = df[hb_col].values.astype(np.float32)
        ids = df["participant_id"].values
        col_means = np.nanmean(X, axis=0)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(col_means, nan_idx[1])
        return X, y, ids

    X_tr, y_tr, _ = make_arrays(train_df)
    X_va, y_va, _ = make_arrays(val_df)
    X_te, y_te, ids_te = make_arrays(test_df)

    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    results_for_plot = []

    # ── Dummy baseline ────────────────────────────────────────────────────────
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_tr_s, y_tr)
    dummy_pred = dummy.predict(X_te_s)
    m = regression_metrics(y_te, dummy_pred)
    m["model"] = "Dummy (mean)"
    print_regression_metrics(m, "Dummy")
    record_metric("Dummy Regressor", "regression", dataset_name, "train/test",
                  len(train_df), len(test_df), m, n_validation=len(val_df))
    results_for_plot.append(m)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    rf_pred = rf.predict(X_te_s)
    m_rf = regression_metrics(y_te, rf_pred)
    m_rf["model"] = "Random Forest"
    print_regression_metrics(m_rf, "RandomForest")
    record_metric("Random Forest Regressor", "regression", dataset_name, "train/test",
                  len(train_df), len(test_df), m_rf, n_validation=len(val_df))
    results_for_plot.append(m_rf)

    # ── Extra Trees ───────────────────────────────────────────────────────────
    et = ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    et.fit(X_tr_s, y_tr)
    et_pred = et.predict(X_te_s)
    m_et = regression_metrics(y_te, et_pred)
    m_et["model"] = "Extra Trees"
    print_regression_metrics(m_et, "ExtraTrees")
    record_metric("Extra Trees Regressor", "regression", dataset_name, "train/test",
                  len(train_df), len(test_df), m_et, n_validation=len(val_df))
    results_for_plot.append(m_et)

    # ── Save models and predictions ───────────────────────────────────────────
    slug = modality_label.lower().replace(" ", "_")
    save_obj = {"scaler": scaler, "rf": rf, "et": et, "feature_cols": feat_cols}
    with open(OUT_MODELS / f"{slug}_handcrafted_regression.pkl", "wb") as f:
        pickle.dump(save_obj, f)

    save_predictions(
        ids_te, y_te, rf_pred,
        OUT_METRICS / f"{slug}_rf_reg_predictions.csv",
        extra={"et_pred": et_pred, "image_path": test_df["image_path"].values}
    )
    save_metrics(m_rf, OUT_METRICS / f"{slug}_rf_reg_metrics.json")
    save_metrics(m_et, OUT_METRICS / f"{slug}_et_reg_metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    best_pred = rf_pred if m_rf["R2"] >= m_et["R2"] else et_pred
    best_name = "Random Forest" if m_rf["R2"] >= m_et["R2"] else "Extra Trees"
    plot_actual_vs_predicted(
        y_te, best_pred,
        title=f"{modality_label} — Handcrafted {best_name} (Hb Regression)",
        path=OUT_PLOTS / f"{slug}_actual_vs_predicted.png",
    )
    plot_residuals(
        y_te, best_pred,
        title=f"{modality_label} — {best_name}",
        path=OUT_PLOTS / f"{slug}_residuals.png",
    )
    plot_regression_comparison(
        results_for_plot,
        path=OUT_PLOTS / f"{slug}_regression_comparison.png",
        title=f"{modality_label} — Regression Model Comparison",
    )
    plot_feature_importance(
        feat_cols, rf.feature_importances_,
        path=OUT_PLOTS / f"{slug}_feature_importance.png",
        title=f"{modality_label} — RF Feature Importance (Top 20)",
    )

    return {"rf": rf, "et": et, "scaler": scaler, "feat_cols": feat_cols}


# ══════════════════════════════════════════════════════════════════════════════
#  TASK B — XGBoost REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def task_b_xgboost_regression(eye_feat, nail_feat, full_feat):
    """Train XGBoost on handcrafted features only.

    A separate CNN-embedding experiment lives in extract_m3_cnn_embeddings.py
    and train_m3_embedding_xgboost.py. Those artifacts are stored under
    outputs/metrics/cnn_embedding_experiment/ and are not overwritten here.
    """
    print("\n" + "="*70)
    print("TASK B — XGBoost Regression")
    print("="*70)

    def run_xgb(feat_df, meta_cols, dataset_name, modality_label):
        hb_col = "hb_g_dl"
        if hb_col not in feat_df.columns or feat_df[hb_col].isna().all():
            print(f"  ⚠ No Hb for {dataset_name} — skipping XGBoost.")
            return

        feat_df = feat_df[feat_df[hb_col].notna()].copy()
        train_df, val_df, test_df = split_df(feat_df)
        if len(train_df) < 10 or len(test_df) < 5:
            print(f"  ⚠ Too few samples — skipping.")
            return

        actual_meta = [c for c in meta_cols if c in feat_df.columns]
        feat_cols = get_feature_columns(feat_df, actual_meta)

        def make_arrays(df):
            X = df[feat_cols].values.astype(np.float32)
            y = df[hb_col].values.astype(np.float32)
            ids = df["participant_id"].values
            col_means = np.nanmean(X, axis=0)
            nan_idx = np.where(np.isnan(X))
            X[nan_idx] = np.take(col_means, nan_idx[1])
            return X, y, ids

        X_tr, y_tr, _ = make_arrays(train_df)
        X_va, y_va, _ = make_arrays(val_df)
        X_te, y_te, ids_te = make_arrays(test_df)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        print(f"\n  XGBoost — {dataset_name}")
        print(f"  Features: {len(feat_cols)}  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

        # Simple grid of key hyperparameters — validated on validation set
        best_val_mae = np.inf
        best_params  = {}
        param_grid = [
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,  "subsample": 0.8},
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.8},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10,  "subsample": 0.9},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,  "subsample": 0.7},
        ]
        from sklearn.metrics import mean_absolute_error as _mae_fn
        for params in param_grid:
            model = xgb.XGBRegressor(
                **params,
                random_state=RANDOM_SEED,
                tree_method="hist",
                verbosity=0,
            )
            model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
            val_mae = _mae_fn(y_va, model.predict(X_va_s))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params  = params
                best_model   = model

        print(f"  Best hyperparams (val MAE={best_val_mae:.4f}): {best_params}")

        xgb_pred = best_model.predict(X_te_s)
        m_xgb = regression_metrics(y_te, xgb_pred)
        m_xgb["model"] = f"XGBoost ({dataset_name})"
        print_regression_metrics(m_xgb, "XGBoost")
        if dataset_name in {"Eye", "Nail"}:
            caveat = (
                "Handcrafted features. A separate CNN-embedding XGBoost experiment "
                "exists for this modality; see outputs/metrics/cnn_embedding_experiment/."
            )
        else:
            caveat = (
                "Handcrafted features. No Member 3 full-hand regression checkpoint was supplied."
            )
        record_metric("XGBoost Regressor", "regression", dataset_name, "train/val/test",
                      len(train_df), len(test_df), m_xgb,
                      caveats=caveat,
                      n_validation=len(val_df))

        slug = modality_label.lower().replace(" ", "_")
        save_obj = {"scaler": scaler, "xgb": best_model, "feat_cols": feat_cols}
        with open(OUT_MODELS / f"{slug}_xgb_regression.pkl", "wb") as f:
            pickle.dump(save_obj, f)

        save_predictions(ids_te, y_te, xgb_pred,
                         OUT_METRICS / f"{slug}_xgb_reg_predictions.csv",
                         extra={"image_path": test_df["image_path"].values})
        save_metrics(m_xgb, OUT_METRICS / f"{slug}_xgb_reg_metrics.json")

        plot_actual_vs_predicted(
            y_te, xgb_pred,
            title=f"{modality_label} — XGBoost Hb Regression",
            path=OUT_PLOTS / f"{slug}_xgb_actual_vs_predicted.png",
        )
        plot_feature_importance(
            feat_cols, best_model.feature_importances_,
            path=OUT_PLOTS / f"{slug}_xgb_feature_importance.png",
            title=f"{modality_label} — XGBoost Feature Importance (Top 20)",
        )
        return m_xgb

    run_xgb(eye_feat,  META_COLS_EYE,  "Eye",  "Eye")
    run_xgb(nail_feat, META_COLS_NAIL, "Nail", "Nail")
    run_xgb(full_feat, META_COLS_NAIL, "Full-hand", "Full_hand")


# ══════════════════════════════════════════════════════════════════════════════
#  TASK C — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def train_classification_models(feat_df, meta_cols, dataset_name, modality_label,
                                 has_measured_labels: bool = True):
    """Train Dummy + RandomForest + ExtraTrees classifiers."""
    print(f"\n  Classification — {dataset_name}")

    label_col = "anemia_label"
    if label_col not in feat_df.columns or feat_df[label_col].isna().all():
        print(f"  ⚠ No labels for {dataset_name} — skipping.")
        return

    feat_df = feat_df[feat_df[label_col].notna()].copy()
    train_df, val_df, test_df = split_df(feat_df)

    if len(test_df) < 5:
        print(f"  ⚠ Test set too small ({len(test_df)}) — skipping.")
        return

    actual_meta = [c for c in meta_cols if c in feat_df.columns]
    feat_cols = get_feature_columns(feat_df, actual_meta)

    class_dist = pd.Series(train_df[label_col]).value_counts().to_dict()
    print(f"  Feature count: {len(feat_cols)}  |  "
          f"train={len(train_df)} {class_dist}, val={len(val_df)}, test={len(test_df)}")

    caveats = ""
    if not has_measured_labels:
        caveats = "Labels from filenames — not clinically verified"
    if len(class_dist) < 2:
        caveats += " | Single class in training set — cannot train classifier"
        print(f"  ⚠ Only one class present in training set. Skipping.")
        return

    def make_arrays(df):
        X = df[feat_cols].values.astype(np.float32)
        y = df[label_col].values.astype(int)
        ids = df["participant_id"].values
        col_means = np.nanmean(X, axis=0)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(col_means, nan_idx[1])
        return X, y, ids

    X_tr, y_tr, _ = make_arrays(train_df)
    X_va, y_va, _ = make_arrays(val_df)
    X_te, y_te, ids_te = make_arrays(test_df)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # Class weights for imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw = dict(zip(classes, weights))

    results_for_plot = []

    # ── Dummy baseline ────────────────────────────────────────────────────────
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(X_tr_s, y_tr)
    dummy_pred = dummy.predict(X_te_s)
    m_d = classification_metrics(y_te, dummy_pred)
    m_d["model"] = "Dummy (majority)"
    print_classification_metrics(m_d, "Dummy")
    record_metric("Dummy Classifier", "classification", dataset_name, "train/test",
                  len(train_df), len(test_df), m_d, caveats, n_validation=len(val_df))
    results_for_plot.append(m_d)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, class_weight=cw,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_tr_s, y_tr)
    rf_pred  = rf.predict(X_te_s)
    rf_prob  = rf.predict_proba(X_te_s)[:, 1]
    m_rf = classification_metrics(y_te, rf_pred, rf_prob)
    m_rf["model"] = "Random Forest"
    print_classification_metrics(m_rf, "RandomForest")
    record_metric("Random Forest Classifier", "classification", dataset_name, "train/test",
                  len(train_df), len(test_df), m_rf, caveats, n_validation=len(val_df))
    results_for_plot.append(m_rf)

    # ── Extra Trees ───────────────────────────────────────────────────────────
    et = ExtraTreesClassifier(
        n_estimators=300, class_weight=cw,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    et.fit(X_tr_s, y_tr)
    et_pred = et.predict(X_te_s)
    et_prob = et.predict_proba(X_te_s)[:, 1]
    m_et = classification_metrics(y_te, et_pred, et_prob)
    m_et["model"] = "Extra Trees"
    print_classification_metrics(m_et, "ExtraTrees")
    record_metric("Extra Trees Classifier", "classification", dataset_name, "train/test",
                  len(train_df), len(test_df), m_et, caveats, n_validation=len(val_df))
    results_for_plot.append(m_et)

    # ── Save ──────────────────────────────────────────────────────────────────
    slug = modality_label.lower().replace(" ", "_")
    save_obj = {"scaler": scaler, "rf": rf, "et": et, "feat_cols": feat_cols}
    with open(OUT_MODELS / f"{slug}_handcrafted_classification.pkl", "wb") as f:
        pickle.dump(save_obj, f)

    save_predictions(ids_te, y_te, rf_pred,
                     OUT_METRICS / f"{slug}_rf_cls_predictions.csv",
                     extra={"rf_prob": rf_prob, "et_pred": et_pred, "et_prob": et_prob,
                            "image_path": test_df["image_path"].values})
    save_metrics(m_rf, OUT_METRICS / f"{slug}_rf_cls_metrics.json")
    save_metrics(m_et, OUT_METRICS / f"{slug}_et_cls_metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    best_m = m_rf if (m_rf["f1_score"] or 0) >= (m_et["f1_score"] or 0) else m_et
    best_pred_cls = rf_pred if best_m is m_rf else et_pred
    best_prob_cls = rf_prob if best_m is m_rf else et_prob
    best_name = "Random Forest" if best_m is m_rf else "Extra Trees"

    cm_arr = confusion_matrix(y_te, best_pred_cls, labels=[0, 1])
    plot_confusion_matrix(
        cm_arr, class_names=["Non-Anemic", "Anemic"],
        path=OUT_PLOTS / f"{slug}_cls_confusion_matrix.png",
        title=f"{modality_label} — {best_name} Confusion Matrix",
    )

    if len(np.unique(y_te)) == 2:
        plot_roc_curve(
            y_te, best_prob_cls,
            path=OUT_PLOTS / f"{slug}_cls_roc_curve.png",
            title=f"{modality_label} — {best_name} ROC Curve",
            auc=best_m.get("roc_auc"),
        )

    plot_classification_comparison(
        results_for_plot,
        path=OUT_PLOTS / f"{slug}_classification_comparison.png",
        title=f"{modality_label} — Classification Model Comparison",
    )
    plot_feature_importance(
        feat_cols, rf.feature_importances_,
        path=OUT_PLOTS / f"{slug}_cls_feature_importance.png",
        title=f"{modality_label} — RF Feature Importance (Classification, Top 20)",
    )

    return {"rf": rf, "et": et, "scaler": scaler, "feat_cols": feat_cols}


def task_c_classification(eye_feat, nail_feat, palm_feat):
    print("\n" + "="*70)
    print("TASK C — Anemia Classification")
    print("="*70)

    train_classification_models(eye_feat,  META_COLS_EYE,  "Eye",  "Eye",
                                  has_measured_labels=True)
    train_classification_models(nail_feat, META_COLS_NAIL, "Nail", "Nail",
                                  has_measured_labels=True)
    train_classification_models(palm_feat, META_COLS_PALM, "Palm", "Palm",
                                  has_measured_labels=False)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK D — MODEL COMPARISON TABLE (vs Member 3 CNN results)
# ══════════════════════════════════════════════════════════════════════════════

def task_d_comparison():
    print("\n" + "="*70)
    print("TASK D — Model Comparison (Handcrafted vs CNN)")
    print("="*70)

    unpaired_note = (
        "Member3 saved test predictions; not row-aligned with Member 4 splits "
        "(no image/participant IDs). Not a paired comparison."
    )
    eye_reg = pd.read_csv(EYE_REG_PREDICTIONS_CSV)
    nail_reg = pd.read_csv(NAIL_REG_PREDICTIONS_CSV)
    eye_cls = pd.read_csv(EYE_CLS_PREDICTIONS_CSV)
    nail_cls = pd.read_csv(NAIL_CLS_PREDICTIONS_CSV)
    m3_eye_reg = regression_metrics(eye_reg["true_hb"], eye_reg["predicted_hb"])
    m3_nail_reg = regression_metrics(nail_reg["true_hb"], nail_reg["predicted_hb"])
    m3_eye_cls = classification_metrics(eye_cls["true_label"], eye_cls["predicted_label"], eye_cls["prob_anemic"])
    m3_nail_cls = classification_metrics(nail_cls["true_label"], nail_cls["predicted_label"], nail_cls["prob_anemic"])

    m3_reg = [
        {"model": "M3 CNN (Eye)", "task": "regression", "dataset": "Eye",
         "MAE_g/dL": m3_eye_reg["MAE_g/dL"], "RMSE_g/dL": m3_eye_reg["RMSE_g/dL"],
         "R2": m3_eye_reg["R2"], "n_train": 446, "n_validation": 93,
         "n_test": m3_eye_reg["n_samples"], "source": unpaired_note},
        {"model": "M3 CNN (Nail)", "task": "regression", "dataset": "Nail",
         "MAE_g/dL": m3_nail_reg["MAE_g/dL"], "RMSE_g/dL": m3_nail_reg["RMSE_g/dL"],
         "R2": m3_nail_reg["R2"], "n_train": 503, "n_validation": 107,
         "n_test": m3_nail_reg["n_samples"], "source": unpaired_note},
    ]
    m3_cls = [
        {"model": "M3 CNN (Eye)", "task": "classification", "dataset": "Eye",
         "accuracy": m3_eye_cls["accuracy"], "precision": m3_eye_cls["precision"],
         "recall": m3_eye_cls["recall"], "specificity": m3_eye_cls["specificity"],
         "f1_score": m3_eye_cls["f1_score"], "roc_auc": m3_eye_cls["roc_auc"],
         "n_train": 446, "n_validation": 93, "n_test": m3_eye_cls["n_samples"],
         "source": unpaired_note},
        {"model": "M3 CNN (Nail)", "task": "classification", "dataset": "Nail",
         "accuracy": m3_nail_cls["accuracy"], "precision": m3_nail_cls["precision"],
         "recall": m3_nail_cls["recall"], "specificity": m3_nail_cls["specificity"],
         "f1_score": m3_nail_cls["f1_score"], "roc_auc": m3_nail_cls["roc_auc"],
         "n_train": 503, "n_validation": 107, "n_test": m3_nail_cls["n_samples"],
         "source": unpaired_note,
         "caveats": "Collapsed to majority class — class imbalance"},
    ]
    for row in m3_reg + m3_cls:
        ALL_METRICS.append(row)

    print("  Member 3 Eye/Nail CNN metrics recomputed from saved prediction CSVs.")
    print("  Those CSVs have no image IDs, so they are not paired with Member 4 rows.")
    print("  Member 3 Palm result excluded: exact image duplicates cross splits,")
    print("  including images assigned conflicting filename labels.")
    print("  Handcrafted XGBoost remains in this pipeline. CNN-embedding XGBoost")
    print("  is a separate experiment under outputs/metrics/cnn_embedding_experiment/.")


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL — SAVE SUMMARY AND REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_summary():
    print("\n" + "="*70)
    print("Saving Summary Metrics Table")
    print("="*70)

    if not ALL_METRICS:
        print("  No metrics collected.")
        return

    summary_df = pd.DataFrame(ALL_METRICS)
    summary_path = OUT_METRICS / "all_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary saved → {summary_path}")
    print(f"  Total rows: {len(summary_df)}")

    # Print a readable table
    print("\n  === METRICS SUMMARY ===")
    for _, row in summary_df.iterrows():
        task = row.get("task", "")
        model = row.get("model", "")
        dataset = row.get("dataset", "")
        n_test = row.get("n_test") or row.get("n_samples", "?")
        if task == "regression":
            units = "g/dL"
            mae_key = f"MAE_{units}"
            rmse_key = f"RMSE_{units}"
            mae  = row.get(mae_key, "N/A")
            rmse = row.get(rmse_key, "N/A")
            r2   = row.get("R2", "N/A")
            print(f"  [{task}] {model} ({dataset}) n={n_test} | MAE={mae} RMSE={rmse} R²={r2}")
        else:
            acc = row.get("accuracy", "N/A")
            f1  = row.get("f1_score", "N/A")
            auc = row.get("roc_auc", "N/A")
            print(f"  [{task}] {model} ({dataset}) n={n_test} | Acc={acc} F1={f1} AUC={auc}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("Member 4 — ML Modelling & Evaluation Pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print("="*70)
    t_start = time.time()

    # Task A: Feature extraction
    eye_feat, nail_feat, full_feat, palm_feat = task_a_extract_features()

    # Task A continued: Handcrafted regression baselines
    print("\n" + "="*70)
    print("TASK A (cont.) — Handcrafted Regression Baselines")
    print("="*70)
    train_regression_models(eye_feat,  META_COLS_EYE,  "Eye",       "Eye")
    train_regression_models(nail_feat, META_COLS_NAIL, "Nail",      "Nail")
    train_regression_models(full_feat, META_COLS_NAIL, "Full-hand", "Full_hand")

    # Task B: XGBoost regression
    task_b_xgboost_regression(eye_feat, nail_feat, full_feat)

    # Task C: Classification
    task_c_classification(eye_feat, nail_feat, palm_feat)

    # Task D: Comparison
    task_d_comparison()

    # Save summary
    save_summary()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Pipeline completed in {elapsed/60:.1f} minutes.")
    print(f"Outputs in: {M4_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
