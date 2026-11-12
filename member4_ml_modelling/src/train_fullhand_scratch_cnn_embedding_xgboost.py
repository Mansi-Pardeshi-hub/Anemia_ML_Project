"""Fit/validate a separate XGBoost regressor on Full-hand scratch-CNN embeddings."""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "member4_ml_modelling" / "outputs"
EMBED = OUT / "embeddings" / "fullhand_scratch_cnn_embeddings.csv"
PROVENANCE = OUT / "embeddings" / "fullhand_scratch_cnn_embeddings_provenance.json"
BASELINE = OUT / "metrics" / "full_hand_xgb_reg_predictions.csv"
METRICS_DIR = OUT / "metrics" / "fullhand_cnn_experiment"
MODEL_DIR = OUT / "models" / "fullhand_cnn_scratch"
PLOT_DIR = OUT / "plots" / "fullhand_cnn_experiment"
GRID = [
    {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
    {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
    {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0},
]


def normalize_path(value):
    return str(value).replace("/", "\\").rstrip("\\").casefold()


def score(y, prediction):
    return {
        "MAE_g/dL": float(mean_absolute_error(y, prediction)),
        "RMSE_g/dL": float(mean_squared_error(y, prediction) ** 0.5),
        "R2": float(r2_score(y, prediction)),
        "n_samples": int(len(y)),
    }


def main():
    embeddings = pd.read_csv(EMBED)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    feature_cols = sorted(c for c in embeddings.columns if c.startswith("embedding_"))
    required = {"image_id", "image_path", "participant_id", "target_hb_g_dl", "split", "checkpoint_sha256"}
    if not required.issubset(embeddings.columns) or len(feature_cols) != 64:
        raise ValueError("Scratch-CNN embedding schema/dimension is invalid")
    if not embeddings.image_id.is_unique or not embeddings.image_path.is_unique:
        raise ValueError("Duplicate Full-hand image IDs or paths")
    if embeddings.groupby("participant_id").split.nunique().max() != 1:
        raise ValueError("Full-hand participant leakage across splits")
    matrix = embeddings[feature_cols].to_numpy(dtype=np.float32)
    target = pd.to_numeric(embeddings.target_hb_g_dl, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite CNN features or Hb targets")

    train = embeddings.split.eq("train").to_numpy()
    validation = embeddings.split.eq("validation").to_numpy()
    test = embeddings.split.eq("test").to_numpy()
    if (int(train.sum()), int(validation.sum()), int(test.sum())) != (175, 37, 38):
        raise ValueError("Unexpected Full-hand split counts")
    scaler = StandardScaler().fit(matrix[train])
    x_train, x_validation = scaler.transform(matrix[train]), scaler.transform(matrix[validation])
    y_train, y_validation = target[train], target[validation]

    best_model, best_params, best_val_mae = None, None, float("inf")
    validation_results = []
    for params in GRID:
        candidate = xgb.XGBRegressor(**params, random_state=42, tree_method="hist", verbosity=0, n_jobs=4)
        candidate.fit(x_train, y_train, verbose=False)
        val_mae = float(mean_absolute_error(y_validation, candidate.predict(x_validation)))
        validation_results.append({"params": params, "validation_mae_g_dl": val_mae})
        if val_mae < best_val_mae:
            best_model, best_params, best_val_mae = candidate, params, val_mae

    # The test data is first used for metrics only after validation has selected the model.
    x_test = scaler.transform(matrix[test])
    test_prediction = best_model.predict(x_test)
    test_rows = embeddings.loc[test].copy()
    test_rows["_path_key"] = test_rows.image_path.map(normalize_path)
    baseline = pd.read_csv(BASELINE)
    baseline["_path_key"] = baseline.image_path.map(normalize_path)
    if not baseline._path_key.is_unique:
        raise ValueError("Handcrafted Full-hand test paths are not unique")
    paired = test_rows.merge(baseline, on="_path_key", how="outer", suffixes=("_cnn", "_hand"), indicator=True, validate="one_to_one")
    if len(paired) != int(test.sum()) or not paired._merge.eq("both").all():
        raise ValueError("CNN embeddings and handcrafted predictions do not use identical held-out images")
    if not np.allclose(paired.target_hb_g_dl, paired.true, atol=1e-8):
        raise ValueError("CNN/handcrafted Hb targets do not match")
    if not np.array_equal(pd.to_numeric(paired.participant_id).to_numpy(), pd.to_numeric(paired.sample_id).to_numpy()):
        raise ValueError("Image participant IDs do not match the handcrafted sample IDs")
    prediction_map = dict(zip(test_rows._path_key, test_prediction))
    paired["cnn_embedding_prediction"] = paired._path_key.map(prediction_map)
    if paired.cnn_embedding_prediction.isna().any():
        raise ValueError("Unmatched CNN test prediction")
    cnn_metrics = score(paired.true, paired.cnn_embedding_prediction)
    handcrafted_metrics = score(paired.true, paired.predicted)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "dataset_id": paired.dataset_id,
        "image_id": paired.image_id,
        "image_path": paired.image_path_cnn,
        "participant_id": paired.participant_id,
        "split": "test",
        "target_hb_g_dl": paired.true,
        "fullhand_scratch_cnn_xgb_prediction_g_dl": paired.cnn_embedding_prediction,
        "handcrafted_xgb_prediction_g_dl": paired.predicted,
    })
    prediction_path = METRICS_DIR / "fullhand_scratch_cnn_xgb_predictions.csv"
    predictions.to_csv(prediction_path, index=False, float_format="%.12g")
    model_path = MODEL_DIR / "fullhand_scratch_cnn_embedding_xgb.pkl"
    with model_path.open("wb") as stream:
        pickle.dump({
            "scaler": scaler,
            "xgb": best_model,
            "embedding_columns": feature_cols,
            "cnn_checkpoint": provenance["checkpoint"],
            "cnn_checkpoint_sha256": provenance["checkpoint_sha256"],
            "dataset_id": "figshare_fullhand",
            "selection_metric": "validation MAE g/dL",
            "selected_parameters": best_params,
        }, stream)

    report = {
        "dataset": "Full-hand",
        "task": "Hb regression from source-linked numeric g/dL targets",
        "status": "Full-hand-trained scratch CNN embeddings; exploratory due to n=250 and one held-out split",
        "cnn_feature_source": "Member 4 compact FullHandScratchCNN trained from random initialization on Full-hand training participants only",
        "cnn_checkpoint": provenance["checkpoint"],
        "cnn_checkpoint_sha256": provenance["checkpoint_sha256"],
        "cnn_architecture": provenance["architecture"],
        "cnn_training_provenance": provenance,
        "target_source": "Source-linked numeric Figshare Hb g/dL; no folder-derived anemia label used",
        "split_counts": {s: int(embeddings.split.eq(s).sum()) for s in ("train", "validation", "test")},
        "participants": int(embeddings.participant_id.nunique()),
        "xgboost_validation_grid": validation_results,
        "selected_xgboost_parameters": best_params,
        "selected_xgboost_validation_mae_g_dl": best_val_mae,
        "test_rows_paired_to_handcrafted": len(predictions),
        "test_metrics": {"fullhand_scratch_cnn_embedding_xgboost": cnn_metrics, "existing_handcrafted_xgboost": handcrafted_metrics},
        "limitations": [
            "Only 175 training, 37 validation and 38 test participants are available; performance is highly uncertain and is not clinical validation.",
            "CNN checkpoint selection used validation MAE; XGBoost selection used validation MAE; test participants were not used for either choice.",
            "CNN was trained from scratch with no pretrained weights. Results are seed- and split-specific.",
            "No multimodal fusion or clinical deployment is claimed.",
        ],
        "artifacts": {"embeddings": str(EMBED.relative_to(ROOT)), "predictions": str(prediction_path.relative_to(ROOT)), "model": str(model_path.relative_to(ROOT))},
    }
    metrics_path = METRICS_DIR / "fullhand_scratch_cnn_xgb_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    low = min(float(paired.true.min()), float(paired.cnn_embedding_prediction.min()), float(paired.predicted.min()))
    high = max(float(paired.true.max()), float(paired.cnn_embedding_prediction.max()), float(paired.predicted.max()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pred_col, title, color in (
        (axes[0], "cnn_embedding_prediction", "Full-hand scratch-CNN embeddings + XGBoost", "steelblue"),
        (axes[1], "predicted", "Existing handcrafted XGBoost", "darkorange"),
    ):
        ax.scatter(paired.true, paired[pred_col], alpha=0.8, s=28, color=color)
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set_title(f"{title}\nMAE={score(paired.true, paired[pred_col])['MAE_g/dL']:.3f}")
        ax.set_xlabel("Laboratory Hb (g/dL)")
        ax.set_ylabel("Predicted Hb (g/dL)")
        ax.grid(alpha=0.2)
    fig.suptitle("Full-hand — same held-out participants/images (n=38)")
    fig.tight_layout()
    plot_path = PLOT_DIR / "fullhand_scratch_cnn_xgb_vs_handcrafted.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    report["artifacts"]["comparison_plot"] = str(plot_path.relative_to(ROOT))
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_path = OUT / "metrics" / "cnn_embedding_experiment" / "cnn_embedding_comparison_summary.csv"
    summary = pd.read_csv(summary_path)
    summary = summary[~(
        (summary.dataset == "full_hand")
        & summary.model.isin(("fullhand_scratch_cnn_embedding_xgboost", "existing_handcrafted_xgboost"))
    )]
    new = pd.DataFrame([{
        "dataset": "full_hand", "model": "fullhand_scratch_cnn_embedding_xgboost",
        "feature_type": "Member 4 Full-hand CNN trained from scratch, 64-D embedding",
        "split": "held-out test", "n_test": cnn_metrics["n_samples"],
        "MAE_g/dL": cnn_metrics["MAE_g/dL"], "RMSE_g/dL": cnn_metrics["RMSE_g/dL"], "R2": cnn_metrics["R2"],
    }, {
        "dataset": "full_hand", "model": "existing_handcrafted_xgboost",
        "feature_type": "Member 4 handcrafted", "split": "held-out test", "n_test": handcrafted_metrics["n_samples"],
        "MAE_g/dL": handcrafted_metrics["MAE_g/dL"], "RMSE_g/dL": handcrafted_metrics["RMSE_g/dL"], "R2": handcrafted_metrics["R2"],
    }])
    pd.concat([summary, new], ignore_index=True).to_csv(summary_path, index=False)
    print(f"Full-hand CNN+XGB test MAE/RMSE/R2={cnn_metrics['MAE_g/dL']:.4f}/{cnn_metrics['RMSE_g/dL']:.4f}/{cnn_metrics['R2']:.4f}; handcrafted={handcrafted_metrics['MAE_g/dL']:.4f}/{handcrafted_metrics['RMSE_g/dL']:.4f}/{handcrafted_metrics['R2']:.4f}; n={len(predictions)}")


if __name__ == "__main__":
    main()
