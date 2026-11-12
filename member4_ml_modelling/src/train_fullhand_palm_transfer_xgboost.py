"""Fit validation-selected XGBoost on exploratory transferred Palm-CNN features."""

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
EMBED = OUT / "embeddings" / "m3_palm_classifier_fullhand_transfer_embeddings.csv"
PROVENANCE = OUT / "embeddings" / "m3_palm_classifier_fullhand_transfer_embeddings_provenance.json"
BASELINE = OUT / "metrics" / "full_hand_xgb_reg_predictions.csv"
METRICS_DIR = OUT / "metrics" / "cnn_embedding_experiment"
MODEL_DIR = OUT / "models" / "cnn_embedding"
PLOT_DIR = OUT / "plots" / "cnn_embedding"
GRID = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.9},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.7},
]


def normpath(value):
    return str(value).replace("/", "\\").rstrip("\\").casefold()


def metrics(y, pred):
    return {
        "MAE_g/dL": float(mean_absolute_error(y, pred)),
        "RMSE_g/dL": float(mean_squared_error(y, pred) ** 0.5),
        "R2": float(r2_score(y, pred)),
        "n_samples": int(len(y)),
    }


def main():
    data = pd.read_csv(EMBED)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    feature_cols = sorted(c for c in data.columns if c.startswith("embedding_"))
    required = {"image_id", "image_path", "participant_id", "target_hb_g_dl", "split", "checkpoint_sha256"}
    if not required.issubset(data.columns) or len(feature_cols) != 1024:
        raise ValueError("Transferred Full-hand embedding table has an invalid schema")
    if not data.image_id.is_unique or not data.image_path.is_unique:
        raise ValueError("Duplicate image ID/path in embedding rows")
    if data.groupby("participant_id").split.nunique().max() != 1:
        raise ValueError("Full-hand participant leakage across splits")
    matrix = data[feature_cols].to_numpy(dtype=np.float32)
    target = pd.to_numeric(data.target_hb_g_dl, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite Full-hand embedding or numeric Hb target")

    train = data.split.eq("train").to_numpy()
    val = data.split.eq("validation").to_numpy()
    test = data.split.eq("test").to_numpy()
    if [int(train.sum()), int(val.sum()), int(test.sum())] != [175, 37, 38]:
        raise ValueError("Unexpected Full-hand split counts")
    scaler = StandardScaler().fit(matrix[train])
    x_train, x_val, x_test = scaler.transform(matrix[train]), scaler.transform(matrix[val]), scaler.transform(matrix[test])
    y_train, y_val, y_test = target[train], target[val], target[test]

    best, best_params, best_val = None, None, float("inf")
    for params in GRID:
        candidate = xgb.XGBRegressor(**params, random_state=42, tree_method="hist", verbosity=0)
        candidate.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        score = mean_absolute_error(y_val, candidate.predict(x_val))
        if score < best_val:
            best, best_params, best_val = candidate, params, float(score)
    pred = best.predict(x_test)

    test_rows = data.loc[test].copy()
    test_rows["_path_key"] = test_rows.image_path.map(normpath)
    baseline = pd.read_csv(BASELINE)
    baseline["_path_key"] = baseline.image_path.map(normpath)
    if not baseline._path_key.is_unique:
        raise ValueError("Handcrafted Full-hand test predictions have duplicate image paths")
    paired = test_rows.merge(baseline, on="_path_key", how="outer", indicator=True, validate="one_to_one")
    if len(paired) != int(test.sum()) or not paired._merge.eq("both").all():
        raise ValueError("Transferred features and handcrafted XGBoost have different held-out images")
    if not np.allclose(paired.target_hb_g_dl, paired.true, atol=1e-8):
        raise ValueError("Full-hand Hb targets differ between paired models")
    pred_by_path = dict(zip(test_rows._path_key, pred))
    paired["transfer_prediction"] = paired._path_key.map(pred_by_path)
    transfer_metrics = metrics(paired.true, paired.transfer_prediction)
    baseline_metrics = metrics(paired.true, paired.predicted)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "dataset_id": paired.dataset_id,
        "image_id": paired.image_id,
        "image_path": paired.image_path_x,
        "participant_id": paired.participant_id,
        "split": "test",
        "target_hb_g_dl": paired.true,
        "palm_checkpoint_transfer_xgb_prediction_g_dl": paired.transfer_prediction,
        "handcrafted_xgb_prediction_g_dl": paired.predicted,
    })
    pred_path = METRICS_DIR / "full_hand_palm_checkpoint_transfer_xgb_predictions.csv"
    predictions.to_csv(pred_path, index=False, float_format="%.12g")
    model_path = MODEL_DIR / "m3_palm_checkpoint_transfer_fullhand_xgb.pkl"
    with model_path.open("wb") as stream:
        pickle.dump({
            "scaler": scaler,
            "xgb": best,
            "embedding_columns": feature_cols,
            "checkpoint": provenance["checkpoint"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "dataset_id": "figshare_fullhand",
            "experiment_type": "exploratory cross-dataset CNN feature transfer; checkpoint not Full-hand-trained",
        }, stream)

    report = {
        "dataset": "Full-hand",
        "dataset_id": "figshare_fullhand",
        "task": "Hb regression (g/dL)",
        "status": "exploratory transfer; no Full-hand-trained CNN checkpoint was supplied",
        "feature_source": "M3 LightweightCNN backbone trained on a different Palm dataset for folder-derived two-class classification; pooled 1,024-D features before classifier",
        "checkpoint": provenance["checkpoint"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_safe_load": provenance["checkpoint_safe_load"],
        "checkpoint_training_note": provenance["source_training_task"],
        "target_provenance": "Figshare source-provided numerical Hb in g/dL; no folder-class target used for regression",
        "preprocessing": provenance["preprocessing"],
        "split_counts": {s: int(data.split.eq(s).sum()) for s in ("train", "validation", "test")},
        "participants": int(data.participant_id.nunique()),
        "test_rows_same_as_handcrafted_xgb": len(predictions),
        "validation_selection_metric": "MAE g/dL",
        "best_validation_mae_g_dl": best_val,
        "selected_xgboost_parameters": best_params,
        "test_metrics": {"palm_checkpoint_transfer_xgboost": transfer_metrics, "existing_handcrafted_xgboost": baseline_metrics},
        "limitations": [
            "The checkpoint was trained on a different Palm image dataset for unverified folder-label classification, not Full-hand Hb regression.",
            "Features were transferred without fine-tuning; therefore this is exploratory cross-dataset transfer, not the missing Full-hand-trained CNN experiment.",
            "Full-hand has 250 participants, including only 38 held-out test images; results are small-sample and not clinical validation.",
            "Validation MAE selected among four configurations; test samples were not used for model selection.",
        ],
        "artifacts": {
            "embeddings": str(EMBED.relative_to(ROOT)),
            "predictions": str(pred_path.relative_to(ROOT)),
            "model": str(model_path.relative_to(ROOT)),
        },
    }
    metrics_path = METRICS_DIR / "full_hand_palm_checkpoint_transfer_xgb_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lo = min(float(paired.true.min()), float(paired.transfer_prediction.min()), float(paired.predicted.min()))
    hi = max(float(paired.true.max()), float(paired.transfer_prediction.max()), float(paired.predicted.max()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, col, title, color in (
        (axes[0], "transfer_prediction", "Palm CNN transfer", "steelblue"),
        (axes[1], "predicted", "Handcrafted XGBoost", "darkorange"),
    ):
        ax.scatter(paired.true, paired[col], alpha=0.8, s=28, color=color)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("Laboratory Hb (g/dL)")
        ax.set_ylabel("Predicted Hb (g/dL)")
        ax.set_title(f"{title}\nMAE={metrics(paired.true, paired[col])['MAE_g/dL']:.3f}")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Full-hand exploratory transferred features — same held-out images (n={len(paired)})")
    fig.tight_layout()
    plot_path = PLOT_DIR / "full_hand_palm_checkpoint_transfer_vs_handcrafted.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    report["artifacts"]["comparison_plot"] = str(plot_path.relative_to(ROOT))
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_path = METRICS_DIR / "cnn_embedding_comparison_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    summary = summary[~summary.dataset.eq("full_hand")] if not summary.empty else summary
    new_rows = pd.DataFrame([
        {"dataset": "full_hand", "model": "palm_checkpoint_transfer_xgboost", "feature_type": "M3 Palm classifier transferred embedding (exploratory)", "split": "held-out test", "n_test": transfer_metrics["n_samples"], "MAE_g/dL": transfer_metrics["MAE_g/dL"], "RMSE_g/dL": transfer_metrics["RMSE_g/dL"], "R2": transfer_metrics["R2"]},
        {"dataset": "full_hand", "model": "existing_handcrafted_xgboost", "feature_type": "Member 4 handcrafted", "split": "held-out test", "n_test": baseline_metrics["n_samples"], "MAE_g/dL": baseline_metrics["MAE_g/dL"], "RMSE_g/dL": baseline_metrics["RMSE_g/dL"], "R2": baseline_metrics["R2"]},
    ])
    pd.concat([summary, new_rows], ignore_index=True).to_csv(summary_path, index=False)
    print(
        f"Full-hand transferred features train/val/test=175/37/38; val MAE={best_val:.4f}; "
        f"transfer test MAE/RMSE/R2={transfer_metrics['MAE_g/dL']:.4f}/{transfer_metrics['RMSE_g/dL']:.4f}/{transfer_metrics['R2']:.4f}; "
        f"handcrafted={baseline_metrics['MAE_g/dL']:.4f}/{baseline_metrics['RMSE_g/dL']:.4f}/{baseline_metrics['R2']:.4f}"
    )


if __name__ == "__main__":
    main()
