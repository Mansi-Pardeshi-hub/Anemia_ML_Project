"""Train separate XGBoost regressors on verified Member 3 CNN embeddings.

This never modifies the original Member 4 baseline artifacts or summary CSV.
Run with the Member 4 Python 3.10 environment from the project root.
"""

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
EMBED = OUT / "embeddings"
METRICS = OUT / "metrics" / "cnn_embedding_experiment"
MODELS = OUT / "models" / "cnn_embedding"
PLOTS = OUT / "plots" / "cnn_embedding"
BASELINES = {
    "eye": OUT / "metrics" / "eye_xgb_reg_predictions.csv",
    "nail": OUT / "metrics" / "nail_xgb_reg_predictions.csv",
}
DATASET_IDS = {"eye": "eye_edi", "nail": "figshare_nail"}
GRID = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.9},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.7},
]


def normpath(value: str) -> str:
    return str(value).replace("/", "\\").rstrip("\\").casefold()


def metrics(y, prediction):
    return {
        "MAE_g/dL": float(mean_absolute_error(y, prediction)),
        "RMSE_g/dL": float(mean_squared_error(y, prediction) ** 0.5),
        "R2": float(r2_score(y, prediction)),
        "n_samples": int(len(y)),
    }


def train_dataset(dataset: str) -> dict:
    embedding_path = EMBED / f"m3_{dataset}_regression_embeddings.csv"
    provenance_path = EMBED / f"m3_{dataset}_regression_embeddings_provenance.json"
    embeddings = pd.read_csv(embedding_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    feature_columns = sorted(c for c in embeddings.columns if c.startswith("embedding_"))
    required = {
        "dataset_id", "image_id", "image_path", "participant_id",
        "target_hb_g_dl", "split", "checkpoint_id", "checkpoint_sha256",
    }
    if not required.issubset(embeddings.columns) or len(feature_columns) != 1024:
        raise ValueError(f"{dataset}: embedding schema/dimension invalid")
    if not embeddings.image_id.is_unique or not embeddings.image_path.is_unique:
        raise ValueError(f"{dataset}: duplicate image identifiers/paths")
    matrix = embeddings[feature_columns].to_numpy(dtype=np.float32)
    target = pd.to_numeric(embeddings.target_hb_g_dl, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError(f"{dataset}: non-finite target or embedding")
    if embeddings.groupby("participant_id").split.nunique().max() != 1:
        raise ValueError(f"{dataset}: participant leakage across splits")
    if set(embeddings.dataset_id) != {DATASET_IDS[dataset]}:
        raise ValueError(f"{dataset}: dataset ID mismatch")

    train_mask = embeddings.split.eq("train").to_numpy()
    val_mask = embeddings.split.eq("validation").to_numpy()
    test_mask = embeddings.split.eq("test").to_numpy()
    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise ValueError(f"{dataset}: missing train/validation/test partition")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(matrix[train_mask])
    x_val = scaler.transform(matrix[val_mask])
    x_test = scaler.transform(matrix[test_mask])
    y_train, y_val, y_test = target[train_mask], target[val_mask], target[test_mask]

    best_model = None
    best_params = None
    best_val_mae = float("inf")
    for params in GRID:
        candidate = xgb.XGBRegressor(
            **params,
            random_state=42,
            tree_method="hist",
            verbosity=0,
        )
        candidate.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        val_mae = mean_absolute_error(y_val, candidate.predict(x_val))
        if val_mae < best_val_mae:
            best_model, best_params, best_val_mae = candidate, params, float(val_mae)
    cnn_prediction = best_model.predict(x_test)

    test_embeddings = embeddings.loc[test_mask].copy()
    test_embeddings["_path_key"] = test_embeddings.image_path.map(normpath)
    baseline = pd.read_csv(BASELINES[dataset])
    baseline["_path_key"] = baseline.image_path.map(normpath)
    if not baseline._path_key.is_unique:
        raise ValueError(f"{dataset}: baseline test paths are not unique")
    aligned = test_embeddings.merge(baseline, on="_path_key", how="outer", suffixes=("_cnn", "_hand"), indicator=True, validate="one_to_one")
    if len(aligned) != len(test_embeddings) or not aligned._merge.eq("both").all():
        raise ValueError(f"{dataset}: CNN embeddings and handcrafted baseline test rows differ")
    if not np.allclose(aligned.target_hb_g_dl, aligned.true, atol=1e-8):
        raise ValueError(f"{dataset}: test Hb target differs from handcrafted baseline")
    prediction_by_path = dict(zip(test_embeddings._path_key, cnn_prediction))
    aligned["cnn_embedding_prediction"] = aligned._path_key.map(prediction_by_path)
    if aligned.cnn_embedding_prediction.isna().any():
        raise ValueError(f"{dataset}: a test prediction did not align by image path")
    baseline_metrics = metrics(aligned.true.to_numpy(), aligned.predicted.to_numpy())
    cnn_metrics = metrics(aligned.true.to_numpy(), aligned.cnn_embedding_prediction.to_numpy())

    predictions = pd.DataFrame({
        "dataset_id": aligned.dataset_id,
        "image_id": aligned.image_id,
        "image_path": aligned.image_path_cnn,
        "participant_id": aligned.participant_id,
        "split": "test",
        "target_hb_g_dl": aligned.true,
        "cnn_embedding_xgb_prediction_g_dl": aligned.cnn_embedding_prediction,
        "handcrafted_xgb_prediction_g_dl": aligned.predicted,
    })
    METRICS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    prediction_path = METRICS / f"{dataset}_cnn_embedding_xgb_predictions.csv"
    predictions.to_csv(prediction_path, index=False, float_format="%.12g")
    model_path = MODELS / f"m3_{dataset}_cnn_embedding_xgb.pkl"
    with model_path.open("wb") as stream:
        pickle.dump({
            "scaler": scaler,
            "xgb": best_model,
            "embedding_columns": feature_columns,
            "checkpoint": provenance["checkpoint"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "dataset_id": DATASET_IDS[dataset],
        }, stream)

    report = {
        "dataset": dataset,
        "dataset_id": DATASET_IDS[dataset],
        "task": "Hb regression (g/dL)",
        "feature_source": "Member 3 scratch-trained RegressionCNN pooled 1,024-D embedding",
        "checkpoint": provenance["checkpoint"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_safe_load": provenance["checkpoint_load"],
        "preprocessing": provenance["preprocessing"],
        "split_counts": {s: int((embeddings.split == s).sum()) for s in ("train", "validation", "test")},
        "participants": int(embeddings.participant_id.nunique()),
        "test_rows_same_as_handcrafted_xgb": int(len(predictions)),
        "validation_selection_metric": "MAE g/dL",
        "best_validation_mae_g_dl": best_val_mae,
        "selected_xgboost_parameters": best_params,
        "test_metrics": {"cnn_embedding_xgboost": cnn_metrics, "existing_handcrafted_xgboost": baseline_metrics},
        "metric_delta_cnn_minus_handcrafted": {
            key: cnn_metrics[key] - baseline_metrics[key] for key in ("MAE_g/dL", "RMSE_g/dL", "R2")
        },
        "artifacts": {
            "embeddings": str(embedding_path.relative_to(ROOT)),
            "predictions": str(prediction_path.relative_to(ROOT)),
            "model": str(model_path.relative_to(ROOT)),
        },
        "limitations": [
            "Research prototype; not clinical validation.",
            "Member 3 checkpoint was trained on the same training/validation partitions; test was held out.",
            "Eye uses the duplicate-corrected 89-row test split; Nail uses 109 test nail crops from participant-disjoint splits.",
            "No full-hand regression checkpoint exists in the supplied archive; this experiment covers Eye and Nail only.",
            "No paired modalities; this is not multimodal fusion.",
        ],
    }
    metrics_path = METRICS / f"{dataset}_cnn_embedding_xgb_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    low = min(float(aligned.true.min()), float(aligned.cnn_embedding_prediction.min()), float(aligned.predicted.min()))
    high = max(float(aligned.true.max()), float(aligned.cnn_embedding_prediction.max()), float(aligned.predicted.max()))
    axes[0].scatter(aligned.true, aligned.cnn_embedding_prediction, alpha=0.75, s=24)
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set_title(f"CNN embeddings XGBoost\nMAE={cnn_metrics['MAE_g/dL']:.3f}; RMSE={cnn_metrics['RMSE_g/dL']:.3f}")
    axes[1].scatter(aligned.true, aligned.predicted, alpha=0.75, s=24, color="darkorange")
    axes[1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1].set_title(f"Handcrafted XGBoost\nMAE={baseline_metrics['MAE_g/dL']:.3f}; RMSE={baseline_metrics['RMSE_g/dL']:.3f}")
    for ax in axes:
        ax.set_xlabel("Laboratory Hb (g/dL)")
        ax.set_ylabel("Predicted Hb (g/dL)")
        ax.grid(alpha=0.2)
    fig.suptitle(f"{dataset.title()} — identical held-out test images (n={len(predictions)})")
    fig.tight_layout()
    plot_path = PLOTS / f"{dataset}_cnn_embedding_xgb_vs_handcrafted.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    report["artifacts"]["comparison_plot"] = str(plot_path.relative_to(ROOT))
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"{dataset}: train/val/test={report['split_counts']['train']}/"
        f"{report['split_counts']['validation']}/{report['split_counts']['test']} "
        f"val MAE={best_val_mae:.4f}; CNN XGB test MAE/RMSE/R2="
        f"{cnn_metrics['MAE_g/dL']:.4f}/{cnn_metrics['RMSE_g/dL']:.4f}/{cnn_metrics['R2']:.4f}; "
        f"handcrafted={baseline_metrics['MAE_g/dL']:.4f}/{baseline_metrics['RMSE_g/dL']:.4f}/{baseline_metrics['R2']:.4f}"
    )
    return report


def main() -> None:
    reports = [train_dataset(dataset) for dataset in ("eye", "nail")]
    summary = []
    for report in reports:
        for model_name, result in report["test_metrics"].items():
            summary.append({
                "dataset": report["dataset"],
                "model": model_name,
                "feature_type": "M3 CNN embedding" if model_name.startswith("cnn_") else "Member 4 handcrafted",
                "split": "held-out test",
                "n_test": result["n_samples"],
                "MAE_g/dL": result["MAE_g/dL"],
                "RMSE_g/dL": result["RMSE_g/dL"],
                "R2": result["R2"],
            })
    summary_path = METRICS / "cnn_embedding_comparison_summary.csv"
    if summary_path.is_file():
        previous = pd.read_csv(summary_path)
        preserved = previous[previous.dataset.eq("full_hand")]
        if not preserved.empty:
            summary = pd.concat([pd.DataFrame(summary), preserved], ignore_index=True).to_dict("records")
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"Separate CNN comparison summary saved to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
