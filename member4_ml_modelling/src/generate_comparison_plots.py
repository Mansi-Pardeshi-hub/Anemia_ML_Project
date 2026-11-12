"""Regenerate cross-model plots from saved metrics and Member 3 predictions."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplcache"))

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config import OUT_METRICS, OUT_PLOTS, EYE_REG_PREDICTIONS_CSV, NAIL_REG_PREDICTIONS_CSV
from plotting import plot_actual_vs_predicted, plot_regression_comparison, plot_classification_comparison


def _records(frame, columns):
    return [
        {key: row[key] for key in columns if key in row.index}
        for _, row in frame.iterrows()
    ]


def main():
    summary_path = OUT_METRICS / "all_metrics_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Run the pipeline first; missing {summary_path}")
    summary = pd.read_csv(summary_path)

    reg_cols = ["model", "MAE_g/dL", "RMSE_g/dL", "R2"]
    for dataset in ["Eye", "Nail", "Full-hand"]:
        rows = summary[(summary.task == "regression") & (summary.dataset == dataset)]
        if len(rows) >= 2:
            plot_regression_comparison(
                _records(rows, reg_cols),
                OUT_PLOTS / f"{dataset.lower().replace('-', '_')}_all_models_regression.png",
                title=f"{dataset} — saved test-set regression results",
            )

    cls_cols = ["model", "accuracy", "f1_score", "roc_auc"]
    for dataset in ["Eye", "Nail", "Palm"]:
        rows = summary[(summary.task == "classification") & (summary.dataset == dataset)]
        if len(rows) >= 2:
            plot_classification_comparison(
                _records(rows, cls_cols),
                OUT_PLOTS / f"{dataset.lower()}_all_models_classification.png",
                title=f"{dataset} — saved test-set classification results",
            )

    for modality, prediction_path in [("eye", EYE_REG_PREDICTIONS_CSV), ("nail", NAIL_REG_PREDICTIONS_CSV)]:
        if prediction_path.is_file():
            predictions = pd.read_csv(prediction_path)
            if {"true_hb", "predicted_hb"}.issubset(predictions.columns):
                plot_actual_vs_predicted(
                    predictions["true_hb"], predictions["predicted_hb"],
                    title=f"Member 3 saved CNN predictions — {modality.title()} (not rerun by Member 4)",
                    path=OUT_PLOTS / f"m3_{modality}_cnn_actual_vs_predicted.png",
                )
    print("Comparison plots regenerated from saved metrics and prediction files.")


if __name__ == "__main__":
    main()

