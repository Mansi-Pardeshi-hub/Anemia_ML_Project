"""
evaluation.py — Shared evaluation utilities for regression and classification.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)


# ── Regression ─────────────────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred, units: str = "g/dL") -> dict:
    """Return MAE, RMSE, R² for regression."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {
        f"MAE_{units}":  round(mae, 4),
        f"RMSE_{units}": round(rmse, 4),
        "R2":            round(r2, 4),
        "n_samples":     int(len(y_true)),
        "units":         units,
    }


def print_regression_metrics(metrics: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    units  = metrics.get("units", "g/dL")
    print(f"{prefix}MAE={metrics[f'MAE_{units}']:.4f} {units}  "
          f"RMSE={metrics[f'RMSE_{units}']:.4f} {units}  "
          f"R²={metrics['R2']:.4f}  n={metrics['n_samples']}")


# ── Classification ─────────────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred, y_prob=None, pos_label: int = 1) -> dict:
    """
    Return comprehensive classification metrics.
    Handles edge cases (single-class predictions / test sets) gracefully.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    acc = accuracy_score(y_true, y_pred)

    zero_div_warn = len(unique_pred) < 2 or len(unique_true) < 2

    prec = precision_score(y_true, y_pred, pos_label=pos_label,
                           zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=pos_label,
                        zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp + 1e-9)
    else:
        spec = float("nan")
        warnings.warn("Confusion matrix not 2×2 — specificity undefined.")

    roc_auc = float("nan")
    if y_prob is not None and len(unique_true) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception as e:
            warnings.warn(f"ROC-AUC could not be computed: {e}")

    result = {
        "accuracy":    round(acc, 4),
        "precision":   round(prec, 4),
        "recall":      round(rec, 4),
        "specificity": round(float(spec), 4) if not np.isnan(spec) else None,
        "f1_score":    round(f1, 4),
        "roc_auc":     round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "n_samples":   int(len(y_true)),
        "confusion_matrix": cm.tolist(),
    }
    if zero_div_warn:
        result["warning"] = (
            "Single-class predictions or single-class test set — "
            "precision/recall/F1 unreliable."
        )
    return result


def print_classification_metrics(metrics: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    auc_str = f"  ROC-AUC={metrics['roc_auc']:.4f}" if metrics.get("roc_auc") else ""
    print(f"{prefix}Acc={metrics['accuracy']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
          f"F1={metrics['f1_score']:.4f}{auc_str}  n={metrics['n_samples']}")
    if "warning" in metrics:
        print(f"  ⚠ {metrics['warning']}")


# ── Persistence ────────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, path: Path) -> None:
    """Save metrics dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {path}")


def save_predictions(ids, y_true, y_pred, path: Path, extra: dict | None = None) -> None:
    """Save predictions CSV with sample identifiers."""
    df = pd.DataFrame({"sample_id": ids, "true": y_true, "predicted": y_pred})
    if extra:
        for k, v in extra.items():
            df[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Predictions saved → {path}")

