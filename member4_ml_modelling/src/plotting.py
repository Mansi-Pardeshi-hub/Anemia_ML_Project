"""
plotting.py — Publication-appropriate evaluation visualisations.
All functions save to OUT_PLOTS and return the file path.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")
    return path


# ── Regression ─────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true, y_pred,
    title: str,
    path: Path,
    units: str = "g/dL",
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4, s=40)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel(f"Actual Hb ({units})", fontsize=12)
    ax.set_ylabel(f"Predicted Hb ({units})", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    return _save(fig, path)


def plot_residuals(
    y_true, y_pred,
    title: str,
    path: Path,
    units: str = "g/dL",
) -> Path:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.4, s=40)
    axes[0].axhline(0, color="r", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel(f"Predicted Hb ({units})", fontsize=11)
    axes[0].set_ylabel(f"Residual (Pred − True) ({units})", fontsize=11)
    axes[0].set_title(f"{title} — Residual vs Predicted", fontsize=12)

    axes[1].hist(residuals, bins=20, edgecolor="k", alpha=0.75)
    axes[1].axvline(0, color="r", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel(f"Residual ({units})", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(f"{title} — Residual Distribution", fontsize=12)

    fig.tight_layout()
    return _save(fig, path)


def plot_regression_comparison(
    results: list[dict],
    path: Path,
    title: str = "Regression Model Comparison",
) -> Path:
    """
    results: list of dicts with keys 'model', 'MAE_g/dL', 'RMSE_g/dL', 'R2'.
    """
    labels = [r["model"] for r in results]
    maes   = [r.get("MAE_g/dL", r.get("MAE_g/dl", float("nan"))) for r in results]
    rmses  = [r.get("RMSE_g/dL", r.get("RMSE_g/dl", float("nan"))) for r in results]
    r2s    = [r.get("R2", float("nan")) for r in results]

    x = np.arange(len(labels))
    width = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, vals, ylabel, color in zip(
        axes,
        [maes, rmses, r2s],
        ["MAE (g/dL)", "RMSE (g/dL)", "R²"],
        ["steelblue", "coral", "mediumseagreen"]
    ):
        bars = ax.bar(x, vals, color=color, alpha=0.85, edgecolor="k", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    path: Path,
    title: str = "Feature Importance",
    top_n: int = 20,
) -> Path:
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color="steelblue", alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


# ── Classification ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: list | np.ndarray,
    class_names: list[str],
    path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, path)


def plot_roc_curve(
    y_true, y_prob,
    path: Path,
    title: str = "ROC Curve",
    auc: float | None = None,
) -> Path | None:
    from sklearn.metrics import roc_curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception as e:
        warnings.warn(f"ROC curve could not be plotted: {e}")
        return None

    auc_str = f"AUC={auc:.4f}" if auc is not None else ""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC {auc_str}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    return _save(fig, path)


def plot_classification_comparison(
    results: list[dict],
    path: Path,
    title: str = "Classification Model Comparison",
) -> Path:
    """results: list of dicts with 'model', 'accuracy', 'f1_score', 'roc_auc'."""
    labels  = [r["model"] for r in results]
    metrics = ["accuracy", "f1_score", "roc_auc"]
    titles  = ["Accuracy", "F1-Score", "ROC-AUC"]
    colors  = ["steelblue", "coral", "mediumseagreen"]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, met, met_title, color in zip(axes, metrics, titles, colors):
        raw_vals = [r.get(met) for r in results]
        valid = [v is not None and not pd.isna(v) for v in raw_vals]
        vals = [float(v) if ok else 0.0 for v, ok in zip(raw_vals, valid)]
        bars = ax.bar(x, vals, color=color, alpha=0.85, edgecolor="k", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(met_title, fontsize=11)
        ax.set_title(met_title, fontsize=12)
        for bar, v, ok in zip(bars, vals, valid):
            if not ok:
                bar.set_alpha(0.0)
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "N/A",
                        ha="center", va="bottom", fontsize=8, color="gray")
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)

