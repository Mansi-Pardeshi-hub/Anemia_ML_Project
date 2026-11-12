"""
evaluation.py — Consolidated evaluation and metric verification utilities for Member 5.

Aggregates Member 4 handcrafted, CNN-embedding, scratch-CNN, and Member 5 fusion
metrics into a single reproducible comparison table and verifies all metrics
against saved prediction files.

Outputs:
  member5_integration/outputs/metrics/member5_consolidated_comparison.csv
  member5_integration/outputs/metrics/member5_consolidated_comparison.json
  member5_integration/outputs/predictions/sample_adapter_verification.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from paths import (
    M4_METRICS, M5_FUSION, M5_METRICS, M5_PREDICTIONS,
    EYE_FEAT_PARQUET, NAIL_FEAT_PARQUET, FULLHAND_FEAT_PARQUET,
)


def compute_regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE_g_dl":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE_g_dl": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":        round(float(r2_score(y_true, y_pred)), 4),
        "n_samples": int(len(y_true)),
    }


def build_consolidated_comparison() -> pd.DataFrame:
    """
    Build the required comparison table across:
      - Eye handcrafted XGBoost
      - Eye Member 3 CNN embedding XGBoost
      - Nail handcrafted XGBoost (crop-level and participant-level)
      - Nail Member 3 CNN embedding XGBoost (crop-level)
      - Full-hand handcrafted XGBoost
      - Full-hand scratch-CNN embedding XGBoost
      - Fused Nail + Full-hand XGBoost (participant-level)
    """
    rows = []

    # 1. Eye handcrafted XGBoost
    eye_xgb = pd.read_csv(M4_METRICS / "eye_xgb_reg_predictions.csv")
    m = compute_regression_metrics(eye_xgb["true"], eye_xgb["predicted"])
    rows.append({
        "model_id": "eye_handcrafted_xgboost",
        "model_name": "Eye Handcrafted XGBoost",
        "modality": "Eye / Conjunctiva",
        "feature_type": "130 handcrafted image features",
        "evaluation_unit": "image (deduplicated)",
        "split_counts_train_val_test": "432 / 89 / 89 images (202 participants)",
        "n_test_images_or_crops": m["n_samples"],
        "n_test_participants": 31,
        "MAE_g_dl": m["MAE_g_dl"],
        "RMSE_g_dl": m["RMSE_g_dl"],
        "R2": m["R2"],
        "notes": "Verified laboratory Hb (g/dL); exact-duplicate variant images removed.",
    })

    # 2. Eye CNN embedding XGBoost
    eye_cnn = pd.read_csv(M4_METRICS / "cnn_embedding_experiment" / "eye_cnn_embedding_xgb_predictions.csv")
    m = compute_regression_metrics(eye_cnn["target_hb_g_dl"], eye_cnn["cnn_embedding_xgb_prediction_g_dl"])
    rows.append({
        "model_id": "eye_cnn_embedding_xgboost",
        "model_name": "Eye Member 3 CNN-Embedding XGBoost",
        "modality": "Eye / Conjunctiva",
        "feature_type": "1024-D Member 3 RegressionCNN embedding",
        "evaluation_unit": "image (deduplicated)",
        "split_counts_train_val_test": "432 / 89 / 89 images (202 participants)",
        "n_test_images_or_crops": m["n_samples"],
        "n_test_participants": int(eye_cnn["participant_id"].nunique()),
        "MAE_g_dl": m["MAE_g_dl"],
        "RMSE_g_dl": m["RMSE_g_dl"],
        "R2": m["R2"],
        "notes": "Paired on the exact same 89 test images as Eye Handcrafted XGBoost.",
    })

    # 3. Nail handcrafted XGBoost (crop-level)
    nail_xgb = pd.read_csv(M4_METRICS / "nail_xgb_reg_predictions.csv")
    m = compute_regression_metrics(nail_xgb["true"], nail_xgb["predicted"])
    rows.append({
        "model_id": "nail_handcrafted_xgboost_crop",
        "model_name": "Nail Handcrafted XGBoost (Crop-Level)",
        "modality": "Nail (Figshare)",
        "feature_type": "130 handcrafted image features",
        "evaluation_unit": "nail crop (1–3 per participant)",
        "split_counts_train_val_test": "503 / 107 / 109 crops (175 / 37 / 38 participants)",
        "n_test_images_or_crops": m["n_samples"],
        "n_test_participants": int(nail_xgb["sample_id"].nunique()),
        "MAE_g_dl": m["MAE_g_dl"],
        "RMSE_g_dl": m["RMSE_g_dl"],
        "R2": m["R2"],
        "notes": "Crop-level metric (109 crops across 38 test participants); do not compare directly to 38-participant Full-hand without aggregation.",
    })

    # 4. Nail CNN embedding XGBoost (crop-level)
    nail_cnn = pd.read_csv(M4_METRICS / "cnn_embedding_experiment" / "nail_cnn_embedding_xgb_predictions.csv")
    m = compute_regression_metrics(nail_cnn["target_hb_g_dl"], nail_cnn["cnn_embedding_xgb_prediction_g_dl"])
    rows.append({
        "model_id": "nail_cnn_embedding_xgboost_crop",
        "model_name": "Nail Member 3 CNN-Embedding XGBoost (Crop-Level)",
        "modality": "Nail (Figshare)",
        "feature_type": "1024-D Member 3 RegressionCNN embedding",
        "evaluation_unit": "nail crop (1–3 per participant)",
        "split_counts_train_val_test": "503 / 107 / 109 crops (175 / 37 / 38 participants)",
        "n_test_images_or_crops": m["n_samples"],
        "n_test_participants": int(nail_cnn["participant_id"].nunique()),
        "MAE_g_dl": m["MAE_g_dl"],
        "RMSE_g_dl": m["RMSE_g_dl"],
        "R2": m["R2"],
        "notes": "Paired on the exact same 109 test nail crops as Nail Handcrafted XGBoost.",
    })

    # 5. Full-hand handcrafted XGBoost
    fh_xgb = pd.read_csv(M4_METRICS / "full_hand_xgb_reg_predictions.csv")
    m = compute_regression_metrics(fh_xgb["true"], fh_xgb["predicted"])
    rows.append({
        "model_id": "full_hand_handcrafted_xgboost",
        "model_name": "Full-hand Handcrafted XGBoost",
        "modality": "Full-hand (Figshare)",
        "feature_type": "130 handcrafted image features",
        "evaluation_unit": "participant (1 image per participant)",
        "split_counts_train_val_test": "175 / 37 / 38 participants",
        "n_test_images_or_crops": m["n_samples"],
        "n_test_participants": m["n_samples"],
        "MAE_g_dl": m["MAE_g_dl"],
        "RMSE_g_dl": m["RMSE_g_dl"],
        "R2": m["R2"],
        "notes": "1-to-1 participant-level evaluation on 38 held-out Figshare participants.",
    })

    # 6. Full-hand scratch-CNN embedding XGBoost
    fh_cnn = pd.read_csv(M4_METRICS / "fullhand_cnn_experiment" / "fullhand_scratch_cnn_xgb_predictions.csv")
    m = compute_regression_metrics(fh_cnn["target_hb_g_dl"], fh_cnn["fullhand_scratch_cnn_xgb_prediction_g_dl"])
    rows.append({
        "model_id": "full_hand_scratch_cnn_embedding_xgboost",
        "model_name": "Full-hand Scratch-CNN (64-D) + XGBoost",
        "modality": "Full-hand (Figshare)",
        "feature_type": "64-D Scratch-CNN embedding",
        "evaluation_unit": "participant (1 image per participant)",
        "split_counts_train_val_test": "175 / 37 / 38 participants",
        "n_test_images_or_crops": m["n_samples"],
        "n_test_participants": m["n_samples"],
        "MAE_g_dl": m["MAE_g_dl"],
        "RMSE_g_dl": m["RMSE_g_dl"],
        "R2": m["R2"],
        "notes": "Compact CNN trained from scratch on 175 training participants; small improvement over handcrafted (1.5599 vs 1.5819) is within n=38 sampling uncertainty.",
    })

    # 7. Nail participant-level & Fused Nail + Full-hand (from fusion_predictions.csv)
    fusion_preds_path = M5_FUSION / "fusion_predictions.csv"
    if fusion_preds_path.exists():
        f_df = pd.read_csv(fusion_preds_path)
        m_nail_part = compute_regression_metrics(f_df["true_hb_g_dl"], f_df["nail_pred"])
        rows.append({
            "model_id": "nail_handcrafted_xgboost_participant",
            "model_name": "Nail Handcrafted XGBoost (Participant-Level Mean)",
            "modality": "Nail (Figshare, aggregated)",
            "feature_type": "130 mean-aggregated Nail handcrafted features",
            "evaluation_unit": "participant (mean of 1–3 crops)",
            "split_counts_train_val_test": "175 / 37 / 38 participants",
            "n_test_images_or_crops": 109,
            "n_test_participants": m_nail_part["n_samples"],
            "MAE_g_dl": m_nail_part["MAE_g_dl"],
            "RMSE_g_dl": m_nail_part["RMSE_g_dl"],
            "R2": m_nail_part["R2"],
            "notes": "Fair participant-level Nail baseline on the exact same 38 test participants as Full-hand.",
        })

        m_fused = compute_regression_metrics(f_df["true_hb_g_dl"], f_df["fused_pred"])
        rows.append({
            "model_id": "fused_full_hand_and_nail_xgboost",
            "model_name": "Fused Full-hand + Nail Handcrafted XGBoost",
            "modality": "Paired Full-hand + Nail (same source photo)",
            "feature_type": "260-D concatenated (130 Full-hand + 130 mean Nail)",
            "evaluation_unit": "participant (paired)",
            "split_counts_train_val_test": "175 / 37 / 38 participants",
            "n_test_images_or_crops": 147,
            "n_test_participants": m_fused["n_samples"],
            "MAE_g_dl": m_fused["MAE_g_dl"],
            "RMSE_g_dl": m_fused["RMSE_g_dl"],
            "R2": m_fused["R2"],
            "notes": "Same-photo multi-view fusion. Improved over Nail-only (1.6792 vs 1.8289) but did not beat Full-hand-only (1.5819).",
        })

    comp_df = pd.DataFrame(rows)
    M5_METRICS.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(M5_METRICS / "member5_consolidated_comparison.csv", index=False)
    (M5_METRICS / "member5_consolidated_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return comp_df


def verify_saved_predictions_vs_adapters() -> pd.DataFrame:
    """
    Verify that each Member 5 ModelAdapter produces the exact same prediction
    as the saved Member 4 test predictions CSV for the first held-out test image.
    """
    from model_adapters import (
        FullHandXGBAdapter, NailXGBAdapter, EyeXGBAdapter, FullHandScratchCNNXGBAdapter
    )

    records = []
    checks = [
        ("full_hand_handcrafted_xgb", FullHandXGBAdapter, M4_METRICS / "full_hand_xgb_reg_predictions.csv", "predicted"),
        ("nail_handcrafted_xgb",      NailXGBAdapter,     M4_METRICS / "nail_xgb_reg_predictions.csv",      "predicted"),
        ("eye_handcrafted_xgb",       EyeXGBAdapter,      M4_METRICS / "eye_xgb_reg_predictions.csv",       "predicted"),
        ("full_hand_scratch_cnn_xgb", FullHandScratchCNNXGBAdapter, M4_METRICS / "fullhand_cnn_experiment" / "fullhand_scratch_cnn_xgb_predictions.csv", "fullhand_scratch_cnn_xgb_prediction_g_dl"),
    ]

    for label, cls, csv_path, pred_col in checks:
        df = pd.read_csv(csv_path)
        row0 = df.iloc[0]
        img_path = Path(str(row0["image_path"]))
        saved_pred = float(row0[pred_col])
        true_val = float(row0.get("true", row0.get("target_hb_g_dl")))

        adapter = cls().load()
        res = adapter.predict_image(img_path)
        adapter_pred = float(res["hb_g_dl"])
        diff = abs(adapter_pred - saved_pred)

        records.append({
            "adapter": label,
            "image_path": str(img_path),
            "true_hb_g_dl": true_val,
            "saved_prediction_g_dl": saved_pred,
            "adapter_prediction_g_dl": adapter_pred,
            "abs_diff": diff,
            "exact_match": bool(diff < 1e-4),
        })

    out_df = pd.DataFrame(records)
    M5_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(M5_PREDICTIONS / "sample_adapter_verification.csv", index=False)
    return out_df


if __name__ == "__main__":
    comp = build_consolidated_comparison()
    print("=== Consolidated Comparison Table ===")
    print(comp[["model_name", "evaluation_unit", "n_test_participants", "MAE_g_dl", "RMSE_g_dl", "R2"]].to_string(index=False))
    ver = verify_saved_predictions_vs_adapters()
    print("\n=== Adapter vs Saved Prediction Verification ===")
    print(ver[["adapter", "true_hb_g_dl", "saved_prediction_g_dl", "adapter_prediction_g_dl", "abs_diff", "exact_match"]].to_string(index=False))

