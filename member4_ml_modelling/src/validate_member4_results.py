"""Recalculate saved Member 4 metrics without retraining.

Run from the project root with the Member 4 Python 3.10 interpreter:

    python member4_ml_modelling\\src\\validate_member4_results.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config import M3_RESULTS_DIR, M4_DIR, OUT_FEATURES, OUT_METRICS, OUT_MODELS
from evaluation import classification_metrics, regression_metrics


LEDGER_PATH = OUT_METRICS / "member4_validation_ledger.json"


def _close(a, b, ndigits=4) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return round(float(a), ndigits) == round(float(b), ndigits)


def _cmp(saved: dict, recomputed: dict, keys: tuple[str, ...]) -> list[dict]:
    mismatches = []
    for key in keys:
        if key not in saved:
            continue
        if not _close(saved[key], recomputed[key]):
            mismatches.append({"key": key, "saved": saved[key], "recomputed": recomputed[key]})
    return mismatches


def _norm(path: str) -> str:
    return str(path).replace("/", "\\").rstrip("\\").casefold()


def check_handcrafted() -> list[dict]:
    out = []
    for slug in ("eye", "nail", "full_hand"):
        preds = pd.read_csv(OUT_METRICS / f"{slug}_rf_reg_predictions.csv")
        for label, col, json_name in (
            ("rf", "predicted", f"{slug}_rf_reg_metrics.json"),
            ("et", "et_pred", f"{slug}_et_reg_metrics.json"),
        ):
            saved = json.loads((OUT_METRICS / json_name).read_text(encoding="utf-8"))
            recomputed = regression_metrics(preds["true"], preds[col])
            mismatches = _cmp(saved, recomputed, ("MAE_g/dL", "RMSE_g/dL", "R2", "n_samples"))
            out.append({"check": f"{slug}_{label}_regression", "ok": not mismatches, "recomputed": recomputed, "mismatches": mismatches})
        xgb_preds = pd.read_csv(OUT_METRICS / f"{slug}_xgb_reg_predictions.csv")
        saved = json.loads((OUT_METRICS / f"{slug}_xgb_reg_metrics.json").read_text(encoding="utf-8"))
        recomputed = regression_metrics(xgb_preds["true"], xgb_preds["predicted"])
        mismatches = _cmp(saved, recomputed, ("MAE_g/dL", "RMSE_g/dL", "R2", "n_samples"))
        out.append({"check": f"{slug}_xgb_regression", "ok": not mismatches, "recomputed": recomputed, "mismatches": mismatches})

    for slug in ("eye", "nail", "palm"):
        preds = pd.read_csv(OUT_METRICS / f"{slug}_rf_cls_predictions.csv")
        for label, pred_col, prob_col, json_name in (
            ("rf", "predicted", "rf_prob", f"{slug}_rf_cls_metrics.json"),
            ("et", "et_pred", "et_prob", f"{slug}_et_cls_metrics.json"),
        ):
            saved = json.loads((OUT_METRICS / json_name).read_text(encoding="utf-8"))
            recomputed = classification_metrics(preds["true"], preds[pred_col], preds[prob_col])
            mismatches = _cmp(saved, recomputed, ("accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc", "n_samples"))
            if saved.get("confusion_matrix") != recomputed.get("confusion_matrix"):
                mismatches.append({"key": "confusion_matrix", "saved": saved.get("confusion_matrix"), "recomputed": recomputed.get("confusion_matrix")})
            out.append({
                "check": f"{slug}_{label}_classification",
                "ok": not mismatches,
                "recomputed": {k: recomputed[k] for k in ("accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc", "n_samples")},
                "mismatches": mismatches,
            })
    return out


def check_member3_saved() -> list[dict]:
    out = []
    eye_reg = pd.read_csv(M3_RESULTS_DIR / "eye_reg_test_predictions.csv")
    nail_reg = pd.read_csv(M3_RESULTS_DIR / "nail_reg_test_predictions.csv")
    eye_cls = pd.read_csv(M3_RESULTS_DIR / "eye_cls_test_predictions.csv")
    nail_cls = pd.read_csv(M3_RESULTS_DIR / "nail_cls_test_predictions.csv")
    palm_cls = pd.read_csv(M3_RESULTS_DIR / "palm_test_predictions.csv")

    eye_reg_m = regression_metrics(eye_reg["true_hb"], eye_reg["predicted_hb"])
    nail_reg_m = regression_metrics(nail_reg["true_hb"], nail_reg["predicted_hb"])
    eye_cls_m = classification_metrics(eye_cls["true_label"], eye_cls["predicted_label"], eye_cls["prob_anemic"])
    nail_cls_m = classification_metrics(nail_cls["true_label"], nail_cls["predicted_label"], nail_cls["prob_anemic"])
    palm_cls_m = classification_metrics(palm_cls["true_label"], palm_cls["predicted_label"], palm_cls.get("prob_anemic"))

    summary = pd.read_csv(OUT_METRICS / "all_metrics_summary.csv")
    expected = {
        "M3 CNN (Eye)|regression": {"MAE_g/dL": 1.8578, "RMSE_g/dL": 2.4230, "R2": 0.2202, "n_test": 92},
        "M3 CNN (Nail)|regression": {"MAE_g/dL": 2.0916, "RMSE_g/dL": 2.5914, "R2": -0.1917, "n_test": 109},
        "M3 CNN (Eye)|classification": {"accuracy": 0.8587, "f1_score": 0.8571, "roc_auc": 0.9381, "n_test": 92},
        "M3 CNN (Nail)|classification": {"accuracy": 0.8257, "f1_score": 0.0, "roc_auc": 0.5216, "n_test": 109},
    }
    recomputed_map = {
        "M3 CNN (Eye)|regression": {**eye_reg_m, "n_test": len(eye_reg)},
        "M3 CNN (Nail)|regression": {**nail_reg_m, "n_test": len(nail_reg)},
        "M3 CNN (Eye)|classification": {**eye_cls_m, "n_test": len(eye_cls)},
        "M3 CNN (Nail)|classification": {**nail_cls_m, "n_test": len(nail_cls)},
    }
    for key, saved in expected.items():
        rec = recomputed_map[key]
        mismatches = []
        for field, value in saved.items():
            actual = rec.get(field, rec.get("n_samples"))
            if field == "n_test":
                actual = rec["n_test"]
            if not _close(value, actual):
                mismatches.append({"key": field, "saved": value, "recomputed": actual})
        model, task = key.split("|")
        row = summary[(summary.model == model) & (summary.task == task)]
        out.append({
            "check": key,
            "ok": not mismatches,
            "has_image_ids": False,
            "n_rows": rec["n_test"],
            "summary_rows": int(len(row)),
            "recomputed": {k: rec.get(k) for k in ("MAE_g/dL", "RMSE_g/dL", "R2", "accuracy", "f1_score", "roc_auc", "n_test")},
            "mismatches": mismatches,
        })
    out.append({
        "check": "M3 CNN (Palm)|classification",
        "ok": True,
        "note": "Palm CNN predictions exist (n=667) but are excluded from Member 4 comparison because original splits contain exact duplicates and conflicting labels.",
        "n_rows": int(len(palm_cls)),
        "recomputed_accuracy": palm_cls_m["accuracy"],
        "has_image_ids": bool({"image_path", "image_id"} & set(palm_cls.columns)),
    })
    return out


def check_cnn_experiment(dataset: str, expected_test: int) -> dict:
    embed = pd.read_csv(M4_DIR / "outputs" / "embeddings" / f"m3_{dataset}_regression_embeddings.csv")
    preds = pd.read_csv(OUT_METRICS / "cnn_embedding_experiment" / f"{dataset}_cnn_embedding_xgb_predictions.csv")
    report = json.loads((OUT_METRICS / "cnn_embedding_experiment" / f"{dataset}_cnn_embedding_xgb_metrics.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(OUT_METRICS / f"{dataset}_xgb_reg_predictions.csv")
    issues = []
    cols = [c for c in embed.columns if c.startswith("embedding_")]
    if len(cols) != 1024:
        issues.append(f"embedding dimension {len(cols)}")
    if not np.isfinite(embed[cols].to_numpy(dtype=float)).all():
        issues.append("non-finite embeddings")
    if int(embed.groupby("participant_id").split.nunique().max()) != 1:
        issues.append("participant split leakage")
    if int((embed.split == "test").sum()) != expected_test:
        issues.append(f"unexpected test count {(embed.split == 'test').sum()}")

    cnn_mae = float(mean_absolute_error(preds["target_hb_g_dl"], preds["cnn_embedding_xgb_prediction_g_dl"]))
    cnn_rmse = float(mean_squared_error(preds["target_hb_g_dl"], preds["cnn_embedding_xgb_prediction_g_dl"]) ** 0.5)
    cnn_r2 = float(r2_score(preds["target_hb_g_dl"], preds["cnn_embedding_xgb_prediction_g_dl"]))
    hand_mae = float(mean_absolute_error(preds["target_hb_g_dl"], preds["handcrafted_xgb_prediction_g_dl"]))
    hand_rmse = float(mean_squared_error(preds["target_hb_g_dl"], preds["handcrafted_xgb_prediction_g_dl"]) ** 0.5)
    hand_r2 = float(r2_score(preds["target_hb_g_dl"], preds["handcrafted_xgb_prediction_g_dl"]))
    saved_cnn = report["test_metrics"]["cnn_embedding_xgboost"]
    saved_hand = report["test_metrics"]["existing_handcrafted_xgboost"]
    for key, actual, saved in (
        ("MAE_g/dL", cnn_mae, saved_cnn["MAE_g/dL"]),
        ("RMSE_g/dL", cnn_rmse, saved_cnn["RMSE_g/dL"]),
        ("R2", cnn_r2, saved_cnn["R2"]),
        ("hand_MAE", hand_mae, saved_hand["MAE_g/dL"]),
        ("hand_RMSE", hand_rmse, saved_hand["RMSE_g/dL"]),
        ("hand_R2", hand_r2, saved_hand["R2"]),
    ):
        # CSV serialization rounds predictions; this tolerance is below one micro g/dL.
        if abs(float(actual) - float(saved)) > 1e-6:
            issues.append(f"{key} drift saved={saved} recomputed={actual}")

    embed_test = set(embed.loc[embed.split.eq("test"), "image_path"].map(_norm))
    pred_paths = set(preds["image_path"].map(_norm))
    base_paths = set(baseline["image_path"].map(_norm))
    if not (embed_test == pred_paths == base_paths):
        issues.append("held-out image paths are not identical")
    merged = preds.copy()
    merged["_k"] = merged.image_path.map(_norm)
    base = baseline.copy()
    base["_k"] = base.image_path.map(_norm)
    joined = merged.merge(base, on="_k", suffixes=("_cnn", "_base"), validate="one_to_one")
    if not np.allclose(joined["target_hb_g_dl"], joined["true"], atol=1e-8):
        issues.append("Hb values differ from handcrafted baseline")
    return {
        "check": f"{dataset}_cnn_embedding_xgboost",
        "ok": not issues,
        "issues": issues,
        "split_counts": {k: int(v) for k, v in embed.split.value_counts().items()},
        "participants": int(embed.participant_id.nunique()),
        "paired_n": int(len(preds)),
        "cnn": {"MAE_g/dL": cnn_mae, "RMSE_g/dL": cnn_rmse, "R2": cnn_r2},
        "handcrafted": {"MAE_g/dL": hand_mae, "RMSE_g/dL": hand_rmse, "R2": hand_r2},
    }


def check_fullhand_transfer() -> dict:
    embed = pd.read_csv(M4_DIR / "outputs" / "embeddings" / "m3_palm_classifier_fullhand_transfer_embeddings.csv")
    preds = pd.read_csv(OUT_METRICS / "cnn_embedding_experiment" / "full_hand_palm_checkpoint_transfer_xgb_predictions.csv")
    report = json.loads((OUT_METRICS / "cnn_embedding_experiment" / "full_hand_palm_checkpoint_transfer_xgb_metrics.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(OUT_METRICS / "full_hand_xgb_reg_predictions.csv")
    issues = []
    cols = [c for c in embed.columns if c.startswith("embedding_")]
    if len(cols) != 1024 or not np.isfinite(embed[cols].to_numpy(dtype=float)).all():
        issues.append("embedding matrix invalid or non-finite")
    if embed.groupby("participant_id").split.nunique().max() != 1:
        issues.append("participant split leakage")
    if report.get("status", "").find("exploratory transfer") < 0:
        issues.append("transfer caveat missing from report")
    if report.get("checkpoint_safe_load") != "torch.load(map_location='cpu', weights_only=True)":
        issues.append("checkpoint safe-load provenance missing")

    key = "palm_checkpoint_transfer_xgb_prediction_g_dl"
    actual = {
        "MAE_g/dL": float(mean_absolute_error(preds.target_hb_g_dl, preds[key])),
        "RMSE_g/dL": float(mean_squared_error(preds.target_hb_g_dl, preds[key]) ** 0.5),
        "R2": float(r2_score(preds.target_hb_g_dl, preds[key])),
    }
    saved = report["test_metrics"]["palm_checkpoint_transfer_xgboost"]
    for metric, value in actual.items():
        if abs(value - float(saved[metric])) > 1e-6:
            issues.append(f"{metric} mismatch: saved={saved[metric]} recomputed={value}")

    paths = lambda frame: set(frame.image_path.map(_norm))
    if paths(preds) != paths(baseline) or paths(preds) != paths(embed[embed.split.eq("test")]):
        issues.append("held-out Full-hand image paths differ across embeddings/predictions/baseline")
    pred_join = preds.copy()
    base_join = baseline.copy()
    pred_join["_k"] = pred_join.image_path.map(_norm)
    base_join["_k"] = base_join.image_path.map(_norm)
    joined = pred_join.merge(base_join, on="_k", suffixes=("_transfer", "_base"), validate="one_to_one")
    if not np.allclose(joined.target_hb_g_dl, joined.true, atol=1e-8):
        issues.append("Full-hand Hb targets differ from baseline predictions")

    return {"check": "full_hand_palm_classifier_transfer_xgboost", "ok": not issues, "issues": issues,
            "paired_n": len(preds), "transfer": actual,
            "handcrafted": report["test_metrics"]["existing_handcrafted_xgboost"]}


def check_fullhand_scratch_cnn() -> dict:
    embed_path = M4_DIR / "outputs" / "embeddings" / "fullhand_scratch_cnn_embeddings.csv"
    provenance_path = M4_DIR / "outputs" / "embeddings" / "fullhand_scratch_cnn_embeddings_provenance.json"
    checkpoint_path = OUT_MODELS / "fullhand_cnn_scratch" / "fullhand_scratch_cnn_regressor.pth"
    predictions_path = OUT_METRICS / "fullhand_cnn_experiment" / "fullhand_scratch_cnn_xgb_predictions.csv"
    report_path = OUT_METRICS / "fullhand_cnn_experiment" / "fullhand_scratch_cnn_xgb_metrics.json"
    embed = pd.read_csv(embed_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    preds = pd.read_csv(predictions_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline = pd.read_csv(OUT_METRICS / "full_hand_xgb_reg_predictions.csv")
    issues = []
    cols = [c for c in embed.columns if c.startswith("embedding_")]
    counts = {s: int(embed.split.eq(s).sum()) for s in ("train", "validation", "test")}
    if len(embed) != 250 or counts != {"train": 175, "validation": 37, "test": 38}:
        issues.append(f"unexpected embedding rows/splits: rows={len(embed)}, counts={counts}")
    if len(cols) != 64 or not np.isfinite(embed[cols].to_numpy(dtype=float)).all():
        issues.append("scratch-CNN embedding matrix is invalid or non-finite")
    if not embed.image_id.is_unique or not embed.image_path.is_unique:
        issues.append("duplicate image ID/path in embeddings")
    if embed.groupby("participant_id").split.nunique().max() != 1:
        issues.append("participant leakage across splits")
    if provenance.get("pretrained") is not False or provenance.get("best_epoch") is None:
        issues.append("scratch-training/validation-selection provenance missing")
    if not checkpoint_path.is_file() or not provenance.get("checkpoint_sha256"):
        issues.append("scratch CNN checkpoint missing or unhashed")
    if not report.get("status", "").startswith("Full-hand-trained scratch CNN"):
        issues.append("report does not identify actual Full-hand scratch training")

    cnn_col = "fullhand_scratch_cnn_xgb_prediction_g_dl"
    actual_cnn = {
        "MAE_g/dL": float(mean_absolute_error(preds.target_hb_g_dl, preds[cnn_col])),
        "RMSE_g/dL": float(mean_squared_error(preds.target_hb_g_dl, preds[cnn_col]) ** 0.5),
        "R2": float(r2_score(preds.target_hb_g_dl, preds[cnn_col])),
    }
    actual_hand = {
        "MAE_g/dL": float(mean_absolute_error(preds.target_hb_g_dl, preds.handcrafted_xgb_prediction_g_dl)),
        "RMSE_g/dL": float(mean_squared_error(preds.target_hb_g_dl, preds.handcrafted_xgb_prediction_g_dl) ** 0.5),
        "R2": float(r2_score(preds.target_hb_g_dl, preds.handcrafted_xgb_prediction_g_dl)),
    }
    saved = report["test_metrics"]["fullhand_scratch_cnn_embedding_xgboost"]
    saved_hand = report["test_metrics"]["existing_handcrafted_xgboost"]
    for key, value in actual_cnn.items():
        if abs(value - float(saved[key])) > 1e-6:
            issues.append(f"CNN {key} mismatch: saved={saved[key]} recomputed={value}")
    for key, value in actual_hand.items():
        if abs(value - float(saved_hand[key])) > 1e-6:
            issues.append(f"handcrafted {key} mismatch: saved={saved_hand[key]} recomputed={value}")

    pathset = lambda frame: set(frame.image_path.map(_norm))
    if pathset(preds) != pathset(baseline) or pathset(preds) != pathset(embed[embed.split.eq("test")]):
        issues.append("test image paths are not identical across embeddings/predictions/baseline")
    if not np.allclose(preds.target_hb_g_dl, preds.image_id.map(dict(zip(embed.image_id, embed.target_hb_g_dl))), atol=1e-8):
        issues.append("saved test Hb labels do not match the keyed embedding table")
    pred_join, base_join = preds.copy(), baseline.copy()
    pred_join["_k"], base_join["_k"] = pred_join.image_path.map(_norm), base_join.image_path.map(_norm)
    joined = pred_join.merge(base_join, on="_k", suffixes=("_cnn", "_base"), validate="one_to_one")
    if not np.allclose(joined.target_hb_g_dl, joined.true, atol=1e-8):
        issues.append("CNN/handcrafted held-out Hb values differ")
    if not np.array_equal(pd.to_numeric(joined.participant_id).to_numpy(), pd.to_numeric(joined.sample_id).to_numpy()):
        issues.append("CNN/handcrafted participant IDs differ")

    return {"check": "fullhand_scratch_cnn_embedding_xgboost", "ok": not issues, "issues": issues,
            "split_counts": counts, "paired_n": len(preds), "cnn": actual_cnn, "handcrafted": actual_hand}


def reload_models() -> list[dict]:
    results = []
    mapping = [
        ("eye_handcrafted_regression.pkl", "eye_handcrafted.parquet"),
        ("nail_handcrafted_regression.pkl", "nail_handcrafted.parquet"),
        ("full_hand_handcrafted_regression.pkl", "full_handcrafted.parquet"),
        ("eye_xgb_regression.pkl", "eye_handcrafted.parquet"),
        ("nail_xgb_regression.pkl", "nail_handcrafted.parquet"),
        ("full_hand_xgb_regression.pkl", "full_handcrafted.parquet"),
        ("eye_handcrafted_classification.pkl", "eye_handcrafted.parquet"),
        ("nail_handcrafted_classification.pkl", "nail_handcrafted.parquet"),
        ("palm_handcrafted_classification.pkl", "palm_handcrafted.parquet"),
    ]
    for model_name, cache_name in mapping:
        with (OUT_MODELS / model_name).open("rb") as stream:
            obj = pickle.load(stream)
        cache = pd.read_parquet(OUT_FEATURES / cache_name)
        feat_cols = obj.get("feat_cols") or obj.get("feature_cols")
        sample = cache[feat_cols].head(5).to_numpy(dtype=float)
        if "xgb" in obj:
            pred = obj["xgb"].predict(obj["scaler"].transform(sample))
        else:
            pred = obj["rf"].predict(obj["scaler"].transform(sample))
        results.append({"model": model_name, "ok": bool(np.isfinite(np.asarray(pred, dtype=float)).all()), "n_pred": int(len(pred))})
    for dataset in ("eye", "nail"):
        path = OUT_MODELS / "cnn_embedding" / f"m3_{dataset}_cnn_embedding_xgb.pkl"
        embed = pd.read_csv(M4_DIR / "outputs" / "embeddings" / f"m3_{dataset}_regression_embeddings.csv")
        with path.open("rb") as stream:
            obj = pickle.load(stream)
        sample = embed[obj["embedding_columns"]].head(5)
        pred = obj["xgb"].predict(obj["scaler"].transform(sample))
        results.append({"model": path.name, "ok": bool(np.isfinite(np.asarray(pred, dtype=float)).all()), "n_pred": int(len(pred))})
    transfer_path = OUT_MODELS / "cnn_embedding" / "m3_palm_checkpoint_transfer_fullhand_xgb.pkl"
    transfer_embed = pd.read_csv(M4_DIR / "outputs" / "embeddings" / "m3_palm_classifier_fullhand_transfer_embeddings.csv")
    transfer_predictions = pd.read_csv(OUT_METRICS / "cnn_embedding_experiment" / "full_hand_palm_checkpoint_transfer_xgb_predictions.csv")
    with transfer_path.open("rb") as stream:
        obj = pickle.load(stream)
    test_embed = transfer_embed[transfer_embed.split.eq("test")].copy()
    test_embed["_k"] = test_embed.image_path.map(_norm)
    saved_predictions = transfer_predictions.copy()
    saved_predictions["_k"] = saved_predictions.image_path.map(_norm)
    reloaded = obj["xgb"].predict(obj["scaler"].transform(test_embed[obj["embedding_columns"]]))
    reloaded_map = dict(zip(test_embed._k, reloaded))
    saved_map = dict(zip(saved_predictions._k, saved_predictions["palm_checkpoint_transfer_xgb_prediction_g_dl"]))
    exact = reloaded_map.keys() == saved_map.keys() and all(abs(float(reloaded_map[k]) - float(saved_map[k])) < 1e-6 for k in reloaded_map)
    results.append({"model": transfer_path.name, "ok": bool(exact), "n_pred": len(reloaded), "reproduces_saved_test_predictions": bool(exact)})
    scratch_model_path = OUT_MODELS / "fullhand_cnn_scratch" / "fullhand_scratch_cnn_embedding_xgb.pkl"
    scratch_embed = pd.read_csv(M4_DIR / "outputs" / "embeddings" / "fullhand_scratch_cnn_embeddings.csv")
    scratch_predictions = pd.read_csv(OUT_METRICS / "fullhand_cnn_experiment" / "fullhand_scratch_cnn_xgb_predictions.csv")
    with scratch_model_path.open("rb") as stream:
        obj = pickle.load(stream)
    scratch_test = scratch_embed[scratch_embed.split.eq("test")].copy()
    scratch_test["_k"] = scratch_test.image_path.map(_norm)
    scratch_saved = scratch_predictions.copy()
    scratch_saved["_k"] = scratch_saved.image_path.map(_norm)
    scratch_pred = obj["xgb"].predict(obj["scaler"].transform(scratch_test[obj["embedding_columns"]]))
    scratch_pred_map = dict(zip(scratch_test._k, scratch_pred))
    scratch_saved_map = dict(zip(scratch_saved._k, scratch_saved["fullhand_scratch_cnn_xgb_prediction_g_dl"]))
    scratch_exact = scratch_pred_map.keys() == scratch_saved_map.keys() and all(abs(float(scratch_pred_map[k]) - float(scratch_saved_map[k])) < 1e-6 for k in scratch_pred_map)
    results.append({"model": scratch_model_path.name, "ok": bool(scratch_exact), "n_pred": len(scratch_pred), "reproduces_saved_test_predictions": bool(scratch_exact)})
    return results


def main() -> int:
    ledger = {
        "handcrafted": check_handcrafted(),
        "member3_saved_predictions": check_member3_saved(),
        "cnn_embedding_experiment": [check_cnn_experiment("eye", 89), check_cnn_experiment("nail", 109)],
        "full_hand_transfer_experiment": check_fullhand_transfer(),
        "fullhand_scratch_cnn_experiment": check_fullhand_scratch_cnn(),
        "model_reload": reload_models(),
        "full_hand_cnn": {
            "ok": True,
            "supported": True,
            "experiment_type": "compact Member 4 CNN regressor trained from scratch on Full-hand training participants; XGBoost fits its learned embeddings",
            "exploratory": True,
            "reason": "The archive had no Member 3 Full-hand checkpoint; this separate Full-hand-trained scratch model is small-sample exploratory work and is not equivalent to Member 3's model.",
        },
    }
    failures = []
    for group, items in ledger.items():
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if not item.get("ok", False):
                failures.append({"group": group, "item": item})
    ledger["ok"] = not failures
    ledger["failure_count"] = len(failures)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {LEDGER_PATH}")
    print(f"Validation {'PASSED' if ledger['ok'] else 'FAILED'} ({ledger['failure_count']} mismatched checks)")
    if failures:
        print(json.dumps(failures, indent=2, default=str)[:4000])
        return 1
    for item in ledger["cnn_embedding_experiment"]:
        print(
            f"{item['check']}: n={item['paired_n']} "
            f"CNN MAE={item['cnn']['MAE_g/dL']:.4f} vs handcrafted MAE={item['handcrafted']['MAE_g/dL']:.4f}"
        )
    transfer = ledger["full_hand_transfer_experiment"]
    print(
        f"{transfer['check']}: n={transfer['paired_n']} "
        f"transferred-feature MAE={transfer['transfer']['MAE_g/dL']:.4f} "
        f"vs handcrafted MAE={transfer['handcrafted']['MAE_g/dL']:.4f}"
    )
    scratch = ledger["fullhand_scratch_cnn_experiment"]
    print(
        f"{scratch['check']}: n={scratch['paired_n']} "
        f"scratch-CNN embedding MAE={scratch['cnn']['MAE_g/dL']:.4f} "
        f"vs handcrafted MAE={scratch['handcrafted']['MAE_g/dL']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
