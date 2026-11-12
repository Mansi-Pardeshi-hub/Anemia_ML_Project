"""Load Member 3 regression checkpoints and compare two embeddings to the saved tables.

Run with the isolated cnn_env interpreter from the project root:

    cnn_env\\Scripts\\python.exe member4_ml_modelling\\src\\smoke_test_m3_cnn_embeddings.py

This does not re-extract the full embedding tables or modify checkpoints.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from extract_m3_cnn_embeddings import FEATURE_NAMES, load_model_classes, preprocess, safe_checkpoint_load


ROOT = Path(__file__).resolve().parents[2]
EMBED = ROOT / "member4_ml_modelling" / "outputs" / "embeddings"
OUT = ROOT / "member4_ml_modelling" / "outputs" / "metrics" / "cnn_embedding_experiment" / "cnn_checkpoint_smoke_test.json"

SPECS = (
    ("eye", "datasets/member3_extracted/models/eye/best_regression_model.pth", "m3_eye_regression_embeddings.csv"),
    ("nail", "datasets/member3_extracted/models/nail_skin/best_nail_regression_model.pth", "m3_nail_regression_embeddings.csv"),
)


def compare_dataset(dataset: str, checkpoint_rel: str, embedding_name: str) -> dict:
    checkpoint_path = ROOT / checkpoint_rel
    rows = list(csv.DictReader((EMBED / embedding_name).open(newline="", encoding="utf-8-sig")))
    test_rows = [row for row in rows if row["split"] == "test"][:2]
    if len(test_rows) != 2:
        raise RuntimeError(f"{dataset}: need two held-out rows for the smoke test")
    checkpoint = safe_checkpoint_load(checkpoint_path, dataset)
    model = load_model_classes()(dropout=0.4).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    max_abs = []
    with torch.inference_mode():
        batch = torch.stack([preprocess(row["image_path"]) for row in test_rows])
        if tuple(batch.shape) != (2, 3, 224, 224):
            raise RuntimeError(f"{dataset}: unexpected input tensor {tuple(batch.shape)}")
        features = model.backbone.get_features(batch).cpu().numpy().astype(np.float64)
    if features.shape != (2, 1024) or not np.isfinite(features).all():
        raise RuntimeError(f"{dataset}: invalid smoke-test embeddings {features.shape}")
    for i, row in enumerate(test_rows):
        saved = np.asarray([float(row[name]) for name in FEATURE_NAMES], dtype=np.float64)
        max_abs.append(float(np.max(np.abs(saved - features[i]))))
    return {
        "dataset": dataset,
        "checkpoint": checkpoint_rel,
        "ok": all(delta < 1e-5 for delta in max_abs),
        "n_compared": 2,
        "input_shape": [2, 3, 224, 224],
        "embedding_shape": [2, 1024],
        "max_abs_diff": max_abs,
        "image_ids": [row["image_id"] for row in test_rows],
        "weights_only": True,
    }


def main() -> int:
    torch.manual_seed(0)
    reports = [compare_dataset(*spec) for spec in SPECS]
    payload = {"ok": all(item["ok"] for item in reports), "datasets": reports}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
