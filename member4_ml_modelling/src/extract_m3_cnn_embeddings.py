"""Safely extract Member 3 CNN regression-backbone embeddings.

Run with the project's isolated cnn_env Python, from any working directory:
    cnn_env\\Scripts\\python.exe member4_ml_modelling\\src\\extract_m3_cnn_embeddings.py

The checkpoint is loaded with ``weights_only=True`` only. The Eye regression
checkpoint contains NumPy scalar metadata, so only the two globals found by
static inspection and the specific Float64 dtype class needed to rebuild that
scalar are allowlisted.
"""

import ast
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
M3_SRC = ROOT / "datasets" / "member3_extracted" / "src" / "utils.py"
MODEL_DIR = ROOT / "datasets" / "member3_extracted" / "models"
EMBED_DIR = ROOT / "member4_ml_modelling" / "outputs" / "embeddings"
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
FEATURE_NAMES = [f"embedding_{i:04d}" for i in range(1024)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_classes():
    """Compile only the reviewed model class definitions from Member 3 source."""
    tree = ast.parse(M3_SRC.read_text(encoding="utf-8"), filename=str(M3_SRC))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"LightweightCNN", "RegressionCNN"}
    ]
    if {node.name for node in selected} != {"LightweightCNN", "RegressionCNN"}:
        raise RuntimeError("Expected LightweightCNN and RegressionCNN definitions were not found")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"nn": torch.nn}
    exec(compile(module, str(M3_SRC), "exec"), namespace)
    return namespace["RegressionCNN"]


def safe_checkpoint_load(checkpoint_path: Path, dataset: str) -> dict:
    found = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path))
    expected = {"numpy.dtype", "numpy.core.multiarray.scalar"} if dataset == "eye" else set()
    if found != expected:
        raise RuntimeError(f"Unexpected checkpoint globals for {dataset}: {sorted(found)}")
    if dataset == "eye":
        # This restricts weights_only=True to the two inert NumPy scalar
        # reconstruction globals statically found in the reviewed checkpoint.
        safe = [
            (np.dtype, "numpy.dtype"),
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            (np.dtypes.Float64DType, "numpy.dtypes.Float64DType"),
        ]
        with torch.serialization.safe_globals(safe):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise RuntimeError("Checkpoint did not contain the expected model_state_dict")
    return checkpoint


def preprocess(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float()


def extract(dataset: str, checkpoint_rel: str) -> None:
    input_path = EMBED_DIR / f"{dataset}_regression_input_manifest.csv"
    checkpoint_path = ROOT / checkpoint_rel
    output_path = EMBED_DIR / f"m3_{dataset}_regression_embeddings.csv"
    provenance_path = EMBED_DIR / f"m3_{dataset}_regression_embeddings_provenance.json"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    with input_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty input manifest: {input_path}")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image_id values in {input_path}")
    for row in rows:
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        if row["checkpoint_id"] != f"m3_{dataset}_regression":
            raise ValueError(f"Checkpoint mismatch for {row['image_id']}")

    checkpoint = safe_checkpoint_load(checkpoint_path, dataset)
    RegressionCNN = load_model_classes()
    model = RegressionCNN(dropout=0.4).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    batch_size = 8
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            batch = torch.stack([preprocess(row["image_path"]) for row in batch_rows])
            features = model.backbone.get_features(batch).cpu().numpy()
            if features.shape != (len(batch_rows), 1024):
                raise RuntimeError(f"Unexpected embedding shape: {features.shape}")
            if not np.isfinite(features).all():
                raise RuntimeError(f"Non-finite embedding values in batch starting at {start}")
            vectors.append(features.astype(np.float32, copy=False))
            print(f"{dataset}: extracted {min(start + len(batch_rows), len(rows))}/{len(rows)}")

    matrix = np.concatenate(vectors, axis=0)
    if matrix.shape != (len(rows), 1024) or len(matrix) != len(ids):
        raise RuntimeError("Embedding rows lost their input-manifest alignment")

    out_rows = []
    for row, vector in zip(rows, matrix):
        record = {k: row[k] for k in (
            "dataset_id", "image_id", "image_path", "source_relative_path",
            "participant_id", "target_hb_g_dl", "split", "checkpoint_id",
        )}
        record["checkpoint_sha256"] = sha256(checkpoint_path)
        record.update({name: format(float(value), ".9g") for name, value in zip(FEATURE_NAMES, vector)})
        out_rows.append(record)

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0])
    temporary = output_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    temporary.replace(output_path)

    split_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    provenance = {
        "dataset": dataset,
        "dataset_id": rows[0]["dataset_id"],
        "checkpoint": checkpoint_rel,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_load": "torch.load(map_location='cpu', weights_only=True)",
        "checkpoint_static_unsafe_globals": sorted(
            torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
        ),
        "safe_globals_allowlisted": (
            [] if dataset == "nail" else [
                "numpy.dtype", "numpy.core.multiarray.scalar", "numpy.dtypes.Float64DType"
            ]
        ),
        "member3_architecture_source": str(M3_SRC.relative_to(ROOT)),
        "architecture": "Member 3 LightweightCNN + RegressionCNN, reconstructed by compiling only these class definitions from utils.py; strict state_dict load",
        "pretrained": False,
        "embedding_layer": "RegressionCNN.backbone.get_features() after adaptive average pooling and before regression classifier",
        "embedding_dimension": 1024,
        "preprocessing": "RGB; Pillow bilinear resize 224x224; float32 /255; ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]; no augmentation at inference",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": __import__("torchvision").__version__, "device": "cpu"},
        "input_manifest": str(input_path.relative_to(ROOT)),
        "output_embeddings": str(output_path.relative_to(ROOT)),
        "rows": len(rows),
        "split_counts": split_counts,
        "participants": len({row["participant_id"] for row in rows}),
        "all_values_finite": bool(np.isfinite(matrix).all()),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"{dataset}: saved {matrix.shape} embeddings to {output_path.relative_to(ROOT)}")


def main() -> None:
    torch.manual_seed(0)
    extract("eye", "datasets/member3_extracted/models/eye/best_regression_model.pth")
    extract("nail", "datasets/member3_extracted/models/nail_skin/best_nail_regression_model.pth")


if __name__ == "__main__":
    main()
