"""Extract Full-hand embeddings using M3's non-Full-hand Palm classifier safely.

The checkpoint is only a transferred representation from a separately trained
Palm folder-label classifier. It is not Full-hand-trained or pretrained.
"""

import ast
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
EMBED = ROOT / "member4_ml_modelling" / "outputs" / "embeddings"
MANIFEST = EMBED / "full_hand_palm_transfer_input_manifest.csv"
CHECKPOINT = ROOT / "datasets" / "member3_extracted" / "models" / "palm" / "best_model.pth"
SOURCE = ROOT / "datasets" / "member3_extracted" / "src" / "utils.py"
OUTPUT = EMBED / "m3_palm_classifier_fullhand_transfer_embeddings.csv"
PROVENANCE = EMBED / "m3_palm_classifier_fullhand_transfer_embeddings_provenance.json"
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
FEATURES = [f"embedding_{i:04d}" for i in range(1024)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reviewed_model_class():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LightweightCNN"]
    if len(classes) != 1:
        raise RuntimeError("Expected one reviewed LightweightCNN definition")
    namespace = {"nn": torch.nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["LightweightCNN"]


def preprocess(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float()


def main() -> None:
    if not MANIFEST.is_file() or not CHECKPOINT.is_file():
        raise FileNotFoundError("Full-hand manifest or supplied Palm checkpoint is missing")
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(CHECKPOINT)
    if unsafe:
        raise RuntimeError(f"Unexpected checkpoint globals; refusing load: {unsafe}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain model_state_dict")
    config = checkpoint.get("config", {})
    if config.get("num_classes") != 2 or config.get("hb_regression") is not False:
        raise RuntimeError("Checkpoint metadata is not the expected two-class Palm classifier")

    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 250:
        raise RuntimeError(f"Expected 250 Full-hand records; found {len(rows)}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise RuntimeError("Manifest image IDs are not unique")
    for row in rows:
        if row["checkpoint_id"] != "m3_palm_classifier_transfer_exploratory":
            raise RuntimeError("Unexpected manifest checkpoint ID")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])

    Model = load_reviewed_model_class()
    model = Model(num_classes=2, dropout=0.3).cpu()
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    vectors = []
    batch_size = 8
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            batch = torch.stack([preprocess(row["image_path"]) for row in batch_rows])
            values = model.get_features(batch).cpu().numpy()
            if values.shape != (len(batch_rows), 1024) or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid embedding batch at row {start}: {values.shape}")
            vectors.append(values.astype(np.float32, copy=False))
            print(f"Full-hand transfer: {min(start + len(batch_rows), len(rows))}/{len(rows)}")
    matrix = np.concatenate(vectors)

    output_rows = []
    checkpoint_hash = sha256(CHECKPOINT)
    for row, vector in zip(rows, matrix):
        record = {key: row[key] for key in (
            "dataset_id", "image_id", "image_path", "source_relative_path",
            "participant_id", "target_hb_g_dl", "split", "checkpoint_id", "target_source",
        )}
        record["checkpoint_sha256"] = checkpoint_hash
        record.update({key: format(float(value), ".9g") for key, value in zip(FEATURES, vector)})
        output_rows.append(record)

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    split_counts = {s: sum(row["split"] == s for row in rows) for s in ("train", "validation", "test")}
    provenance = {
        "status": "exploratory cross-dataset feature transfer; not a Full-hand-trained CNN",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_safe_load": "torch.load(map_location='cpu', weights_only=True)",
        "checkpoint_static_unsafe_globals": unsafe,
        "source_training_task": "M3 Palm binary classifier on filename-derived folder classes; config says labels not clinically verified and no measured Hb",
        "source_training_data_root": config.get("data_root"),
        "target_training_data": "Figshare Full-hand images with source-provided numeric Hb g/dL",
        "architecture": "M3 LightweightCNN(num_classes=2, dropout=0.3), strict state_dict load; pooled get_features() before class head",
        "pretrained": False,
        "embedding_dimension": 1024,
        "preprocessing": "RGB; Pillow bilinear resize 224x224; float32 /255; ImageNet mean/std; no augmentation at inference",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "rows": len(rows),
        "split_counts": split_counts,
        "participants": len({row["participant_id"] for row in rows}),
        "all_embeddings_finite": bool(np.isfinite(matrix).all()),
        "input_manifest": str(MANIFEST.relative_to(ROOT)),
        "output_embeddings": str(OUTPUT.relative_to(ROOT)),
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Saved {matrix.shape} transferred features to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
