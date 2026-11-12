"""Train a compact Full-hand Hb CNN from scratch and export keyed embeddings.

This is separate from Member 3's checkpoints and trains weights only on the
provided Full-hand training participants. Validation MAE selects the checkpoint.
The test partition is read only after checkpoint selection for embedding export.
Run with cnn_env\Scripts\python.exe from the project root.
"""

import csv
import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
EMBED_DIR = ROOT / "member4_ml_modelling" / "outputs" / "embeddings"
MODEL_DIR = ROOT / "member4_ml_modelling" / "outputs" / "models" / "fullhand_cnn_scratch"
METRIC_DIR = ROOT / "member4_ml_modelling" / "outputs" / "metrics" / "fullhand_cnn_experiment"
MANIFEST = EMBED_DIR / "full_hand_scratch_cnn_input_manifest.csv"
CHECKPOINT = MODEL_DIR / "fullhand_scratch_cnn_regressor.pth"
HISTORY = METRIC_DIR / "fullhand_scratch_cnn_training_history.json"
PROVENANCE = EMBED_DIR / "fullhand_scratch_cnn_embeddings_provenance.json"
OUTPUT = EMBED_DIR / "fullhand_scratch_cnn_embeddings.csv"
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 16


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform(path, augment=False):
    with Image.open(path) as source:
        image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    if augment:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float()


class HbImageDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return transform(row["image_path"], self.augment), torch.tensor(float(row["target_hb_g_dl"]), dtype=torch.float32)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FullHandScratchCNN(nn.Module):
    """Compact 64-D CNN regression model; randomly initialized, no pretrained weights."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 24, stride=2),
            nn.MaxPool2d(2),
            ConvBlock(24, 48, stride=2),
            nn.MaxPool2d(2),
            ConvBlock(48, 96, stride=2),
            nn.MaxPool2d(2),
            ConvBlock(96, 128, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(inplace=True))
        self.regressor = nn.Sequential(nn.Dropout(0.4), nn.Linear(64, 1))

    def get_embedding(self, x):
        return self.embedding(self.features(x))

    def forward(self, x):
        return self.regressor(self.get_embedding(x)).squeeze(1)


def load_rows():
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 250 or {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")} != {
        "train": 175, "validation": 37, "test": 38
    }:
        raise ValueError("Full-hand manifest split counts changed; refusing to train/evaluate with an unexpected split")
    for row in rows:
        if row["checkpoint_id"] != "member4_fullhand_cnn_scratch":
            raise ValueError("Incorrect Full-hand scratch-CNN manifest")
        if not Path(row["image_path"]).is_file() or not np.isfinite(float(row["target_hb_g_dl"])):
            raise ValueError(f"Missing image or numeric Hb target for {row['image_id']}")
    if len({r["image_id"] for r in rows}) != len(rows) or len({r["image_path"].casefold() for r in rows}) != len(rows):
        raise ValueError("Duplicate image identifiers or paths")
    if len({r["participant_id"] for r in rows}) != 250:
        raise ValueError("Expected 250 distinct Full-hand participants")
    return rows


def mae(y, p):
    return float(np.abs(np.asarray(y) - np.asarray(p)).mean())


def main():
    seed_everything(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    if set(r["participant_id"] for r in train_rows) & set(r["participant_id"] for r in val_rows):
        raise ValueError("Train/validation participant overlap")

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(HbImageDataset(train_rows, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(HbImageDataset(val_rows, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = FullHandScratchCNN()
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.SmoothL1Loss(beta=0.5)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for images, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        val_target, val_prediction = [], []
        with torch.inference_mode():
            for images, target in val_loader:
                val_prediction.extend(model(images).cpu().numpy().tolist())
                val_target.extend(target.cpu().numpy().tolist())
        val_mae = mae(val_target, val_prediction)
        scheduler.step(val_mae)
        row = {"epoch": epoch, "train_smooth_l1": float(np.mean(train_losses)), "validation_mae_g_dl": val_mae, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"epoch {epoch:03d}: train_loss={row['train_smooth_l1']:.4f} val_MAE={val_mae:.4f} lr={row['learning_rate']:.6g}")
        if val_mae < best_val - 1e-6:
            best_val, best_epoch, stale = val_mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("No validation-selected CNN checkpoint was produced")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "architecture": "FullHandScratchCNN", "embedding_dim": 64}, CHECKPOINT)
    HISTORY.write_text(json.dumps({"epochs": history, "best_epoch": best_epoch, "best_validation_mae_g_dl": best_val}, indent=2), encoding="utf-8")

    # Reload the weights-only checkpoint before extracting all split features.
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    final_model = FullHandScratchCNN().cpu()
    final_model.load_state_dict(saved["state_dict"], strict=True)
    final_model.eval()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start:start + BATCH_SIZE]
            batch = torch.stack([transform(r["image_path"], augment=False) for r in batch_rows])
            values = final_model.get_embedding(batch).cpu().numpy()
            if values.shape != (len(batch_rows), 64) or not np.isfinite(values).all():
                raise RuntimeError("Invalid or nonfinite Full-hand CNN embeddings")
            embeddings.extend(values.tolist())
            print(f"embedding extraction {min(start + len(batch_rows), len(rows))}/{len(rows)}")
    checkpoint_hash = sha256(CHECKPOINT)
    output_rows = []
    for row, vector in zip(rows, embeddings):
        record = {k: row[k] for k in ("dataset_id", "image_id", "image_path", "source_relative_path", "participant_id", "target_hb_g_dl", "split", "target_source")}
        record["checkpoint_id"] = "member4_fullhand_cnn_scratch"
        record["checkpoint_sha256"] = checkpoint_hash
        record.update({f"embedding_{i:03d}": format(float(x), ".10g") for i, x in enumerate(vector)})
        output_rows.append(record)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    provenance = {
        "experiment": "Full-hand Hb CNN trained from scratch; separate from Member 3 checkpoints",
        "architecture": "4 compact ConvBlock stages with channels 24/48/96/128, global average pooling, 64-D projection, dropout 0.4, linear Hb head",
        "parameter_count": params,
        "pretrained": False,
        "seed": SEED,
        "image_preprocessing": "RGB; Pillow bilinear resize 224x224; float32 /255; ImageNet mean/std normalization; train-only horizontal flip and brightness factor 0.9-1.1",
        "training": {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.01, "loss": "SmoothL1Loss(beta=0.5)", "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE, "selection": "minimum validation MAE; model weights fit only on training participants"},
        "best_epoch": best_epoch,
        "best_validation_mae_g_dl": best_val,
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")},
        "participants": len({r["participant_id"] for r in rows}),
        "target_source": "Source-linked numeric Figshare Hb g/dL from the supplied participant split table; no folder-derived class labels used",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_load": "torch.load(map_location='cpu', weights_only=True); strict state_dict load",
        "training_history": str(HISTORY.relative_to(ROOT)),
        "embeddings": str(OUTPUT.relative_to(ROOT)),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"best epoch={best_epoch}; validation MAE={best_val:.4f}; params={params}; embeddings={len(rows)}x64")


if __name__ == "__main__":
    main()
