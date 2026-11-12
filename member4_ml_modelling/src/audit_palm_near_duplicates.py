"""Audit perceptual near-duplicates in Palm's exact-deduplicated manifest.

Run from the project root:
    python member4_ml_modelling/src/audit_palm_near_duplicates.py

Uses a 64-bit DCT perceptual hash on grayscale 32x32 images. Pairs at
Hamming distance <= 4 creates review candidates. Only cross-split candidates
with SSIM >= 0.98 trigger a test-side exact-hash exclusion; source files are
never modified.
"""

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from config import PALM_DIR, PALM_SPLITS_CSV, OUT_METRICS
from data_loading import load_palm_splits

THRESHOLD = 4


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path: str) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float32)
    low = dctn(pixels, norm="ortho")[:8, :8].ravel()
    bits = low[1:] > np.median(low[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def resized_gray(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((128, 128), Image.Resampling.LANCZOS))


def main() -> None:
    frame = load_palm_splits(PALM_SPLITS_CSV, PALM_DIR)
    paths = frame["abs_image_path"].astype(str).tolist()
    hashes = [phash(path) for path in paths]
    rows = []
    for i, a in enumerate(hashes):
        for j in range(i + 1, len(hashes)):
            distance = (a ^ hashes[j]).bit_count()
            if distance <= THRESHOLD:
                left, right = frame.iloc[i], frame.iloc[j]
                similarity = structural_similarity(
                    resized_gray(paths[i]), resized_gray(paths[j]), data_range=255
                )
                rows.append({
                    "left_path": paths[i], "right_path": paths[j],
                    "left_split": left["split"], "right_split": right["split"],
                    "left_group_id": left["participant_id"],
                    "right_group_id": right["participant_id"],
                    "left_label": int(left["anemia_label"]),
                    "right_label": int(right["anemia_label"]),
                    "hamming_distance": distance,
                    "ssim_128_gray": float(similarity),
                    "cross_split": left["split"] != right["split"],
                    "cross_group": left["participant_id"] != right["participant_id"],
                    "label_conflict": left["anemia_label"] != right["anemia_label"],
                })
    result = pd.DataFrame(rows, columns=[
        "left_path", "right_path", "left_split", "right_split",
        "left_group_id", "right_group_id", "left_label", "right_label",
        "hamming_distance", "ssim_128_gray", "cross_split", "cross_group", "label_conflict",
    ])
    out_csv = OUT_METRICS / "palm_near_duplicate_candidates.csv"
    result.to_csv(out_csv, index=False)
    summary = {
        "method": "64-bit DCT perceptual hash; grayscale 32x32 Lanczos resize; median threshold over non-DC 8x8 coefficients",
        "candidate_threshold_hamming_distance_inclusive": THRESHOLD,
        "automatic_exclusion_rule": "cross-split candidate with pHash distance <= 4 and SSIM >= 0.98; remove test-side exact-hash copies only",
        "unique_exact_content_images_scanned": len(frame),
        "candidate_pairs": len(result),
        "cross_split_pairs": int(result["cross_split"].sum()) if len(result) else 0,
        "high_confidence_ssim_threshold": 0.98,
        "high_confidence_cross_split_pairs": int((result["cross_split"] & (result["ssim_128_gray"] >= 0.98)).sum()) if len(result) else 0,
        "cross_group_pairs": int(result["cross_group"].sum()) if len(result) else 0,
        "label_conflict_pairs": int(result["label_conflict"].sum()) if len(result) else 0,
        "distance_counts": {str(k): int(v) for k, v in result["hamming_distance"].value_counts().sort_index().items()} if len(result) else {},
        "candidate_pairs_csv": str(out_csv.relative_to(ROOT)),
    }
    out_json = OUT_METRICS / "palm_near_duplicate_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # A conservative corrective action: when a test image is near-identical
    # (SSIM >= .98) to an image in a different partition, exclude only the
    # test-side image from evaluation. Never alter original image files.
    confirmed = result[
        result["cross_split"] & (result["ssim_128_gray"] >= 0.98)
    ] if len(result) else result
    test_paths = set()
    for _, row in confirmed.iterrows():
        if row["left_split"] == "test":
            test_paths.add(row["left_path"])
        if row["right_split"] == "test":
            test_paths.add(row["right_path"])
    exclusions = frame[frame["abs_image_path"].isin(test_paths)].copy()
    if len(exclusions):
        exclusions["exclusion_reason"] = "cross_split near-duplicate; pHash distance <= 4 and SSIM >= 0.98"
        digest_map = {path: sha256_file(path)
                      for path in exclusions["abs_image_path"].astype(str)}
        exclusions["duplicate_sha256"] = exclusions["abs_image_path"].map(digest_map)
        exclusion_path = OUT_METRICS / "palm_near_duplicate_test_exclusions.csv"
        exclusions.to_csv(exclusion_path, index=False)
        summary["test_images_excluded"] = len(exclusions)
        raw = pd.read_csv(PALM_SPLITS_CSV)
        raw_paths = raw["processed_path"].map(lambda p: str(PALM_DIR / str(p)))
        excluded_hashes = set(exclusions["duplicate_sha256"])
        summary["raw_source_rows_removed_by_exclusion_hashes"] = sum(
            sha256_file(path) in excluded_hashes for path in raw_paths
        )
        summary["test_exclusion_ledger"] = str(exclusion_path.relative_to(ROOT))
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Pair details: {out_csv}")


if __name__ == "__main__":
    main()
