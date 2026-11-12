"""
feature_extraction.py — Handcrafted image feature extraction for anemia modalities.

Features extracted per image:
  • RGB channel statistics (mean, std, percentiles 10/25/50/75/90)
  • HSV/LAB channel statistics
  • Colour histogram (binned, normalised)
  • Grey-level co-occurrence matrix (GLCM) texture features on L channel
  • Paleness / redness ratio

All features are deterministic and do not require a GPU.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available — GLCM features will be skipped.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_rgb(image_path: Path) -> np.ndarray | None:
    """Load image as uint8 RGB array (H, W, 3). Returns None on failure."""
    try:
        img = Image.open(image_path).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        warnings.warn(f"Cannot load {image_path}: {exc}")
        return None


def _channel_stats(arr: np.ndarray, prefix: str) -> dict:
    """Return mean, std, and percentiles for a 1-D or 2-D channel array."""
    flat = arr.flatten().astype(np.float32)
    p = np.percentile(flat, [10, 25, 50, 75, 90])
    return {
        f"{prefix}_mean":  float(flat.mean()),
        f"{prefix}_std":   float(flat.std()),
        f"{prefix}_p10":   float(p[0]),
        f"{prefix}_p25":   float(p[1]),
        f"{prefix}_p50":   float(p[2]),
        f"{prefix}_p75":   float(p[3]),
        f"{prefix}_p90":   float(p[4]),
    }


def _histogram_features(arr: np.ndarray, prefix: str, bins: int = 16) -> dict:
    """Normalised histogram over [0, 255]."""
    hist, _ = np.histogram(arr.flatten(), bins=bins, range=(0, 255))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-9
    return {f"{prefix}_hist_{i}": float(v) for i, v in enumerate(hist)}


def _glcm_features(grey: np.ndarray) -> dict:
    """GLCM texture features on a grey-level image."""
    if not SKIMAGE_AVAILABLE:
        return {}
    grey8 = (grey / 16).astype(np.uint8)   # reduce to 16 levels
    distances  = [1, 2]
    angles     = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(grey8, distances=distances, angles=angles,
                        levels=16, symmetric=True, normed=True)
    feats = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).flatten()
        feats[f"glcm_{prop}_mean"] = float(vals.mean())
        feats[f"glcm_{prop}_std"]  = float(vals.std())
    return feats


# ── Main extraction function ───────────────────────────────────────────────────

def extract_features(image_path: Path) -> dict | None:
    """
    Extract handcrafted features from one image.
    Returns a flat dict of feature_name → float, or None if image fails to load.
    """
    rgb = _load_rgb(image_path)
    if rgb is None:
        return None

    feats: dict = {}

    # ── RGB statistics ────────────────────────────────────────────────────────
    for i, ch in enumerate(["R", "G", "B"]):
        feats.update(_channel_stats(rgb[:, :, i], ch))
        feats.update(_histogram_features(rgb[:, :, i], ch, bins=16))

    # Paleness index: high G+B relative to R → less redness → potential pallor
    r_mean = feats["R_mean"]
    g_mean = feats["G_mean"]
    b_mean = feats["B_mean"]
    feats["paleness_index"] = float((g_mean + b_mean) / (r_mean + 1e-9))
    feats["redness_ratio"]  = float(r_mean / (r_mean + g_mean + b_mean + 1e-9))

    # ── HSV statistics ────────────────────────────────────────────────────────
    # Convert via PIL to avoid importing cv2 dependency
    hsv_img = Image.fromarray(rgb).convert("HSV")
    hsv = np.array(hsv_img, dtype=np.float32)
    for i, ch in enumerate(["H", "S", "V"]):
        feats.update(_channel_stats(hsv[:, :, i], f"HSV_{ch}"))

    # ── LAB statistics ────────────────────────────────────────────────────────
    try:
        lab_img = Image.fromarray(rgb).convert("LAB")
        lab = np.array(lab_img, dtype=np.float32)
        for i, ch in enumerate(["L", "A", "B_lab"]):
            feats.update(_channel_stats(lab[:, :, i], f"LAB_{ch}"))
    except Exception:
        pass   # LAB not always supported by Pillow build

    # ── Greyscale + GLCM texture ─────────────────────────────────────────────
    grey_img = Image.fromarray(rgb).convert("L")
    grey = np.array(grey_img, dtype=np.uint8)
    feats.update(_channel_stats(grey, "GREY"))
    feats.update(_glcm_features(grey))

    return feats


# ── Batch extraction ───────────────────────────────────────────────────────────

def extract_feature_matrix(
    manifest: pd.DataFrame,
    image_col: str,
    id_col: str,
    base_dir: Path | None = None,
    max_images: int | None = None,
) -> pd.DataFrame:
    """
    Extract features for every row in `manifest`.

    Parameters
    ----------
    manifest   : DataFrame with at minimum `id_col` and `image_col` columns.
    image_col  : Column holding the relative (or absolute) image path.
    id_col     : Column holding the sample/participant identifier.
    base_dir   : If provided, prepend to relative paths in `image_col`.
    max_images : Cap number of images (for quick testing).

    Returns
    -------
    DataFrame with one row per successfully processed image.
    The first two columns are `id_col` and `image_path`.
    All other columns are numerical features.
    All metadata columns from `manifest` are carried forward.
    """
    records = []
    failed  = 0
    rows    = manifest if max_images is None else manifest.head(max_images)

    for _, row in rows.iterrows():
        img_rel = str(row[image_col])
        if base_dir is not None:
            img_path = base_dir / img_rel
        else:
            img_path = Path(img_rel)

        feats = extract_features(img_path)
        if feats is None:
            failed += 1
            continue

        record = {c: row[c] for c in manifest.columns}
        record["image_path"] = str(img_path)
        record.update(feats)
        records.append(record)

    print(f"  Extracted features: {len(records)} ok, {failed} failed.")
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


FEATURE_COLS_CACHE: list[str] | None = None

def get_feature_columns(df: pd.DataFrame, metadata_cols: list[str]) -> list[str]:
    """Return the list of pure feature columns (excluding metadata)."""
    return [c for c in df.columns if c not in metadata_cols and df[c].dtype != object]

