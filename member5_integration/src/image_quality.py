"""
image_quality.py — Engineering-level image quality checks for uploaded images.

Returns a structured dict; thresholds documented inline.
These are engineering warnings only. They do not clinically validate an image.
"""
import warnings
import numpy as np
from pathlib import Path
from PIL import Image


# ── Documented thresholds ─────────────────────────────────────────────────────
MIN_DIM_PX          = 64        # Images smaller than this in either dimension
MAX_ASPECT_RATIO    = 10.0      # Width/Height or Height/Width ratio ceiling
DARK_THRESHOLD      = 30.0      # Mean brightness below → "nearly black"
BRIGHT_THRESHOLD    = 220.0     # Mean brightness above → "nearly white"
LOW_CONTRAST_STD    = 15.0      # Std of greyscale below → low dynamic range
BLUR_THRESHOLD      = 50.0      # Variance of Laplacian below → blurry
SATURATION_HIGH     = 240.0     # HSV S-channel mean above → oversaturated


def check_image_quality(image_path) -> dict:
    """
    Run all quality checks on a single image.

    Parameters
    ----------
    image_path : str or Path

    Returns
    -------
    dict with keys:
        is_readable, warnings, brightness_mean, contrast_std,
        blur_score, width, height, channels, aspect_ratio,
        saturation_mean (or None if not computable)
    """
    path = Path(image_path)
    result = {
        "is_readable":      False,
        "warnings":         [],
        "brightness_mean":  None,
        "contrast_std":     None,
        "blur_score":       None,
        "width":            None,
        "height":           None,
        "channels":         None,
        "aspect_ratio":     None,
        "saturation_mean":  None,
        "file_path":        str(path),
    }

    # 1. File readability
    if not path.exists():
        result["warnings"].append(f"File not found: {path}")
        return result
    if path.stat().st_size == 0:
        result["warnings"].append("File is empty (0 bytes).")
        return result

    try:
        img = Image.open(path)
        img.verify()          # check for truncation
    except Exception as exc:
        result["warnings"].append(f"File cannot be opened as image: {exc}")
        return result

    # Re-open (verify() closes the file)
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        result["warnings"].append(f"RGB conversion failed: {exc}")
        return result

    result["is_readable"] = True
    w, h = img.size
    result["width"]  = w
    result["height"] = h
    result["channels"] = 3

    # 2. Dimensions
    if w < MIN_DIM_PX or h < MIN_DIM_PX:
        result["warnings"].append(
            f"Image very small ({w}x{h}). Minimum recommended: {MIN_DIM_PX}px each side."
        )

    # 3. Aspect ratio
    ar = max(w, h) / max(min(w, h), 1)
    result["aspect_ratio"] = round(ar, 2)
    if ar > MAX_ASPECT_RATIO:
        result["warnings"].append(
            f"Extreme aspect ratio ({ar:.1f}:1). Expected roughly square or portrait images."
        )

    arr = np.array(img, dtype=np.float32)   # H x W x 3

    # 4. Brightness
    grey = arr.mean(axis=2)   # H x W
    brightness = float(grey.mean())
    result["brightness_mean"] = round(brightness, 2)
    if brightness < DARK_THRESHOLD:
        result["warnings"].append(
            f"Image appears very dark (mean brightness {brightness:.1f}/255). "
            f"Threshold: <{DARK_THRESHOLD}."
        )
    elif brightness > BRIGHT_THRESHOLD:
        result["warnings"].append(
            f"Image appears very bright/overexposed (mean {brightness:.1f}/255). "
            f"Threshold: >{BRIGHT_THRESHOLD}."
        )

    # 5. Contrast (std of greyscale)
    contrast_std = float(grey.std())
    result["contrast_std"] = round(contrast_std, 2)
    if contrast_std < LOW_CONTRAST_STD:
        result["warnings"].append(
            f"Very low contrast (greyscale std {contrast_std:.1f}). "
            f"Threshold: <{LOW_CONTRAST_STD}."
        )

    # 6. Blur — variance of discrete 2-D Laplacian (applied to greyscale)
    try:
        if grey.shape[0] >= 3 and grey.shape[1] >= 3:
            lap_arr = (
                grey[:-2, 1:-1]
                + grey[2:, 1:-1]
                + grey[1:-1, :-2]
                + grey[1:-1, 2:]
                - 4.0 * grey[1:-1, 1:-1]
            )
            blur_score = float(lap_arr.var())
        else:
            blur_score = 0.0
        result["blur_score"] = round(blur_score, 2)
        if blur_score < BLUR_THRESHOLD:
            result["warnings"].append(
                f"Image may be blurry (Laplacian variance {blur_score:.1f}). "
                f"Threshold: <{BLUR_THRESHOLD}."
            )
    except Exception as exc:
        result["warnings"].append(f"Blur check failed: {exc}")

    # 7. Colour saturation via HSV
    try:
        hsv = img.convert("HSV")
        hsv_arr = np.array(hsv, dtype=np.float32)
        sat_mean = float(hsv_arr[:, :, 1].mean())
        result["saturation_mean"] = round(sat_mean, 2)
        if sat_mean > SATURATION_HIGH:
            result["warnings"].append(
                f"Unusual colour saturation (HSV S mean={sat_mean:.1f}). "
                f"May indicate colour-processed or non-standard image."
            )
    except Exception:
        pass   # HSV conversion not always available

    # 8. Framing / central-content heuristic
    if w >= 16 and h >= 16:
        y0, y1 = h // 4, (3 * h) // 4
        x0, x1 = w // 4, (3 * w) // 4
        center_std = float(grey[y0:y1, x0:x1].std())
        if center_std < LOW_CONTRAST_STD * 0.5:
            result["warnings"].append(
                f"Framing warning: central 50% region is nearly uniform (std={center_std:.1f}). "
                f"Ensure the target tissue occupies the center of the frame."
            )

    return result


def summarise_quality(result: dict) -> str:
    """Human-readable one-line quality summary."""
    if not result["is_readable"]:
        return "UNREADABLE — " + "; ".join(result["warnings"])
    if result["warnings"]:
        return f"OK with {len(result['warnings'])} warning(s): " + "; ".join(result["warnings"])
    return "OK — no quality issues detected."

