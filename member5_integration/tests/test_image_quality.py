"""
tests/test_image_quality.py — Automated tests for engineering image quality checks.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

_THIS = Path(__file__).resolve().parent
_SRC  = _THIS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from image_quality import check_image_quality, summarise_quality


def _make_temp_image(arr: np.ndarray, suffix: str = ".png") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    Image.fromarray(arr.astype(np.uint8)).save(tmp.name)
    return Path(tmp.name)


def test_unreadable_and_corrupt_files():
    # Non-existent file
    res_missing = check_image_quality("/nonexistent/file_12345.jpg")
    assert not res_missing["is_readable"]
    assert any("not found" in w.lower() for w in res_missing["warnings"])

    # Empty 0-byte file
    tmp_empty = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_empty.close()
    p_empty = Path(tmp_empty.name)
    try:
        res_empty = check_image_quality(p_empty)
        assert not res_empty["is_readable"]
        assert any("empty" in w.lower() for w in res_empty["warnings"])
    finally:
        p_empty.unlink(missing_ok=True)

    # Corrupt non-image bytes
    tmp_corrupt = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_corrupt.write(b"not an image payload at all")
    tmp_corrupt.close()
    p_corrupt = Path(tmp_corrupt.name)
    try:
        res_corrupt = check_image_quality(p_corrupt)
        assert not res_corrupt["is_readable"]
        assert "UNREADABLE" in summarise_quality(res_corrupt)
    finally:
        p_corrupt.unlink(missing_ok=True)


def test_very_small_and_extreme_aspect_ratio():
    # Very small image (16x16)
    small_arr = np.full((16, 16, 3), 120, dtype=np.uint8)
    p_small = _make_temp_image(small_arr)
    try:
        res_small = check_image_quality(p_small)
        assert res_small["is_readable"]
        assert res_small["width"] == 16 and res_small["height"] == 16
        assert any("small" in w.lower() for w in res_small["warnings"])
    finally:
        p_small.unlink(missing_ok=True)

    # Extreme aspect ratio (800x64 -> 12.5:1)
    wide_arr = np.full((64, 800, 3), 120, dtype=np.uint8)
    p_wide = _make_temp_image(wide_arr)
    try:
        res_wide = check_image_quality(p_wide)
        assert res_wide["is_readable"]
        assert res_wide["aspect_ratio"] > 10.0
        assert any("aspect ratio" in w.lower() for w in res_wide["warnings"])
    finally:
        p_wide.unlink(missing_ok=True)


def test_very_dark_and_very_bright():
    # Nearly black (brightness ~ 5)
    dark_arr = np.full((128, 128, 3), 5, dtype=np.uint8)
    p_dark = _make_temp_image(dark_arr)
    try:
        res_dark = check_image_quality(p_dark)
        assert res_dark["is_readable"]
        assert res_dark["brightness_mean"] < 30.0
        assert any("dark" in w.lower() for w in res_dark["warnings"])
    finally:
        p_dark.unlink(missing_ok=True)

    # Nearly white (brightness ~ 250)
    bright_arr = np.full((128, 128, 3), 250, dtype=np.uint8)
    p_bright = _make_temp_image(bright_arr)
    try:
        res_bright = check_image_quality(p_bright)
        assert res_bright["is_readable"]
        assert res_bright["brightness_mean"] > 220.0
        assert any("bright" in w.lower() or "overexposed" in w.lower() for w in res_bright["warnings"])
    finally:
        p_bright.unlink(missing_ok=True)


def test_blur_and_sharp_images():
    # Uniform/smooth gradient image -> blur score < 50
    x = np.linspace(60, 180, 128, dtype=np.float32)
    grad = np.broadcast_to(x, (128, 128))
    blurry_arr = np.stack([grad, grad, grad], axis=-1)
    p_blurry = _make_temp_image(blurry_arr)
    try:
        res_blur = check_image_quality(p_blurry)
        assert res_blur["is_readable"]
        assert res_blur["blur_score"] < 50.0
        assert any("blurry" in w.lower() for w in res_blur["warnings"])
    finally:
        p_blurry.unlink(missing_ok=True)

    # High-frequency textured image -> no blur warning
    rng = np.random.default_rng(42)
    sharp_arr = rng.integers(50, 200, size=(128, 128, 3), dtype=np.uint8)
    p_sharp = _make_temp_image(sharp_arr)
    try:
        res_sharp = check_image_quality(p_sharp)
        assert res_sharp["is_readable"]
        assert res_sharp["blur_score"] >= 50.0
        assert len(res_sharp["warnings"]) == 0
    finally:
        p_sharp.unlink(missing_ok=True)


if __name__ == "__main__":
    for fn in [
        test_unreadable_and_corrupt_files,
        test_very_small_and_extreme_aspect_ratio,
        test_very_dark_and_very_bright,
        test_blur_and_sharp_images,
    ]:
        fn()
        print(f"  [PASS] {fn.__name__}")

