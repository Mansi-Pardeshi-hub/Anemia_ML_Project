"""
report_utils.py — Helper utilities for generating Member 5 integration reports
and sample image quality audits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from paths import (
    M5_QUALITY, M5_REPORTS, M5_METRICS, M5_PREDICTIONS, M5_FUSION,
    FULLHAND_FEAT_PARQUET, NAIL_FEAT_PARQUET, EYE_FEAT_PARQUET,
)
from image_quality import check_image_quality


def generate_quality_sample_audit() -> pd.DataFrame:
    """Run image_quality checks on a sample of test images across modalities and save to outputs/quality/."""
    records = []
    for modality, pq_path in [
        ("full_hand", FULLHAND_FEAT_PARQUET),
        ("nail", NAIL_FEAT_PARQUET),
        ("eye", EYE_FEAT_PARQUET),
    ]:
        if not pq_path.exists():
            continue
        df = pd.read_parquet(pq_path)
        test_df = df[df["split"] == "test"].head(5) if "split" in df.columns else df.head(5)
        img_col = "abs_image_path" if "abs_image_path" in test_df.columns else "image_path"
        for _, row in test_df.iterrows():
            qr = check_image_quality(row[img_col])
            records.append({
                "modality": modality,
                "participant_id": str(row.get("participant_id", "")),
                "is_readable": qr["is_readable"],
                "width": qr["width"],
                "height": qr["height"],
                "brightness_mean": qr["brightness_mean"],
                "contrast_std": qr["contrast_std"],
                "blur_score": qr["blur_score"],
                "saturation_mean": qr["saturation_mean"],
                "warning_count": len(qr["warnings"]),
                "warnings": "; ".join(qr["warnings"]),
            })

    out_df = pd.DataFrame(records)
    M5_QUALITY.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(M5_QUALITY / "sample_image_quality_audit.csv", index=False)
    (M5_QUALITY / "sample_image_quality_audit.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return out_df


if __name__ == "__main__":
    df = generate_quality_sample_audit()
    print(f"Saved {len(df)} quality audit rows -> {M5_QUALITY / 'sample_image_quality_audit.csv'}")

