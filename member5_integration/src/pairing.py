"""
pairing.py — Verifies the Nail–Full-hand participant pairing for fusion.

Saves:
  member5_integration/outputs/fusion/pairing_audit.csv
  member5_integration/reports/pairing_audit.md
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from paths import (
    NAIL_SPLITS_CSV, FULL_SPLITS_CSV,
    M5_FUSION, M5_REPORTS,
)


def run_pairing_audit(nail_csv: Path = NAIL_SPLITS_CSV,
                      full_csv: Path = FULL_SPLITS_CSV) -> dict:
    """
    Verify Nail–Full-hand pairing.
    Returns a result dict; also saves CSV and Markdown reports.
    """
    result = {
        "passed": False,
        "issues": [],
        "nail_crop_rows": 0,
        "full_participant_rows": 0,
        "paired_participants": 0,
        "hb_mismatches": 0,
        "split_mismatches": 0,
        "source_mismatches": 0,
        "train_participants": 0,
        "val_participants": 0,
        "test_participants": 0,
        "nail_crops_per_split": {},
        "crops_per_participant_min": 0,
        "crops_per_participant_max": 0,
        "crops_per_participant_mean": 0.0,
    }

    if not nail_csv.exists():
        result["issues"].append(f"Nail splits CSV not found: {nail_csv}")
        _save_audit(result, pd.DataFrame())
        return result

    if not full_csv.exists():
        result["issues"].append(f"Full splits CSV not found: {full_csv}")
        _save_audit(result, pd.DataFrame())
        return result

    nail = pd.read_csv(nail_csv)
    full = pd.read_csv(full_csv)

    # Normalise participant ID
    nail["pid"] = nail["participant_id"].astype(str).str.strip()
    full["pid"] = full["participant_id"].astype(str).str.strip()

    result["nail_crop_rows"]       = len(nail)
    result["full_participant_rows"] = len(full)

    # ── Intersection ─────────────────────────────────────────────────────────
    nail_pids = set(nail["pid"].unique())
    full_pids = set(full["pid"].unique())
    paired    = nail_pids & full_pids
    result["paired_participants"] = len(paired)

    if len(paired) != len(full_pids):
        result["issues"].append(
            f"{len(full_pids) - len(paired)} Full-hand participants have no Nail records."
        )
    if len(paired) != len(nail_pids):
        result["issues"].append(
            f"{len(nail_pids) - len(paired)} Nail participants have no Full-hand record."
        )

    # ── Hb target verification ────────────────────────────────────────────────
    hb_col = "hb_g_per_dl" if "hb_g_per_dl" in nail.columns else "hb_g_dl"
    nail_hb = nail.drop_duplicates("pid").set_index("pid")[hb_col]
    full_hb = full.set_index("pid")[hb_col]

    merged_hb = full_hb.to_frame("hb_full").join(nail_hb.to_frame("hb_nail"), how="inner")
    hb_mismatch = (merged_hb["hb_full"] != merged_hb["hb_nail"]).sum()
    result["hb_mismatches"] = int(hb_mismatch)
    if hb_mismatch > 0:
        result["issues"].append(f"Hb target mismatches across {hb_mismatch} participants.")

    # ── Split alignment ───────────────────────────────────────────────────────
    nail_split = nail.drop_duplicates("pid").set_index("pid")["split"]
    full_split = full.set_index("pid")["split"]
    merged_split = full_split.to_frame("split_full").join(
        nail_split.to_frame("split_nail"), how="inner"
    )
    split_mismatch = (merged_split["split_full"] != merged_split["split_nail"]).sum()
    result["split_mismatches"] = int(split_mismatch)
    if split_mismatch > 0:
        result["issues"].append(
            f"Split assignment mismatches across {split_mismatch} participants."
        )

    # ── Source-image references ───────────────────────────────────────────────
    if "source_image" in nail.columns and "source_image" in full.columns:
        nail_src = nail.drop_duplicates("pid").set_index("pid")["source_image"]
        full_src = full.set_index("pid")["source_image"]
        merged_src = full_src.to_frame("src_full").join(
            nail_src.to_frame("src_nail"), how="inner"
        )
        src_mismatch = (merged_src["src_full"] != merged_src["src_nail"]).sum()
        result["source_mismatches"] = int(src_mismatch)
        if src_mismatch > 0:
            result["issues"].append(
                f"Source-image reference mismatches: {src_mismatch} participants."
            )

    # ── Split participant counts (using Full-hand as ground truth) ────────────
    for sp in ["train", "validation", "test"]:
        cnt = int((full["split"] == sp).sum())
        result[f"{sp}_participants"] = cnt
    result["val_participants"] = result["validation_participants"]

    # ── Nail crop counts ──────────────────────────────────────────────────────
    for sp in ["train", "validation", "test"]:
        result["nail_crops_per_split"][sp] = int((nail["split"] == sp).sum())

    crops_per = nail.groupby("pid").size()
    result["crops_per_participant_min"]  = int(crops_per.min())
    result["crops_per_participant_max"]  = int(crops_per.max())
    result["crops_per_participant_mean"] = round(float(crops_per.mean()), 2)

    # ── Pass / Fail ───────────────────────────────────────────────────────────
    result["passed"] = (
        hb_mismatch == 0
        and split_mismatch == 0
        and len(paired) == 250
        and not result["issues"]
    )

    # ── Build audit DataFrame ─────────────────────────────────────────────────
    audit_df = (
        merged_hb.join(merged_split, how="left")
        .join(merged_src, how="left")
        .join(crops_per.rename("nail_crop_count"), how="left")
        .reset_index()
        .rename(columns={"pid": "participant_id"})
    )

    _save_audit(result, audit_df)
    return result


def _save_audit(result: dict, audit_df: pd.DataFrame):
    M5_FUSION.mkdir(parents=True, exist_ok=True)
    M5_REPORTS.mkdir(parents=True, exist_ok=True)

    # CSV
    if not audit_df.empty:
        audit_df.to_csv(M5_FUSION / "pairing_audit.csv", index=False)

    # Markdown
    status = "PASSED" if result["passed"] else "FAILED"
    issues_md = "\n".join(f"- {i}" for i in result["issues"]) if result["issues"] else "None."

    md = f"""# Nail–Full-hand Pairing Audit

**Status: {status}**  
**Date:** Auto-generated by member5_integration/src/pairing.py

## Summary

| Item | Value |
|------|-------|
| Nail crop rows | {result['nail_crop_rows']} |
| Full-hand participant rows | {result['full_participant_rows']} |
| Paired participants | {result['paired_participants']} |
| Hb target mismatches | {result['hb_mismatches']} |
| Split assignment mismatches | {result['split_mismatches']} |
| Source-image mismatches | {result['source_mismatches']} |

## Split Counts (Full-hand participants)

| Split | Participants |
|-------|-------------|
| Train | {result['train_participants']} |
| Validation | {result['val_participants']} |
| Test | {result['test_participants']} |

## Nail Crops per Split

| Split | Nail Crop Count |
|-------|----------------|
| Train | {result['nail_crops_per_split'].get('train', 0)} |
| Validation | {result['nail_crops_per_split'].get('validation', 0)} |
| Test | {result['nail_crops_per_split'].get('test', 0)} |

## Nail Crops per Participant

| Statistic | Value |
|-----------|-------|
| Min | {result['crops_per_participant_min']} |
| Max | {result['crops_per_participant_max']} |
| Mean | {result['crops_per_participant_mean']} |

## Issues

{issues_md}

## Notes

- Nail and Full-hand images are **paired views** from the same source photograph — not independent clinical sensors.
- Both datasets originate from the same 250 Figshare participants.
- Fusion is scientifically defensible only because the pairings are verified above.
- Small test set (38 participants) limits generalisability of any metric differences.
"""
    (M5_REPORTS / "pairing_audit.md").write_text(md, encoding="utf-8")
    print(f"Pairing audit saved -> {M5_FUSION / 'pairing_audit.csv'}")
    print(f"Pairing report saved -> {M5_REPORTS / 'pairing_audit.md'}")
    print(f"Status: {status}")


if __name__ == "__main__":
    r = run_pairing_audit()
    for k, v in r.items():
        print(f"  {k}: {v}")

