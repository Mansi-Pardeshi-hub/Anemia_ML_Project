"""
tests/test_pairing.py — Automated tests for Nail–Full-hand participant pairing
and split integrity.
"""

import sys
from pathlib import Path
import pandas as pd

_THIS = Path(__file__).resolve().parent
_SRC  = _THIS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paths import NAIL_SPLITS_CSV, FULL_SPLITS_CSV, M5_FUSION
from pairing import run_pairing_audit


def test_pairing_audit_counts_and_alignment():
    res = run_pairing_audit()
    assert res["passed"] is True
    assert res["nail_crop_rows"] == 719
    assert res["full_participant_rows"] == 250
    assert res["paired_participants"] == 250
    assert res["hb_mismatches"] == 0
    assert res["split_mismatches"] == 0
    assert res["source_mismatches"] == 0
    assert res["train_participants"] == 175
    assert res["val_participants"] == 37
    assert res["test_participants"] == 38
    assert res["nail_crops_per_split"] == {"train": 503, "validation": 107, "test": 109}
    assert res["crops_per_participant_min"] == 1
    assert res["crops_per_participant_max"] == 3


def test_one_to_one_participant_joins_and_no_split_leakage():
    nail = pd.read_csv(NAIL_SPLITS_CSV)
    full = pd.read_csv(FULL_SPLITS_CSV)

    # Full-hand has exactly 1 row per participant
    assert full["participant_id"].is_unique
    assert len(full) == 250

    # No participant appears in more than 1 split
    assert nail.groupby("participant_id")["split"].nunique().max() == 1
    assert full.groupby("participant_id")["split"].nunique().max() == 1

    # Saved audit CSV matches 250 rows
    audit_csv = pd.read_csv(M5_FUSION / "pairing_audit.csv")
    assert len(audit_csv) == 250
    assert audit_csv["participant_id"].is_unique


if __name__ == "__main__":
    for fn in [
        test_pairing_audit_counts_and_alignment,
        test_one_to_one_participant_joins_and_no_split_leakage,
    ]:
        fn()
        print(f"  [PASS] {fn.__name__}")

