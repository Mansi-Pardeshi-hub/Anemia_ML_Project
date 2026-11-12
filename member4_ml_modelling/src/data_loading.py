"""
data_loading.py — Load and validate split manifests for all three modalities.

Returns clean DataFrames with:
  • participant_id (str key for leakage prevention)
  • image_path    (relative path within the dataset dir)
  • split         (train / validation / test)
  • hb_g_dl       (float, when available)
  • anemia_label  (int 0/1, when available)
  • gender        (str, when available)
"""

import warnings
import hashlib
import pandas as pd
from pathlib import Path


def _deduplicate_exact_images(
    df: pd.DataFrame,
    image_col: str,
    label_cols: list[str],
    audit_path: Path | None,
    dataset_name: str,
    drop_within_split_duplicates: bool = True,
) -> pd.DataFrame:
    """Keep one deterministic copy of each image and log duplicate exclusions.

    Conflicting targets for byte-identical images invalidate all copies.
    Otherwise prefer train, then validation, then test if duplicates cross
    partitions so no exact train image is retained in evaluation.
    """
    hashes = []
    for image_path in df[image_col].astype(str):
        digest = hashlib.sha256()
        with open(image_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes.append(digest.hexdigest())

    work = df.copy()
    work["_exact_image_sha256"] = hashes
    priority = {"train": 0, "validation": 1, "test": 2}
    keep_indices = []
    excluded = []

    for image_hash, group in work.groupby("_exact_image_sha256", sort=False):
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue
        conflicting = any(
            col in group and group[col].dropna().astype(str).nunique() > 1
            for col in label_cols
        )
        if conflicting:
            for _, row in group.iterrows():
                record = row.drop(labels=["_exact_image_sha256"]).to_dict()
                record.update({"duplicate_sha256": image_hash,
                               "resolution": "excluded_all_copies_conflicting_targets",
                               "kept_image_path": ""})
                excluded.append(record)
            continue

        if not drop_within_split_duplicates and group["split"].nunique() == 1:
            keep_indices.extend(group.index.tolist())
            for _, row in group.iterrows():
                record = row.drop(labels=["_exact_image_sha256"]).to_dict()
                record.update({"duplicate_sha256": image_hash,
                               "resolution": "retained_same_split_duplicate_for_m3_comparability",
                               "kept_image_path": str(row[image_col])})
                excluded.append(record)
            continue

        ordered = group.assign(
            _split_priority=group["split"].map(priority).fillna(99),
            _image_sort=group[image_col].astype(str),
        ).sort_values(["_split_priority", "_image_sort"])
        keep_index = ordered.index[0]
        keep_indices.append(keep_index)
        kept_path = str(work.loc[keep_index, image_col])
        for index in group.index:
            if index == keep_index:
                continue
            record = work.loc[index].drop(labels=["_exact_image_sha256"]).to_dict()
            record.update({"duplicate_sha256": image_hash,
                           "resolution": "excluded_duplicate_copy",
                           "kept_image_path": kept_path})
            excluded.append(record)

    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(excluded).to_csv(audit_path, index=False)
    if excluded:
        print(f"{dataset_name}: logged {len(excluded)} exact-duplicate audit row(s); "
              f"details: {audit_path}")
    return work.loc[keep_indices].drop(columns=["_exact_image_sha256"]).copy()


# ── Eye / Conjunctiva ──────────────────────────────────────────────────────────

def load_eye_splits(
    splits_csv: Path,
    eye_preprocessed_dir: Path,
    duplicate_audit_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load the eye split manifest produced by Member 3.

    Expected columns: pid, country, participant_id, processed_relative_path,
                      full_image_path, image_variant, split, hb_g_dl,
                      anemia_label, gender
    """
    df = pd.read_csv(splits_csv)
    df = df.rename(columns={"hb_g_dl": "hb_g_dl"})   # already correct name

    # Build absolute paths using the local extraction directory
    df["abs_image_path"] = df["processed_relative_path"].apply(
        lambda p: str(eye_preprocessed_dir / p)
    )

    # Verify existence
    df["image_exists"] = df["abs_image_path"].apply(lambda p: Path(p).is_file())
    missing = (~df["image_exists"]).sum()
    if missing > 0:
        warnings.warn(f"Eye: {missing}/{len(df)} image paths not found on disk.")

    df = df[df["image_exists"]].copy()
    df["participant_id"] = df["pid"].astype(str)
    df["modality"] = "eye"
    df = _deduplicate_exact_images(
        df, "abs_image_path", ["hb_g_dl", "anemia_label"],
        duplicate_audit_path, "Eye", drop_within_split_duplicates=True,
    )

    print(f"Eye splits loaded: {len(df)} images, "
          f"{df['participant_id'].nunique()} participants, "
          f"splits={df['split'].value_counts().to_dict()}")
    return df


# ── Nail (Figshare) ────────────────────────────────────────────────────────────

def load_nail_splits(
    nail_splits_csv: Path,
    figshare_dir: Path,
) -> pd.DataFrame:
    """
    Load nail-crop split manifest.
    """
    df = pd.read_csv(nail_splits_csv)
    df = df.rename(columns={"hb_g_per_dl": "hb_g_dl"})   # standardise Hb column name
    df["abs_image_path"] = df["image_path"].apply(
        lambda p: str(figshare_dir / p)
    )
    df["image_exists"] = df["abs_image_path"].apply(lambda p: Path(p).is_file())
    missing = (~df["image_exists"]).sum()
    if missing > 0:
        warnings.warn(f"Nail: {missing}/{len(df)} image paths not found.")
    df = df[df["image_exists"]].copy()
    df["participant_id"] = df["participant_id"].astype(str)
    df["modality"] = "nail"

    print(f"Nail splits loaded: {len(df)} images, "
          f"{df['participant_id'].nunique()} participants, "
          f"splits={df['split'].value_counts().to_dict()}")
    return df


# ── Full-hand (Figshare) ────────────────────────────────────────────────────────

def load_full_splits(
    full_splits_csv: Path,
    figshare_dir: Path,
) -> pd.DataFrame:
    """
    Load full-image split manifest (one image per participant).
    """
    df = pd.read_csv(full_splits_csv)
    df = df.rename(columns={"hb_g_per_dl": "hb_g_dl"})   # standardise Hb column name
    df["abs_image_path"] = df["image_path"].apply(
        lambda p: str(figshare_dir / p)
    )
    df["image_exists"] = df["abs_image_path"].apply(lambda p: Path(p).is_file())
    missing = (~df["image_exists"]).sum()
    if missing > 0:
        warnings.warn(f"Full images: {missing}/{len(df)} paths not found.")
    df = df[df["image_exists"]].copy()
    df["participant_id"] = df["participant_id"].astype(str)
    df["modality"] = "full_image"

    print(f"Full-image splits loaded: {len(df)} images, "
          f"{df['participant_id'].nunique()} participants, "
          f"splits={df['split'].value_counts().to_dict()}")
    return df


# ── Palm ───────────────────────────────────────────────────────────────────────

def load_palm_splits(
    palm_splits_csv: Path,
    palm_dir: Path,
    duplicate_audit_path: Path | None = None,
    near_duplicate_exclusions_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load palm split manifest. Labels come from filenames; no Hb values.
    Expected columns (Member 3 palm_splits.csv):
      processed_path, class_label_from_filename, split, inferred_group_id, label
    """
    df = pd.read_csv(palm_splits_csv)

    # Known column names from Member 3's palm_splits.csv
    img_col   = "processed_path"
    id_col    = "inferred_group_id"
    label_col = "label"

    if img_col not in df.columns:
        # Fall back to auto-detection
        for c in df.columns:
            cl = c.lower()
            if "path" in cl or "file" in cl:
                img_col = c
                break

    # Build absolute paths
    df["abs_image_path"] = df[img_col].apply(
        lambda p: str(palm_dir / str(p)) if not Path(str(p)).is_absolute() else str(p)
    )
    df["image_exists"] = df["abs_image_path"].apply(lambda p: Path(p).is_file())
    missing = (~df["image_exists"]).sum()
    if missing > 0:
        warnings.warn(f"Palm: {missing}/{len(df)} image paths not found.")
    df = df[df["image_exists"]].copy()

    # Standardise participant_id
    if id_col in df.columns:
        df["participant_id"] = df[id_col].astype(str)
    else:
        df["participant_id"] = df[img_col].apply(lambda p: Path(p).parent.name)

    # Standardise label
    if label_col in df.columns:
        df["anemia_label"] = df[label_col].astype(int)
    elif "anemia_label" not in df.columns:
        df["anemia_label"] = df["abs_image_path"].apply(
            lambda p: 1 if "anemic" in Path(p).parts[-2].lower()
                      and "non" not in Path(p).parts[-2].lower() else 0
        )

    df["modality"] = "palm"
    df["hb_g_dl"] = float("nan")   # No measured Hb for palm
    if near_duplicate_exclusions_path is not None and near_duplicate_exclusions_path.is_file():
        exclusion_ledger = pd.read_csv(near_duplicate_exclusions_path)
        excluded_hashes = set(exclusion_ledger["duplicate_sha256"].astype(str))
        def exact_hash(path: str) -> str:
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        before = len(df)
        df = df[~df["abs_image_path"].map(exact_hash).isin(excluded_hashes)].copy()
        print(f"Palm: excluded {before - len(df)} exact-content copy/copies of confirmed cross-split "
              f"near-duplicate test image(s); "
              f"ledger: {near_duplicate_exclusions_path}")
    df = _deduplicate_exact_images(
        df, "abs_image_path", ["anemia_label"],
        duplicate_audit_path, "Palm",
    )

    print(f"Palm splits loaded: {len(df)} images, "
          f"{df['participant_id'].nunique()} participants, "
          f"splits={df['split'].value_counts().to_dict()}")
    return df


# ── Leakage check ──────────────────────────────────────────────────────────────

def verify_no_participant_leakage(df: pd.DataFrame, dataset_name: str) -> bool:
    """
    Verify that no participant appears in more than one split.
    Returns True if clean, False if leakage detected.
    """
    splits = df.groupby("participant_id")["split"].nunique()
    leakers = splits[splits > 1]
    if len(leakers) > 0:
        warnings.warn(
            f"{dataset_name}: LEAKAGE — {len(leakers)} participant(s) appear "
            f"in multiple splits: {list(leakers.index[:5])}"
        )
        return False
    print(f"  {dataset_name}: No participant leakage detected [OK]")
    return True
