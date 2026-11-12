"""Validate current Member 4 rows against Member 3 splits and export CNN inputs."""

from pathlib import Path

import pandas as pd

from config import EYE_PREPROCESSED_DIR, FIGSHARE_DIR, M3_DIR, OUT_FEATURES, M4_DIR

OUT = M4_DIR / "outputs" / "embeddings"


def normalize_relative(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def write_manifest(dataset: str) -> None:
    if dataset == "eye":
        features = pd.read_parquet(OUT_FEATURES / "eye_handcrafted.parquet")
        source = pd.read_csv(M3_DIR / "configs" / "eye_splits.csv")
        key_m4, key_m3 = "processed_relative_path", "processed_relative_path"
        target_m4, target_m3 = "hb_g_dl", "hb_g_dl"
        participant_m4, participant_m3 = "participant_id", "pid"
        root = EYE_PREPROCESSED_DIR
        dataset_id = "eye_edi"
        checkpoint_id = "m3_eye_regression"
    elif dataset == "nail":
        features = pd.read_parquet(OUT_FEATURES / "nail_handcrafted.parquet")
        source = pd.read_csv(M3_DIR / "configs" / "nail_skin_nail_splits.csv")
        key_m4, key_m3 = "image_path", "image_path"
        target_m4, target_m3 = "hb_g_dl", "hb_g_per_dl"
        participant_m4 = participant_m3 = "participant_id"
        root = FIGSHARE_DIR
        dataset_id = "figshare_nail"
        checkpoint_id = "m3_nail_regression"
    else:
        raise ValueError(dataset)

    m4 = features.copy()
    m3 = source.copy()

    def m4_relative(value: str) -> str:
        path = Path(str(value))
        if path.is_absolute():
            return path.resolve().relative_to(root.resolve()).as_posix()
        return normalize_relative(value)

    m4["_key"] = m4[key_m4].map(m4_relative)
    m3["_key"] = m3[key_m3].map(normalize_relative)
    if not m4["_key"].is_unique or not m3["_key"].is_unique:
        raise ValueError(f"{dataset}: duplicate image key in a source manifest")

    joined = m4.merge(
        m3[["_key", "split", target_m3, participant_m3]],
        on="_key", how="left", suffixes=("_m4", "_m3"), validate="one_to_one", indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError(f"{dataset}: {int((joined['_merge'] != 'both').sum())} rows did not join to Member 3 splits")

    split4 = "split_m4" if "split_m4" in joined else "split"
    split3 = "split_m3" if "split_m3" in joined else "split"
    target4 = f"{target_m4}_m4" if f"{target_m4}_m4" in joined else target_m4
    target3 = f"{target_m3}_m3" if f"{target_m3}_m3" in joined else target_m3
    participant4 = f"{participant_m4}_m4" if f"{participant_m4}_m4" in joined else participant_m4
    participant3 = f"{participant_m3}_m3" if f"{participant_m3}_m3" in joined else participant_m3

    if not (joined[split4].astype(str) == joined[split3].astype(str)).all():
        raise ValueError(f"{dataset}: split mismatch against Member 3 records")
    target_delta = (pd.to_numeric(joined[target4]) - pd.to_numeric(joined[target3])).abs()
    if target_delta.isna().any() or target_delta.max() > 1e-8:
        raise ValueError(f"{dataset}: Hb target mismatch against Member 3 records")
    if not (joined[participant4].astype(str) == joined[participant3].astype(str)).all():
        raise ValueError(f"{dataset}: participant ID mismatch against Member 3 records")
    if joined[target4].isna().any() or joined["abs_image_path"].isna().any():
        raise ValueError(f"{dataset}: missing image path or Hb target")
    if not joined["abs_image_path"].map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError(f"{dataset}: manifest contains missing image files")
    if joined.groupby(participant4)[split4].nunique().max() != 1:
        raise ValueError(f"{dataset}: participant appears in more than one split")

    manifest = pd.DataFrame({
        "dataset_id": dataset_id,
        "image_id": [f"{dataset_id}:{key}" for key in joined["_key"]],
        "image_path": joined["abs_image_path"].astype(str),
        "source_relative_path": joined["_key"],
        "participant_id": joined[participant4].astype(str),
        "target_hb_g_dl": pd.to_numeric(joined[target4]),
        "split": joined[split4].astype(str),
        "checkpoint_id": checkpoint_id,
    })
    if not manifest["image_id"].is_unique:
        raise ValueError(f"{dataset}: image IDs are not unique")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}_regression_input_manifest.csv"
    manifest.to_csv(path, index=False)
    print(
        f"{dataset}: wrote {path.relative_to(M4_DIR)} rows={len(manifest)} "
        f"splits={manifest['split'].value_counts().to_dict()} "
        f"participants={manifest['participant_id'].nunique()} "
        f"Hb delta max={target_delta.max():.1g}; paths and image IDs unique"
    )


def main() -> None:
    write_manifest("eye")
    write_manifest("nail")


if __name__ == "__main__":
    main()
