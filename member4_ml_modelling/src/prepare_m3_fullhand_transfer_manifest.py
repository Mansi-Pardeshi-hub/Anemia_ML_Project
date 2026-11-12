"""Build a verified Full-hand feature manifest for exploratory Palm-CNN transfer."""

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
M4 = ROOT / "member4_ml_modelling"
FEATURES = M4 / "outputs" / "features" / "full_handcrafted.parquet"
SOURCE = ROOT / "datasets" / "member3_extracted" / "configs" / "nail_skin_full_splits.csv"
IMAGE_ROOT = ROOT / "datasets" / "figshare_extracted" / "Figshare_Hb_Preprocessed"
OUTPUT = M4 / "outputs" / "embeddings" / "full_hand_palm_transfer_input_manifest.csv"
SCRATCH_OUTPUT = M4 / "outputs" / "embeddings" / "full_hand_scratch_cnn_input_manifest.csv"


def main() -> None:
    m4 = pd.read_parquet(FEATURES)
    m3 = pd.read_csv(SOURCE)
    m4["_key"] = m4["image_path"].map(lambda p: Path(p).resolve().relative_to(IMAGE_ROOT.resolve()).as_posix())
    m3["_key"] = m3["image_path"].astype(str).str.replace("\\", "/", regex=False).str.lstrip("./")
    if not m4["_key"].is_unique or not m3["_key"].is_unique:
        raise ValueError("Full-hand source contains duplicate image keys")
    source = m3[["_key", "participant_id", "split", "hb_g_per_dl", "hb_g_per_l"]].rename(columns={
        "hb_g_per_dl": "source_hb_g_dl",
        "hb_g_per_l": "source_hb_g_per_l",
    })
    joined = m4.merge(
        source,
        on="_key", how="outer", suffixes=("_m4", "_m3"), indicator=True, validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Full-hand Member 4 cache and Member 3 source rows do not align one-to-one")
    for m4_col, m3_col in (("participant_id_m4", "participant_id_m3"), ("split_m4", "split_m3")):
        if not (joined[m4_col].astype(str) == joined[m3_col].astype(str)).all():
            raise ValueError(f"Full-hand mismatch between {m4_col} and {m3_col}")
    for m4_col, m3_col in (("hb_g_dl", "source_hb_g_dl"), ("hb_g_per_l", "source_hb_g_per_l")):
        delta = (pd.to_numeric(joined[m4_col]) - pd.to_numeric(joined[m3_col])).abs()
        if delta.isna().any() or delta.max() > 1e-8:
            raise ValueError(f"Full-hand measured Hb mismatch: {m4_col} vs {m3_col}")
    if joined.groupby("participant_id_m4")["split_m4"].nunique().max() != 1:
        raise ValueError("Participant leakage across Full-hand splits")

    manifest = pd.DataFrame({
        "dataset_id": "figshare_fullhand",
        "image_id": joined["_key"].map(lambda k: f"figshare_fullhand:{k}"),
        "image_path": joined["abs_image_path"].astype(str),
        "source_relative_path": joined["_key"],
        "participant_id": joined["participant_id_m4"].astype(str),
        "target_hb_g_dl": pd.to_numeric(joined["hb_g_dl"]),
        "split": joined["split_m4"].astype(str),
        "checkpoint_id": "m3_palm_classifier_transfer_exploratory",
        "target_source": "Figshare source split numeric hb_g_per_dl; no derived folder label used",
    })
    if not manifest["image_id"].is_unique or not manifest["image_path"].is_unique:
        raise ValueError("Full-hand manifest image IDs or paths are not unique")
    if not manifest["image_path"].map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError("Full-hand manifest references a missing image")
    def image_hash(value: str) -> str:
        digest = hashlib.sha256()
        with Path(value).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    hashes = manifest["image_path"].map(image_hash)
    if hashes.duplicated().any():
        raise ValueError("Exact duplicate Full-hand image contents found; investigate before evaluation")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT, index=False)
    scratch = manifest.copy()
    scratch["checkpoint_id"] = "member4_fullhand_cnn_scratch"
    scratch.to_csv(SCRATCH_OUTPUT, index=False)
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)}: rows={len(manifest)}, "
        f"splits={manifest['split'].value_counts().to_dict()}, "
        f"participants={manifest['participant_id'].nunique()}, max Hb delta=0"
    )
    print(f"Also wrote scratch-CNN manifest {SCRATCH_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
