# Dataset Audit and Preprocessing Findings — Member 4

**Audit date:** 2026-10-27. Findings below are based on the extracted manifests, source split CSVs, archive listings, and Member 4 run outputs. Original archives were preserved.

## Archive and extracted data inventory

| Archive | ZIP entries | Compressed size (approx.) | Extracted contents / task use |
|---|---:|---:|---|
| `datasets/dataset_edi_eyes.zip` | 1,989 | 671 MB | Raw India/Italy eye records and preprocessed eye data with source label workbook exports, manifests, and quality issue log |
| `datasets/PalmDataset.zip` | 4,264 | 30 MB | Preprocessed 224×224 palm images in class folders, manifest, and quality issue log |
| `datasets/Figshare_Hb_Preprocessed_nail_skin.zip` | 1,119 | 4.9 MB | Full-hand images, nail/skin crops, Hb labels, manifest, preprocessing report, quality log |
| `datasets/member3_model_training.zip` | 59 | 66 MB | Member 3 source, configs/splits, checkpoints, predictions, metrics, and handover reports |

## Eye / conjunctiva

- Preprocessed image manifest: 675 images, all 224×224. It records source path, image variant, Hb, age, gender, and participant match. Hb is available for 631 rows; the supplied Member 3 split selects those 631 matched image rows (446 train, 93 validation, 92 test) and 202 participant keys (`pid`, country-qualified).
- Numeric Hb is present as `hb_g_dl`. The original EYES-DEFY-ANEMIA paper describes Hb measured by blood count and reported in g/dL; the Member 3 config nevertheless calls the unit assumed, and the local files do not include individual lab reports. Joining all 202 modeled participants to the source label table gives ages 19–88 (19 are over 65) and M/F labels. Member 3 class labels recompute exactly from <13 g/dL male / <12 g/dL female. The current data cannot verify pregnancy status or altitude/smoking adjustments, and the WHO adult table cited applies to ages 15–65; thus the Hb regression target has lab-based provenance while the class labels remain threshold-derived and exploratory. [Source study](https://www.sciencedirect.com/science/article/pii/S0933365722002299); [WHO 2024 guideline](https://www.who.int/publications/i/item/9789240088542).
- The image quality log contains failed source-variant preprocessing entries. The split has 631 image rows; do not infer that all 44 manifest rows outside it were corrupt without row-level reason verification.
- Matched eye images are JPEGs; the raw source archive also includes PNG region crops. The preprocessed manifest specifies 224×224; the pipeline confirms each split path exists and extracts features successfully.

## Figshare nail, full-hand, and skin

- The image manifest contains 1,113 records: 719 nail crops, 250 full-hand images, and 144 skin crops; all listed records have participant Hb in both g/L and g/dL, linked by participant ID. 250 unique participants are represented. In Member 3 splits, 719 nail images are partitioned 503/107/109 (train/validation/test); 250 full-hand images 175/37/38.
- Age and gender are missing for the Figshare rows. Anemia label 1/0 is derived in inherited splits using a universal 11 g/dL rule. This is not a suitable universal adult anemia threshold and is only treated as a weak exploratory label; no clinically grounded anemia-classification claim is made.
- Nail and full-hand files are JPEG crops/images. `image_path` is relative under `Figshare_Hb_Preprocessed/`; local paths from another machine are not used. Member 4 verified all split files load and produced features.
- There are only 144 skin image records in the provided manifest, not the 764 referenced in a prior quality summary. The supplied quality log records small crop issues; prior Member 3 report claims 637 invalid/empty boxes among 764 candidate skin boxes. Because those counts refer to candidate boxes rather than 144 manifest image files, skin was omitted from modeling and the mismatch is retained as an audit caveat.

## Palm

- Manifest and split table list 4,260 JPEG images resized to 224×224, grouped into 2,562 `anemic` and 1,698 `non_anemic` filename/folder labels, across 527 inferred groups. Member 3 split is 2,975/618/667 images. No measured Hb is present (0 non-null Hb values in manifest).
- `inferred_group_id` is parsed from source naming/folders and is used to keep groups disjoint across partitions. It is not independently confirmed as a unique human participant identifier.
- The dataset publisher says the original Ghana cohort was children aged five and below; laboratory staff measured Hb and technicians/medical laboratory officers assigned anemic/non-anemic remarks from those measurements. The publisher also says 710 original images were augmented into 4,260 labelled files. The supplied local image/split files contain folder-derived labels but no Hb values or individual lab records, so the source-reported clinical basis cannot be independently verified row by row. Member 4 therefore evaluates folder-label prediction only, not Hb regression or a clinically validated anemia classifier. [Mendeley Data source record](https://data.mendeley.com/datasets/ccr8cm22vz/1).

## Member 3 artifacts and pairing

- The extracted archive includes Eye, Nail and Palm checkpoints, split/config files, test prediction CSVs and JSON results. Prediction tables have only true/predicted targets and probabilities; they do not contain image IDs or embedding vectors.
- No pre-extracted CNN embedding table is provided. Checkpoints alone cannot demonstrate that embeddings have the same rows/order as Member 4 targets. No embedding or fused model was trained.
- Cross-dataset IDs have no common namespace or verified correspondence. Combining eye, palm and nail rows as paired multimodal examples would be invalid.

## Data quality and preprocessing used by Member 4

- Member 4 reuses the provided preprocessed images and participant/group splits; it does not alter source archives or make a new clinical label. Images are loaded one at a time and converted to RGB for feature extraction.
- Current cached feature tables contain 610 Eye, 719 Nail, 250 Full-hand and 2,056 Palm model rows, with no duplicate image paths. Eye and Palm rows reflect the final exact/confirmed near-duplicate corrections. Model feature columns have no missing values in the current cache. Nail/full-hand age/gender metadata are all missing and excluded from model features; palm Hb metadata is missing and excluded. Eye features are complete.
- Member 4's run reported no participant/group crossing split boundaries for the supplied split manifests. This does not establish the absence of near-duplicate imagery across files. The palm split check applies only to inferred groups.
- Counts are images/crops except the full-hand table (one image per participant); nail metrics are image/crop-level although held-out participants are disjoint.

## Suitability summary

| Dataset | Numerical Hb regression | Classification | Participant pairing / fusion |
|---|---|---|---|
| Eye | Yes, with unit/protocol caveat | Inherited sex-specific label; limited missing clinical context | Eye participant key and grouped split available |
| Figshare nail | Yes; repeated crops per participant | Weak 11 g/dL-derived labels with demographics missing | Participant IDs and grouped split available within Figshare |
| Figshare full-hand | Yes; one image per participant | Same weak label caveat | Participant IDs and grouped split available within Figshare |
| Figshare skin | Hb metadata listed, but crop-quality counts inconsistent and only 144 image records | Not modeled | Not modeled |
| Palm | No measured Hb | Folder-derived class only; exploratory | Inferred groups; no cross-dataset pairing |

See [implementation audit](implementation_audit_report.md) for verification commands and [results report](results_report.md) for metrics actually produced.

## Final duplicate audit update (2026-10-27)

A streaming SHA-256 check found exact-content duplicates; a subsequent Palm perceptual-hash/SSIM audit assessed near-duplicates. Original archives, source manifests, and image files were preserved. Corrected modeling manifests/caches omit specified rows only.

- **Eye:** The original selected split contained 631 rows (446/93/92) across 202 country-qualified participant IDs. There are 21 exact-content pairs, all same participant and same partition, with equal Hb/class labels. The separate variant paths are duplicate image inputs, not separate participant records. One deterministic copy per exact hash is retained, removing 21 redundant rows (including three test rows). Current Eye data: 610 images, 202 participant IDs; 432/89/89 train/validation/test. Exclusions are logged in `outputs/metrics/eye_exact_duplicate_audit.csv`.
- **Palm exact duplicates:** 1,391 duplicated-content groups among the original 4,260 records, including 2,196 excess copies. Twenty groups crossed split boundaries and four exact-content groups had conflicting folder labels. The loader prefers train over validation over test for consistent cross-split hashes and removes every copy of conflicting-label hashes.
- **Palm near duplicates:** DCT pHash (grayscale 32×32, 64-bit; candidate distance <=4) found 36 candidate pairs among 2,060 exact-unique images, of which 17 crossed splits. Four test/validation pairs across inferred groups AnemicP-001 and AnemicP-010 had SSIM 0.987–0.988 at grayscale 128×128. Those four test image hashes and all 12 exact-byte raw manifest rows bearing those hashes were excluded. The other 32 pairs were reviewed from their hash/SSIM/split/label fields and in `outputs/metrics/palm_near_duplicate_residual_review.png`: 13 cross splits (SSIM .594–.908); the highest same-split SSIM is .949. None reaches the predeclared .98 exclusion threshold; three label conflicts are within train (SSIM .65–.72). The remaining visually similar pairs show related palm-crease patterns but no clear duplicate captures, so no further exclusion is justified. Current Palm data: 2,056 unique image contents / 514 inferred groups; 1,452/312/292 images and 363/78/73 inferred groups across train/validation/test.
- **Nail/full-hand:** no exact duplicate content was found in the selected Member 4 manifests.

Audit evidence: `outputs/metrics/eye_exact_duplicate_audit.csv`, `outputs/metrics/palm_exact_duplicate_exclusions.csv`, `outputs/metrics/palm_near_duplicate_candidates.csv`, `outputs/metrics/palm_near_duplicate_test_exclusions.csv`, and `outputs/metrics/palm_near_duplicate_summary.json`. The Palm near-duplicate scan is reproducible with `python member4_ml_modelling\src\audit_palm_near_duplicates.py`.

The former 4,260-row Palm evaluation is not a clean held-out result and Member 3 Palm metrics are not used as comparison results. Member 4's current Palm test metrics use 292 images/73 inferred groups after exact and confirmed near-duplicate exclusions.

## Clinical label provenance and claim limits

- Eye class labels were inherited from the supplied split and recompute from <13 g/dL male / <12 g/dL female. The source study reports laboratory Hb in g/dL, supporting the numeric target despite the Member 3 config wording “assumed.” In the 202-participant modeled cohort, ages are 19–88 and 19 participants are over 65. The inherited <13/<12 class mapping matches WHO 2024 cutoffs for men and non-pregnant women aged 15–65 only; pregnancy and adjustment context are unavailable. Describe this as lab-Hb-derived threshold classification with population caveats, not validated clinical performance. [Source study](https://www.sciencedirect.com/science/article/pii/S0933365722002299); [WHO 2024 guideline](https://www.who.int/publications/i/item/9789240088542)
- The official Figshare record describes 250 patients with known Hb; the supplied numeric values and g/L-to-g/dL conversion are consistent. The inherited <11 g/dL classes are generated in the Member 3 split, not provided as a clinical diagnosis label; age/sex are absent from the modeled data. They are exploratory low-Hb labels only. [Figshare dataset record](https://springernature.figshare.com/articles/dataset/Dataset_of_human_skin_and_fingernails_images_for_non-invasive_haemoglobin_level_assessment/25867432).
- Palm has no Hb values in the supplied local archive. Its folder classes trace to a source dataset whose publisher says original remarks were assigned from laboratory Hb, but the local augmented images do not carry the underlying Hb, age/sex, or patient-level lab record needed to validate those labels. Member 4 therefore reports folder-label prediction only, not an independently validated anemia target. [Mendeley Data source record](https://data.mendeley.com/datasets/ccr8cm22vz/1).

