# Member 4 Results and Experimental Method

**Verified run:** 2026-10-27  
**Seed:** 42  
**Current model data:** exact/confirmed near-duplicate corrected; original source images remain unchanged.

## Evaluation setup

Member 4 uses supplied train/validation/test manifests with participant/group IDs kept within one partition. Eye IDs are country-qualified; Figshare uses participant IDs; Palm IDs are inferred from names/folders and are not verified human identities. Handcrafted features comprise RGB/color statistics and histograms, HSV/LAB/gray intensity statistics, pallor/redness ratios, and gray-level co-occurrence texture features.

Random Forest and Extra Trees settings are fixed. XGBoost hyperparameters are selected using validation MAE and scored on the held-out test split. Results are image/crop-level, not participant-aggregated; multiple samples from one held-out participant are not independent observations.

| Dataset | Train images (groups) | Validation images (groups) | Test images (groups) |
|---|---:|---:|---:|
| Eye after exact-content deduplication | 432 (141) | 89 (30) | 89 (31) |
| Nail crops | 503 (175) | 107 (37) | 109 (38) |
| Full-hand | 175 (175) | 37 (37) | 38 (38) |
| Palm after exact and confirmed near-duplicate exclusions | 1,452 (363 inferred) | 312 (78) | 292 (73 inferred) |

## Hemoglobin regression

| Dataset / input | Model | Test n | MAE (g/dL) | RMSE (g/dL) | R² |
|---|---|---:|---:|---:|---:|
| Eye, handcrafted | Extra Trees | 89 | 1.4370 | 1.9832 | 0.4867 |
| Eye, handcrafted | XGBoost | 89 | 1.4177 | 1.9744 | 0.4912 |
| Nail crops, handcrafted | XGBoost | 109 | 1.7171 | 2.3079 | 0.0548 |
| Full-hand, handcrafted | Extra Trees | 38 | 1.5215 | 1.8952 | 0.4068 |
| Full-hand, handcrafted | XGBoost | 38 | 1.5819 | 2.0355 | 0.3157 |

The Nail score is only slightly better than a mean predictor. No Palm regression is possible because Palm has no measured Hb target. No CNN-embedding XGBoost result exists yet.

## Classification and target provenance

| Dataset / target | Model | Test n | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Eye inherited sex-specific threshold | Extra Trees | 89 | 0.8876 | 0.9487 | 0.8222 | 0.9545 | 0.8810 | 0.9702 |
| Nail inherited <11 g/dL threshold | Extra Trees | 109 | 0.7798 | 0.2222 | 0.1053 | 0.9222 | 0.1429 | 0.6099 |
| Palm filename/folder label | Extra Trees | 292 | 0.7500 | 0.7844 | 0.7798 | 0.7097 | 0.7821 | 0.7788 |

Eye labels recompute exactly from the supplied <13 g/dL male / <12 g/dL female rule. The source study reports laboratory-measured Hb in g/dL; Member 3 config still says “assumed.” Joining the source label table shows cohort ages 19–88, including 19 participants over 65. WHO 2024 uses these cutoffs for men and non-pregnant women aged 15–65; pregnancy and altitude/smoking context are unavailable. Treat this as lab-Hb-derived threshold classification with population caveats, not validated clinical performance. [Source study](https://www.sciencedirect.com/science/article/pii/S0933365722002299); [WHO guideline](https://www.who.int/publications/i/item/9789240088542)

Figshare’s official dataset record describes 250 patients with known Hb; the numeric target and supplied g/L-to-g/dL conversion are consistent. Its <11 g/dL class is an inherited split-derived threshold without age/sex context, so report it only as low-Hb exploration. The Palm publisher says original labels were entered from laboratory Hb remarks in a cohort of children aged five and below, but the supplied folder-based augmented data lacks Hb/patient records to verify those labels row by row. Member 4 reports Palm folder-label prediction only. [Figshare record](https://springernature.figshare.com/articles/dataset/Dataset_of_human_skin_and_fingernails_images_for_non-invasive_haemoglobin_level_assessment/25867432); [Palm source record](https://data.mendeley.com/datasets/ccr8cm22vz/1). None of these experiments establishes diagnosis, clinical utility or replacement of laboratory CBC testing.

## CNN comparison boundary

Member 3 supplied saved CNN test predictions for Eye and Nail. Aggregate metrics recalculate from those files, but the CSVs contain no image or participant IDs. Eye's Member 3 test file has 92 rows whereas Member 4 now uses 89 unique Eye test images after duplicate correction; those numbers are not a paired comparison. Nail has the same row count but still cannot be row-matched without IDs. Member 3 Palm results are not used because the original test partition contained exact duplicate content across splits and conflicting folder labels.

Image-keyed 1,024-D embeddings were extracted from Member 3's supplied scratch-trained Eye and Nail regression checkpoints using the reconstructed `LightweightCNN`/`RegressionCNN` architecture and safe checkpoint loading. CNN-embedding XGBoost ran on the same held-out images as handcrafted XGBoost and scored lower on both datasets. No Member 3 Full-hand-trained regression checkpoint exists. Member 4 additionally trained a compact CNN from scratch on 175 Full-hand training participants, selected the checkpoint by validation MAE, extracted keyed 64-D embeddings, and fit validation-selected XGBoost. On the same 38 held-out images it scored MAE/RMSE/R² 1.5599/1.8946/0.4072 versus handcrafted 1.5819/2.0355/0.3157. This is a small-sample exploratory result, not clinical evidence. Separately, the frozen M3 Palm folder-classification backbone transfer scored 2.0970/2.6163/-0.1305 and did not improve; it is not Full-hand-trained. Detailed methods and artifacts are in `cnn_embedding_experiment_report.md` and the latest `member4_final_completion_audit.md`.

## Duplicate handling

### Eye

The three excess test rows were exact byte-identical image pairs stored under separate anatomical-variant paths. Each pair shares participant, partition, Hb and class. One deterministic image per SHA-256 is retained, with no source-file deletion; current Eye split is 432/89/89 and test n=89. Exclusions are in `outputs/metrics/eye_exact_duplicate_audit.csv`.

### Palm

The scan uses a 64-bit DCT pHash after grayscale 32×32 Lanczos resize, candidate Hamming distance <=4, then grayscale SSIM after resize to 128×128. It found 36 candidates, 17 crossing partitions. Four cross-split test/validation pairs between inferred groups AnemicP-001 and AnemicP-010 had SSIM .987–.988. Member 4 excludes those four test-side image hashes and all 12 exact-byte source-manifest rows carrying the hashes. Looser candidates are retained and documented; similarity alone is not used to claim common participants. The corrected set has 2,056 unique image contents / 514 inferred groups and test n=292.

Reproduction command: `python member4_ml_modelling\src\audit_palm_near_duplicates.py`. Candidate and exclusion ledgers are `outputs/metrics/palm_near_duplicate_candidates.csv` and `outputs/metrics/palm_near_duplicate_test_exclusions.csv`; exact-content exclusions are `outputs/metrics/palm_exact_duplicate_exclusions.csv`.

## Artifacts and reproducibility

- Current metric summary: `outputs/metrics/all_metrics_summary.csv`
- Current result page: `reports/results_dashboard.html` (regenerate from metrics with `python member4_ml_modelling\src\generate_results_dashboard.py`)
- Per-model JSON and saved predictions: `outputs/metrics/`
- Fitted Member 4 models: `outputs/models/*.pkl`
- Evaluation figures: `outputs/plots/`
- Dataset/method audit: `dataset_audit_report.md`

The end-to-end pipeline, CNN embedding experiments and comparison generators completed after the corrections. The saved validation ledger passes all metric, sample-pairing and model-reload checks (including Full-hand scratch-CNN and the separate exploratory transfer). A clean Python 3.10 venv was created, but installing `requirements.txt` failed at PyPI DNS resolution (`getaddrinfo failed`), so fresh-install reproducibility remains unverified. The notebook's nine plain-Python code cells were executed with the fallback runner and outputs saved; standard Jupyter/nbconvert execution remains unavailable. All results are research prototypes and do not demonstrate clinical validity or external generalization.
