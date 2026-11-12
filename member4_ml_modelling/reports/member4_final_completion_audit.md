# Member 4 Final Completion Audit

**Audit date:** 2026-10-30  
**Project:** Hybrid Multimodal AI Framework for Non-Invasive Anemia Detection and Hemoglobin Estimation  
**Workspace:** `C:\Users\nethr\Downloads\Anemia_ML_Project`

## Verdict

# INCOMPLETE — Locally executable Member 4 work and the guarded launcher are complete; clean-install and real Jupyter-kernel checks remain blocked by unavailable package access.

Existing handcrafted and Eye/Nail baselines remain in their original output paths. In addition to Eye/Nail Member 3 embeddings and the separately labeled Palm-checkpoint transfer, a compact Member 4 CNN was trained from scratch on Full-hand training participants, selected by validation MAE, and used to extract features for XGBoost. On the same 38 test images, scratch-CNN embeddings slightly improved over handcrafted XGBoost (MAE 1.5599 vs 1.5819); the small sample makes this exploratory. The archive still has no Member 3 Full-hand checkpoint. Clean installation remains unverified because PyPI DNS fails.

## Requirement completion matrix

| Member 4 requirement | Status | Evidence and remaining work |
|---|---|---|
| Inspect data, schemas, targets and usable splits | COMPLETE LOCALLY | Extracted archives, source manifests, split rows, targets, image paths and counts are in `dataset_audit_report.md`. Original archives and source images were preserved. License/source documentation is not available in the workspace; Member 1/source documentation must confirm it. |
| Handcrafted feature baseline | COMPLETE AND VERIFIED | Features and conventional regressors/classifiers ran on Eye, Nail, Full-hand and Palm. The current feature caches match the final deduplicated manifests. |
| XGBoost Hb regression on available numeric Hb | COMPLETE AND VERIFIED | Eye, Nail and Full-hand use train/validation/test partitions; parameter choice uses validation MAE, held-out test scoring is separate. Predictions, metrics, models and plots were regenerated. |
| XGBoost using CNN embeddings | COMPLETE FOR AVAILABLE DATA; EXPERIMENTAL | Eye/Nail use supplied Member 3 regression-CNN checkpoints. Full-hand uses a separate compact Member 4 CNN trained from scratch on the 175 training participants; its validation-selected 64-D embeddings feed XGBoost. All compare against handcrafted XGBoost on identical held-out images. The supplied archive still has no Member 3 Full-hand checkpoint; the separate Palm-checkpoint transfer remains exploratory and is not treated as equivalent. |
| Regression evaluation | COMPLETE FOR AVAILABLE MODELS | MAE, RMSE and R² are saved. Nine model prediction variants (RF, Extra Trees, XGBoost across applicable datasets) recalculate to the JSON metrics. |
| Experimental anemia/low-Hb classification | PARTIAL — METRICS VERIFIED, CLINICAL CLAIMS RESTRICTED | Metrics recalculate from saved predictions. Eye numeric Hb has source-study laboratory provenance in g/dL; class labels are inherited thresholds, with age >65 and missing pregnancy/adjustment context. Figshare's <11 g/dL labels are exploratory only. Palm labels are folder classes with source-reported Hb basis but no local row-level clinical verification. |
| Eye duplicate investigation | COMPLETE AND CORRECTED | 21 exact byte-identical pairs are separate variant rows but share participant, split, Hb and class. They are duplicate image inputs, not separate participant records. One deterministic copy per hash is used; all source images remain. The selected Eye data now has 610 images/202 participants, split 432/89/89. |
| Palm near-duplicate analysis | COMPLETE; LOW-CONFIDENCE CANDIDATES RETAINED | DCT pHash scan and SSIM confirmation found 36 candidate pairs, 17 crossing splits. Four test/validation pairs across inferred groups AnemicP-001 and AnemicP-010 were high similarity (SSIM 0.987–0.988). Four test-side hashes and all 12 raw manifest rows containing their exact byte copies are excluded. Lower-similarity candidates are recorded, not assumed duplicates. Final Palm data has 2,056 images/514 inferred groups, split 1,452/312/292 (train/validation/test). |
| Group separation, comparison and no fusion claim | PARTIAL | Current supplied groups are split-disjoint. Palm groups are inferred IDs, and near-duplicate matches show they can fail to capture related imagery. Member 3 Eye/Nail outputs have no image/person IDs, so exact paired comparisons cannot be proven. No shared participant keys exist across modalities; fusion is unsupported. |
| Saved model, prediction, metric and plot artifacts | COMPLETE AND VERIFIED FOR AVAILABLE EXPERIMENTS | Nine original Member 4 models, two Eye/Nail CNN XGBoost models, the Palm-transfer XGBoost model, and the Full-hand scratch-CNN XGBoost model reload; the latter reproduces saved test predictions. The scratch-CNN state-dict also reloads safely before embedding extraction. Validation recalculates all four CNN experiment comparisons from saved paired predictions. Baseline paths remain separate and unchanged by these CNN scripts. |
| Reproducible clean install | BLOCKED BY PACKAGE ACCESS | Existing Python 3.10 has all nine exact pinned Member 4 packages. `cnn_env` lacks the non-CNN dependencies; pip configuration has no alternate index and pip cache has no usable wheels (the two wheel-named local files are zero-byte placeholders). The previously recorded single PyPI DNS check failed, and no repeated retry or environment modification was made; clean installation is not verified. |
| Notebook execution | FALLBACK VERIFIED; REAL JUPYTER BLOCKED | Inspected all 18 cells; the 9 code cells contain plain Python. The launcher re-executed them sequentially and saved outputs to the `.ipynb`. Jupyter, nbconvert, nbclient, nbformat, ipykernel and IPython are absent in Python 3.10, Python 3.13 and `cnn_env`; fallback execution is not a Jupyter-kernel run. Reproduce the fallback with `py -3.10 member4_ml_modelling\\src\\execute_notebook.py`. |
| Root launcher and final artifact validation | COMPLETE AND VERIFIED | Updated launcher exited 0, reused valid baseline/CNN artifacts without training, executed the notebook fallback, and regenerated the dashboard. Final validation reports zero mismatches and 13/13 model reload/inference checks passing. |
| Fusion, clinical deployment, upload/predict application | NOT A MEMBER 4 DELIVERABLE / UNSUPPORTED | No paired cohort exists for fusion; clinical deployment is explicitly out of scope. Streamlit/demo integration is assigned to Member 5. |

## Corrected experiments and current results

The project pipeline completed after the duplicate corrections; it refreshed metrics, saved predictions and models, and regenerated plots. The subsequent comparison-plot generator completed. Metrics below use held-out image/crop rows and do not represent independent patient-level aggregation.

### Hb regression

| Dataset/input | Model | Train / validation / test | MAE (g/dL) | RMSE (g/dL) | R² |
|---|---|---:|---:|---:|---:|
| Eye handcrafted | Extra Trees | 432 / 89 / 89 | 1.4370 | 1.9832 | 0.4867 |
| Eye handcrafted | XGBoost | 432 / 89 / 89 | 1.4177 | 1.9744 | 0.4912 |
| Eye Member 3 CNN embeddings | XGBoost | 432 / 89 / 89 | 1.7452 | 2.2907 | 0.3152 |
| Nail handcrafted | XGBoost | 503 / 107 / 109 | 1.7171 | 2.3079 | 0.0548 |
| Nail Member 3 CNN embeddings | XGBoost | 503 / 107 / 109 | 2.0425 | 2.6010 | -0.2006 |
| Full-hand handcrafted | Extra Trees | 175 / 37 / 38 | 1.5215 | 1.8952 | 0.4068 |
| Full-hand scratch CNN embeddings | XGBoost | 175 / 37 / 38 | 1.5599 | 1.8946 | 0.4072 |
| Full-hand exploratory Palm-checkpoint transfer | XGBoost | 175 / 37 / 38 | 2.0970 | 2.6163 | -0.1305 |
| Full-hand handcrafted | XGBoost | 175 / 37 / 38 | 1.5819 | 2.0355 | 0.3157 |

### Classification (exploratory targets)

| Dataset/target | Model | Test n | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Eye inherited sex-specific threshold | Extra Trees | 89 | 0.8876 | 0.9487 | 0.8222 | 0.9545 | 0.8810 | 0.9702 |
| Figshare Nail inherited <11 g/dL threshold | Extra Trees | 109 | 0.7798 | 0.2222 | 0.1053 | 0.9222 | 0.1429 | 0.6099 |
| Palm folder-derived label | Extra Trees | 292 | 0.7500 | 0.7844 | 0.7798 | 0.7097 | 0.7821 | 0.7788 |

The Nail classifier's sensitivity is poor. Palm predicts folder labels only. Eye is threshold-derived and exploratory, not clinically validated. Full-hand scratch-CNN XGBoost is a small-sample image regression experiment; its 38-person test result does not establish clinical utility. These results must not be presented as a diagnosis or replacement for CBC.

## Duplicate investigation evidence

### Eye

The three excess test rows resolve to three exact-content image pairs under separate anatomical-variant paths. Each pair has the same participant ID, split, Hb and label; no pair crosses participants or train/validation/test boundaries. They are duplicate image inputs created/retained under variant names. The loader now retains only one deterministic copy per exact SHA-256 image hash; the source images and original manifest are untouched. The 21 redundant copies across all Eye partitions are logged at `outputs/metrics/eye_exact_duplicate_audit.csv`. Current participant groups remain 202; row counts are 432 train, 89 validation and 89 test.

### Palm

The reproducible scan is `python member4_ml_modelling\src\audit_palm_near_duplicates.py`. It operates on the exact-content-deduplicated Palm manifest using grayscale 32×32 Lanczos resizing, a 64-bit DCT perceptual hash and Hamming threshold <=4. Candidate pairs are compared with SSIM on grayscale 128×128 images. The current audit found 36 pairs: 17 cross-split; distances 0 (7), 2 (5), and 4 (24). Four cross-split test/validation pairs have SSIM 0.987–0.988 and map to inferred groups AnemicP-001/AnemicP-010. The test-side four image hashes (12 raw manifest rows counting byte-identical copies) are excluded from modeling/evaluation; source data stays unchanged. Remaining low-similarity candidates, including three same-train label-conflict candidates with SSIM about 0.65–0.72, were not removed because these scores do not establish duplicate imagery. Pair details and exclusion record are in `outputs/metrics/palm_near_duplicate_candidates.csv` and `outputs/metrics/palm_near_duplicate_test_exclusions.csv`; summary is `outputs/metrics/palm_near_duplicate_summary.json`.

Exact Palm deduplication also removes identical hashes crossing partitions and all copies of four exact-hash groups with conflicting folder labels. After combining exact and confirmed near-duplicate exclusions, the current data contains 2,056 image contents / 514 inferred groups: train 1,452/363, validation 312/78, test 292/73. Exact exclusions are logged in `outputs/metrics/palm_exact_duplicate_exclusions.csv`.

## Clinical label provenance review

- **Eye:** The supplied split derives class labels using <13 g/dL for male and <12 g/dL for female. This agrees with WHO 2024 cutoffs only for adults aged 15–65, men and non-pregnant women, respectively. The source study describes blood-count Hb in g/dL, although Member 3 config says assumed. Joining the source table shows ages 19–88, with 19 participants over 65. The <13/<12 class rule matches WHO 2024 cutoffs for men and non-pregnant women aged 15–65 only; pregnancy and altitude/smoking adjustment context is missing. This supports a lab-Hb-derived exploratory target, not validation across the full cohort. [Source study](https://www.sciencedirect.com/science/article/pii/S0933365722002299); [WHO 2024 guideline](https://www.who.int/publications/i/item/9789240088542)
- **Figshare:** the official dataset record describes 250 patients with known Hb; numeric targets and supplied unit conversion are consistent. The <11 g/dL class is split-derived, lacks age/sex context and is low-Hb exploration only. [Source record](https://springernature.figshare.com/articles/dataset/Dataset_of_human_skin_and_fingernails_images_for_non-invasive_haemoglobin_level_assessment/25867432).
- **Palm:** no Hb or patient lab records exist in the supplied local copy. The source publisher says the original cohort was children aged five and below and that lab staff/medical officers assigned remarks from Hb measurements. Local folder labels therefore have source-reported clinical provenance but cannot be rechecked per image; call this folder-label prediction, not verified clinical classification. [Source record](https://data.mendeley.com/datasets/ccr8cm22vz/1).

## Member 3 checkpoint and CNN-embedding experiment

Inspected the 59-entry `datasets/member3_model_training.zip`, its training source/configs/dependencies, and all five checkpoint containers. Member 3's README states pretrained weights were unavailable; `LightweightCNN` is scratch-trained, with 11 MobileNet-inspired depthwise-separable blocks and 1,024 pooled features. Both Eye and Nail regression scripts instantiate `RegressionCNN(dropout=0.4)`. Inference preprocessing is RGB, 224×224 bilinear resize, [0,1] scaling and ImageNet mean/std normalization, without augmentation. The supplied checkpoints have matching `backbone.features.*` and `backbone.classifier.*` state-dictionary keys; Palm has only a classifier checkpoint. No Member 3 Full-hand regression checkpoint is supplied; Member 4 separately trained a compact Full-hand CNN from scratch, reported below.

The isolated `cnn_env` runs Python 3.10.0, PyTorch 2.14.1+cpu and torchvision 0.29.1+cpu; CPU inference works (CUDA unavailable). The Eye checkpoint safely loaded with `weights_only=True` after PyTorch static inspection reported exactly `numpy.dtype` and `numpy.core.multiarray.scalar`; the safe-global allowlist also included the specific `numpy.dtypes.Float64DType` needed to rebuild its stored training Hb scalar. Nail loaded with `weights_only=True` and no extra globals. No `weights_only=False` call was used. The reviewed Member 3 model classes were compiled from `utils.py`; strict state-dictionary loading succeeded for both regression checkpoints.

Current Member 4 samples were joined one-to-one to Member 3 split records by image path. Eye: 610 rows, split 432/89/89, 202 participants, zero Hb differences. Nail: 719 rows, split 503/107/109, 250 participants, zero Hb differences. All paths exist; participant IDs and splits agree; targets are present; participant groups stay within one split. The 1,024-D embedding tables retain dataset/image IDs, paths, participant IDs, Hb targets, split, checkpoint ID and hash; all values are finite. Test paths and target values exactly match the existing handcrafted-XGBoost predictions (Eye 89/89, Nail 109/109).

XGBoost used the same four parameter candidates as the handcrafted pipeline, selected by validation MAE, with the scaler fit on training rows only. It trained on training embeddings and evaluated on those same held-out images. Eye CNN embedding result: MAE 1.7452/RMSE 2.2907/R² 0.3152 vs handcrafted XGBoost 1.4177/1.9744/0.4912. Nail CNN embedding result: 2.0425/2.6010/-0.2006 vs handcrafted 1.7171/2.3079/0.0548. The CNN-embedding models did not improve. Full-hand now has a distinct scratch-trained CNN experiment; there is still no supplied Member 3 Full-hand regression checkpoint. The Nail checkpoint was not repurposed out-of-domain. No multimodal fusion or clinical classification claim was added. Detailed method and artifact paths are in `cnn_embedding_experiment_report.md`.

## Notebook and reproduction commands

The notebook is a walkthrough over existing artifacts. It has now been executed top-to-bottom through the reviewed plain-Python fallback, which saved outputs into the `.ipynb`. This does not create a Jupyter kernel log; Jupyter/nbformat/nbclient are unavailable. The existing Python 3.10 runtime and project dependencies suffice because the notebook cells are plain Python. Reproduce outputs from the project root with:

```powershell
python member4_ml_modelling\src\execute_notebook.py
```

For standard Jupyter-kernel execution in an environment with package-index access, the optional commands remain:

```powershell
python -m pip install notebook nbconvert ipykernel
python -m nbconvert --to notebook --execute --inplace member4_ml_modelling\notebooks\member4_analysis.ipynb
```

For the verified pipeline, use `Run_Project.bat`, or from project-root PowerShell:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python member4_ml_modelling\src\run_pipeline.py
python member4_ml_modelling\src\generate_comparison_plots.py
```

To repeat Palm near-duplicate analysis, run the audit script above before the pipeline; it writes its candidate and test-exclusion ledgers. A clean install could not be tested in this workspace because the package index is unreachable.

## Evidence checked in this audit

- Eye duplicate pair membership/participant/split/labels from the duplicate ledger and source manifest; no cross-participant or cross-split exact duplicates.
- Palm near-duplicate summary and pairwise SSIM values; test-side exclusion hashes; resulting 2,056-row cache and 292-row Palm test predictions.
- Final pipeline exit code 0; updated Eye/Palm predictions, model metrics and plots; comparison-plot generator exit code 0.
- Recalculated MAE/RMSE/R² for all RF/Extra Trees/XGBoost regression prediction files; recalculated RF/Extra Trees classification metrics for Eye/Nail/Palm; values match saved JSON metrics after rounding.
- All nine current Member 4 pickle model artifacts reloaded and returned finite predictions on five rows from the corresponding current feature cache.
- Fresh CNN inference environment has Python 3.10.0, torch 2.14.1+cpu and torchvision 0.29.1+cpu; it was sufficient for safe checkpoint inference. Full-requirements install remains unverified because PyPI DNS resolution failed. The original global interpreter was left unchanged.
- Notebook JSON and all cell sources were inspected; all nine plain-Python code cells executed sequentially and outputs were saved. The fallback runner is `src/execute_notebook.py`; a standard Jupyter kernel was not available.

## Final responsibility statement

**The Full-hand-trained CNN embedding experiment is now implemented locally as a small-sample research experiment.** The Full-hand CNN weights were initialized and trained from scratch on the supplied 175 training participants, with the 37 validation participants selecting the checkpoint and the 38 test participants held untouched until final comparison. A matching Member 3 Full-hand checkpoint remains absent; the separate Palm-classifier transfer is not equivalent. Remaining environment checks are incomplete: a clean installation is blocked by PyPI DNS failure, and a real Jupyter-kernel run is unavailable because no installed Python runtime has Jupyter/nbformat/nbclient/ipykernel. Fusion remains unsupported without verified paired modalities.

## Final execution addendum (2026-10-29)

- Created one-to-one input manifests joining the corrected Member 4 modeling rows to Member 3 image paths, participant IDs, splits and numeric Hb targets. Eye: 610 rows (432/89/89), 202 participants, zero target disagreements. Nail: 719 rows (503/107/109), 250 participants, zero target disagreements.
- Used the isolated `cnn_env` (Python 3.10.0, torch 2.14.1+cpu, torchvision 0.29.1+cpu) to safely load only the supplied Eye and Nail regression checkpoints with `weights_only=True`, run the reviewed Member 3 architectures and preprocessing, and extract finite 1,024-D keyed embeddings. No unsafe checkpoint deserialization was used.
- Validated Full-hand’s 250 unique images against the source split file: numeric source Hb values, participant IDs and splits agree exactly; the partition is 175/37/38 across 250 participants with no exact duplicate image contents or group leakage.
- Safely loaded the supplied M3 Palm two-class classifier checkpoint (static inspection found no unsafe globals; `weights_only=True`; checkpoint metadata records epoch 2, validation accuracy 0.5987, filename-derived class labels and no measured Hb). Extracted finite 1,024-D features from Full-hand images using the M3 non-augmented RGB/224/ImageNet-normalized preprocessing. The Palm checkpoint was not trained on Full-hand images or Hb regression.
- Trained validation-selected XGBoost on the transferred Full-hand features; compared on the same 38 test images. Transfer MAE/RMSE/R² 2.0970/2.6163/-0.1305 vs handcrafted 1.5819/2.0355/0.3157. Transfer was worse and remains strictly exploratory. Saved keyed predictions, model, metrics and plot separately.
- Trained a separate compact Full-hand CNN from random initialization (428,905 parameters; no pretrained or M3 checkpoint weights). It used only the 175 training participants for gradient updates, train-only augmentation, and stopped at epoch 39 with epoch 24 selected by validation MAE 1.3779 g/dL. The best state-dict was safely reloaded with `weights_only=True` before producing keyed 64-D embeddings for the 250 rows.
- XGBoost was selected among four configurations by validation MAE, with StandardScaler fit on train rows only. On the same 38 held-out images as handcrafted XGBoost, scratch-CNN embeddings scored MAE/RMSE/R² 1.5599/1.8946/0.4072 vs 1.5819/2.0355/0.3157. All metrics recalculate from saved predictions, and the reloaded XGBoost reproduces the 38 test predictions; the validation ledger passed with zero mismatches.
- Trained separate XGBoost regressors using validation MAE selection and train-only scaling. Compared them against handcrafted XGBoost on identical test images. Eye: CNN MAE/RMSE/R² 1.7452/2.2907/0.3152 vs handcrafted 1.4177/1.9744/0.4912 (n=89). Nail: 2.0425/2.6010/-0.2006 vs 1.7171/2.3079/0.0548 (n=109). CNN embeddings did not improve these results. All three CNN experiment metric sets independently recalculate from saved outputs; saved models reload and reproduce their paired test predictions.
- New artifacts are under `member4_ml_modelling/outputs/embeddings/`, `outputs/metrics/cnn_embedding_experiment/`, `outputs/models/cnn_embedding/`, and `outputs/plots/cnn_embedding/`. Detailed methodology is in `cnn_embedding_experiment_report.md`. The dashboard at `reports/results_dashboard.html` was regenerated to show Eye/Nail comparisons, the Full-hand scratch-CNN result, and the separately caveated Palm-classifier transfer.
- The M3 archive still contains no Full-hand checkpoint; the separate Member 4 scratch-trained Full-hand CNN is actual trained work and must not be conflated with the exploratory Palm-checkpoint transfer.
- The isolated `cnn_env` contains torch/torchvision but not pandas, scikit-learn, XGBoost, pyarrow or plotting dependencies. The global Python 3.10 environment has the nine pinned Member 4 dependencies. A single DNS check failed for `pypi.org` (`getaddrinfo failed`), so no clean environment installation was attempted. Jupyter, nbformat, nbclient and ipykernel were absent in the project Python 3.10 and available Python 3.13 runtimes; Python 3.12 is not installed. The notebook fallback remains available. Existing handcrafted models and predictions were not overwritten by the new CNN scripts.

Reproduce all separate embedding experiments from the project root (the commands below do not rerun the handcrafted baseline):

```powershell
python member4_ml_modelling\src\prepare_m3_embedding_manifests.py
cnn_env\Scripts\python.exe member4_ml_modelling\src\extract_m3_cnn_embeddings.py
python member4_ml_modelling\src\train_m3_embedding_xgboost.py
python member4_ml_modelling\src\prepare_m3_fullhand_transfer_manifest.py
cnn_env\Scripts\python.exe member4_ml_modelling\src\extract_m3_palm_transfer_embeddings.py
python member4_ml_modelling\src\train_fullhand_palm_transfer_xgboost.py
cnn_env\Scripts\python.exe member4_ml_modelling\src\train_fullhand_scratch_cnn.py
python member4_ml_modelling\src\train_fullhand_scratch_cnn_embedding_xgboost.py
python member4_ml_modelling\src\validate_member4_results.py
python member4_ml_modelling\src\generate_results_dashboard.py
```

## Final execution addendum (2026-10-30)

- Re-read this audit, the completion checklist and the CNN experiment report before proceeding. The existing Full-hand scratch-CNN experiment was not rebuilt.
- Rechecked all nine pinned versions in the installed Python 3.10 runtime: numpy 2.2.4, pandas 2.2.3, Pillow 11.2.1, matplotlib 3.10.1, scipy 1.15.3, scikit-learn 1.7.2, xgboost 3.2.0, scikit-image 0.25.2 and pyarrow 19.0.1. They matched `requirements.txt` exactly.
- Inspected both available interpreters (Python 3.10 and 3.13), the CNN virtual environment, pip configuration and cache. No alternative package index is configured; pip cache has no usable package wheels. The only local `.whl` files found in `cnn_env` are zero-byte placeholders for NumPy and SciPy. A previously recorded single PyPI DNS check failed with `getaddrinfo failed`; no second network retry or package installation was attempted. A separate clean environment cannot be installed from the currently available local sources.
- Jupyter, nbconvert, nbclient, nbformat, ipykernel and IPython are absent from Python 3.10, Python 3.13 and `cnn_env`. The actual `.ipynb` was rerun through the established plain-Python fallback by the launcher, which completed all nine code cells and saved their outputs. This remains distinct from real Jupyter-kernel execution.
- Replaced the root launcher with a guarded, idempotent sequence. It checks baseline artifacts and refuses to retrain/overwrite them if incomplete; it validates existing CNN artifacts before reuse and reruns CNN experiment stages only if outputs are missing or validation fails; it then executes the notebook with nbconvert if available (plain-Python fallback otherwise), and regenerates the dashboard. It uses `py -3.10` for Member 4 and `cnn_env` for CNN image processing/training.
- `cmd.exe /c Run_Project.bat` completed with exit code 0. It skipped all model training because baseline and CNN artifacts were present and passed validation; fallback notebook execution completed 9/9 code cells; dashboard generation completed.
- Final ledger has `ok=true`, zero mismatches and 13/13 model reload/inference checks passing. Full-hand scratch-CNN predictions recalculate to MAE 1.5598706672, RMSE 1.8946331154 and R² 0.4071558064 (n=38); the same images' handcrafted XGBoost predictions recalculate to MAE 1.5818585921, RMSE 2.0354729441 and R² 0.3157402553. The regenerated dashboard contains both paired result rows. Baseline files were not rewritten by this launcher run.
- Remaining external/unsupported items: a clean `requirements.txt` installation and real Jupyter-kernel run require package-index access; source-license/provenance confirmation requires authoritative dataset documentation not present locally; fusion is unsupported without verified paired participants/modalities. Member 5's interface remains outside Member 4 scope. These are not marked complete.

Reproduce the available-workflow run from the project root with `cmd.exe /c Run_Project.bat` (or double-click `Run_Project.bat`). With package access restored, create an isolated environment using `py -3.10 -m venv member4_ml_modelling\.venv-clean`, install with `member4_ml_modelling\.venv-clean\Scripts\python.exe -m pip install -r member4_ml_modelling\requirements.txt notebook nbconvert ipykernel`, then execute with `member4_ml_modelling\.venv-clean\Scripts\python.exe -m nbconvert --to notebook --execute --inplace member4_ml_modelling\notebooks\member4_analysis.ipynb`.
