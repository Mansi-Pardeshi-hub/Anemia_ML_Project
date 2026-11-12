# Member 4 — ML Modeling and Evaluation

This folder contains the Member 4 handcrafted-feature baselines, Hb regression experiments, exploratory classification experiments, and evaluation artifacts. Read the [final completion audit](reports/member4_final_completion_audit.md), [implementation audit](reports/implementation_audit_report.md), and [results report](reports/results_report.md) for verified findings and limitations.

## Requirements

- Python 3.10 or newer
- Extracted project datasets under `datasets/` (the original ZIP archives are preserved)
- Pinned direct dependencies in `requirements.txt`

The launcher uses Python 3.10 for Member 4 scripts and the existing `cnn_env` containing CPU PyTorch/torchvision for CNN image processing. The isolated CNN environment used for the verified run was Python 3.10.0, torch 2.14.1+cpu, torchvision 0.29.1+cpu. The installed Python 3.10 has all nine exact versions in `requirements.txt`. A clean recreation was not possible: no alternate index or usable cached wheels were found, and the previously recorded PyPI DNS check failed.

To create a separate clean environment when package access is available (the existing project environments are left untouched):

```powershell
py -3.10 -m venv member4_ml_modelling\.venv-clean
member4_ml_modelling\.venv-clean\Scripts\python.exe -m pip install -r member4_ml_modelling\requirements.txt
```

## Run the pipeline

On this project machine, double-click `Run_Project.bat` in the project root. It checks the existing baselines, validates and reuses saved CNN artifacts when they are present, executes only missing/invalid CNN experiment stages, runs the notebook with nbconvert when available (otherwise the documented plain-Python fallback), validates the saved artifacts, and regenerates the results page. It does not retrain verified handcrafted baselines. If baseline result files are incomplete, the launcher stops rather than overwrite existing baseline files. No administrator access is needed.

From project-root PowerShell, the same command is:

```powershell
.\Run_Project.bat
```

Or run from PowerShell at the project root:

```powershell
cmd.exe /c Run_Project.bat
```

The pipeline loads provided split manifests, checks group partitions and exact duplicate content, validates/rebuilds cached handcrafted features, trains conventional baselines and XGBoost regressors for Eye, Nail and Full-hand Hb targets, and runs exploratory classification experiments. It saves metrics, test predictions with image paths, trained models, feature caches, and evaluation plots. Feature extraction processes one image at a time.

## Data and task boundaries

- Eye: measured numeric Hb values and sex labels are in the supplied split; exact-content deduplication leaves 610 modeling images from the same 202 country-qualified participant groups (432/89/89 train/validation/test). Eye class labels use inherited sex-specific cutoffs. The source study reports blood-count Hb in g/dL, although Member 3 config still says assumed. Joined labels show ages 19–88 (19 participants over 65); pregnancy and altitude/smoking context are unavailable, so inherited threshold classes remain exploratory. [Source study](https://www.sciencedirect.com/science/article/pii/S0933365722002299).
- Figshare nail/full-hand: the publisher describes 250 patients with known Hb; numeric Hb is supplied in g/L and g/dL. Splits contain 719 nail crops and 250 full-hand images from 250 participants. [Dataset record](https://springernature.figshare.com/articles/dataset/Dataset_of_human_skin_and_fingernails_images_for_non-invasive_haemoglobin_level_assessment/25867432). Nail crops repeat per participant and are grouped across splits.
- Figshare nail classification uses an inherited universal 11 g/dL threshold without age/sex context; treat it as low-Hb threshold exploration only.
- Palm: exact-duplicate filtering and the conservative cross-split near-duplicate correction leave 2,056 unique image contents across 514 inferred groups (test n=292). Four near-identical test images and all exact byte copies were excluded from evaluation; source images remain untouched. The source publisher says the original pediatric cohort labels were assigned from laboratory Hb remarks and 710 photos were augmented into 4,260 files. The supplied augmented files have folder labels but no Hb records for row-wise verification; report only folder-label prediction. [Source record](https://data.mendeley.com/datasets/ccr8cm22vz/1). No measured Hb is present in the local split.
- Modality datasets have no verified shared participant keys. No multimodal fusion was performed. Member 4 has separately run image-keyed CNN-embedding XGBoost experiments for Eye and Nail using Member 3's regression checkpoints. Full-hand has a clearly labeled exploratory transfer experiment using the Member 3 Palm classifier backbone; those weights were trained on a different dataset for folder-derived classes, not Full-hand Hb. See `reports/cnn_embedding_experiment_report.md` and the completion audit.

Features include color statistics/histograms, HSV/LAB/grayscale intensity statistics, pallor/redness ratios, and GLCM texture features. The run uses seed 42 and provided train/validation/test partitions; XGBoost selection uses validation data and final scoring uses held-out test data. Palm groups are inferred from filenames/folders, not confirmed participant identifiers.

## Main files

- `src/config.py`: dataset and output paths
- `src/data_loading.py`: split loaders, exact-image duplicate filtering and group checks
- `src/feature_extraction.py`: handcrafted image features
- `src/evaluation.py`, `src/plotting.py`: metrics and figures
- `src/run_pipeline.py`: model/evaluation pipeline
- `src/generate_comparison_plots.py`: comparison plots generated from saved metric files
- `src/generate_results_dashboard.py`: self-contained HTML page generated from the current metrics summary
- `src/prepare_m3_embedding_manifests.py`, `src/extract_m3_cnn_embeddings.py`, `src/train_m3_embedding_xgboost.py`: verified Eye/Nail Member 3 regression-CNN feature extraction and XGBoost comparison
- `src/prepare_m3_fullhand_transfer_manifest.py`, `src/extract_m3_palm_transfer_embeddings.py`, `src/train_fullhand_palm_transfer_xgboost.py`: exploratory Full-hand feature transfer using the separate Palm folder-classification checkpoint; not a Full-hand-trained CNN
- `src/train_fullhand_scratch_cnn.py`, `src/train_fullhand_scratch_cnn_embedding_xgboost.py`: separate compact Full-hand CNN trained from scratch and validation-selected XGBoost regression on its learned embeddings
- `src/execute_notebook.py`: execute the reviewed plain-Python notebook cells and save their standard outputs without installing Jupyter
- `reports/dataset_audit_report.md`: source dataset audit
- `reports/implementation_audit_report.md`: implementation and execution evidence
- `reports/results_report.md`: verified Member 4 results and methodology
- `reports/member4_final_completion_audit.md`: requirement-by-requirement verdict
- `reports/member4_completion_checklist.md`: concise status checklist
- `outputs/features/*.parquet`: current cached feature rows with manifest metadata; `palm_handcrafted_pre_near_duplicate.parquet` is a preserved pre-correction snapshot and is not used by the pipeline
- `outputs/metrics/`: metrics, test predictions and duplicate audit ledgers
- `outputs/models/*.pkl`: fitted models and feature-column/scaler metadata
- `outputs/plots/`: evaluation figures
- `reports/results_dashboard.html`: latest model comparison page; regenerated by `Run_Project.bat` or with `python member4_ml_modelling\src\generate_results_dashboard.py`

All findings are research prototypes; they do not establish clinical validity or generalization. The Full-hand Palm-checkpoint transfer is exploratory and uses weights trained on a different dataset and a different folder-label classification task; it is not a Full-hand-trained Hb model. The notebook is a review/walkthrough companion, not the pipeline runner. Its nine plain-Python code cells have been executed in sequence using the current Python 3.10 environment; stdout and the Hb distribution figure are saved as cell outputs. Jupyter/nbformat/nbclient are not installed, so this was not a Jupyter-kernel execution. Reproduce the saved outputs from the project root with:

```powershell
python member4_ml_modelling\src\execute_notebook.py
```

For standard Jupyter-kernel notebook execution after package access is available, install the notebook runtime into the separate environment and run:

```powershell
member4_ml_modelling\.venv-clean\Scripts\python.exe -m pip install notebook nbconvert ipykernel
member4_ml_modelling\.venv-clean\Scripts\python.exe -m nbconvert --to notebook --execute --inplace member4_ml_modelling\notebooks\member4_analysis.ipynb
```

The current machine has no `nbconvert`, `nbclient`, `nbformat`, `ipykernel` or Jupyter installation in Python 3.10, Python 3.13 or `cnn_env`. The launcher therefore runs `src/execute_notebook.py`, which executes the notebook's plain-Python cells and saves their outputs; that fallback is not described as a Jupyter-kernel run.

The Palm near-duplicate scan can be reproduced with `python member4_ml_modelling\src\audit_palm_near_duplicates.py`; it records DCT pHash candidates (Hamming distance <= 4), compares candidate pairs using grayscale SSIM, and excludes test-side exact-hash copies only when a cross-split pair has SSIM >= 0.98. Member 5's upload-and-predict interface is outside Member 4's scope.
