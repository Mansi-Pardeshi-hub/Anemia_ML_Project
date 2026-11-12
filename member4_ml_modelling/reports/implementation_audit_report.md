# Member 4 Implementation and Verification Audit

**Audit date:** 2026-10-27  
**Workspace:** `C:\Users\nethr\Downloads\Anemia_ML_Project`

## Final corrected run

The final corrected root launcher `Run_Project.bat` was run after the duplicate corrections. It exited with status 0, ran the pipeline, and regenerated the comparison plots. The direct component commands used to verify the final pipeline were:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python member4_ml_modelling\src\run_pipeline.py
python member4_ml_modelling\src\generate_comparison_plots.py
```

Both commands completed successfully. The pipeline regenerated current model predictions/metrics and relevant model/plot artifacts for the corrected Eye and Palm samples; comparison figures were regenerated from the saved outputs. Eye uses 610 exact-unique images with test n=89. Palm uses 2,056 rows with test n=292 after exact and confirmed near-duplicate exclusion. The root launcher was rerun after the corrections and completed with exit status 0; the direct component commands also completed successfully.

## Checks completed

- `python -m compileall -q member4_ml_modelling\src` passed on the final source tree; notebook JSON and all code-cell syntax were also checked.
- Recomputed MAE, RMSE and R² from all saved RF/Extra Trees/XGBoost regression prediction columns; all values match per-model JSON after rounding.
- Recomputed accuracy, precision, sensitivity/recall, specificity, F1, AUC and confusion counts from Eye/Nail/Palm RF and Extra Trees predictions; values match per-model JSON after rounding.
- The final run completed successfully and regenerated the Eye/Palm predictions and evaluation figures.
- All nine current Member 4 pickle artifacts reloaded and returned finite predictions on five feature rows each.
- Eye exact content ledger shows 21 redundant rows removed; no exact duplicate pair crosses participant or partition.
- Palm pHash/SSIM near-duplicate scan returned 36 candidate pairs, 17 cross-split; four high-similarity (SSIM .987–.988) test/validation pairs drove test-side hash exclusion. All 12 source manifest rows bearing those four hashes were filtered; no source images were modified.
- Current group counts are disjoint under supplied IDs: Eye 202 participants, Nail 250, Full-hand 250, Palm 514 inferred groups. Palm group IDs are not verified person IDs.
- Eye/Nail/Figshare/Palm target provenance was reviewed; reports now separate measured Hb, inherited threshold labels and Palm folder labels. No clinical-validation claim is made.

## CNN checkpoint / embedding blocker

Member 3's archive has Eye classification/regression, Nail classification/regression, and Palm classification `.pth` checkpoints. `datasets/member3_extracted/src/utils.py` defines `LightweightCNN` (MobileNet-inspired, scratch-trained because pretrained weights were unavailable) and `RegressionCNN`; the pooled `get_features()` output is 1,024 dimensions. README/utils specify RGB 224×224 bilinear resize, [0,1] scaling and ImageNet mean/std normalization. No image-keyed feature/embedding table is supplied. `torch`/`torchvision` are absent in the Member 4 environment and PyPI is not reachable here, so checkpoint loading and embedding extraction were not performed. No CNN retraining was attempted.

A valid follow-up requires embeddings keyed to image path/ID, participant/group, split and Hb target, with checkpoint/training provenance, architecture/layer/dimension and preprocessing documented. Member 3 must supply this table or an extraction handoff and usable PyTorch runtime. Member 4 can then tune XGBoost using training/validation data and score the same corrected held-out participants. Until then, the existing XGBoost results use handcrafted features only.

## Fresh environment and notebook

A clean Python 3.10 virtual environment was created in the project without touching the existing environment. `ensurepip` succeeded, then `pip install -r member4_ml_modelling\requirements.txt` failed while resolving `numpy==2.2.4` because PyPI DNS lookup returned `getaddrinfo failed`. Therefore README install/run portability is not verified in a fresh environment; rerun where package access is available.

The notebook has 18 cells/9 code cells. All nine plain-Python cells were executed sequentially with shared state in the existing Python 3.10 runtime; stdout and one figure output were saved into the notebook. The current cache counts and 25-row metrics summary agree with the pipeline. Jupyter/nbformat/nbclient remain absent, so this was not a standard Jupyter-kernel run. Reproduce via `python member4_ml_modelling\src\execute_notebook.py`; standard `nbconvert` commands remain optional.

## File evidence

- Current reports: `dataset_audit_report.md`, `results_report.md`, `member4_final_completion_audit.md`, and `member4_completion_checklist.md`
- Current metrics/predictions and exact/near duplicate ledgers: `outputs/metrics/`
- Models: `outputs/models/`
- Handcrafted feature caches: `outputs/features/`
- Evaluation figures: `outputs/plots/`
- Root launcher: `Run_Project.bat`
- Results page generator/output: `src/generate_results_dashboard.py` / `reports/results_dashboard.html`
- Requirements: `requirements.txt`
- Walkthrough notebook: `notebooks/member4_analysis.ipynb`
- Notebook output runner: `src/execute_notebook.py` (sequential plain-Python fallback; does not require Jupyter)

## Scope boundary

Handcrafted modeling/evaluation and notebook-cell fallback execution are complete locally. Eye/Nail CNN-embedding XGBoost has been completed with safely loaded Member 3 regression checkpoints and keyed embeddings. A separate exploratory M3 Palm-to-Full-hand transfer and a Member 4 Full-hand scratch-CNN embedding/XGBoost experiment are complete. The latter is trained on the 175 training participants and evaluated on the same 38 held-out images as handcrafted XGBoost; no Member 3 Full-hand checkpoint exists. A clean full-requirements install and standard Jupyter-kernel execution remain unverified because package-index DNS fails. Member 5 owns the Streamlit upload-and-predict demo. No paired dataset supports multimodal fusion; see `member4_final_completion_audit.md` for current evidence.
