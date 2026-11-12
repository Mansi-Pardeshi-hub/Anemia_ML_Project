# Member 3 CNN Embedding → XGBoost Regression Experiment

**Run date:** 2026-10-29  
**Status:** Eye/Nail Member 3 regression-checkpoint experiments, exploratory Palm-checkpoint transfer, and a separate Member 4 Full-hand scratch-CNN embedding experiment are complete. No Member 3 Full-hand checkpoint was supplied.

## Method and data integrity

The 59-entry `datasets/member3_model_training.zip` and its extracted source/configs were inspected. `src/utils.py` defines a scratch-trained `LightweightCNN` with `RegressionCNN`; its pooled `backbone.get_features()` representation is 1,024 dimensions. The Eye and Nail regression training scripts instantiate the model with dropout 0.4. Both use RGB, 224×224 bilinear resize, float32 `/255`, then ImageNet mean/std normalization; inference uses no augmentation. These are Member 3's learned features, not a pretrained MobileNet/EfficientNet model.

The regression checkpoints were loaded only with `torch.load(..., weights_only=True)` and strict `load_state_dict`. PyTorch's static unsafe-global inspection found no extra globals in Nail; Eye found `numpy.dtype` and `numpy.core.multiarray.scalar`. For Eye, only those plus the specific `numpy.dtypes.Float64DType` required to rebuild the stored train Hb scalar were allowlisted. No `weights_only=False` load was used. The reconstructed classes were compiled directly from the reviewed Member 3 source; strict state-dictionary loading passed for both models.

The isolated `cnn_env` runs Python 3.10.0, PyTorch 2.14.1+cpu and torchvision 0.29.1+cpu; CPU inference succeeded. The existing global Python 3.10 environment supplied pandas, scikit-learn, XGBoost, pyarrow and plotting dependencies. No environment was modified to install packages.

Before extraction, the Member 4 feature rows were joined one-to-one to Member 3 split records by image path. Every image path exists; IDs, participant, split and numeric Hb agree exactly; there are no missing targets or participant split crossings. Embedding rows retain `dataset_id`, unique `image_id`, path, participant, Hb target (g/dL), split, checkpoint ID/hash and 1,024 numeric embedding columns. All values are finite. The corrected held-out image paths and targets match the existing handcrafted-XGBoost prediction files exactly: Eye 89/89 and Nail 109/109.

For XGBoost, the same four parameter configurations used by the handcrafted pipeline were scored on validation MAE. StandardScaler was fit on training embeddings only. The chosen XGBoost model was fit on training rows and evaluated once on the held-out test rows. The saved handcrafted-XGBoost predictions were used as a same-image comparator. Existing baseline metrics and model files were not overwritten.

## Results

| Dataset | Features | Train / val / test | CNN-embedding XGBoost test MAE | RMSE | R² | Handcrafted XGBoost MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Eye | Member 3 Eye regression CNN, 1,024-D | 432 / 89 / 89 | 1.7452 | 2.2907 | 0.3152 | 1.4177 | 1.9744 | 0.4912 |
| Nail | Member 3 Nail regression CNN, 1,024-D | 503 / 107 / 109 | 2.0425 | 2.6010 | -0.2006 | 1.7171 | 2.3079 | 0.0548 |
| Full-hand | Member 4 CNN trained from scratch, 64-D | 175 / 37 / 38 | 1.5599 | 1.8946 | 0.4072 | 1.5819 | 2.0355 | 0.3157 |
| Full-hand | M3 Palm folder-classifier transfer, 1,024-D (exploratory; not Full-hand-trained) | 175 / 37 / 38 | 2.0970 | 2.6163 | -0.1305 | 1.5819 | 2.0355 | 0.3157 |

The Eye/Nail Member 3 embeddings and Full-hand Palm-checkpoint transfer did **not** improve over handcrafted XGBoost. The separate Full-hand scratch-CNN embedding result was slightly better on this held-out split; its margin is small and the test set has only 38 participants. These results are exploratory image/crop-level estimates, not clinical validation. Eye labels have the previously documented age/pregnancy/adjustment caveats; Figshare Hb is numeric but its inherited classification threshold lacks demographic context. No classification was trained from these regression embeddings. There is no Full-hand regression checkpoint in the supplied archive, so neither Eye nor Nail weights were repurposed out of domain. Per the explicit exploratory-transfer allowance, the distinct M3 Palm classification checkpoint was used as a frozen feature extractor for Full-hand: it had been trained on another Palm image collection for filename-derived binary folder classes (no measured Hb; checkpoint metadata reports epoch 2 and validation accuracy 0.5987). Full-hand has 250 exact source-linked numeric Hb targets, unique image contents, 250 participant IDs, and participant-disjoint 175/37/38 splits. The Palm transfer XGBoost selected among four parameter sets by validation MAE, fit scaling on train only, and scored the same 38 held-out images as handcrafted XGBoost. It yielded MAE/RMSE/R² 2.0970/2.6163/-0.1305 versus handcrafted 1.5819/2.0355/0.3157. It did not improve the baseline and is only an exploratory cross-dataset representation transfer, not equivalent to a Full-hand-trained CNN experiment. In addition, a compact Member 4 CNN was initialized randomly and trained on the 175 Full-hand training participants only (428,905 parameters; 64-D embedding). Training used SmoothL1, AdamW, train-only flip/brightness augmentation, and validation-MAE checkpoint selection; best epoch 24 had validation MAE 1.3779 g/dL and early stopping ended at epoch 39. The best weights were saved and safely reloaded before keyed embeddings were produced. XGBoost settings were selected by validation MAE among four configurations, and scaling was fitted to training embeddings only. On the untouched 38 test images, CNN-embedding XGBoost scored MAE/RMSE/R² 1.5599/1.8946/0.4072 vs handcrafted 1.5819/2.0355/0.3157. The 0.0221 g/dL MAE difference is small on a 38-person test set; this is feasibility evidence, not reliable clinical performance. No cross-modality pairing or fusion is claimed.

## Reproduction and artifacts

From the project root, create manifests using the existing Member 4 Python 3.10 environment, then extract with `cnn_env`, then train/compare with global Python 3.10:

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

- Input manifests and 1,024-D embeddings: `outputs/embeddings/`
- Checkpoint/preprocessing/runtime provenance: `outputs/embeddings/m3_*_embeddings_provenance.json`
- Test predictions, JSON metrics and separate comparison CSV: `outputs/metrics/cnn_embedding_experiment/`
- New models: `outputs/models/cnn_embedding/`
- Paired-test plots: `outputs/plots/cnn_embedding/`
- Updated HTML comparison page: `reports/results_dashboard.html`

A previously recorded PyPI DNS check failed. `cnn_env` lacks the other Member 4 requirements, but separate interpreters were sufficient for inference and XGBoost, so a package installation was unnecessary. The clean all-requirements environment remains unverified. Full-hand Palm-transfer artifacts use `full_hand_palm_checkpoint_transfer_*`; scratch-CNN artifacts use `fullhand_scratch_cnn_*` in separate output directories. The trained CNN checkpoint is `outputs/models/fullhand_cnn_scratch/fullhand_scratch_cnn_regressor.pth`.

## Final project run verification (2026-10-30)

The current saved Full-hand experiment was revalidated without retraining. The participant split remains 175/37/38 with no participant overlap; all 38 test image paths and targets pair one-to-one with the handcrafted XGBoost predictions. Saved Full-hand scratch-CNN embeddings and the XGBoost model reload; the reloaded model reproduces the saved test predictions. Recalculated CNN metrics are MAE 1.5598706672 g/dL, RMSE 1.8946331154 g/dL, R² 0.4071558064; paired handcrafted metrics are MAE 1.5818585921, RMSE 2.0354729441, R² 0.3157402553. These remain a small-sample exploratory comparison.

The revised `Run_Project.bat` completed with exit code 0. Since baseline and CNN artifacts were present and valid, it skipped model training, executed the notebook's documented fallback (9/9 code cells), and regenerated the results dashboard. Final validation reports zero mismatches and all 13 saved Member 4 model reload/inference checks pass. The dashboard includes the paired Full-hand scratch-CNN and handcrafted rows; the separate Palm-checkpoint transfer remains labeled exploratory.

Environment inspection found all nine exact pinned Member 4 packages in Python 3.10. No alternate pip index is configured, pip cache has no usable wheels, and the only wheel-named files under `cnn_env` are zero bytes. Python 3.10, Python 3.13 and `cnn_env` lack Jupyter, nbconvert, nbclient, nbformat, ipykernel and IPython. One PyPI DNS failure was already recorded in the preceding audit; it was not retried. Therefore a clean installation and real Jupyter-kernel execution remain unverified for this external package-access blocker. The plain-Python notebook fallback does not satisfy the Jupyter-kernel requirement.
