@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher ^(py^) was not found. Install Python 3.10 and its launcher.
  exit /b 1
)
py -3.10 -c "import numpy,pandas,sklearn,xgboost,matplotlib,pyarrow; print('Member 4 Python 3.10 dependencies are available.')"
if errorlevel 1 (
  echo Required Member 4 packages are unavailable in Python 3.10.
  echo Install member4_ml_modelling\requirements.txt when package access is available.
  exit /b 1
)
if not exist "cnn_env\Scripts\python.exe" (
  echo The isolated CNN runtime is missing: cnn_env\Scripts\python.exe
  exit /b 1
)

echo Checking existing artifacts before running any training.
set "BASELINE_MISSING=0"
for %%F in (eye_rf_reg_predictions.csv eye_et_reg_metrics.json eye_xgb_reg_predictions.csv nail_rf_reg_predictions.csv nail_et_reg_metrics.json nail_xgb_reg_predictions.csv full_hand_rf_reg_predictions.csv full_hand_et_reg_metrics.json full_hand_xgb_reg_predictions.csv all_metrics_summary.csv) do (
  if not exist "member4_ml_modelling\outputs\metrics\%%F" set "BASELINE_MISSING=1"
)
for %%F in (eye_handcrafted_regression.pkl nail_handcrafted_regression.pkl full_hand_handcrafted_regression.pkl eye_xgb_regression.pkl nail_xgb_regression.pkl full_hand_xgb_regression.pkl eye_handcrafted_classification.pkl nail_handcrafted_classification.pkl palm_handcrafted_classification.pkl) do (
  if not exist "member4_ml_modelling\outputs\models\%%F" set "BASELINE_MISSING=1"
)
for %%F in (eye_handcrafted.parquet nail_handcrafted.parquet full_handcrafted.parquet palm_handcrafted.parquet) do (
  if not exist "member4_ml_modelling\outputs\features\%%F" set "BASELINE_MISSING=1"
)
if "%BASELINE_MISSING%"=="1" (
  echo Baseline result files are incomplete. Refusing to retrain or overwrite existing baseline artifacts automatically.
  echo Restore the missing baseline artifacts before using this launcher.
  exit /b 1
)
echo Existing handcrafted baselines found; they will not be retrained or overwritten.

set "CNN_MISSING=0"
for %%F in (m3_eye_regression_embeddings.csv m3_nail_regression_embeddings.csv m3_palm_classifier_fullhand_transfer_embeddings.csv fullhand_scratch_cnn_embeddings.csv) do (
  if not exist "member4_ml_modelling\outputs\embeddings\%%F" set "CNN_MISSING=1"
)
if not exist "member4_ml_modelling\outputs\metrics\cnn_embedding_experiment\eye_cnn_embedding_xgb_metrics.json" set "CNN_MISSING=1"
if not exist "member4_ml_modelling\outputs\metrics\cnn_embedding_experiment\nail_cnn_embedding_xgb_metrics.json" set "CNN_MISSING=1"
if not exist "member4_ml_modelling\outputs\metrics\cnn_embedding_experiment\full_hand_palm_checkpoint_transfer_xgb_metrics.json" set "CNN_MISSING=1"
if not exist "member4_ml_modelling\outputs\metrics\fullhand_cnn_experiment\fullhand_scratch_cnn_xgb_metrics.json" set "CNN_MISSING=1"
if not exist "member4_ml_modelling\outputs\models\fullhand_cnn_scratch\fullhand_scratch_cnn_regressor.pth" set "CNN_MISSING=1"
if not exist "member4_ml_modelling\outputs\models\fullhand_cnn_scratch\fullhand_scratch_cnn_embedding_xgb.pkl" set "CNN_MISSING=1"
if not exist "member4_ml_modelling\outputs\plots\cnn_embedding\eye_cnn_embedding_xgb_vs_handcrafted.png" set "CNN_MISSING=1"
if not exist "member4_ml_modelling\outputs\plots\cnn_embedding\nail_cnn_embedding_xgb_vs_handcrafted.png" set "CNN_MISSING=1"
if not exist "member4_ml_modelling\outputs\plots\cnn_embedding\full_hand_palm_checkpoint_transfer_vs_handcrafted.png" set "CNN_MISSING=1"
if not exist "member4_ml_modelling\outputs\plots\fullhand_cnn_experiment\fullhand_scratch_cnn_xgb_vs_handcrafted.png" set "CNN_MISSING=1"
if "%CNN_MISSING%"=="1" goto run_cnn_stages

echo Required CNN embeddings, models, and metrics are present. Verifying saved results before reuse.
py -3.10 member4_ml_modelling\src\validate_member4_results.py
if errorlevel 1 goto run_cnn_stages
goto artifacts_verified

:run_cnn_stages
echo CNN artifacts are missing or failed validation. Rebuilding CNN experiments only; handcrafted baselines remain untouched.
py -3.10 member4_ml_modelling\src\prepare_m3_embedding_manifests.py
if errorlevel 1 goto failed
cnn_env\Scripts\python.exe member4_ml_modelling\src\extract_m3_cnn_embeddings.py
if errorlevel 1 goto failed
py -3.10 member4_ml_modelling\src\train_m3_embedding_xgboost.py
if errorlevel 1 goto failed
py -3.10 member4_ml_modelling\src\prepare_m3_fullhand_transfer_manifest.py
if errorlevel 1 goto failed
cnn_env\Scripts\python.exe member4_ml_modelling\src\extract_m3_palm_transfer_embeddings.py
if errorlevel 1 goto failed
py -3.10 member4_ml_modelling\src\train_fullhand_palm_transfer_xgboost.py
if errorlevel 1 goto failed
cnn_env\Scripts\python.exe member4_ml_modelling\src\train_fullhand_scratch_cnn.py
if errorlevel 1 goto failed
py -3.10 member4_ml_modelling\src\train_fullhand_scratch_cnn_embedding_xgboost.py
if errorlevel 1 goto failed
py -3.10 member4_ml_modelling\src\validate_member4_results.py
if errorlevel 1 goto failed

:artifacts_verified
echo Saved models, paired predictions, and metrics passed validation.

set "PLOTS_MISSING=0"
for %%F in (eye_all_models_regression.png nail_all_models_regression.png full_hand_all_models_regression.png eye_all_models_classification.png nail_all_models_classification.png palm_all_models_classification.png) do (
  if not exist "member4_ml_modelling\outputs\plots\%%F" set "PLOTS_MISSING=1"
)
if "%PLOTS_MISSING%"=="1" (
  echo Regenerating comparison plots from verified saved metrics.
  py -3.10 member4_ml_modelling\src\generate_comparison_plots.py
  if errorlevel 1 goto failed
) else echo Existing comparison plots are present; regeneration skipped.

py -3.10 -m nbconvert --version >nul 2>nul
if errorlevel 1 goto notebook_fallback
echo Executing the notebook with the installed Jupyter nbconvert kernel.
py -3.10 -m nbconvert --to notebook --execute --inplace member4_ml_modelling\notebooks\member4_analysis.ipynb
if errorlevel 1 goto failed
goto dashboard

:notebook_fallback
echo Jupyter nbconvert is unavailable; running the documented plain-Python notebook fallback.
py -3.10 member4_ml_modelling\src\execute_notebook.py
if errorlevel 1 goto failed

:dashboard
py -3.10 member4_ml_modelling\src\generate_results_dashboard.py
if errorlevel 1 goto failed

echo.
echo Run complete. Validation passed; existing baselines were preserved.
echo Open member4_ml_modelling\reports\results_dashboard.html.
exit /b 0

:failed
echo.
echo The launcher stopped because a required step failed. See the error above.
exit /b 1
