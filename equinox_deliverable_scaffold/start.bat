@echo off
TITLE Equinox — Emergency Corridor & Road Blockage Intelligence

echo ========================================================================
echo   EQUINOX — Emergency Corridor & Road Blockage Intelligence
echo   Track 1: Smart Mobility & Road Incident Intelligence
echo ========================================================================
echo.

:: Ensure we are inside the equinox directory
cd /d "%~dp0"
if exist "equinox" (
    cd equinox
)

:: Step 1: Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.9+ to run Equinox.
    pause
    exit /b 1
)

:: Step 2: Install dependencies if missing
echo [1/3] Checking Python dependencies...
python -c "import cv2, pandas, sklearn, fastapi, uvicorn" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing missing Python packages...
    pip install -r requirements.txt
) else (
    echo [OK] All required Python packages are installed.
)

:: Step 3: Run pipeline data generation if missing
echo.
echo [2/3] Verifying Equinox Pipeline & Model Data...
if not exist "models\severity_model.pkl" (
    echo Running end-to-end demo dataset generation & pipeline training...
    python pipeline/generate_demo_data.py
    python -c "import os; os.remove('data/manifest.csv') if os.path.exists('data/manifest.csv') else None"
    python pipeline/ingestion/ingest.py --raw_dir data/raw --out data/manifest.csv
    python pipeline/preprocessing/preprocess.py --manifest data/manifest.csv --out data/frames --fps 3
    python pipeline/features/feature_engineer.py --detections data/detections/detections.csv --out data/features
    python pipeline/training/train_severity.py --features data/features/incident_windows.parquet --labels data/labels/labels.csv --out models
    python pipeline/validation/validate.py --model models/severity_model.pkl --features data/features/incident_windows.parquet --labels data/labels/labels.csv --report_out docs/validation_report.md
) else (
    echo [OK] Pipeline models and dataset ready.
)

:: Step 4: Open Browser and Launch Server
echo.
echo [3/3] Launching FastAPI Backend Server & Dashboard...
echo Opening Operator Dashboard in default browser: http://127.0.0.1:8000 ...
timeout /t 2 >nul
start "" "http://127.0.0.1:8000"

python -m uvicorn pipeline.deployment.api:app --host 127.0.0.1 --port 8000 --reload

pause
