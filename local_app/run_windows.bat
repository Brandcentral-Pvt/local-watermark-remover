@echo off
REM ---------------------------------------------------------------------------
REM  Watermark Remover - one-click launcher for Windows
REM  First run creates a virtual environment, installs dependencies and
REM  downloads the 208 MB model, then opens the app in your browser.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Install Python 3.10+ from https://www.python.org/downloads/ and tick
    echo "Add python.exe to PATH" during setup, then run this file again.
    pause & exit /b 1
)

if not exist ".venv" (
    echo [1/3] creating virtual environment...
    python -m venv .venv || (echo failed to create venv & pause & exit /b 1)
)

echo [2/3] installing dependencies (first run only)...
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo dependency install failed & pause & exit /b 1)

if not exist "models\lama_fp32.onnx" (
    echo [3/3] downloading the model ^(208 MB, once^)...
    ".venv\Scripts\python.exe" watermark_gui.py --download || (echo model download failed & pause & exit /b 1)
)

echo starting the app - your browser will open at http://127.0.0.1:7860
".venv\Scripts\python.exe" watermark_gui.py --port 7860
pause
