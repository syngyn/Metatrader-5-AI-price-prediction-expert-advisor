@echo off
setlocal enabledelayedexpansion

REM ============================================
REM GGTH Predictor v2.0 - Smart Launcher
REM Handles Python detection and setup
REM ============================================

REM Change to script directory
cd /d "%~dp0"

echo.
echo ================================================
echo  GGTH Predictor v2.0 - Starting Up...
echo  (unified_predictor_v8.py)
echo ================================================
echo.

REM --- Step 1: Check for Python ---
echo [1/4] Checking for Python installation...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ================================================
    echo  ERROR: Python Not Found!
    echo ================================================
    echo.
    echo Python is required but not installed or not in PATH.
    echo.
    echo Please install Python 3.9, 3.10, or 3.11 from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check the box that says:
    echo "Add Python to PATH"
    echo.
    echo After installing Python, run this script again.
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo    Found Python version: %PYVER%

REM --- Step 2: Create Virtual Environment ---
echo [2/4] Setting up virtual environment...

if not exist ".venv" (
    echo    Creating new virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to create virtual environment.
        echo This usually means Python venv module is not available.
        echo.
        echo Try reinstalling Python and ensure "pip" is included.
        echo.
        pause
        exit /b 1
    )
    echo    Virtual environment created successfully!
) else (
    echo    Virtual environment already exists.
)

REM --- Step 3: Activate and Install Dependencies ---
echo [3/4] Activating virtual environment...

call ".venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to activate virtual environment.
    echo Try deleting the .venv folder and running this script again.
    echo.
    pause
    exit /b 1
)

echo    Virtual environment activated.
echo.
echo [4/4] Installing/updating required packages...
echo    This may take a few minutes on first run...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip --quiet

REM Install requirements
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some packages may have failed to install.
    echo Attempting to continue...
    echo.
)

echo.
echo ================================================
echo  Setup Complete! Launching GUI...
echo ================================================
echo.

REM --- Launch GUI ---
python ggth_gui.py

REM If GUI exits with error, pause so user can see the error
if %errorlevel% neq 0 (
    echo.
    echo ================================================
    echo  GUI Exited with Error Code: %errorlevel%
    echo ================================================
    echo.
    echo If you see import errors, try running:
    echo    pip install -r requirements.txt
    echo.
    pause
)

endlocal
