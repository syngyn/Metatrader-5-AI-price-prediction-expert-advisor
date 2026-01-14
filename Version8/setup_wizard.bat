@echo off
setlocal enabledelayedexpansion

REM ============================================
REM GGTH Predictor - Quick Setup Wizard
REM Helps configure MT5 path for first-time users
REM ============================================

echo.
echo ================================================
echo  GGTH Predictor - Quick Setup Wizard
echo ================================================
echo.
echo This wizard will help you configure your MT5 path.
echo.

REM Change to script directory
cd /d "%~dp0"

REM --- Check if config.json already exists ---
if exist "config.json" (
    echo Config file already exists!
    echo.
    choice /C YN /M "Do you want to reconfigure the MT5 path"
    if errorlevel 2 goto :show_current
    if errorlevel 1 goto :configure
) else (
    goto :configure
)

:show_current
echo.
echo Current configuration:
type config.json
echo.
echo.
echo You can also use the GUI to change the MT5 path.
echo Click "Browse..." next to "MT5 Files:" and then "Save MT5 Path"
echo.
pause
exit /b 0

:configure
echo.
echo ================================================
echo  Finding your MT5 Files directory...
echo ================================================
echo.

REM Try to auto-detect MT5 path
set "FOUND_PATH="
set "APPDATA_PATH=%APPDATA%\MetaQuotes\Terminal"

if exist "%APPDATA_PATH%" (
    echo Checking: %APPDATA_PATH%
    
    REM Look for Terminal folders
    for /d %%D in ("%APPDATA_PATH%\*") do (
        if exist "%%D\MQL5\Files" (
            set "FOUND_PATH=%%D\MQL5\Files"
            echo.
            echo ================================================
            echo  Found MT5 installation!
            echo ================================================
            echo.
            echo Path: !FOUND_PATH!
            echo.
            goto :ask_confirm
        )
    )
)

echo Could not auto-detect MT5 installation.
echo.
goto :manual_entry

:ask_confirm
choice /C YN /M "Is this the correct MT5 Files directory"
if errorlevel 2 goto :manual_entry
if errorlevel 1 goto :save_config

:manual_entry
echo.
echo ================================================
echo  Manual MT5 Path Entry
echo ================================================
echo.
echo Please enter your MT5 Files directory path.
echo.
echo How to find it:
echo   1. Open MetaTrader 5
echo   2. Click File ^> Open Data Folder
echo   3. Navigate to MQL5\Files
echo   4. Copy the full path from the address bar
echo.
echo Example:
echo   C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\HASH\MQL5\Files
echo.
echo Paste your path below (or type Q to quit):
echo.
set /p "FOUND_PATH=MT5 Files Path: "

if /i "%FOUND_PATH%"=="Q" (
    echo Setup cancelled.
    pause
    exit /b 1
)

REM Remove quotes if user pasted with quotes
set "FOUND_PATH=%FOUND_PATH:"=%"

REM Check if directory exists
if not exist "%FOUND_PATH%" (
    echo.
    echo ERROR: Directory does not exist!
    echo Path: %FOUND_PATH%
    echo.
    echo Please verify the path and try again.
    echo.
    pause
    exit /b 1
)

REM Check if it looks like an MT5 directory
echo %FOUND_PATH% | findstr /C:"MQL5" >nul
if errorlevel 1 (
    echo.
    echo WARNING: This path does not contain "MQL5"
    echo Are you sure this is the correct MT5 Files directory?
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 (
        echo Setup cancelled.
        pause
        exit /b 1
    )
)

:save_config
echo.
echo ================================================
echo  Creating configuration file...
echo ================================================
echo.

REM Escape backslashes for JSON
set "JSON_PATH=%FOUND_PATH:\=\\%"

REM Create config.json
(
echo {
echo   "mt5_files_path": "%JSON_PATH%",
echo   "version": "1.2",
echo   "use_kalman": true,
echo   "default_symbol": "USDJPY",
echo   "prediction_interval_minutes": 60
echo }
) > config.json

if exist config.json (
    echo.
    echo ================================================
    echo  Configuration Saved Successfully!
    echo ================================================
    echo.
    echo MT5 Files Path: %FOUND_PATH%
    echo Config file: %CD%\config.json
    echo.
    echo You're all set! You can now:
    echo   1. Launch the GGTH Predictor GUI
    echo   2. Train your models
    echo   3. Start making predictions
    echo.
    echo You can change the MT5 path anytime through:
    echo   - The GUI ^(Browse... ^> Save MT5 Path^)
    echo   - Editing config.json manually
    echo   - Running this setup wizard again
    echo.
) else (
    echo.
    echo ERROR: Failed to create config.json
    echo Please check file permissions.
    echo.
    pause
    exit /b 1
)

REM Ask if user wants to launch GUI
echo.
choice /C YN /M "Would you like to launch the GGTH Predictor GUI now"
if errorlevel 2 goto :end
if errorlevel 1 goto :launch_gui

:launch_gui
echo.
echo Launching GGTH Predictor GUI...
echo.
start "" "run_ggth_gui.bat"
goto :end

:end
echo.
echo Setup complete!
echo.
pause
exit /b 0
