@echo off
REM ============================================================================
REM  build.bat - one-click build of the AI POS Scale Windows executable
REM
REM  Usage:   build.bat            -> dist\AIPosScale\AIPosScale.exe   (onedir, fast start-up)
REM           build.bat /onefile   -> dist\AIPosScale.exe              (single file, slower start-up)
REM           build.bat /skipdeps  -> do not reinstall requirements
REM           build.bat /skiptests -> skip the regression suite
REM
REM  Requirements: 64-bit Python 3.10+ in PATH, internet for the first run
REM  (pip packages + MobileNetV2 weights, ~14 MB). Everything is bundled, so
REM  the target POS machine needs no Python and no internet.
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set APP_NAME=AIPosScale
set MODE=onedir
set SKIPDEPS=0
set SKIPTESTS=0
for %%A in (%*) do (
    if /I "%%~A"=="/onefile"  set MODE=onefile
    if /I "%%~A"=="--onefile" set MODE=onefile
    if /I "%%~A"=="/skipdeps" set SKIPDEPS=1
    if /I "%%~A"=="/skiptests" set SKIPTESTS=1
)

echo.
echo ==== [1/6] Python check ====================================================
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python.exe not found in PATH. Install 64-bit Python 3.10+ and tick "Add to PATH".
    exit /b 1
)
python -c "import struct,sys; assert struct.calcsize('P')*8==64, '64-bit Python is required (PyTorch has no 32-bit wheels)'; print('Python', sys.version.split()[0], '64-bit OK')"
if errorlevel 1 exit /b 1

echo.
echo ==== [2/6] Installing requirements ========================================
if "%SKIPDEPS%"=="1" (
    echo skipped ^(/skipdeps^)
) else (
    python -m pip install --upgrade pip >nul 2>nul
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: pip install failed.
        exit /b 1
    )
)

echo.
echo ==== [3/6] MobileNetV2 weights for offline use ============================
if not exist models mkdir models
python -c "from ai_engine import ensure_weights; print('weights:', ensure_weights('models/mobilenet_v2-b0353104.pth', print))"
if errorlevel 1 (
    echo ERROR: could not obtain MobileNetV2 weights ^(internet needed once^).
    exit /b 1
)

echo.
echo ==== [4/6] Camera SDK =====================================================
if not exist RTKCamSDK.dll (
    if exist "%USERPROFILE%\Desktop\RTKCamSDK_v1.0\Release\RTKCamSDK.dll" (
        copy /Y "%USERPROFILE%\Desktop\RTKCamSDK_v1.0\Release\RTKCamSDK.dll" . >nul
        echo copied RTKCamSDK.dll from the SDK folder
    )
)
set ADD_BIN=
if exist RTKCamSDK.dll (
    set ADD_BIN=--add-binary "RTKCamSDK.dll;."
    echo RTKCamSDK.dll will be embedded.
) else (
    echo WARNING: RTKCamSDK.dll not found - the app will use the OpenCV camera path only.
)

echo.
echo ==== [5/6] Self-test and regression suite =================================
python app.py --selftest
if errorlevel 1 (
    echo WARNING: self-test reported problems ^(see selftest_report.txt^). Continuing the build.
)
if /I not "%SKIPTESTS%"=="1" (
    python tests\run_all.py
    if errorlevel 1 (
        echo ERROR: regression tests failed. Fix them or rebuild with /skiptests.
        exit /b 1
    )
)

echo.
echo ==== [6/6] PyInstaller ^(%MODE%^) ==========================================
if exist build rmdir /S /Q build
python -m PyInstaller --noconfirm --clean --log-level WARN ^
    --name %APP_NAME% ^
    --noconsole ^
    --%MODE% ^
    %ADD_BIN% ^
    --add-data "models;models" ^
    --add-data "README.md;." ^
    --hidden-import torch ^
    --hidden-import serial ^
    --hidden-import serial.tools.list_ports ^
    --hidden-import serial.win32 ^
    --hidden-import serial.serialwin32 ^
    --hidden-import PIL ^
    --hidden-import PIL.Image ^
    --hidden-import PIL.ImageDraw ^
    --hidden-import PIL.ImageFont ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import win32print ^
    --hidden-import win32api ^
    --hidden-import pywintypes ^
    --hidden-import arabic_reshaper ^
    --hidden-import bidi.algorithm ^
    --collect-data arabic_reshaper ^
    --exclude-module torchvision ^
    --exclude-module PyQt5 ^
    --exclude-module PySide6 ^
    --exclude-module PySide2 ^
    --exclude-module tkinter ^
    --exclude-module matplotlib ^
    --exclude-module IPython ^
    --exclude-module jupyter ^
    --exclude-module notebook ^
    --exclude-module pytest ^
    --exclude-module scipy ^
    --exclude-module pandas ^
    --exclude-module sympy ^
    --exclude-module onnx ^
    --exclude-module tensorboard ^
    app.py
if errorlevel 1 (
    echo ERROR: PyInstaller failed.
    exit /b 1
)

echo.
echo ==== DONE =================================================================
if "%MODE%"=="onefile" (
    echo Executable: %CD%\dist\%APP_NAME%.exe
) else (
    echo Executable: %CD%\dist\%APP_NAME%\%APP_NAME%.exe
    echo Copy the whole dist\%APP_NAME% folder to the POS machine.
)
echo config.json, items_db.pkl and the invoices folder are created next to the exe on first run.
echo Run "%APP_NAME%.exe --selftest" on the target machine to write selftest_report.txt.
endlocal
