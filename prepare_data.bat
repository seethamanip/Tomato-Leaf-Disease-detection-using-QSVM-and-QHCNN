@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment in .venv ...
  py -3 -m venv .venv 2>nul
  if errorlevel 1 (
    python -m venv .venv
    if errorlevel 1 goto :error
  )
)

set "PY=%~dp0.venv\Scripts\python.exe"

echo [setup] Installing dependencies ...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Enter the path to your PlantVillage Tomato folder.
echo Example: C:\Users\You\Downloads\PlantVillage\Tomato
set /p SRC="PlantVillage Tomato path: "

if "%SRC%"=="" (
  echo [error] No path provided.
  pause
  exit /b 1
)

echo.
echo [data] Copying target classes into data\tomato ...
"%PY%" scripts\prepare_plantvillage.py --src "%SRC%" --overwrite
if errorlevel 1 goto :error

echo.
echo [data] Done. You can now run run_app.bat
pause
exit /b 0

:error
echo.
echo [error] Failed with exit code %errorlevel%.
pause
exit /b %errorlevel%

