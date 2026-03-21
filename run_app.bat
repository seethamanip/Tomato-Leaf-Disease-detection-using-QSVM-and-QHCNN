@echo off
setlocal

REM Always work from the project root (this folder)
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "PY=%VENV_DIR%\Scripts\python.exe"

REM Create virtual environment if it does not exist
if not exist "%VENV_DIR%" (
  echo [setup] Creating virtual environment in .venv ...
  py -3 -m venv "%VENV_DIR%" 2>nul
  if errorlevel 1 (
    python -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
  )
)

echo [setup] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo [setup] Installing / updating dependencies from requirements.txt ...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

REM Ensure artifacts directory exists
set "ARTIFACTS_DIR=%~dp0artifacts"
set "MODEL_PATH=%ARTIFACTS_DIR%\model.joblib"

if not exist "%ARTIFACTS_DIR%" (
  mkdir "%ARTIFACTS_DIR%"
)

echo.
echo [info] When the server starts, the app will be at: http://127.0.0.1:5000
echo [model] Training now with latest Python code on the full dataset. This can take several minutes...
set "PYTHONPATH=%~dp0"
"%PY%" run_training.py
if errorlevel 1 goto :error

echo.
echo [server] Starting Flask app (HTML/CSS/JS frontend, no Streamlit) ...
echo [server] Once started, open: http://127.0.0.1:5000
start "" http://127.0.0.1:5000

"%PY%" server.py
goto :eof

:error
echo.
echo [error] Setup or training failed with exit code %errorlevel%.
pause
exit /b %errorlevel%

