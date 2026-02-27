@echo off
REM FLUX.2 Windows Deployment Script using NSSM
REM Requires: NSSM (Non-Sucking Service Manager) installed
REM Usage: Run as Administrator

setlocal enabledelayedexpansion

set FLUX2_ROOT=C:\ai\flux2
set FLUX2_VENV=%FLUX2_ROOT%\.venv
set NSSM_PATH=C:\Program Files\nssm\win64\nssm.exe

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator
    exit /b 1
)

REM Check if NSSM is installed
if not exist "%NSSM_PATH%" (
    echo ERROR: NSSM not found at %NSSM_PATH%
    echo Please install NSSM from: https://nssm.cc/download
    exit /b 1
)

echo ============================================================
echo FLUX.2 Windows Service Deployment
echo ============================================================

REM ============================================================
REM UI WORKER 1
REM ============================================================
echo.
echo [1/5] Installing FLUX2-UI-Worker-1...
%NSSM_PATH% install FLUX2-UI-Worker-1 "%FLUX2_VENV%\Scripts\streamlit.exe" ^
    "run ui_flux2_professional.py --server.port=8501 --server.address=127.0.0.1"
%NSSM_PATH% set FLUX2-UI-Worker-1 AppDirectory "%FLUX2_ROOT%"
%NSSM_PATH% set FLUX2-UI-Worker-1 AppEnvironmentExtra CUDA_VISIBLE_DEVICES=0
%NSSM_PATH% set FLUX2-UI-Worker-1 AppNoConsole 0
%NSSM_PATH% set FLUX2-UI-Worker-1 AppPriority HIGH
%NSSM_PATH% set FLUX2-UI-Worker-1 AppExit Default Restart
%NSSM_PATH% start FLUX2-UI-Worker-1
echo FLUX2-UI-Worker-1 installed and started

REM ============================================================
REM UI WORKER 2
REM ============================================================
echo.
echo [2/5] Installing FLUX2-UI-Worker-2...
%NSSM_PATH% install FLUX2-UI-Worker-2 "%FLUX2_VENV%\Scripts\streamlit.exe" ^
    "run ui_flux2_professional.py --server.port=8502 --server.address=127.0.0.1"
%NSSM_PATH% set FLUX2-UI-Worker-2 AppDirectory "%FLUX2_ROOT%"
%NSSM_PATH% set FLUX2-UI-Worker-2 AppEnvironmentExtra CUDA_VISIBLE_DEVICES=0
%NSSM_PATH% set FLUX2-UI-Worker-2 AppNoConsole 0
%NSSM_PATH% set FLUX2-UI-Worker-2 AppPriority HIGH
%NSSM_PATH% set FLUX2-UI-Worker-2 AppExit Default Restart
%NSSM_PATH% start FLUX2-UI-Worker-2
echo FLUX2-UI-Worker-2 installed and started

REM ============================================================
REM UI WORKER 3
REM ============================================================
echo.
echo [3/5] Installing FLUX2-UI-Worker-3...
%NSSM_PATH% install FLUX2-UI-Worker-3 "%FLUX2_VENV%\Scripts\streamlit.exe" ^
    "run ui_flux2_professional.py --server.port=8503 --server.address=127.0.0.1"
%NSSM_PATH% set FLUX2-UI-Worker-3 AppDirectory "%FLUX2_ROOT%"
%NSSM_PATH% set FLUX2-UI-Worker-3 AppEnvironmentExtra CUDA_VISIBLE_DEVICES=0
%NSSM_PATH% set FLUX2-UI-Worker-3 AppNoConsole 0
%NSSM_PATH% set FLUX2-UI-Worker-3 AppPriority HIGH
%NSSM_PATH% set FLUX2-UI-Worker-3 AppExit Default Restart
%NSSM_PATH% start FLUX2-UI-Worker-3
echo FLUX2-UI-Worker-3 installed and started

REM ============================================================
REM MODEL WORKER 4B
REM ============================================================
echo.
echo [4/5] Installing FLUX2-Model-Worker-4B...
%NSSM_PATH% install FLUX2-Model-Worker-4B "%FLUX2_VENV%\Scripts\python.exe" ^
    "scripts\model_worker.py"
%NSSM_PATH% set FLUX2-Model-Worker-4B AppDirectory "%FLUX2_ROOT%"
%NSSM_PATH% set FLUX2-Model-Worker-4B AppEnvironmentExtra MODEL_WORKER_HOST=127.0.0.1
%NSSM_PATH% set FLUX2-Model-Worker-4B AppEnvironmentExtra MODEL_WORKER_PORT=8600
%NSSM_PATH% set FLUX2-Model-Worker-4B AppEnvironmentExtra MODEL_SERVICE_NAME=klein-4b-service
%NSSM_PATH% set FLUX2-Model-Worker-4B AppEnvironmentExtra MODEL_KEY=flux.2-klein-4b
%NSSM_PATH% set FLUX2-Model-Worker-4B AppEnvironmentExtra CUDA_VISIBLE_DEVICES=1
%NSSM_PATH% set FLUX2-Model-Worker-4B AppNoConsole 0
%NSSM_PATH% set FLUX2-Model-Worker-4B AppPriority HIGH
%NSSM_PATH% set FLUX2-Model-Worker-4B AppExit Default Restart
%NSSM_PATH% start FLUX2-Model-Worker-4B
echo FLUX2-Model-Worker-4B installed and started

REM ============================================================
REM MODEL WORKER 9B
REM ============================================================
echo.
echo [5/5] Installing FLUX2-Model-Worker-9B...
%NSSM_PATH% install FLUX2-Model-Worker-9B "%FLUX2_VENV%\Scripts\python.exe" ^
    "scripts\model_worker.py"
%NSSM_PATH% set FLUX2-Model-Worker-9B AppDirectory "%FLUX2_ROOT%"
%NSSM_PATH% set FLUX2-Model-Worker-9B AppEnvironmentExtra MODEL_WORKER_HOST=127.0.0.1
%NSSM_PATH% set FLUX2-Model-Worker-9B AppEnvironmentExtra MODEL_WORKER_PORT=8601
%NSSM_PATH% set FLUX2-Model-Worker-9B AppEnvironmentExtra MODEL_SERVICE_NAME=klein-9b-service
%NSSM_PATH% set FLUX2-Model-Worker-9B AppEnvironmentExtra MODEL_KEY=flux.2-klein-9b
%NSSM_PATH% set FLUX2-Model-Worker-9B AppEnvironmentExtra CUDA_VISIBLE_DEVICES=2
%NSSM_PATH% set FLUX2-Model-Worker-9B AppNoConsole 0
%NSSM_PATH% set FLUX2-Model-Worker-9B AppPriority HIGH
%NSSM_PATH% set FLUX2-Model-Worker-9B AppExit Default Restart
%NSSM_PATH% start FLUX2-Model-Worker-9B
echo FLUX2-Model-Worker-9B installed and started

echo.
echo ============================================================
echo Services Installed Successfully!
echo ============================================================
echo.
echo Installed Services:
echo   - FLUX2-UI-Worker-1 (port 8501)
echo   - FLUX2-UI-Worker-2 (port 8502)
echo   - FLUX2-UI-Worker-3 (port 8503)
echo   - FLUX2-Model-Worker-4B (port 8600)
echo   - FLUX2-Model-Worker-9B (port 8601)
echo.
echo Management Commands:
echo   sc query | findstr FLUX2
echo   net start FLUX2-UI-Worker-1
echo   net stop FLUX2-UI-Worker-1
echo   %NSSM_PATH% status FLUX2-UI-Worker-1
echo.
pause
