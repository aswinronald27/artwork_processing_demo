@echo off
setlocal
cd /d "%~dp0"
title ThreadOps AI Art Studio (Local GPU)
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_DISABLE_TELEMETRY=1
echo ThreadOps AI Art Studio - http://127.0.0.1:7860
echo.
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.12 virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto :failed
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :failed
python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'"
if errorlevel 1 goto :failed
rem Reinstall only when the requirements content has changed.
if not exist ".venv\threadops-requirements.txt" goto :install
fc /b requirements.txt .venv\threadops-requirements.txt >nul 2>&1
if errorlevel 1 goto :install
goto :launch
:install
echo Installing pinned dependencies, including CUDA 12.4 PyTorch...
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
copy /y requirements.txt .venv\threadops-requirements.txt >nul
:launch
python -c "import torch; print('PyTorch:', torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver and CUDA PyTorch installation'; print('GPU:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto :failed
echo Preloading FP16 models before starting the web server...
python preload_models.py
if errorlevel 1 goto :failed
echo Launching ThreadOps at http://127.0.0.1:7860
python app.py
if errorlevel 1 goto :failed
exit /b 0
:failed
echo.
echo Studio could not start. Review the error above.
pause
exit /b 1
