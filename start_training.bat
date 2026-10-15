@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Run start_demo.bat to install the environment first.
    pause
    exit /b 1
)
echo ThreadOps local LoRA training: 50 artworks, 1000 steps, rank 8.
echo Checkpoints save every 100 steps. Close the inference app to free GPU memory.
echo Progress: training\checkpoints\threadops-flat\status.json
".venv\Scripts\python.exe" -u training\train_style.py --resume
if errorlevel 1 pause
