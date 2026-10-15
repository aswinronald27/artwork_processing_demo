@echo off
setlocal
cd /d "%~dp0"
set THREADOPS_MODEL_FAMILY=sdxl
set THREADOPS_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
set THREADOPS_REMBG_MODEL=u2net
set THREADOPS_LORA_PATH=
echo Local SDXL portrait profile: 768px, compact Canny ControlNet and flat illustration LoRA.
call start_demo.bat
exit /b %errorlevel%
