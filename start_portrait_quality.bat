@echo off
setlocal
cd /d "%~dp0"
rem Optional illustration backbone and larger CPU background model.
rem The first launch downloads additional weights; existing SD 1.5 stays cached.
set THREADOPS_MODEL_ID=Lykon/dreamshaper-8
set THREADOPS_REMBG_MODEL=u2net
echo Portrait quality profile: DreamShaper 8 + U2Net CPU isolation.
echo Additional model downloads are required on first launch.
call start_demo.bat
exit /b %errorlevel%
