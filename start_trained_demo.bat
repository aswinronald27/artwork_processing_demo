@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" launch_trained.py
if errorlevel 1 pause
