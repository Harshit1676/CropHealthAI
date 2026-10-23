@echo off
title CropHealthAI - Backend
cd /d "%~dp0"

echo ========================================
echo       CropHealthAI Backend
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Starting FastAPI backend on port 8000...
echo Backend: http://127.0.0.1:8000
echo.
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

pause
