@echo off
title CropHealthAI - Full Application
cd /d "%~dp0"

echo ========================================
echo       CropHealthAI Application
echo ========================================
echo.
echo Backend : http://127.0.0.1:8000
echo Frontend: http://localhost:8501
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found.
    pause
    exit /b 1
)

start "CropHealthAI Backend" cmd /k call "%~dp0backend.bat"
timeout /t 3 /nobreak >nul
start "CropHealthAI Frontend" cmd /k call "%~dp0frontend.bat"

echo Both services started.
echo Open http://localhost:8501
echo.
pause
