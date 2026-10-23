@echo off
title CropHealthAI - Frontend
cd /d "%~dp0"

echo ========================================
echo       CropHealthAI Frontend
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Starting Streamlit frontend on port 8501...
echo Frontend: http://localhost:8501
echo.
python -m streamlit run frontend\Home.py --server.address 127.0.0.1 --server.port 8501

pause
