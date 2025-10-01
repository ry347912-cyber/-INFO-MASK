@echo off
echo ================================================
echo InfoMask - Steganography Web Application
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Starting the application...
echo The website will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo ------------------------------------------------
echo.

python app.py

pause 