@echo off
REM Quick Start Script for Disease Prediction System with Authentication (Windows)

echo ================================================================
echo  Disease Prediction System - Quick Start
echo ================================================================
echo.
echo This script will help you set up the authentication system
echo.

REM Check if .env exists
if exist ".env" (
    echo [32menv file found[0m
) else (
    echo [33m.env file not found[0m
    echo.
    echo You need to create a .env file with your database URL
    echo.
    echo Quick options:
    echo 1. Run: python setup_auth.py (interactive setup)
    echo 2. Copy .env.example to .env and edit it manually
    echo.
    set /p response="Run interactive setup now? (y/N): "
    if /i "%response%"=="y" (
        python setup_auth.py
    ) else (
        echo Please create .env file manually
        exit /b 1
    )
)

echo.
echo Checking dependencies...

REM Check if required packages are installed
python -c "import bcrypt, pymongo, dotenv, sqlalchemy" 2>nul
if %errorlevel% equ 0 (
    echo [32mAll dependencies installed[0m
) else (
    echo [33mSome dependencies missing[0m
    set /p response="Install dependencies now? (Y/n): "
    if /i not "%response%"=="n" (
        pip install -r requirements.txt
    )
)

echo.
echo ================================================================
echo Starting Application...
echo ================================================================
echo.
echo The application will open in your browser
echo.
echo Quick Guide:
echo    1. Click 'Register' to create an account
echo    2. Fill in the registration form
echo    3. Login with your credentials
echo    4. Access disease prediction modules from the sidebar
echo.
echo ================================================================
echo.

REM Start Streamlit
streamlit run Home.py
