@echo off
REM Disease Prediction System - MongoDB Edition
REM Quick Start Batch File for Windows

echo.
echo ====================================================================
echo    Disease Prediction System - MongoDB Authentication
echo ====================================================================
echo.

REM Check if virtual environment exists
if exist "env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call env\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python...
)

REM Check if secrets.toml exists
if not exist ".streamlit\secrets.toml" (
    echo.
    echo [WARNING] MongoDB configuration not found!
    echo.
    echo Please run setup first:
    echo    python setup_mongodb.py
    echo.
    echo Or create .streamlit\secrets.toml manually
    echo.
    pause
    exit /b 1
)

REM Run the application
echo.
echo Starting application...
echo.
echo Once loaded, open your browser to: http://localhost:8501
echo.
echo Demo Login:
echo   Username: demo_user
echo   Password: demo123
echo.
echo Press Ctrl+C to stop the application
echo.
echo ====================================================================
echo.

streamlit run app_mongodb.py

pause

