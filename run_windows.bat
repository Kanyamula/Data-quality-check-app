@echo off
REM Double-click this file, or run it from Command Prompt with: run_windows.bat
REM Sets up a virtual environment (first run only), installs dependencies,
REM and starts the app in your browser.

cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not find "python". Install Python 3.9+ from https://python.org
        echo and make sure "Add python.exe to PATH" is checked during setup.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

echo Installing/updating dependencies (first run may take a few minutes)...
pip install -r requirements.txt

echo.
echo Starting the app... Streamlit will open it in your browser automatically.
echo Press Ctrl+C in this window to stop the app.
echo.
streamlit run app.py

pause
