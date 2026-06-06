@echo off
setlocal

cd /d "%~dp0"

echo ========================================
echo Gartic OpenCV Drawer
echo ========================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found.
        echo Please install Python 3.10 or newer, then run this launcher again.
        echo https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )
)

echo Installing / updating dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to update pip.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    echo.
    pause
    exit /b 1
)

echo.
echo Starting Gartic OpenCV Drawer...
".venv\Scripts\python.exe" gartic_auto_drawer.py

if errorlevel 1 (
    echo.
    echo The app closed with an error.
    pause
)

endlocal
