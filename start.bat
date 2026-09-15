@echo off
setlocal EnableDelayedExpansion

REM SenseNova U1.5 Lite Image Generation Service Launcher
REM Usage: start.bat [port] (default 8000)

set PORT=8000
if not "%~1"=="" set PORT=%~1

echo.
echo ========================================
echo  SenseNova U1.5 Lite Image Generation
echo ========================================
echo.
echo  Port: !PORT!
echo  URL:  http://127.0.0.1:!PORT!
echo  API:  http://127.0.0.1:!PORT!/docs
echo.
echo  Press Ctrl+C to stop
echo ========================================
echo.

REM Check dependencies
python -c "import fastapi, uvicorn, requests" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt -q
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation failed
        pause
        exit /b 1
    )
)

REM Start server
python -m uvicorn app:app --host 0.0.0.0 --port !PORT!
