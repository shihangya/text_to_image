# SenseNova U1.5 Lite Image Generation - Launcher
# Usage: .\start.ps1 [port] (default 8000)

$Port = 8000
if ($args.Count -gt 0) {
    $Port = [int]$args[0]
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " SenseNova U1.5 Lite Image Generation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Port: $Port"
Write-Host "  URL:  http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "  API:  http://127.0.0.1:$Port/docs" -ForegroundColor Green
Write-Host ""
Write-Host "  Press Ctrl+C to stop"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check dependencies
python -c "import fastapi, uvicorn, requests" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Dependency installation failed" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Start server
python -m uvicorn app:app --host 0.0.0.0 --port $Port
