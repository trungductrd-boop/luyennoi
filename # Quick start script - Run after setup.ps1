# Quick start script - Run after setup.ps1
# Run: powershell -ExecutionPolicy Bypass -File run.ps1

Write-Host "🚀 Starting Vietnamese Pronunciation API..." -ForegroundColor Cyan

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Check if dependencies installed
python -c "import fastapi" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Dependencies not found. Run setup.ps1 first:" -ForegroundColor Red
    Write-Host "   powershell -ExecutionPolicy Bypass -File setup.ps1" -ForegroundColor Yellow
    exit 1
}

# Run server
Write-Host "✅ Server starting at http://localhost:8000" -ForegroundColor Green
Write-Host "📚 Docs: http://localhost:8000/docs`n" -ForegroundColor Cyan

python -m uvicorn fastapi_server.main:app --reload --host 0.0.0.0 --port 8000
