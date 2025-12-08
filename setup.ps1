# Vietnamese Pronunciation Learning API - Auto Setup Script
# Run: powershell -ExecutionPolicy Bypass -File setup.ps1

Write-Host "🚀 Vietnamese Pronunciation Learning API - Auto Setup" -ForegroundColor Cyan
Write-Host "=" * 60

# 1. Check Python
Write-Host "`n[1/8] Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# 2. Create virtual environment
Write-Host "`n[2/8] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# 3. Activate venv
Write-Host "`n[3/8] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# 4. Upgrade pip
Write-Host "`n[4/8] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✅ Pip upgraded" -ForegroundColor Green

# 5. Install dependencies
Write-Host "`n[5/8] Installing dependencies..." -ForegroundColor Yellow
Write-Host "   This may take a few minutes..." -ForegroundColor Gray

# Install in order to avoid conflicts
pip install --quiet fastapi uvicorn[standard] starlette pydantic python-multipart
pip install --quiet numpy scipy soundfile
pip install --quiet librosa scikit-learn numba requests

Write-Host "✅ Dependencies installed" -ForegroundColor Green

# 6. Create directories
Write-Host "`n[6/8] Creating data directories..." -ForegroundColor Yellow
$dirs = @("data/samples", "data/tmp", "data/users", "logs", "static")
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}
Write-Host "✅ Directories created" -ForegroundColor Green

# 7. Cleanup
Write-Host "`n[7/8] Cleaning up..." -ForegroundColor Yellow
Remove-Item "fastapi_server\.txt" -ErrorAction SilentlyContinue
Remove-Item "fastapi_server\requirements.txt" -ErrorAction SilentlyContinue
Remove-Item "fastapi_server\index.html" -ErrorAction SilentlyContinue
Write-Host "✅ Cleanup completed" -ForegroundColor Green

# 8. Start server
Write-Host "`n[8/8] Starting FastAPI server..." -ForegroundColor Yellow
Write-Host "`n" + "=" * 60
Write-Host "🎤 Server starting at http://localhost:8000" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "💚 Health Check: http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host "=" * 60 + "`n"

# Run server
python -m uvicorn fastapi_server.main:app --reload --host 0.0.0.0 --port 8000
