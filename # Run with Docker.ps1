# Run with Docker
# Run: powershell -ExecutionPolicy Bypass -File docker-run.ps1

Write-Host "🐳 Vietnamese Pronunciation API - Docker Mode" -ForegroundColor Cyan
Write-Host "=" * 60

# Check Docker
docker --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker not found. Please install Docker Desktop" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker found" -ForegroundColor Green

# Build or run
$choice = Read-Host "`n[1] Build new image`n[2] Run existing image`n[3] Build and run`nChoose (1-3)"

switch ($choice) {
    "1" {
        Write-Host "`n🔨 Building Docker image..." -ForegroundColor Yellow
        docker build -t vietnamese-pronunciation-api .
        Write-Host "✅ Build complete" -ForegroundColor Green
    }
    "2" {
        Write-Host "`n🚀 Starting container..." -ForegroundColor Yellow
        docker-compose up -d
        Write-Host "✅ Container started at http://localhost:8000" -ForegroundColor Green
    }
    "3" {
        Write-Host "`n🔨 Building and starting..." -ForegroundColor Yellow
        docker-compose up -d --build
        Write-Host "✅ Container running at http://localhost:8000" -ForegroundColor Green
    }
    default {
        Write-Host "❌ Invalid choice" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "💚 Health: http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "`nTo stop: docker-compose down" -ForegroundColor Gray
