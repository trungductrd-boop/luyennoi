# Run server script for project (PowerShell)
Param(
    [switch]$Dev
)

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
}

if ($Dev) {
    Write-Host "Starting uvicorn in dev mode (reload) from project directory..."
    uvicorn main:app --reload --host 127.0.0.1 --port 8000
} else {
    Write-Host "Starting uvicorn (no-reload) from project directory..."
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
}
