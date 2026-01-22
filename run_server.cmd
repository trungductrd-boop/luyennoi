@echo off
SETLOCAL
if exist ".venv\Scripts\activate.bat" (
  call .venv\Scripts\activate.bat
)

if "%1"=="dev" (
  echo Starting uvicorn in dev mode (reload)...
  uvicorn main:app --reload --host 127.0.0.1 --port 8000
) else (
  echo Starting uvicorn (no-reload)...
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
)

ENDLOCAL
