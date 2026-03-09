@echo off
echo === Annotty-MRI Server ===
echo.

cd /d "%~dp0"

echo Starting FastAPI server on http://localhost:8000 ...
start "Annotty-MRI Server" cmd /k ".venv\Scripts\python server\main.py"

timeout /t 3 /nobreak >nul

echo Starting Cloudflare Tunnel ...
cloudflared tunnel --url http://localhost:8000

pause
