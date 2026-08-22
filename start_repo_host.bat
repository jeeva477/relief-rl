@echo off
setlocal
cd /d "D:\relief-rl-gh"
title Relief-RL Repo Host

set LOGFILE=D:\relief-rl-gh\server.log
if not exist "D:\relief-rl-gh" mkdir "D:\relief-rl-gh" >nul 2>&1

echo [%date% %time%] Restarting Relief-RL repo host... >> "%LOGFILE%"

for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
  taskkill /F /PID %%p >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM --- environment ---
set DEMO_MODE=true
set DATABASE_URL=sqlite:///./relief_rl.db
set MODEL_PATH=rl/checkpoints/best_model.pt
set CORS_ORIGINS=*
set AUTH_SECRET=relief-rl-demo-2026-secret
set ADMIN_EMAIL=admin@example.com
set ADMIN_PASSWORD=admin123
set LOG_LEVEL=INFO

echo [%date% %time%] Starting uvicorn on 0.0.0.0:8000 >> "%LOGFILE%"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 >> "%LOGFILE%" 2>&1