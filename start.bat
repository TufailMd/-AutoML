@echo off
title AutoML Cloud Platform Launcher
echo ====================================================
echo   AutoML Cloud Platform - Unified Startup Control
echo ====================================================
echo.

echo 1. Launching FastAPI Backend Server...
start "AutoML Backend Service" cmd /k "cd backend && venv\Scripts\python run.py"

echo 2. Launching Vite React Frontend...
start "AutoML Frontend Web" cmd /k "cd frontend && npm run dev"

echo.
echo Launching default web browser at http://localhost:5173 ...
timeout /t 3 /nobreak > nul
start http://localhost:5173

echo.
echo Both services launched successfully! Keep this window open
echo to review launcher console or close it to exit launch telemetry.
echo ====================================================
pause
