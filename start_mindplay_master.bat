@echo off
setlocal

set "SCRIPT=e:\FYP_Models\FYP(4)\FYP-MindPlay\start_mindplay_master.ps1"

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
exit /b %ERRORLEVEL%
