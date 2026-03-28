@echo off
title MindPlay Launcher
cd /d "e:\FYP_Models\FYP(4)\FYP-MindPlay"
set "STATE_FILE=e:\FYP_Models\FYP(4)\FYP-MindPlay\gamepad_state.json"

echo.
echo ============================================================
echo MindPlay Overlay + Gyro Detector Launcher
echo ============================================================
echo.
echo Starting Overlay...
start "Overlay" /min .\.venv\Scripts\python.exe .\scripts\gamepad_overlay.py --state-file "%STATE_FILE%" --follow-active-window

echo Waiting 2 seconds...
timeout /t 2 /nobreak

echo Starting Gyro Detector...
start "Gyro Detector" .\.venv\Scripts\python.exe .\scripts\gyro_detector.py ^
  --gyro-channels 5,6,7 ^
  --sfreq 500 ^
  --scale-factor 0.25 ^
  --use-z-for-lr ^
  --z-left-threshold 20 ^
  --z-right-threshold 20 ^
  --vel-forward 30 ^
  --vel-backward 30 ^
  --vel-return 120 ^
  --deadzone-z 15 ^
  --deadzone-y 20 ^
  --smoothing-window 14 ^
  --gamepad-mode ^
  --gamepad-repeat-interval 0.40 ^
  --output-keys ^
  --verbose ^
  --overlay-state-file "%STATE_FILE%"

echo.
echo Both windows are starting!
echo Close this window or press any key to continue...
pause
