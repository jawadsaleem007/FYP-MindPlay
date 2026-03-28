# MindPlay Overlay + Gyro Detector Launcher
# Runs both as admin in separate windows

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "MindPlay Overlay + Gyro Detector Launcher" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Paths
$ROOT_DIR = "e:\FYP_Models\FYP(4)\FYP-MindPlay"
$PYTHON_EXE = "$ROOT_DIR\.venv\Scripts\python.exe"
$OVERLAY_SCRIPT = "$ROOT_DIR\scripts\gamepad_overlay.py"
$GYRO_SCRIPT = "$ROOT_DIR\scripts\gyro_detector.py"
$STATE_FILE = "$ROOT_DIR\gamepad_state.json"

if (-not (Test-Path $PYTHON_EXE)) {
  Write-Error "Python not found at $PYTHON_EXE"
  exit 1
}
if (-not (Test-Path $OVERLAY_SCRIPT)) {
  Write-Error "Overlay script not found at $OVERLAY_SCRIPT"
  exit 1
}
if (-not (Test-Path $GYRO_SCRIPT)) {
  Write-Error "Gyro script not found at $GYRO_SCRIPT"
  exit 1
}

Write-Host "Checking overlay dependency (wxPython)..." -ForegroundColor Cyan
& $PYTHON_EXE -c "import wx" *>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "wxPython missing in launcher venv. Installing..." -ForegroundColor Yellow
  & $PYTHON_EXE -m pip install wxPython
  if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install wxPython in $PYTHON_EXE"
    exit 1
  }
}

Write-Host "Starting Overlay (Admin Window)..." -ForegroundColor Green
# Start Overlay as Admin in new window
Start-Process -Verb RunAs -FilePath $PYTHON_EXE `
  -ArgumentList @(
    $OVERLAY_SCRIPT,
    "--state-file", $STATE_FILE,
    "--follow-active-window"
  ) `
  -WorkingDirectory $ROOT_DIR `
  -WindowStyle Normal

Write-Host "Waiting 2 seconds for overlay to start..."
Start-Sleep -Seconds 2

Write-Host "Starting Gyro Detector (Admin Window)..." -ForegroundColor Green
# Start Gyro Detector as Admin in new window
Start-Process -Verb RunAs -FilePath $PYTHON_EXE `
  -ArgumentList @(
    $GYRO_SCRIPT,
    "--gyro-channels", "5,6,7",
    "--sfreq", "500",
    "--scale-factor", "0.25",
    "--use-z-for-lr",
    "--z-left-threshold", "20",
    "--z-right-threshold", "20",
    "--vel-forward", "30",
    "--vel-backward", "30",
    "--vel-return", "120",
    "--deadzone-z", "15",
    "--deadzone-y", "20",
    "--smoothing-window", "14",
    "--gamepad-mode",
    "--gamepad-repeat-interval", "0.40",
    "--output-keys",
    "--verbose",
    "--overlay-state-file", $STATE_FILE
  ) `
  -WorkingDirectory $ROOT_DIR `
  -WindowStyle Normal

Write-Host ""
Write-Host "Both windows started as admin!" -ForegroundColor Green
Write-Host "Press Ctrl+C in each window to stop." -ForegroundColor Yellow
Write-Host ""

