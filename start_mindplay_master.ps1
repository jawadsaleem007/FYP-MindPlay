param(
  [string]$ModelPath = "",
  [switch]$NoOverlayFollow
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = New-Object Security.Principal.WindowsPrincipal($identity)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Quote-Single([string]$value) {
  return "'" + $value.Replace("'", "''") + "'"
}

function Start-PythonComponent {
  param(
    [string]$Title,
    [string]$PythonExe,
    [string]$ScriptPath,
    [string[]]$Args,
    [string]$WorkingDirectory
  )

  $quotedRoot = Quote-Single $WorkingDirectory
  $quotedPython = Quote-Single $PythonExe
  $quotedScript = Quote-Single $ScriptPath
  $quotedArgs = @()
  foreach ($arg in $Args) {
    $quotedArgs += (Quote-Single $arg)
  }
  $joinedArgs = ($quotedArgs -join " ")

  $safeTitle = $Title.Replace("'", "''")
  $command = "`$host.UI.RawUI.WindowTitle = '$safeTitle'; Set-Location $quotedRoot; & $quotedPython $quotedScript $joinedArgs"

  return Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $command) `
    -WorkingDirectory $WorkingDirectory `
    -WindowStyle Normal `
    -PassThru
}

if (-not (Test-IsAdmin)) {
  Write-Host "Requesting administrator privileges (UAC)..." -ForegroundColor Yellow

  $selfPath = $PSCommandPath
  $elevateArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$selfPath`"")
  if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    $elevateArgs += @("-ModelPath", "`"$ModelPath`"")
  }
  if ($NoOverlayFollow) {
    $elevateArgs += "-NoOverlayFollow"
  }

  Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $elevateArgs
  exit 0
}

$ROOT_DIR = Split-Path -Parent $PSCommandPath
$PARENT_DIR = Split-Path -Parent $ROOT_DIR

$pythonCandidates = @(
  (Join-Path $ROOT_DIR ".venv\Scripts\python.exe"),
  (Join-Path $PARENT_DIR ".venv\Scripts\python.exe")
)

$PYTHON_EXE = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $PYTHON_EXE) {
  throw "Python executable not found. Checked: $($pythonCandidates -join ', ')"
}

$OVERLAY_SCRIPT = Join-Path $ROOT_DIR "scripts\gamepad_overlay.py"
$GYRO_SCRIPT = Join-Path $ROOT_DIR "scripts\gyro_detector.py"
$BLINK_SCRIPT = Join-Path $ROOT_DIR "scripts\blink_detector.py"
$CLASSIFIER_SCRIPT = Join-Path $ROOT_DIR "scripts\real_time_classifier.py"
$STATE_FILE = Join-Path $ROOT_DIR "gamepad_state.json"

$requiredScripts = @($OVERLAY_SCRIPT, $GYRO_SCRIPT, $BLINK_SCRIPT, $CLASSIFIER_SCRIPT)
foreach ($scriptPath in $requiredScripts) {
  if (-not (Test-Path $scriptPath)) {
    throw "Required script not found: $scriptPath"
  }
}

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
  $defaultModel = Join-Path $ROOT_DIR "fbcsp_lda.joblib"
  if (Test-Path $defaultModel) {
    $ModelPath = $defaultModel
  }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "MindPlay Master Launcher (Admin)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Project: $ROOT_DIR"
Write-Host "Python:  $PYTHON_EXE"
if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
  Write-Host "Model:   $ModelPath"
} else {
  Write-Host "Model:   <not provided; real_time_classifier will use its default>" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "Checking required modules..." -ForegroundColor Cyan
$moduleToPackage = @{
  "wx" = "wxPython"
  "numpy" = "numpy"
  "pylsl" = "pylsl"
  "scipy" = "scipy"
}
$missingPackages = @()
foreach ($module in $moduleToPackage.Keys) {
  & $PYTHON_EXE -c "import $module" *>$null
  if ($LASTEXITCODE -ne 0) {
    $missingPackages += $moduleToPackage[$module]
  }
}

if ($missingPackages.Count -gt 0) {
  $missingPackages = $missingPackages | Select-Object -Unique
  Write-Host "Installing missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
  foreach ($package in $missingPackages) {
    & $PYTHON_EXE -m pip install $package
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to install required package: $package"
    }
  }
}

$overlayArgs = @("--state-file", $STATE_FILE)
if (-not $NoOverlayFollow) {
  $overlayArgs += "--follow-active-window"
}

$gyroArgs = @(
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
)

$blinkArgs = @(
  "--sfreq", "500",
  "--picks", "Fp1,Fp2",
  "--window", "0.5",
  "--threshold-uv", "80",
  "--refractory", "0.8"
)

$classifierArgs = @(
  "--sfreq", "500",
  "--window", "3.0",
  "--step", "0.5",
  "--picks", "Cz,C3,C4",
  "--vote-k", "5",
  "--class-names", "0:rest,1:hand_mi",
  "--hand-mi-threshold", "0.97",
  "--hand-mi-consecutive", "3"
)
if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
  if (-not (Test-Path $ModelPath)) {
    throw "Provided model path does not exist: $ModelPath"
  }
  $classifierArgs += @("--model", $ModelPath)
}

Write-Host "Starting overlay..." -ForegroundColor Green
$overlayProc = Start-PythonComponent -Title "MindPlay Overlay" -PythonExe $PYTHON_EXE -ScriptPath $OVERLAY_SCRIPT -Args $overlayArgs -WorkingDirectory $ROOT_DIR
Start-Sleep -Seconds 1

Write-Host "Starting gyro detector..." -ForegroundColor Green
$gyroProc = Start-PythonComponent -Title "MindPlay Gyro Detector" -PythonExe $PYTHON_EXE -ScriptPath $GYRO_SCRIPT -Args $gyroArgs -WorkingDirectory $ROOT_DIR
Start-Sleep -Seconds 1

Write-Host "Starting blink detector..." -ForegroundColor Green
$blinkProc = Start-PythonComponent -Title "MindPlay Blink Detector" -PythonExe $PYTHON_EXE -ScriptPath $BLINK_SCRIPT -Args $blinkArgs -WorkingDirectory $ROOT_DIR
Start-Sleep -Seconds 1

Write-Host "Starting real-time classifier..." -ForegroundColor Green
$classifierProc = Start-PythonComponent -Title "MindPlay Real-Time Classifier" -PythonExe $PYTHON_EXE -ScriptPath $CLASSIFIER_SCRIPT -Args $classifierArgs -WorkingDirectory $ROOT_DIR

Write-Host ""
Write-Host "All MindPlay components launched in admin mode." -ForegroundColor Green
Write-Host "Overlay PID:    $($overlayProc.Id)"
Write-Host "Gyro PID:       $($gyroProc.Id)"
Write-Host "Blink PID:      $($blinkProc.Id)"
Write-Host "Classifier PID: $($classifierProc.Id)"
Write-Host ""
Write-Host "Classifier rule active: hand_mi only when >97% for 3 consecutive windows; otherwise rest(0)." -ForegroundColor Cyan
Write-Host "Use Ctrl+C in each component window to stop it." -ForegroundColor Yellow
