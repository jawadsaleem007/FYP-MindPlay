param(
  [string]$ModelPath = "",
  [switch]$NoOverlayFollow,
  [double]$CommandCooldownSeconds = 2.0,
  [string]$StatusFile = ""
)

$ErrorActionPreference = "Stop"

$script:StatusPath = ""
if (-not [string]::IsNullOrWhiteSpace($StatusFile)) {
  try {
    $script:StatusPath = [System.IO.Path]::GetFullPath($StatusFile)
  } catch {
    $script:StatusPath = $StatusFile
  }
}

$script:LaunchStatus = [ordered]@{
  phase = "init"
  message = "Initializing launcher"
  admin = $false
  updated_at = (Get-Date).ToString("o")
  components = [ordered]@{
    overlay = [ordered]@{ state = "pending"; pid = 0; message = "" }
    gyro = [ordered]@{ state = "pending"; pid = 0; message = "" }
    blink = [ordered]@{ state = "pending"; pid = 0; message = "" }
    classifier = [ordered]@{ state = "pending"; pid = 0; message = "" }
  }
}

function Test-IsAdmin {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = New-Object Security.Principal.WindowsPrincipal($identity)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function ConvertTo-SingleQuotedArgument([string]$value) {
  return "'" + $value.Replace("'", "''") + "'"
}

function Test-PythonModule {
  param(
    [string]$PythonExe,
    [string]$ModuleName
  )

  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $PythonExe
  $psi.Arguments = "-c `"import $ModuleName`""
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.CreateNoWindow = $true

  $process = [System.Diagnostics.Process]::Start($psi)
  $process.WaitForExit()
  return $process.ExitCode
}

function Install-PythonPackage {
  param(
    [string]$PythonExe,
    [string]$PackageName,
    [string]$WorkingDirectory
  )

  $process = Start-Process -FilePath $PythonExe `
    -ArgumentList @("-m", "pip", "install", $PackageName) `
    -WorkingDirectory $WorkingDirectory `
    -NoNewWindow `
    -Wait `
    -PassThru
  return $process.ExitCode
}

function Write-StatusFile {
  if ([string]::IsNullOrWhiteSpace($script:StatusPath)) {
    return
  }
  try {
    $script:LaunchStatus.updated_at = (Get-Date).ToString("o")
    $statusDir = Split-Path -Parent $script:StatusPath
    if (-not [string]::IsNullOrWhiteSpace($statusDir)) {
      New-Item -ItemType Directory -Force -Path $statusDir *>$null
    }
    $jsonText = ($script:LaunchStatus | ConvertTo-Json -Depth 8)
    $tmpPath = "$($script:StatusPath).tmp"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($tmpPath, $jsonText, $utf8NoBom)
    Move-Item -Path $tmpPath -Destination $script:StatusPath -Force
  } catch {
    # Keep launcher running even if status file write fails.
  }
}

function Set-LauncherPhase([string]$phase, [string]$message) {
  $script:LaunchStatus.phase = $phase
  $script:LaunchStatus.message = $message
  Write-StatusFile
}

function Set-ComponentState([string]$name, [string]$state, [int]$processId = 0, [string]$message = "") {
  if (-not $script:LaunchStatus.components.Contains($name)) {
    return
  }
  $script:LaunchStatus.components[$name].state = $state
  $script:LaunchStatus.components[$name].pid = $processId
  $script:LaunchStatus.components[$name].message = $message
  Write-StatusFile
}

function Resolve-ModelPath {
  param(
    [string]$RootDir,
    [string]$RequestedPath
  )

  if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
    $candidate = $RequestedPath
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
      $candidate = Join-Path $RootDir $candidate
    }
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }

    $leaf = [System.IO.Path]::GetFileName($RequestedPath)
    if ($leaf -ieq "fbcsp_lda.joblib") {
      $latest = Get-ChildItem -Path $RootDir -Filter "fbcsp_lda*.joblib" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
      if ($latest) {
        return $latest.FullName
      }
      throw "Model fbcsp_lda.joblib was requested but no matching model exists in project root. Train a model first or pass -ModelPath to an existing .joblib file."
    }

    throw "Provided model path does not exist: $candidate"
  }

  $defaultModel = Join-Path $RootDir "fbcsp_lda.joblib"
  if (Test-Path $defaultModel) {
    return (Resolve-Path $defaultModel).Path
  }

  $latestFallback = Get-ChildItem -Path $RootDir -Filter "fbcsp_lda*.joblib" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($latestFallback) {
    return $latestFallback.FullName
  }

  throw "No model file found in project root. Expected fbcsp_lda.joblib or fbcsp_lda_*.joblib. Train a model first or pass -ModelPath to an existing .joblib file."
}

function Start-PythonComponent {
  param(
    [string]$Title,
    [string]$PythonExe,
    [string]$ScriptPath,
    [string[]]$ComponentArgs,
    [string]$WorkingDirectory
  )

  $quotedRoot = ConvertTo-SingleQuotedArgument $WorkingDirectory
  $quotedPython = ConvertTo-SingleQuotedArgument $PythonExe
  $quotedScript = ConvertTo-SingleQuotedArgument $ScriptPath
  $quotedArgs = @()
  foreach ($arg in $ComponentArgs) {
    $quotedArgs += (ConvertTo-SingleQuotedArgument $arg)
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
  Set-LauncherPhase "elevating" "Requesting administrator privileges (UAC)"
  Write-Host "Requesting administrator privileges (UAC)..." -ForegroundColor Yellow

  $selfPath = $PSCommandPath
  $elevateArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$selfPath`"")
  if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    $elevateArgs += @("-ModelPath", "`"$ModelPath`"")
  }
  if ($NoOverlayFollow) {
    $elevateArgs += "-NoOverlayFollow"
  }
  $elevateArgs += @("-CommandCooldownSeconds", "$CommandCooldownSeconds")
  if (-not [string]::IsNullOrWhiteSpace($StatusFile)) {
    $elevateArgs += @("-StatusFile", "`"$StatusFile`"")
  }

  Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $elevateArgs
  exit 0
}

$script:LaunchStatus.admin = $true
Set-LauncherPhase "admin" "Running as administrator"

try {
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

  $ModelPath = Resolve-ModelPath -RootDir $ROOT_DIR -RequestedPath $ModelPath

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
  Set-LauncherPhase "checking_dependencies" "Checking required modules"

  Write-Host "Checking required modules..." -ForegroundColor Cyan
  $moduleToPackage = @{
    "wx" = "wxPython"
    "numpy" = "numpy"
    "pylsl" = "pylsl"
    "scipy" = "scipy"
  }
  $missingPackages = @()
  foreach ($module in $moduleToPackage.Keys) {
    $moduleExitCode = Test-PythonModule -PythonExe $PYTHON_EXE -ModuleName $module
    if ($moduleExitCode -ne 0) {
      $missingPackages += $moduleToPackage[$module]
    }
  }

  if ($missingPackages.Count -gt 0) {
    $missingPackages = $missingPackages | Select-Object -Unique
    Set-LauncherPhase "installing" "Installing missing packages: $($missingPackages -join ', ')"
    Write-Host "Installing missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
    foreach ($package in $missingPackages) {
      $installExitCode = Install-PythonPackage -PythonExe $PYTHON_EXE -PackageName $package -WorkingDirectory $ROOT_DIR
      if ($installExitCode -ne 0) {
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
  "--command-cooldown", "$CommandCooldownSeconds",
  "--output-keys",
  "--verbose",
  "--overlay-state-file", $STATE_FILE
  )

  $blinkArgs = @(
  "--sfreq", "500",
  "--picks", "0,1",
  "--window", "0.5",
  "--threshold-uv", "80",
  "--refractory", "0.8",
  "--cooldown-state-file", $STATE_FILE
  )

  $classifierArgs = @(
  "--sfreq", "500",
  "--window", "3.0",
  "--step", "0.5",
  "--picks", "2,3,4",
  "--vote-k", "5",
  "--class-names", "0:rest,1:hand_mi",
  "--hand-mi-threshold", "0.97",
  "--hand-mi-consecutive", "3",
  "--cooldown-state-file", $STATE_FILE
  )
  $classifierArgs += @("--model", $ModelPath)

  Set-LauncherPhase "starting" "Launching overlay"
  Set-ComponentState "overlay" "starting" 0 "Launching overlay process"
  Write-Host "Starting overlay..." -ForegroundColor Green
  $overlayProc = Start-PythonComponent -Title "MindPlay Overlay" -PythonExe $PYTHON_EXE -ScriptPath $OVERLAY_SCRIPT -ComponentArgs $overlayArgs -WorkingDirectory $ROOT_DIR
  Set-ComponentState "overlay" "running" $overlayProc.Id "Overlay started"
  Start-Sleep -Seconds 1

  Set-LauncherPhase "starting" "Launching gyro detector"
  Set-ComponentState "gyro" "starting" 0 "Launching gyro process"
  Write-Host "Starting gyro detector..." -ForegroundColor Green
  $gyroProc = Start-PythonComponent -Title "MindPlay Gyro Detector" -PythonExe $PYTHON_EXE -ScriptPath $GYRO_SCRIPT -ComponentArgs $gyroArgs -WorkingDirectory $ROOT_DIR
  Set-ComponentState "gyro" "running" $gyroProc.Id "Gyro started"
  Start-Sleep -Seconds 1

  Set-LauncherPhase "starting" "Launching blink detector"
  Set-ComponentState "blink" "starting" 0 "Launching blink process"
  Write-Host "Starting blink detector..." -ForegroundColor Green
  $blinkProc = Start-PythonComponent -Title "MindPlay Blink Detector" -PythonExe $PYTHON_EXE -ScriptPath $BLINK_SCRIPT -ComponentArgs $blinkArgs -WorkingDirectory $ROOT_DIR
  Set-ComponentState "blink" "running" $blinkProc.Id "Blink started"
  Start-Sleep -Seconds 1

  Set-LauncherPhase "starting" "Launching real-time classifier"
  Set-ComponentState "classifier" "starting" 0 "Launching classifier process"
  Write-Host "Starting real-time classifier..." -ForegroundColor Green
  $classifierProc = Start-PythonComponent -Title "MindPlay Real-Time Classifier" -PythonExe $PYTHON_EXE -ScriptPath $CLASSIFIER_SCRIPT -ComponentArgs $classifierArgs -WorkingDirectory $ROOT_DIR
  Set-ComponentState "classifier" "running" $classifierProc.Id "Classifier started"

  Set-LauncherPhase "ready" "All components launched in admin mode"
  Write-Host ""
  Write-Host "All MindPlay components launched in admin mode." -ForegroundColor Green
  Write-Host "Overlay PID:    $($overlayProc.Id)"
  Write-Host "Gyro PID:       $($gyroProc.Id)"
  Write-Host "Blink PID:      $($blinkProc.Id)"
  Write-Host "Classifier PID: $($classifierProc.Id)"
  Write-Host ""
  Write-Host "Classifier rule active: hand_mi only when >97% for 3 consecutive windows; otherwise rest(0)." -ForegroundColor Cyan
  Write-Host "Gyro command cooldown: $CommandCooldownSeconds second(s) blocking blink/MI after non-center gyro commands." -ForegroundColor Cyan
  Write-Host "Use Ctrl+C in each component window to stop it." -ForegroundColor Yellow
}
catch {
  $msg = ($_ | Out-String).Trim()
  if ([string]::IsNullOrWhiteSpace($msg)) {
    $msg = $_.Exception.Message
  }
  Set-LauncherPhase "error" $msg
  Set-ComponentState "overlay" "error" 0 "Launcher failed before completion"
  Set-ComponentState "gyro" "error" 0 "Launcher failed before completion"
  Set-ComponentState "blink" "error" 0 "Launcher failed before completion"
  Set-ComponentState "classifier" "error" 0 "Launcher failed before completion"
  throw
}
