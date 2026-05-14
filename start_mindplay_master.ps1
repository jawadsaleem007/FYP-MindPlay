param(
  [string]$ModelPath = "",
  [switch]$NoOverlayFollow,
  [string]$StatusFile = "",
  [string]$ConfigFile = ""
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

function Quote-Single([string]$value) {
  return "'" + $value.Replace("'", "''") + "'"
}

function Normalize-PathInput([string]$value) {
  if ([string]::IsNullOrWhiteSpace($value)) {
    return $value
  }

  $clean = $value.Trim()

  # Handle accidental list-like wrapping: [E:\path\model.joblib] or ['E:\path\model.joblib'].
  if ($clean.StartsWith("[") -and $clean.EndsWith("]")) {
    $clean = $clean.Substring(1, $clean.Length - 2).Trim()
  }

  if (($clean.StartsWith('"') -and $clean.EndsWith('"')) -or ($clean.StartsWith("'") -and $clean.EndsWith("'"))) {
    $clean = $clean.Substring(1, $clean.Length - 2).Trim()
  }

  return $clean
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

  $RequestedPath = Normalize-PathInput $RequestedPath

  if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
    $candidate = $RequestedPath
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
      $candidate = Join-Path $RootDir $candidate
    }
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }

    if (-not [System.IO.Path]::IsPathRooted($RequestedPath)) {
      # Handle root-prefixed relative paths, e.g. "FYP-MindPlay\fbcsp_lda_S15.joblib".
      $rootLeaf = [System.IO.Path]::GetFileName($RootDir)
      $normalizedRel = $RequestedPath.TrimStart('\\', '/')
      if ($normalizedRel.StartsWith("$rootLeaf\\", [System.StringComparison]::OrdinalIgnoreCase) -or
          $normalizedRel.StartsWith("$rootLeaf/", [System.StringComparison]::OrdinalIgnoreCase)) {
        $trimmedRel = $normalizedRel.Substring($rootLeaf.Length).TrimStart('\\', '/')
        if (-not [string]::IsNullOrWhiteSpace($trimmedRel)) {
          $trimmedCandidate = Join-Path $RootDir $trimmedRel
          if (Test-Path $trimmedCandidate) {
            return (Resolve-Path $trimmedCandidate).Path
          }
        }
      }

      # Last-resort fallback to basename under project root.
      $leafCandidate = Join-Path $RootDir ([System.IO.Path]::GetFileName($RequestedPath))
      if (Test-Path $leafCandidate) {
        return (Resolve-Path $leafCandidate).Path
      }
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
    [string[]]$ScriptArgs,
    [string]$WorkingDirectory
  )

  $quotedRoot = Quote-Single $WorkingDirectory
  $quotedPython = Quote-Single $PythonExe
  $quotedScript = Quote-Single $ScriptPath
  $quotedArgs = @()
  foreach ($arg in $ScriptArgs) {
    $quotedArgs += (Quote-Single $arg)
  }
  $pythonArgExpr = @($quotedScript)
  if ($quotedArgs.Count -gt 0) {
    $pythonArgExpr += $quotedArgs
  }
  $joinedArgs = ($pythonArgExpr -join ", ")

  $safeTitle = $Title.Replace("'", "''")
  $command = "`$host.UI.RawUI.WindowTitle = '$safeTitle'; Set-Location $quotedRoot; `$pyArgs = @($joinedArgs); & $quotedPython @pyArgs"

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
  $elevateArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $selfPath)
  if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    $elevateArgs += @("-ModelPath", (Normalize-PathInput $ModelPath))
  }
  if ($NoOverlayFollow) {
    $elevateArgs += "-NoOverlayFollow"
  }
  if (-not [string]::IsNullOrWhiteSpace($StatusFile)) {
    $elevateArgs += @("-StatusFile", (Normalize-PathInput $StatusFile))
  }
  if (-not [string]::IsNullOrWhiteSpace($ConfigFile)) {
    $elevateArgs += @("-ConfigFile", (Normalize-PathInput $ConfigFile))
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
    & $PYTHON_EXE -c "import $module" *>$null
    if ($LASTEXITCODE -ne 0) {
      $missingPackages += $moduleToPackage[$module]
    }
  }

  if ($missingPackages.Count -gt 0) {
    $missingPackages = $missingPackages | Select-Object -Unique
    Set-LauncherPhase "installing" "Installing missing packages: $($missingPackages -join ', ')"
    Write-Host "Installing missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
    foreach ($package in $missingPackages) {
      & $PYTHON_EXE -m pip install $package
      if ($LASTEXITCODE -ne 0) {
        throw "Failed to install required package: $package"
      }
    }
  }

  # Load launcher config if provided
  $blinkConfig = $null
  $classifierConfig = $null
  if (-not [string]::IsNullOrWhiteSpace($ConfigFile)) {
    $configPath = [System.IO.Path]::GetFullPath($ConfigFile)
    if (Test-Path $configPath) {
      try {
        $configJson = Get-Content -Path $configPath -Raw | ConvertFrom-Json
        if ($configJson.PSObject.Properties.Name -contains "blink") {
          $blinkConfig = $configJson.blink
        }
        if ($configJson.PSObject.Properties.Name -contains "classifier") {
          $classifierConfig = $configJson.classifier
        }
        Write-Host "Loaded launcher configuration from: $configPath" -ForegroundColor Green
      } catch {
        Write-Host "Warning: Failed to load config file: $_" -ForegroundColor Yellow
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

  # Prepare blink config values (PowerShell 5.1 compatible)
  $blinkSfreq = if ($blinkConfig -and $blinkConfig.sfreq) { $blinkConfig.sfreq } else { "500" }
  $blinkPicks = if ($blinkConfig -and $blinkConfig.picks) { $blinkConfig.picks } else { "Fp1,Fp2" }
  $blinkWindow = if ($blinkConfig -and $blinkConfig.window) { $blinkConfig.window } else { "0.5" }
  $blinkThreshold = if ($blinkConfig -and $blinkConfig.threshold_uv) { $blinkConfig.threshold_uv } else { "140" }
  $blinkRefractory = if ($blinkConfig -and $blinkConfig.refractory) { $blinkConfig.refractory } else { "0.8" }

  $blinkArgs = @(
  "--sfreq", $blinkSfreq,
  "--picks", $blinkPicks,
  "--window", $blinkWindow,
  "--threshold-uv", $blinkThreshold,
  "--refractory", $blinkRefractory
  )
  
  # Add key argument if specified in config
  if ($blinkConfig -and -not [string]::IsNullOrWhiteSpace($blinkConfig.key)) {
    $blinkArgs += @("--key", $blinkConfig.key)
  }
  
  # Add scale-to-uv flag if enabled in config
  if ($blinkConfig.scale_to_uv -eq $true) {
    $blinkArgs += "--scale-to-uv"
  }
  
  # Add extra args if specified
  if ($blinkConfig -and -not [string]::IsNullOrWhiteSpace($blinkConfig.extra_args)) {
    $blinkArgs += $blinkConfig.extra_args
  }

  # Prepare classifier config values (PowerShell 5.1 compatible)
  $classifierSfreq = if ($classifierConfig -and $classifierConfig.sfreq) { $classifierConfig.sfreq } else { "500" }
  $classifierWindow = if ($classifierConfig -and $classifierConfig.window) { $classifierConfig.window } else { "4.0" }
  $classifierStep = if ($classifierConfig -and $classifierConfig.step) { $classifierConfig.step } else { "0.5" }
  $classifierPicks = if ($classifierConfig -and $classifierConfig.picks) { $classifierConfig.picks } else { "Cz,C3,C4" }
  $classifierVoteK = if ($classifierConfig -and $classifierConfig.vote_k) { $classifierConfig.vote_k } else { "5" }
  $classifierClassNames = if ($classifierConfig -and $classifierConfig.class_names) { $classifierConfig.class_names } else { "0:rest,1:hand_mi" }
  $classifierThreshold = if ($classifierConfig -and $classifierConfig.hand_mi_threshold) { $classifierConfig.hand_mi_threshold } else { "0.9" }
  $classifierConsecutive = if ($classifierConfig -and $classifierConfig.hand_mi_consecutive) { $classifierConfig.hand_mi_consecutive } else { "2" }

  $classifierArgs = @(
  "--sfreq", $classifierSfreq,
  "--window", $classifierWindow,
  "--step", $classifierStep,
  "--picks", $classifierPicks,
  "--vote-k", $classifierVoteK,
  "--class-names", $classifierClassNames,
  "--hand-mi-threshold", $classifierThreshold,
  "--hand-mi-consecutive", $classifierConsecutive
  )
  
  # Add scale-to-uV flag if enabled in config
  if ($classifierConfig.scale_to_uV -eq $true) {
    $classifierArgs += "--scale-to-uV"
  }
  
  # Add block flag if enabled in config
  if ($classifierConfig.block -eq $true) {
    $classifierArgs += "--block"
  }
  
  $classifierArgs += @("--model", $ModelPath)

  # Log the parameters being used
  Write-Host ""
  Write-Host "Component Parameters:" -ForegroundColor Cyan
  Write-Host "Blink Detector: --sfreq $($blinkArgs[1]) --picks '$($blinkArgs[3])' --window $($blinkArgs[5]) --threshold-uv $($blinkArgs[7]) --refractory $($blinkArgs[9])" -ForegroundColor Gray
  Write-Host "MI Classifier: --sfreq $($classifierArgs[1]) --window $($classifierArgs[3]) --step $($classifierArgs[5]) --picks '$($classifierArgs[7])' --vote-k $($classifierArgs[9])" -ForegroundColor Gray
  Write-Host ""

  Set-LauncherPhase "starting" "Launching overlay"
  Set-ComponentState "overlay" "starting" 0 "Launching overlay process"
  Write-Host "Starting overlay..." -ForegroundColor Green
  $overlayProc = Start-PythonComponent -Title "MindPlay Overlay" -PythonExe $PYTHON_EXE -ScriptPath $OVERLAY_SCRIPT -ScriptArgs $overlayArgs -WorkingDirectory $ROOT_DIR
  Set-ComponentState "overlay" "running" $overlayProc.Id "Overlay started"
  Start-Sleep -Seconds 1

  Set-LauncherPhase "starting" "Launching gyro detector"
  Set-ComponentState "gyro" "starting" 0 "Launching gyro process"
  Write-Host "Starting gyro detector..." -ForegroundColor Green
  $gyroProc = Start-PythonComponent -Title "MindPlay Gyro Detector" -PythonExe $PYTHON_EXE -ScriptPath $GYRO_SCRIPT -ScriptArgs $gyroArgs -WorkingDirectory $ROOT_DIR
  Set-ComponentState "gyro" "running" $gyroProc.Id "Gyro started"
  Start-Sleep -Seconds 1

  Set-LauncherPhase "starting" "Launching blink detector"
  Set-ComponentState "blink" "starting" 0 "Launching blink process"
  Write-Host "Starting blink detector..." -ForegroundColor Green
  $blinkProc = Start-PythonComponent -Title "MindPlay Blink Detector" -PythonExe $PYTHON_EXE -ScriptPath $BLINK_SCRIPT -ScriptArgs $blinkArgs -WorkingDirectory $ROOT_DIR
  Set-ComponentState "blink" "running" $blinkProc.Id "Blink started"
  Start-Sleep -Seconds 1

  Set-LauncherPhase "starting" "Launching real-time classifier"
  Set-ComponentState "classifier" "starting" 0 "Launching classifier process"
  Write-Host "Starting real-time classifier..." -ForegroundColor Green
  $classifierProc = Start-PythonComponent -Title "MindPlay Real-Time Classifier" -PythonExe $PYTHON_EXE -ScriptPath $CLASSIFIER_SCRIPT -ScriptArgs $classifierArgs -WorkingDirectory $ROOT_DIR
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
  Write-Host "Use Ctrl+C in each component window to stop it." -ForegroundColor Yellow
}
catch {
  $msg = $_.Exception.Message
  Set-LauncherPhase "error" $msg
  Set-ComponentState "overlay" "error" 0 "Launcher failed before completion"
  Set-ComponentState "gyro" "error" 0 "Launcher failed before completion"
  Set-ComponentState "blink" "error" 0 "Launcher failed before completion"
  Set-ComponentState "classifier" "error" 0 "Launcher failed before completion"
  throw
}
