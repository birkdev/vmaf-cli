# Modern VMAF Calculator Launcher
# This script launches the Python-based VMAF calculator

param(
    [string]$Original,
    [string]$Distorted,
    [int]$SegmentDuration,
    [string]$Model,
    [int]$Workers
)

# Check if Python is installed
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}

if (-not $pythonCmd) {
    Write-Host "Error: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

$pythonPath = $pythonCmd.Path
Write-Host "Found Python at $pythonPath" -ForegroundColor Green

# Check if Rich is installed
Write-Host "Checking for required Python packages..." -ForegroundColor Cyan
$null = & $pythonPath -c "import rich" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required package 'rich'..." -ForegroundColor Yellow
    & $pythonPath -m pip install rich
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install 'rich'. Please run: pip install rich" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Verify the Python script exists
$scriptPath = Join-Path $PSScriptRoot "vmaf_modern.py"
if (-not (Test-Path $scriptPath)) {
    Write-Host "Error: vmaf_modern.py not found at $scriptPath" -ForegroundColor Red
    Write-Host "Please save the Python script in the same directory as this launcher." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Build argument list to forward to Python script
$pyArgs = @($scriptPath)

if ($Original)        { $pyArgs += "--original";          $pyArgs += $Original }
if ($Distorted)       { $pyArgs += "--distorted";         $pyArgs += $Distorted }
if ($SegmentDuration) { $pyArgs += "--segment-duration";  $pyArgs += $SegmentDuration }
if ($Model)           { $pyArgs += "--model";             $pyArgs += $Model }
if ($Workers)         { $pyArgs += "--workers";           $pyArgs += $Workers }

# Launch the Python script
Clear-Host
& $pythonPath @pyArgs

# Keep window open if launched from Windows Explorer
if ($Host.Name -eq 'ConsoleHost') {
    Write-Host "`nPress any key to close..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
