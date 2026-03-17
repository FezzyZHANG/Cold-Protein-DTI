param(
    [string]$Python = "",
    [string[]]$Extra = @(),
    [switch]$Recreate,
    [switch]$Frozen
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { Join-Path $ProjectRoot ".venv" }

function Resolve-Python {
    param([string]$RequestedPython)

    if ($RequestedPython) {
        return $RequestedPython
    }

    $candidates = @("python3.10", "python3", "python")
    foreach ($candidate in $candidates) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }

    throw "Could not find a usable Python interpreter."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed or not on PATH."
}

$Python = Resolve-Python -RequestedPython $Python

Set-Location $ProjectRoot

Write-Host "[uv-env] project_root=$ProjectRoot"
Write-Host "[uv-env] venv_dir=$VenvDir"
Write-Host "[uv-env] python=$Python"

if ($Recreate -and (Test-Path $VenvDir)) {
    Write-Host "[uv-env] removing existing environment at $VenvDir"
    Remove-Item -Recurse -Force $VenvDir
}

& uv venv --python $Python $VenvDir

$LockPath = Join-Path $ProjectRoot "uv.lock"
if ($Frozen -and -not (Test-Path $LockPath)) {
    throw "--Frozen was requested but uv.lock does not exist."
}

$SyncArgs = @("sync", "--python", $Python)
if ($Frozen) {
    $SyncArgs += "--frozen"
}
foreach ($item in $Extra) {
    $SyncArgs += @("--extra", $item)
}

Write-Host "[uv-env] running: uv $($SyncArgs -join ' ')"
& uv @SyncArgs

$ActivatePath = Join-Path $VenvDir "Scripts\\Activate.ps1"
Write-Host "[uv-env] done"
Write-Host "[uv-env] activate with: $ActivatePath"
