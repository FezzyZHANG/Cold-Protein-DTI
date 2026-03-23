param(
    [string]$Python = "",
    [string]$ConfigGlob = "config/experiments/exprS1_*.yaml",
    [string]$ResultsRoot = "results/exprS1_windows_test",
    [string]$CudaVisibleDevices = "",
    [switch]$Train = $false,
    [switch]$RunEval = $false
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

trap [System.Management.Automation.PipelineStoppedException] {
    Write-Host "[exprS1-test] interrupt received. Exiting."
    exit 130
}

trap [System.OperationCanceledException] {
    Write-Host $_.Exception.Message
    exit 130
}

function Resolve-Python {
    param([string]$RequestedPython)

    if ($RequestedPython) {
        return $RequestedPython
    }

    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    foreach ($candidate in @("python", "python3")) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }

    throw "Could not find a usable Python interpreter."
}

function Invoke-ExternalCommand {
    param(
        [string]$Description,
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 130) {
        throw [System.OperationCanceledException]::new("[exprS1-test] interrupted during $Description.")
    }
    if ($exitCode -ne 0) {
        throw "[exprS1-test] $Description failed with exit code $exitCode."
    }
}

$Python = Resolve-Python -RequestedPython $Python
$ResolvedGlob = Join-Path $ProjectRoot $ConfigGlob
$ConfigFiles = @(Get-ChildItem -Path $ResolvedGlob | Sort-Object Name)

if ($ConfigFiles.Count -eq 0) {
    throw "No exprS1 config files matched: $ConfigGlob"
}

Push-Location $ProjectRoot
try {
    if ($CudaVisibleDevices) {
        $env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
        Write-Host "[exprS1-test] CUDA_VISIBLE_DEVICES=$CudaVisibleDevices"
    }

    $DryRun = -not $Train.IsPresent

    Write-Host "[exprS1-test] project root: $ProjectRoot"
    Write-Host "[exprS1-test] python: $Python"
    Write-Host "[exprS1-test] configs: $($ConfigFiles.Count)"
    Write-Host "[exprS1-test] mode: $(if ($DryRun) { 'dry-run' } else { 'train' })"
    Write-Host "[exprS1-test] results root override: $ResultsRoot"

    foreach ($ConfigFile in $ConfigFiles) {
        $ConfigName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigFile.Name)
        Write-Host "[exprS1-test] running $ConfigName"

        $TrainArgs = @(
            "-m", "src.train",
            "--config", $ConfigFile.FullName,
            "--set", "output.root_dir=$ResultsRoot"
        )

        if ($DryRun) {
            $TrainArgs += "--dry-run"
        }

        Invoke-ExternalCommand -Description "training for $ConfigName" -FilePath $Python -Arguments $TrainArgs

        if (-not $DryRun -and $RunEval) {
            Write-Host "[exprS1-test] evaluating $ConfigName"
            Invoke-ExternalCommand -Description "evaluation for $ConfigName" -FilePath $Python -Arguments @(
                "-m", "src.eval",
                "--config", $ConfigFile.FullName,
                "--set", "output.root_dir=$ResultsRoot"
            )
        }
    }

    Write-Host "[exprS1-test] all exprS1 configs finished successfully."
} finally {
    Pop-Location
}
