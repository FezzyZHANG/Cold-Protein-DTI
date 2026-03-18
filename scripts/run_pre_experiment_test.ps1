param(
    [string]$Python = "",
    [string]$Mode = "cp-easy",
    [int]$SubsampleN = 500000,
    [string]$InputPath = "",
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

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

if (-not $InputPath) {
    $InputPath = Join-Path $ProjectRoot "data\scope_dti_with_inchikey.parquet"
}

$Python = Resolve-Python -RequestedPython $Python
$SplitDir = Join-Path $ProjectRoot ("data\splits\{0}_sub{1}" -f $Mode, $SubsampleN)
$GraphCachePath = Join-Path $SplitDir "graph_cache.pt"
$RunName = ("preexperiment_{0}_cnn_concat_s42_{1}" -f $Mode.Replace("-", "_"), (Get-Date -Format "yyyyMMdd_HHmmss"))
$ConfigPath = Join-Path $ProjectRoot "config\experiments\preexperiment_cnn_smoke.yaml"

Push-Location $ProjectRoot
try {
    if ($Mode -eq "cp-hard") {
        & $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('esm') else 1)" *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "cp-hard pre-experiment preparation requires both torch and fair-esm. Run `uv sync --extra esm` first."
        }
    }

    Write-Host "[pretest] preparing split files in $SplitDir"
    & $Python "scripts/prepare_dti_splits.py" `
        --input-path $InputPath `
        --output-dir $SplitDir `
        --mode $Mode `
        --seed 42 `
        --subsample-n $SubsampleN `
        --build-graph-cache

    $DryRun = $false
    & $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)" *> $null
    $ImportTorchExitCode = $LASTEXITCODE
    if ($ImportTorchExitCode -ne 0) {
        $DryRun = $true
        Write-Host "[pretest] torch is not installed. Falling back to --dry-run validation."
    }

    if (-not (Test-Path $GraphCachePath)) {
        Write-Host "[pretest] graph cache not found at $GraphCachePath"
        Write-Host "[pretest] build it first with: $Python scripts/prepare_dti_splits.py --input-path $InputPath --output-dir $SplitDir --mode $Mode --seed 42 --subsample-n $SubsampleN --build-graph-cache"
        if (-not $DryRun) {
            $DryRun = $true
            Write-Host "[pretest] falling back to --dry-run validation until the graph cache is available."
        }
    }

    $CommonArgs = @(
        "--config", $ConfigPath,
        "--set", "run_name=$RunName",
        "--set", "data.split_name=$Mode",
        "--set", "data.split_dir=$SplitDir",
        "--set", "data.graph_cache_path=$GraphCachePath",
        "--set", "output.allow_rerun_suffix=false"
    )

    $TrainArgs = @("-m", "src.train") + $CommonArgs
    if ($DryRun) {
        $TrainArgs += "--dry-run"
    }

    Write-Host "[pretest] launching training validation run: $RunName"
    & $Python @TrainArgs

    if (-not $DryRun -and -not $SkipEval) {
        Write-Host "[pretest] launching evaluation validation run: $RunName"
        & $Python -m src.eval @CommonArgs
    }
} finally {
    Pop-Location
}
