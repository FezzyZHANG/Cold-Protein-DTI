param(
    [string]$Python = "",
    [string]$OutputDir = "",
    [string]$Proxy = "",
    [string[]]$Model = @("esm2_t33_650M_UR50D", "esmc_600m", "VESM_650M"),
    [switch]$Force
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

function Invoke-ExternalCommand {
    param(
        [string]$Description,
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "[weights] $Description failed with exit code $exitCode."
    }
}

function Test-NonEmptyFile {
    param([string]$Path)

    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "[weights] expected file is missing: $Path"
    }

    $item = Get-Item $Path
    if ($item.Length -le 0) {
        throw "[weights] expected file is empty: $Path"
    }
}

function Confirm-StagedModel {
    param(
        [string]$ModelName,
        [string]$RootDir
    )

    switch ($ModelName) {
        "esm2_t33_650M_UR50D" {
            Test-NonEmptyFile -Path (Join-Path $RootDir "esm2_t33_650M_UR50D\config.json")
            Test-NonEmptyFile -Path (Join-Path $RootDir "esm2_t33_650M_UR50D\tokenizer_config.json")
            Test-NonEmptyFile -Path (Join-Path $RootDir "esm2_t33_650M_UR50D\vocab.txt")
            Test-NonEmptyFile -Path (Join-Path $RootDir "esm2_t33_650M_UR50D\model.safetensors")
        }
        "esmc_600m" {
            Test-NonEmptyFile -Path (Join-Path $RootDir "esmc_600m\data\weights\esmc_600m_2024_12_v0.pth")
        }
        "VESM_650M" {
            Test-NonEmptyFile -Path (Join-Path $RootDir "VESM_650M\VESM_650M.pth")
            Test-NonEmptyFile -Path (Join-Path $RootDir "esm2_t33_650M_UR50D\model.safetensors")
        }
        default {
            throw "[weights] no verifier is defined for model: $ModelName"
        }
    }
}

if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "artifacts\pretrained"
}

$Python = Resolve-Python -RequestedPython $Python
$DownloadScript = Join-Path $ProjectRoot "scripts\download_pretrained_model.py"

if (-not (Test-Path $DownloadScript -PathType Leaf)) {
    throw "Downloader not found: $DownloadScript"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Push-Location $ProjectRoot
try {
    Write-Host "[weights] project_root=$ProjectRoot"
    Write-Host "[weights] output_dir=$OutputDir"
    Write-Host "[weights] python=$Python"
    if ($Proxy) {
        Write-Host "[weights] proxy=$Proxy"
    }
    Write-Host "[weights] models=$($Model -join ', ')"

    $DownloadArgs = @($DownloadScript, "--output-dir", $OutputDir)
    foreach ($modelName in $Model) {
        $DownloadArgs += @("--model", $modelName)
    }
    if ($Proxy) {
        $DownloadArgs += @("--proxy", $Proxy)
    }
    if ($Force) {
        $DownloadArgs += "--force"
    }

    Invoke-ExternalCommand -Description "model download" -FilePath $Python -Arguments $DownloadArgs

    foreach ($modelName in $Model) {
        Confirm-StagedModel -ModelName $modelName -RootDir $OutputDir
    }

    Write-Host "[weights] staged artifacts verified"
    foreach ($modelName in $Model) {
        $modelPath = Join-Path $OutputDir $modelName
        Write-Host "[weights] $modelName -> $modelPath"
    }
    Write-Host "[weights] sync this directory to the remote project: $OutputDir"
} finally {
    Pop-Location
}
