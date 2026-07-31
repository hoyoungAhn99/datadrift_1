param(
    [string[]]$Methods = @(
        "joint",
        "finetune",
        "replay",
        "icarl",
        "podnet",
        "afc",
        "create",
        "fgp",
        "cscct",
        "casper",
        "sacil"
    ),
    [int]$Gpu = 0,
    [string]$PythonExe = "C:\Users\user\anaconda3\envs\hoyoung\python.exe",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $ProjectRoot "scripts\train_pycil.py"
$Summarizer = Join-Path $ProjectRoot "scripts\summarize_pycil_table1.py"
$LogRoot = Join-Path $ProjectRoot "outputs\pycil\table1\runner_logs_resnet32"
$SummaryPath = Join-Path `
    $ProjectRoot `
    "mds\results\table1_cifar100_b50_inc5_resnet32.md"

$Configs = [ordered]@{
    joint = "configs/pycil/table1/cifar100/joint_nme_b50_inc5_resnet32.json"
    finetune = "configs/pycil/table1/cifar100/finetune_nme_b50_inc5_resnet32.json"
    replay = "configs/pycil/table1/cifar100/replay_nme_b50_inc5_resnet32.json"
    icarl = "configs/pycil/table1/cifar100/icarl_nme_b50_inc5_resnet32.json"
    podnet = "configs/pycil/table1/cifar100/podnet_nme_b50_inc5_resnet32.json"
    afc = "configs/pycil/table1/cifar100/afc_nme_b50_inc5_resnet32.json"
    create = "configs/pycil/table1/cifar100/create_native_b50_inc5_resnet32.json"
    fgp = "configs/pycil/table1/cifar100/fgp_nme_b50_inc5_resnet32.json"
    cscct = "configs/pycil/table1/cifar100/icarl_cscct_nme_b50_inc5_resnet32.json"
    casper = "configs/pycil/table1/cifar100/icarl_casper_nme_b50_inc5_resnet32.json"
    sacil = "configs/pycil/table1/cifar100/proto_sacil_nme_b50_inc5_resnet32.json"
}

function Write-Status {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    [Console]::Out.WriteLine("$timestamp $Message")
}

function Test-Completed {
    param(
        [string]$MarkerPath,
        [string]$LogPath,
        [string]$ConfigHash
    )
    if (-not (Test-Path -LiteralPath $MarkerPath)) {
        return $false
    }
    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $false
    }
    try {
        $marker = Get-Content -LiteralPath $MarkerPath -Raw |
            ConvertFrom-Json
        return $marker.config_sha256 -eq $ConfigHash
    }
    catch {
        return $false
    }
}

foreach ($method in $Methods) {
    if (-not $Configs.Contains($method)) {
        throw (
            "Unknown method '$method'. Valid methods: " +
            ($Configs.Keys -join ", ")
        )
    }
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable does not exist: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$previousCuda = [Environment]::GetEnvironmentVariable(
    "CUDA_VISIBLE_DEVICES",
    "Process"
)
$env:CUDA_VISIBLE_DEVICES = "$Gpu"

try {
    foreach ($method in $Methods) {
        $configRelative = $Configs[$method]
        $configPath = Join-Path $ProjectRoot $configRelative
        $logPath = Join-Path $LogRoot "$method.stdout.log"
        $markerPath = Join-Path $LogRoot "$method.complete.json"
        $configHash = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash

        if (
            -not $Force -and
            (Test-Completed $markerPath $logPath $configHash)
        ) {
            Write-Status "Reusing completed $method"
            continue
        }

        Write-Status "Starting $method on physical GPU $Gpu"
        & $PythonExe `
            $Runner `
            "--config" `
            $configRelative |
            Tee-Object -FilePath $logPath
        if ($LASTEXITCODE -ne 0) {
            throw "$method failed with exit code $LASTEXITCODE"
        }

        [ordered]@{
            method = $method
            config = $configRelative
            config_sha256 = $configHash
            completed_at = (Get-Date).ToString("o")
        } |
            ConvertTo-Json |
            Set-Content -LiteralPath $markerPath -Encoding UTF8
        Write-Status "Completed $method"
    }
}
finally {
    if ($null -eq $previousCuda) {
        Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    }
    else {
        $env:CUDA_VISIBLE_DEVICES = $previousCuda
    }
}

& $PythonExe `
    $Summarizer `
    "--log-root" `
    $LogRoot `
    "--output" `
    $SummaryPath
if ($LASTEXITCODE -ne 0) {
    throw "Table-1 summarization failed with exit code $LASTEXITCODE"
}
Write-Status "Updated $SummaryPath"
