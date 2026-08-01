param(
    [string[]]$Methods = @(
        "finetune",
        "replay",
        "icarl",
        "podnet",
        "prototype_control",
        "global_hap",
        "flat_lrhap",
        "sacil"
    ),
    [int]$Gpu = 0,
    [int]$Seed = 1,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$Configs = [ordered]@{
    finetune = "configs/pycil/official/cifar100/finetune_b50_inc5_resnet32.json"
    replay = "configs/pycil/official/cifar100/replay_b50_inc5_resnet32.json"
    icarl = "configs/pycil/official/cifar100/icarl_b50_inc5_resnet32.json"
    podnet = "configs/pycil/official/cifar100/podnet_b50_inc5_resnet32.json"
    prototype_control = "configs/pycil/controlled/cifar100/prototype_control_nme_b50_inc5_resnet32.json"
    global_hap = "configs/pycil/controlled/cifar100/global_hap_nme_b50_inc5_resnet32.json"
    flat_lrhap = "configs/pycil/controlled/cifar100/flat_lrhap_nme_b50_inc5_resnet32.json"
    sacil = "configs/pycil/controlled/cifar100/sacil_nme_b50_inc5_resnet32.json"
}

foreach ($method in $Methods) {
    if (-not $Configs.Contains($method)) {
        throw "Unknown method '$method'. Valid: $($Configs.Keys -join ', ')"
    }
}

Push-Location $ProjectRoot
try {
    foreach ($method in $Methods) {
        Write-Host "Starting $method on GPU $Gpu, seed $Seed"
        & $PythonExe scripts\train_pycil.py `
            --config $Configs[$method] `
            --device $Gpu `
            --seed $Seed
        if ($LASTEXITCODE -ne 0) {
            throw "$method failed with exit code $LASTEXITCODE"
        }
    }
    & $PythonExe scripts\summarize_official_controlled.py
    if ($LASTEXITCODE -ne 0) {
        throw "summary failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
