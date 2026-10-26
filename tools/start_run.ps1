# Launch a detached cgate training run and write its logs under <workspace>/runs.
#
# The workspace defaults to the repo root (this script's parent), so the run
# works from a fresh clone. Pass -Workspace to point at a checkout elsewhere.
#
# Usage:
#   powershell -File tools\start_run.ps1
#   powershell -File tools\start_run.ps1 -BatchSize 8 -ImgSize 96   # smaller GPU
param(
    [string]$Workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [int]$BatchSize = 32,
    [int]$ImgSize = 128,
    [int]$Epochs = 12,
    [string]$Arch = 'resnet50',
    [int]$Seed = 0
)

$ws  = $Workspace
$log = Join-Path $ws 'runs\run.log'
$err = Join-Path $ws 'runs\run.err'
New-Item -ItemType Directory -Force -Path (Join-Path $ws 'runs') | Out-Null

$py = Join-Path $ws '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

# The workspace path may contain a space, so every path argument must be quoted
# explicitly -- Start-Process -ArgumentList does not quote for us.
$q = { param($p) '"' + $p + '"' }
$runArgs = @(
    '-m', 'cgate.experiments.run',
    '--data', 'brats',
    '--root', (& $q (Join-Path $ws 'data')),
    '--experts', 'torch',
    '--pretrained',
    '--arch', $Arch,
    '--epochs', "$Epochs",
    '--batch-size', "$BatchSize",
    '--img-size', "$ImgSize",
    '--table-cache', (& $q (Join-Path $ws 'runs\table')),
    '--slice-cache', (& $q (Join-Path $ws 'runs\cache')),
    '--out', (& $q (Join-Path $ws "runs\brats_full_seed$Seed.json")),
    '--figures', (& $q (Join-Path $ws 'runs\figs')),
    '--predictions', (& $q (Join-Path $ws 'runs\predictions.json')),
    '--report-system', 'B6',
    '--seed', "$Seed"
)

Start-Process -FilePath $py -ArgumentList $runArgs -WorkingDirectory $ws `
    -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow
Write-Output "launched detached run; logs at $log"
