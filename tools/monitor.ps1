# Live monitor for the detached cgate run + report generation.
# Usage:  powershell -File tools\monitor.ps1 [-Interval 1] [-Workspace <path>]
param([int]$Interval = 1, [string]$Workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path)

$ws    = $Workspace
$log   = Join-Path $ws 'runs\run.log'
$cache = Join-Path $ws 'runs\cache'
$preds = Join-Path $ws 'runs\predictions.json'

function Get-RunState {
    $p = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
         Where-Object { $_.CommandLine -match 'cgate.*experiments.*run' }
    return ($null -ne $p)
}

$baseCacheMB = 0
if (Test-Path $cache) {
    $baseCacheMB = [math]::Round((Get-ChildItem $cache -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum / 1MB, 1)
}

while ($true) {
    Clear-Host
    $alive = Get-RunState
    Write-Output ("=== cgate run monitor  {0} ===" -f (Get-Date -Format 'HH:mm:ss'))

    # Phase / progress from the slice cache (grows during ingest; fixed after).
    $mb = $baseCacheMB
    if (Test-Path $cache) {
        $mb = [math]::Round((Get-ChildItem $cache -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object Length -Sum).Sum / 1MB, 1)
    }
    Write-Output ("ingest slice-cache: {0} MB (1251 cases target ~9000 MB)" -f $mb)

    # GPU
    try {
        $g = nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
        Write-Output ("gpu: {0}" -f $g)
    } catch { Write-Output 'gpu: n/a' }

    # Log tail (buffered, may lag)
    if (Test-Path $log) {
        Write-Output '--- log tail ---'
        Get-Content $log -Tail 5
    }

    # Deliverables
    foreach ($f in @("$ws\runs\brats_full_seed0.json", $preds)) {
        if (Test-Path $f) { Write-Output ("done: {0}" -f $f) }
    }

    if (-not $alive) {
        Write-Output 'RUN NOT ACTIVE (finished or died -- check runs\run.err)'
        break
    }
    Start-Sleep -Seconds $Interval
}
