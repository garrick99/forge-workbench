# GPU-aware throttle daemon for the mower.
#
# Polls nvidia-smi every $PollSeconds, maintains a rolling average of
# GPU utilization over $WindowSec, and writes a target worker count
# to $TargetFile.  The soak supervisor reads $TargetFile at the start
# of each round to decide MOWER_WORKERS.
#
# When the target changes meaningfully (≥ 2 worker delta), this daemon
# kills the running probe-loop python so the supervisor can respawn
# at the new size.  The supervisor's "respawn on any exit" mode is
# what makes that work — it doesn't care WHY python ended.
#
# Bands (rolling avg GPU util →  workers):
#   <  20%  →  12   (genuinely idle, push without maxing)
#   < 50%   →  8    (light external load)
#   < 80%   →  4    (something else is running, e.g. gaming)
#   ≥ 80%   →  2    (heavy external load — give the GPU back)
#
# To stop the daemon: create C:\mower\.throttle_stop  (or kill it).

$ErrorActionPreference = 'Continue'
$PollSeconds = 30
$WindowSec   = 90       # rolling window for averaging
$TargetFile  = 'C:\mower\.workers_target'
$StopFile    = 'C:\mower\.throttle_stop'
$LogFile     = 'C:\mower\logs\gpu_throttle.log'
$ProbeLoopMatch = 'probe-loop'   # CommandLine substring for our mower

function Write-Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'o'), $msg
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

function Get-GpuUtil() {
    $out = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    [int]($out.ToString().Trim())
}

function Pick-Workers($avg) {
    if ($avg -lt 20) { return 12 }
    if ($avg -lt 50) { return  8 }
    if ($avg -lt 80) { return  4 }
    return 2
}

function Read-CurrentTarget() {
    if (-not (Test-Path $TargetFile)) { return $null }
    try {
        return (Get-Content $TargetFile -Raw | ConvertFrom-Json).workers
    } catch { return $null }
}

function Write-Target($workers, $avg) {
    $tmp = "$TargetFile.tmp"
    @{
        workers = $workers
        gpu_avg = [Math]::Round($avg, 1)
        ts = (Get-Date -Format 'o')
    } | ConvertTo-Json -Compress | Set-Content -Path $tmp -Encoding ASCII
    Move-Item $tmp $TargetFile -Force
}

function Kill-Mower() {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match $ProbeLoopMatch } |
        ForEach-Object {
            try {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
                Write-Log "  killed python PID $($_.ProcessId) for respawn"
            } catch {
                Write-Log "  failed to kill PID $($_.ProcessId): $_"
            }
        }
}

# Ensure log dir exists.
$null = New-Item -ItemType Directory -Path (Split-Path $LogFile) -Force -ErrorAction SilentlyContinue
Write-Log "gpu_throttle started. poll=${PollSeconds}s window=${WindowSec}s"

$samples = New-Object System.Collections.Queue
$lastTarget = Read-CurrentTarget
if (-not $lastTarget) { $lastTarget = 8 }

while (-not (Test-Path $StopFile)) {
    $u = Get-GpuUtil
    if ($u -ne $null) {
        $samples.Enqueue(@{ ts = Get-Date; util = $u })
        # Drop samples older than $WindowSec
        $cutoff = (Get-Date).AddSeconds(-$WindowSec)
        while ($samples.Count -gt 0 -and ($samples.Peek().ts -lt $cutoff)) {
            $null = $samples.Dequeue()
        }
        $vals = @($samples | ForEach-Object { $_.util })
        $avg = ($vals | Measure-Object -Average).Average
        $newTarget = Pick-Workers $avg

        if ($newTarget -ne $lastTarget) {
            $delta = [Math]::Abs($newTarget - $lastTarget)
            Write-Log ("util now={0,3}% avg={1,5:F1}%  target {2} -> {3}  (delta={4})" `
                       -f $u, $avg, $lastTarget, $newTarget, $delta)
            Write-Target $newTarget $avg
            # Only kick the running mower if the change is meaningful
            # (≥2 worker delta).  Tiny oscillations stay sticky.
            if ($delta -ge 2) {
                Kill-Mower
            }
            $lastTarget = $newTarget
        }
    } else {
        Write-Log "  nvidia-smi failed (no GPU read this tick)"
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Log "stop signal received; daemon exiting"
Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
