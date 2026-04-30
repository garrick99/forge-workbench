# Live-resolve soak supervisor (PowerShell).
#
# Re-spawns probe-loop on exit code 99 (the scheduler's "git HEAD moved,
# please restart against new code" signal).  Any other exit code is
# terminal — supervisor returns it to its caller and the Scheduled
# Task ends.
#
# Why PowerShell instead of cmd.exe:
#   * PowerShell parses the entire script into memory at start, so
#     in-place edits to this file during a long-running soak are SAFE
#     — they take effect on the next launch, not mid-execution.  The
#     previous cmd.exe wrapper had a re-read race that spawned an
#     orphan python after the file was rewritten mid-run.
#   * No `wmic` dependency.  `wmic.exe` was removed in Windows 11
#     build 26200 (24H2+); the previous cmd wrapper used it for
#     timestamp generation and silently failed on GreenDragon.
#
# Usage (typically via a tiny soak_wrapper.cmd shim invoked by the
# Scheduled Task action):
#     powershell.exe -NoProfile -ExecutionPolicy Bypass `
#                    -File C:\mower\soak_supervisor.ps1
#
# Configuration is via environment variables read at startup; defaults
# are GreenDragon-tuned but BigDaddy or a workstation can override.

$ErrorActionPreference = 'Continue'

# Force UTF-8 for native-command stdout/stderr capture.  Without this,
# PowerShell's `*>>` redirect writes the python output as UTF-16 LE,
# which interleaves with the supervisor's own UTF-8 lines (written via
# Out-File -Encoding utf8) and produces a log file that can't be
# grep'd cleanly.  Match $env:PYTHONIOENCODING set below.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# ---- Configuration with sensible defaults for the GreenDragon mower ----
$ProbeDir   = if ($env:MOWER_PROBE_DIR)  { $env:MOWER_PROBE_DIR }  else { 'C:\mower\probes_long' }
$LogDir     = if ($env:MOWER_LOG_DIR)    { $env:MOWER_LOG_DIR }    else { 'C:\mower\logs' }
$WorkDir    = if ($env:MOWER_WORK_DIR)   { $env:MOWER_WORK_DIR }   else { 'C:\mower\forge-workbench' }
$Workers    = if ($env:MOWER_WORKERS)    { $env:MOWER_WORKERS }    else { '8' }
$Budget     = if ($env:MOWER_BUDGET)     { $env:MOWER_BUDGET }     else { '14400' }
$MaxProbes  = if ($env:MOWER_MAX_PROBES) { $env:MOWER_MAX_PROBES } else { '100000000' }

# ---- Pin the python module path + isel + IO encoding for probe-loop ----
$env:PYTHONIOENCODING = 'utf-8'
if (-not $env:PYTHONPATH) {
    $env:PYTHONPATH = 'C:\mower\openptxas;C:\mower\forge-workbench'
}
if (-not $env:OPENPTXAS_ISEL) {
    $env:OPENPTXAS_ISEL = 'C:\mower\openptxas\sass\isel.py'
}
if (-not $env:MOWER_MAX_WORKERS) {
    $env:MOWER_MAX_WORKERS = '16'
}

Set-Location $WorkDir
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$RESPAWN_EXIT = 99
$TargetFile   = 'C:\mower\.workers_target'   # written by gpu_throttle.ps1
$StopFile     = 'C:\mower\.supervisor_stop'  # touch this to halt the supervisor
$count        = 0

# Restart-rate guard: if probe-loop crashes immediately N times in M
# seconds, treat that as a real failure and exit so the operator sees
# it.  Throttle-induced kills are "normal" but a runaway crash loop
# wastes the GPU.
$consecutiveBadExits = 0
$BAD_EXIT_LIMIT = 5         # 5 in a row → stop
$MIN_RUN_SEC    = 30        # exits before this count as "bad"

while ($true) {
    if (Test-Path $StopFile) {
        Write-Output "[supervisor] stop file present; exiting"
        Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
        exit 0
    }

    # GPU-throttle-aware worker selection.  If gpu_throttle.ps1 has
    # written a target, use it; otherwise fall back to env / default.
    $effectiveWorkers = $Workers
    if (Test-Path $TargetFile) {
        try {
            $tgt = (Get-Content $TargetFile -Raw | ConvertFrom-Json).workers
            if ($tgt -ge 1 -and $tgt -le 16) {
                $effectiveWorkers = "$tgt"
            }
        } catch { }
    }

    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log   = Join-Path $LogDir "soak_$stamp.log"

    # soak.logfile is a pointer file the operator (or other tools) can
    # tail to find the current run's log without scanning the directory.
    $log | Out-File -FilePath (Join-Path (Split-Path $LogDir -Parent) 'soak.logfile') -Encoding ASCII

    @(
        "[supervisor] === probe-loop starting (respawn $count) ==="
        "[supervisor] start:     $(Get-Date -Format 'o')"
        "[supervisor] log:       $log"
        "[supervisor] probe-dir: $ProbeDir  workers: $effectiveWorkers  budget: ${Budget}s"
    ) | Out-File -FilePath $log -Encoding utf8

    $runStart = Get-Date
    $args = @(
        '-m', 'workbench', 'probe-loop',
        '--probe-dir', $ProbeDir,
        '--soak',
        '--budget', $Budget,
        '--max-probes', $MaxProbes,
        '--workers', $effectiveWorkers
    )

    # IMPORTANT: do NOT pipe through ForEach-Object / Out-File.  PowerShell
    # pipelines clobber $LASTEXITCODE — the native python's actual exit
    # code is masked by the pipeline-tail cmdlet's success status, which
    # is always 0.  Use `*>>` (all-streams append) to preserve the
    # native exit code in $LASTEXITCODE.
    # Bug: 2026-04-29 first respawn attempt — probe-loop returned 99 but
    # the pipeline drained it to 0, so the supervisor never respawned.
    & python @args *>> $log
    $code = $LASTEXITCODE

    "[supervisor] exit: $(Get-Date -Format 'o')  code=$code" |
        Add-Content -Path $log -Encoding utf8

    $runDuration = ((Get-Date) - $runStart).TotalSeconds

    if ($code -eq $RESPAWN_EXIT) {
        "[supervisor] respawn requested (code $RESPAWN_EXIT); restarting in 5s" |
            Add-Content -Path $log -Encoding utf8
        $consecutiveBadExits = 0
        $count++
        Start-Sleep -Seconds 5
        continue
    }

    if ($code -eq 0) {
        # Clean budget-exhaust: rotate the soak round.
        "[supervisor] clean budget-exhaust (code 0); restarting for next round in 30s" |
            Add-Content -Path $log -Encoding utf8
        $consecutiveBadExits = 0
        $count++
        Start-Sleep -Seconds 30
        continue
    }

    # Non-zero exit.  Could be: (a) gpu_throttle externally killed the
    # python because workers target changed (this is desired and frequent),
    # or (b) a real crash.  Heuristic: a kill that happened AFTER the
    # mower had been running for ≥ MIN_RUN_SEC is treated as a normal
    # throttle event; an immediate non-zero exit is "bad".
    if ($runDuration -ge $MIN_RUN_SEC) {
        "[supervisor] external exit code=$code after ${runDuration}s (likely throttle); restarting in 5s" |
            Add-Content -Path $log -Encoding utf8
        $consecutiveBadExits = 0
        $count++
        Start-Sleep -Seconds 5
        continue
    }

    # Suspiciously fast exit.
    $consecutiveBadExits++
    "[supervisor] FAST exit code=$code after ${runDuration}s (bad-streak=$consecutiveBadExits/$BAD_EXIT_LIMIT)" |
        Add-Content -Path $log -Encoding utf8
    if ($consecutiveBadExits -ge $BAD_EXIT_LIMIT) {
        "[supervisor] bad-exit limit reached; supervisor stopping for operator review" |
            Add-Content -Path $log -Encoding utf8
        exit $code
    }
    Start-Sleep -Seconds 30
    $count++
}
