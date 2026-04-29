@echo off
REM Thin shim — the Scheduled Task points at THIS file.  All logic lives
REM in soak_supervisor.ps1, which PowerShell parses into memory at start,
REM so editing the .ps1 mid-run can't race a running supervisor.
REM
REM Why a shim and not "just put the .ps1 in the task action":
REM the Scheduled Task config on GreenDragon (and the install pattern in
REM reference_greendragon_mower.md) launches `cmd.exe /c <file>`.  Keeping
REM a .cmd entry-point means the task action doesn't change; we only swap
REM what's INSIDE the cmd.
REM
REM This shim is intentionally trivial (one runnable line) so that even
REM if cmd.exe's mid-execution file re-read does fire, there's nothing
REM here to mis-execute.  The previous wrapper had ~50 lines of logic
REM that re-read mid-run after an in-place edit and spawned an orphan
REM python on Win11 build 26200 (where wmic is missing).
REM
REM Sibling layout (both files in same dir):
REM   <dir>\soak_respawn.cmd      ← this file (Scheduled Task target)
REM   <dir>\soak_supervisor.ps1   ← real supervisor

setlocal
set "_SHIM_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%_SHIM_DIR%soak_supervisor.ps1"
exit /b %ERRORLEVEL%
