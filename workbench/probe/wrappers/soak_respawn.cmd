@echo off
REM Live-resolve soak supervisor: re-spawns the probe-loop when the
REM scheduler exits with code 99 (RESPAWN_EXIT_CODE).  This lets a
REM running soak pick up a code fix without operator intervention.
REM
REM Usage from a Scheduled Task or shell:
REM   soak_respawn.cmd <probe-dir> <log-file> [extra args]
REM
REM The wrapper inherits PYTHONIOENCODING / PYTHONPATH / OPENPTXAS_ISEL
REM / MOWER_MAX_WORKERS from the caller's environment.  Default
REM probe-loop args: --soak --budget 14400 --workers 4.

setlocal
if "%~1"=="" (
    echo usage: soak_respawn.cmd ^<probe-dir^> ^<log-file^> [extra args]
    exit /b 2
)
set "PROBE_DIR=%~1"
set "LOG_FILE=%~2"
shift
shift

set "EXTRA="
:collect_args
if "%~1"=="" goto :run
set "EXTRA=%EXTRA% %1"
shift
goto :collect_args

:run
set "RESPAWN_COUNT=0"
:loop
echo [supervisor] === probe-loop starting (respawn count: %RESPAWN_COUNT%) === >> "%LOG_FILE%"
echo [supervisor] start: %DATE% %TIME% >> "%LOG_FILE%"
python -m workbench probe-loop --probe-dir "%PROBE_DIR%" --soak --budget 14400 --max-probes 100000000 --workers 4 %EXTRA% >> "%LOG_FILE%" 2>&1
set EXITCODE=%ERRORLEVEL%
echo [supervisor] exit: %DATE% %TIME%  code=%EXITCODE% >> "%LOG_FILE%"
if %EXITCODE% EQU 99 (
    echo [supervisor] respawn requested (code 99); restarting in 5s >> "%LOG_FILE%"
    set /a RESPAWN_COUNT+=1
    timeout /t 5 /nobreak > /dev/null
    goto :loop
)
echo [supervisor] terminal exit code %EXITCODE%; supervisor done >> "%LOG_FILE%"
exit /b %EXITCODE%
