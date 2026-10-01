@echo off
setlocal
REM Registers a Windows scheduled task so the scheduler and dashboard start at
REM logon and keep running without a console window to leave open.
REM
REM YOU run this, not the assistant. It writes a persistent entry in Windows
REM Task Scheduler under your account. uninstall-task.bat removes it, and
REM nothing else on the machine is touched.
REM
REM Why this exists: run.bat has to be launched by hand after every restart. The
REM live log accumulated 13 rows in 16 days because that did not happen, and a
REM prediction written after its bar closed cannot prove it was written first.
REM
REM Still no exchange account. There is no API key anywhere in this project and
REM no code path that could place a real order. This schedules paper trading.

cd /d "%~dp0"

set TASK_SCHED=cryptopred scheduler
set TASK_DASH=cryptopred dashboard
set TASK_CHECK=cryptopred model check

if not exist ".venv\Scripts\pythonw.exe" (
    echo.
    echo ERROR: .venv\Scripts\pythonw.exe not found.
    echo Create the environment first:
    echo     uv venv --python 3.12
    echo     uv pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

echo Checking the install...
.venv\Scripts\python.exe -c "import cryptopred" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: the cryptopred package does not import.
    echo Run:  uv pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

echo.
echo This will create two tasks that run at logon:
echo   "%TASK_SCHED%"  - fetches each closed bar, predicts, scores, paper-trades
echo   "%TASK_DASH%"   - serves http://127.0.0.1:8077
echo.
echo And one that runs monthly, on the 1st at 09:00:
echo   "%TASK_CHECK%"  - evaluates the model against fresh data and writes
echo                    a report. It does NOT pass --save, so nothing in the
echo                    registry changes and the live experiment keeps
echo                    measuring the model it has been measuring.
echo.
echo Remove them later with uninstall-task.bat.
echo.
choice /C YN /M "Create them now"
if errorlevel 2 (
    echo Cancelled. Nothing was changed.
    exit /b 0
)

REM pythonw.exe runs without a console window, so a task that starts at logon
REM does not put two black windows on screen every morning. With no console
REM there is nowhere for output to go: the scheduler logs to
REM data\logs\scheduler.log, and uvicorn is told not to probe the console for
REM colour support, which raises when there is no console to probe.
schtasks /Create /TN "%TASK_SCHED%" /SC ONLOGON /RL LIMITED /F ^
  /TR "\"%CD%\.venv\Scripts\pythonw.exe\" -m cryptopred.serve.cli schedule --interval 1h --minute 2"
if errorlevel 1 goto failed

schtasks /Create /TN "%TASK_DASH%" /SC ONLOGON /RL LIMITED /F ^
  /TR "\"%CD%\.venv\Scripts\pythonw.exe\" -m uvicorn cryptopred.serve.api:app --host 127.0.0.1 --port 8077 --no-use-colors"
if errorlevel 1 goto failed

REM schtasks /Create cannot set a task's settings, and its defaults stop a task
REM that has run for 72 hours and stop it when a laptop goes onto battery. A
REM scheduler meant to run for weeks would die on day three - and the dashboard,
REM which carries the watchdog, with it. This lifts both for the two long tasks.
powershell -NoProfile -Command "try { $s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1); foreach ($t in '%TASK_SCHED%', '%TASK_DASH%') { Set-ScheduledTask -TaskName $t -Settings $s -ErrorAction Stop | Out-Null } } catch { Write-Host $_; exit 1 }"
if errorlevel 1 (
    echo.
    echo WARNING: could not change the task settings. The tasks still work, but
    echo Windows will stop them after 3 days and when on battery. To fix by hand:
    echo Task Scheduler, open each cryptopred task, Settings tab, untick
    echo "Stop the task if it runs longer than", and on the Conditions tab untick
    echo "Start the task only if the computer is on AC power".
    echo.
)

REM Monthly, not at logon. Evaluation takes ~25 minutes of CPU, and running it
REM every time the machine boots would spend that for nothing - the answer
REM barely moves in a day.
REM
REM It evaluates and reports. It does not save. A gate is only worth having if
REM somebody reads its output, and on 2026-09-23 an unread gate put a model into
REM the registry at -0.24%% after costs with its own report printing STRATEGY
REM VERDICT: NO-GO one line above. See docs/findings.md.
schtasks /Create /TN "%TASK_CHECK%" /SC MONTHLY /D 1 /ST 09:00 /RL LIMITED /F ^
  /TR "\"%CD%\check-model.bat\" /quiet"
if errorlevel 1 goto failed

echo.
echo Created. Starting the two logon tasks now so you do not have to log out first...
schtasks /Run /TN "%TASK_SCHED%" >nul
schtasks /Run /TN "%TASK_DASH%" >nul
timeout /t 10 /nobreak >nul

echo.
echo Verifying...
REM "call", or control passes to status.bat and never comes back: everything
REM below - what to check if the scheduler is not RUNNING - was never shown.
call status.bat
echo.
echo If the scheduler line above does not say RUNNING, check:
echo     schtasks /Query /TN "%TASK_SCHED%" /V /FO LIST
echo.
echo A task that runs at logon still stops when the machine is off. Bars missed
echo while it was off are refilled and flagged as backfilled, so the gap stays
echo visible rather than silently filling in.
echo.
echo The monthly check was NOT started now - it runs on the 1st. To see what it
echo would say, double-click check-model.bat yourself any time.
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: schtasks failed. The usual cause is that the account is not allowed
echo to create scheduled tasks. If some tasks were created before the failure,
echo run uninstall-task.bat to clear whatever landed.
echo.
pause
exit /b 1
