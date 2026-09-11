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
echo This will create two scheduled tasks that run at logon:
echo   "%TASK_SCHED%"  - fetches each closed bar, predicts, scores, paper-trades
echo   "%TASK_DASH%"   - serves http://127.0.0.1:8077
echo.
echo Remove them later with uninstall-task.bat.
echo.
choice /C YN /M "Create them now"
if errorlevel 2 (
    echo Cancelled. Nothing was changed.
    exit /b 0
)

REM pythonw.exe runs without a console window, so a task that starts at logon
REM does not put two black windows on screen every morning.
schtasks /Create /TN "%TASK_SCHED%" /SC ONLOGON /RL LIMITED /F ^
  /TR "\"%CD%\.venv\Scripts\pythonw.exe\" -m cryptopred.serve.cli schedule --interval 1h --minute 2"
if errorlevel 1 goto failed

schtasks /Create /TN "%TASK_DASH%" /SC ONLOGON /RL LIMITED /F ^
  /TR "\"%CD%\.venv\Scripts\pythonw.exe\" -m uvicorn cryptopred.serve.api:app --host 127.0.0.1 --port 8077"
if errorlevel 1 goto failed

echo.
echo Created. Starting both now so you do not have to log out first...
schtasks /Run /TN "%TASK_SCHED%" >nul
schtasks /Run /TN "%TASK_DASH%" >nul
timeout /t 10 /nobreak >nul

echo.
echo Verifying...
status.bat
echo.
echo If the scheduler line above does not say RUNNING, check:
echo     schtasks /Query /TN "%TASK_SCHED%" /V /FO LIST
echo.
echo A task that runs at logon still stops when the machine is off. Bars missed
echo while it was off are refilled and flagged as backfilled, so the gap stays
echo visible rather than silently filling in.
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: schtasks failed. The usual cause is that the account is not allowed
echo to create scheduled tasks. Nothing partial was left behind if only the
echo first task was created - run uninstall-task.bat to clear it.
echo.
pause
exit /b 1
