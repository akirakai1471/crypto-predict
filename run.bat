@echo off
setlocal
REM Starts the two long-running pieces in their own windows:
REM   scheduler - fetches each closed bar, predicts, scores old predictions,
REM               places and resolves paper limit orders
REM   dashboard - serves http://127.0.0.1:8077
REM
REM Nothing here touches an exchange account: there is no API key and no code
REM path that could place a real order.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: .venv\Scripts\python.exe not found.
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

echo Stopping anything already running...
REM /T takes the python.exe under each window with it. Without it only cmd.exe
REM died, the old scheduler kept running, and a second one started beside it.
taskkill /F /T /FI "WINDOWTITLE eq cryptopred scheduler*" >nul 2>&1
taskkill /F /T /FI "WINDOWTITLE eq cryptopred dashboard*" >nul 2>&1
timeout /t 2 /nobreak >nul

echo Starting scheduler and dashboard...
start "cryptopred scheduler" cmd /k ".venv\Scripts\python.exe -m cryptopred.serve.cli schedule --interval 1h --minute 2"
start "cryptopred dashboard" cmd /k ".venv\Scripts\python.exe -m uvicorn cryptopred.serve.api:app --host 127.0.0.1 --port 8077"

echo Waiting for both to come up...
timeout /t 12 /nobreak >nul

REM Verify rather than assume. A window that opened and died leaves no trace
REM otherwise, and the failure would only surface days later as an empty log.
REM Each piece is counted on its own: one count for both said OK while the
REM scheduler - the piece that matters - had died at start-up. .Where() rather
REM than a pipe: the pipe sits inside cmd's quotes here, where ^ is not an
REM escape, so it reached PowerShell as a stray argument to Get-CimInstance.
REM With no pipe there is nothing for cmd to escape.
set SCHED=0
set DASH=0
for /f %%N in ('powershell -NoProfile -Command ^
  "@((Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\").Where({ $_.CommandLine -like '*cryptopred.serve.cli schedule*' })).Count"') do set SCHED=%%N
for /f %%N in ('powershell -NoProfile -Command ^
  "@((Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\").Where({ $_.CommandLine -like '*uvicorn cryptopred*' })).Count"') do set DASH=%%N

echo.
if "%SCHED%"=="0" echo ERROR: the scheduler is not running.
if "%DASH%"=="0" echo ERROR: the dashboard is not running.
if "%SCHED%"=="0" goto notrunning
if "%DASH%"=="0" goto notrunning

echo OK - scheduler and dashboard are both running.
echo.
echo Dashboard: http://127.0.0.1:8077
echo Results:   status.bat
echo.
echo Leave the two windows open. Closing one stops that piece.
echo If the machine restarts, run this file again - bars missed while it was off
echo are refilled and flagged, so the gap stays visible instead of vanishing.
echo.
start http://127.0.0.1:8077
pause
exit /b 0

:notrunning
echo Look at the window that opened for the error message - it stays open on
echo purpose.
echo.
pause
exit /b 1
