@echo off
setlocal
REM Removes the scheduled tasks created by install-task.bat and stops both
REM processes. Safe to run even if the tasks were never created.

set TASK_SCHED=cryptopred scheduler
set TASK_DASH=cryptopred dashboard
set TASK_CHECK=cryptopred model check

echo Stopping and removing the scheduled tasks...
schtasks /End    /TN "%TASK_SCHED%" >nul 2>&1
schtasks /End    /TN "%TASK_DASH%"  >nul 2>&1
schtasks /Delete /TN "%TASK_SCHED%" /F >nul 2>&1
schtasks /Delete /TN "%TASK_DASH%"  /F >nul 2>&1
schtasks /Delete /TN "%TASK_CHECK%" /F >nul 2>&1

REM /End stops the task, but a process it already spawned can outlive it.
echo Stopping any process still running...
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" | Where-Object { $_.CommandLine -like '*cryptopred.serve.cli schedule*' -or $_.CommandLine -like '*uvicorn cryptopred*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1

echo.
echo Done. Nothing starts at logon any more.
echo Your data is untouched - predictions.db, the parquet store and the model
echo registry are all still there. run.bat still works by hand.
echo.
pause
