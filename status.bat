@echo off
REM What has actually happened since the scheduler started.
REM Reads only the prediction log, whose rows were written before their
REM outcomes existed.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -m cryptopred.serve.cli status --interval 1h
echo.
pause
