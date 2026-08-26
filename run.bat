@echo off
REM Starts the two long-running pieces in their own windows:
REM   scheduler - fetches each closed bar, predicts, scores old predictions,
REM               places and resolves paper limit orders
REM   dashboard - serves http://127.0.0.1:8077
REM
REM Close either window to stop that piece. Nothing here touches an exchange
REM account: there is no API key and no code path that could place a real order.

cd /d "%~dp0"

echo Starting scheduler and dashboard...
start "cryptopred scheduler" cmd /k ".venv\Scripts\python.exe -m cryptopred.serve.cli schedule --interval 1h --minute 2"
timeout /t 3 /nobreak >nul
start "cryptopred dashboard" cmd /k ".venv\Scripts\python.exe -m uvicorn cryptopred.serve.api:app --host 127.0.0.1 --port 8077"

timeout /t 3 /nobreak >nul
start http://127.0.0.1:8077

echo.
echo Both running. To check results later:
echo    status.bat
echo.
