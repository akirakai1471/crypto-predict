@echo off
setlocal
REM Evaluates the current model against fresh data and writes a report.
REM
REM Pass /quiet when running from Task Scheduler: it skips the trailing pause,
REM which would otherwise leave the task hanging on a keypress nobody is there
REM to give, forever, with the scheduler reporting it as still running.
REM
REM It does NOT pass --save, so nothing in the registry changes and the live
REM experiment keeps measuring the model it has been measuring. The point is to
REM be told the model has gone stale, not to have it quietly replaced.
REM
REM Why not just retrain automatically: on 2026-09-23 an automatic-looking
REM retrain put a model into the registry at -0.24% after costs with its own
REM report printing STRATEGY VERDICT: NO-GO one line above. A gate only works
REM if somebody reads it. See docs/findings.md.
REM
REM Still no exchange account. No API key, no order-placement path anywhere.

cd /d "%~dp0"

REM One flag, checked at every exit. A pause left in an error path is worse
REM than one on the happy path: the task hangs precisely when something has
REM gone wrong and nobody is watching.
set "INTERACTIVE=1"
if /i "%~1"=="/quiet" set "INTERACTIVE="

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: .venv\Scripts\python.exe not found. Create the environment first:
    echo     uv venv --python 3.12
    echo     uv pip install -e ".[dev]"
    echo.
    if defined INTERACTIVE pause
    exit /b 1
)

echo Fetching any bars that closed since the last run...
.venv\Scripts\python.exe -m cryptopred.ingest.cli klines
.venv\Scripts\python.exe -m cryptopred.ingest.cli funding

echo.
echo Rebuilding the 1h datasets...
.venv\Scripts\python.exe -m cryptopred.dataset.cli build --config configs\hourly-only.yaml

echo.
echo ============================================================
echo  How is the live model doing right now?
echo ============================================================
.venv\Scripts\python.exe -m cryptopred.serve.cli status

echo.
echo ============================================================
echo  What would a model trained on today's data look like?
echo  (evaluation only - nothing is saved)
echo ============================================================
.venv\Scripts\python.exe -m cryptopred.models.cli train --symbol BTCUSDT --interval 1h --config configs\hourly-only.yaml
.venv\Scripts\python.exe -m cryptopred.models.cli train --symbol ETHUSDT --interval 1h --config configs\hourly-only.yaml

echo.
echo ============================================================
echo  Reports written to data\reports\. Read the two VERDICT lines
echo  for each symbol - the classification one and the strategy one.
echo  BOTH must say GO before a model is worth saving.
echo.
echo  To actually replace the live model, run by hand and read the
echo  output before trusting it:
echo      uv run cryptopred-model train --symbol BTCUSDT --interval 1h --save
echo.
echo  Replacing it resets what the live log is measuring, so the rows
echo  collected against the old model stop being comparable.
echo ============================================================
echo.

if defined INTERACTIVE pause
