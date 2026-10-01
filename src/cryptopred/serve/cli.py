"""Command line entry point for the live service."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import typer
import uvicorn

from cryptopred.config import Config, load_config
from cryptopred.report_io import safe_echo
from cryptopred.serve.runner import run_cycle

app = typer.Typer(help="Run the prediction service and its scheduled job.")
logger = logging.getLogger(__name__)

# A few weeks of hourly cycles each; bounded so a year of them cannot fill a disk.
LOG_FILE_BYTES = 5_000_000
LOG_FILE_COUNT = 3


def scheduler_log_path(cfg: Config) -> Path:
    return Path(cfg.data.root) / "logs" / "scheduler.log"


def setup_scheduler_logging(cfg: Config) -> Path | None:
    """Log to the console when there is one, and always to a file.

    install-task.bat starts the scheduler under pythonw, which has no console:
    sys.stderr is None there, and every log line - the traceback of a failed
    cycle included - went nowhere. The file under data/logs is where to look
    when status.bat says the last cycle is old. Returns its path, or None if it
    could not be opened.
    """
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = []
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    path: Path | None = scheduler_log_path(cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                path, maxBytes=LOG_FILE_BYTES, backupCount=LOG_FILE_COUNT, encoding="utf-8"
            )
        )
    except OSError:
        path = None
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return path


@app.command()
def api(
    host: str = typer.Option("127.0.0.1", help="Bind address. Localhost by default."),
    port: int = typer.Option(8077, help="Port to serve on."),
) -> None:
    """Serve the dashboard and JSON API."""
    uvicorn.run("cryptopred.serve.api:app", host=host, port=port, log_level="info")


@app.command()
def cycle(
    interval: str = typer.Option("1h", help="Bar interval to run a cycle for."),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Run one pass: sync bars, predict, score, and update the paper trader."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = run_cycle(load_config(config), interval=interval)
    typer.echo(
        f"bars={counts['bars']} predictions={counts['predictions']} "
        f"scored={counts['scored']} opened={counts['opened']} closed={counts['closed']}"
    )


@app.command()
def status(
    interval: str = typer.Option("1h", help="Bar interval to report on."),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """What has actually happened since the scheduler started running.

    Reads only the prediction log, whose rows were written before their outcomes
    existed. Sample size is reported before every rate, because a hit rate over
    a handful of signals is noise wearing a percentage sign.
    """
    from cryptopred.serve.status import collect, format_status

    cfg = load_config(config)
    typer.echo(format_status(collect(cfg, interval=interval)))


@app.command()
def doctor(
    send_test: bool = typer.Option(
        False, "--send-test", help="Also send a Telegram test message."
    ),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Check this machine can run unattended: Binance, models, data, alerts."""
    from cryptopred.serve.doctor import format_checks, healthy, run_checks

    checks = run_checks(load_config(config), send_test=send_test)
    safe_echo(format_checks(checks))
    raise typer.Exit(code=0 if healthy(checks) else 1)


@app.command()
def health(config: Path = typer.Option(None, help="Path to a YAML config file.")) -> None:
    """Exit 0 if the scheduler finished a cycle recently, 1 otherwise.

    For container health checks: the same heartbeat test the status report
    and the dashboard use, as an exit code.
    """
    from cryptopred.serve import heartbeat

    beat = heartbeat.status(load_config(config).data.root / "heartbeat.json")
    typer.echo(beat["detail"])
    raise typer.Exit(code=0 if beat["state"] == "alive" else 1)


def _news_job(cfg: Config):
    """The news poller, or None if it cannot start.

    Building it opens data/news.db, and a failure there - a locked file, a full
    disk - must not stop the predictions from being scheduled. News is a
    display feature; the prediction log is the experiment.
    """
    if not cfg.news.feeds:
        return None
    try:
        from cryptopred.news.poller import NewsJob

        return NewsJob(cfg)
    except Exception:  # noqa: BLE001 - news must never cost the prediction job
        logger.exception("news collection disabled: could not start it")
        return None


class QuietNewsRuns(logging.Filter):
    """Drop APScheduler's routine INFO lines for the news job only.

    It logs "Running job" and "executed successfully" for every run. At one
    news poll a minute that is 2,880 lines a day, burying the hourly cycle's
    log that someone reads to see whether predictions happened. Warnings and
    errors about the news job - missed runs, a run still going when the next
    was due - still pass, and the job logs for itself when it stored or failed
    something.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.INFO or not record.args:
            return True
        args = record.args if isinstance(record.args, tuple) else (record.args,)
        return getattr(args[0], "id", None) != "news"


def _quiet_news_runs() -> None:
    executor_log = logging.getLogger("apscheduler.executors.default")
    if not any(isinstance(f, QuietNewsRuns) for f in executor_log.filters):
        executor_log.addFilter(QuietNewsRuns())


def build_scheduler(cfg: Config, interval: str, minute: int):
    """The hourly prediction cycle, plus the news poll every cfg.news.poll_seconds.

    Separate jobs, so separate pool threads: a slow or hung feed delays only the
    next news pass. Both run one instance at a time and coalesce missed runs,
    so neither can stack up behind itself.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_cycle,
        "cron",
        minute=minute,
        args=[cfg, interval],
        id="cycle",
        # A slow cycle must not stack up behind itself.
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    news = _news_job(cfg)
    if news is not None:
        scheduler.add_job(
            news,
            "interval",
            seconds=cfg.news.poll_seconds,
            id="news",
            name="news",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=cfg.news.poll_seconds,
        )
        _quiet_news_runs()
    return scheduler


@app.command()
def schedule(
    interval: str = typer.Option("1h", help="Bar interval to schedule."),
    minute: int = typer.Option(
        2, help="Minutes past the hour to run, giving Binance time to publish the bar."
    ),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Run a cycle after every bar close, and poll news, forever."""
    cfg = load_config(config)
    log_path = setup_scheduler_logging(cfg)
    if log_path is not None:
        logger.info("logging to %s", log_path)

    scheduler = build_scheduler(cfg, interval, minute)
    typer.echo(f"Scheduled: every hour at minute {minute} UTC. Ctrl-C to stop.")
    if scheduler.get_job("news") is not None:
        safe_echo(
            f"Tin tức: quét {len(cfg.news.feeds)} nguồn RSS mỗi {cfg.news.poll_seconds} giây "
            "— chỉ để xem, model không dùng tin."
        )
    first_cycle(cfg, interval)  # run once immediately so the log is current
    scheduler.start()


def first_cycle(cfg: Config, interval: str) -> bool:
    """The immediate cycle at start-up, which must not be able to stop the start.

    Scheduled cycles are wrapped by APScheduler, so one that fails is logged and
    the next hour tries again. This one ran bare: a network blip or an HTTP 451
    at start-up killed the process before the scheduler existed, and under a
    restart policy that became a crash loop re-requesting years of bars every
    few seconds. Now it is logged, and the hourly job carries on.
    """
    try:
        run_cycle(cfg, interval=interval)
        return True
    except Exception:  # noqa: BLE001 - the scheduled job is the retry
        logger.exception(
            "the start-up cycle failed; the scheduler starts anyway and the next "
            "hourly cycle will retry. `cryptopred-serve doctor` checks the usual causes."
        )
        return False


if __name__ == "__main__":
    app()
