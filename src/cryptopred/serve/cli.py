"""Command line entry point for the live service."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import uvicorn

from cryptopred.config import Config, load_config
from cryptopred.report_io import safe_echo
from cryptopred.serve.runner import run_cycle

app = typer.Typer(help="Run the prediction service and its scheduled job.")
logger = logging.getLogger(__name__)


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)

    scheduler = build_scheduler(cfg, interval, minute)
    typer.echo(f"Scheduled: every hour at minute {minute} UTC. Ctrl-C to stop.")
    if scheduler.get_job("news") is not None:
        safe_echo(
            f"Tin tức: quét {len(cfg.news.feeds)} nguồn RSS mỗi {cfg.news.poll_seconds} giây "
            "— chỉ để xem, model không dùng tin."
        )
    run_cycle(cfg, interval=interval)  # run once immediately so the log is current
    scheduler.start()


if __name__ == "__main__":
    app()
