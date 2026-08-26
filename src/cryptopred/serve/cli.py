"""Command line entry point for the live service."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import uvicorn

from cryptopred.config import load_config
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


@app.command()
def schedule(
    interval: str = typer.Option("1h", help="Bar interval to schedule."),
    minute: int = typer.Option(
        2, help="Minutes past the hour to run, giving Binance time to publish the bar."
    ),
    config: Path = typer.Option(None, help="Path to a YAML config file."),
) -> None:
    """Run a cycle after every bar close, forever."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_cycle,
        "cron",
        minute=minute,
        args=[cfg, interval],
        # A slow cycle must not stack up behind itself.
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    typer.echo(f"Scheduled: every hour at minute {minute} UTC. Ctrl-C to stop.")
    run_cycle(cfg, interval=interval)  # run once immediately so the log is current
    scheduler.start()


if __name__ == "__main__":
    app()
