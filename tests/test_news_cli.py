"""`cryptopred-news`: poll, watch, recent. Offline, through MockTransport."""

from pathlib import Path

import httpx
import pandas as pd
import pytest
from typer.testing import CliRunner

from cryptopred.config import FeedConfig
from cryptopred.news import cli
from cryptopred.news.fetch import FeedFetcher
from cryptopred.news.parse import Entry
from cryptopred.news.store import NewsStore

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures" / "news"
RSS = (FIXTURES / "rss2.xml").read_bytes()


def _handler(request: httpx.Request) -> httpx.Response:
    if "dead" in request.url.host:
        raise httpx.ConnectTimeout("no route", request=request)
    return httpx.Response(200, content=RSS)


@pytest.fixture
def config(tmp_path, monkeypatch):
    """A config rooted in tmp_path with two feeds, one dead, and every fetcher
    the CLI builds routed through MockTransport."""
    path = tmp_path / "cfg.yaml"
    path.write_text(
        f"data:\n  root: {tmp_path.as_posix()}\n"
        "news:\n  feeds:\n"
        "    - {name: Live, url: 'https://live.example.com/rss'}\n"
        "    - {name: Dead, url: 'https://dead.example.com/rss'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli, "FeedFetcher",
        lambda feeds: FeedFetcher(feeds, transport=httpx.MockTransport(_handler)),
    )
    return path


def test_help_lists_the_three_commands():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("poll", "watch", "recent"):
        assert command in result.output


def test_poll_prints_per_feed_counts_and_survives_a_dead_feed(config, tmp_path):
    result = runner.invoke(cli.app, ["poll", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "Live" in result.output and "3 mới" in result.output
    assert "Dead" in result.output and "LỖI" in result.output
    assert "Tổng: 3 tin mới · 1/2 nguồn trả lời · 1 lỗi" in result.output
    # The first pass is the RSS backlog, and it says so.
    assert "tồn đọng" in result.output
    assert NewsStore(tmp_path / "news.db").count() == 3


def test_recent_prints_newest_first_with_the_caveat(config):
    runner.invoke(cli.app, ["poll", "--config", str(config)])
    result = runner.invoke(cli.app, ["recent", "--config", str(config), "--limit", "2"])
    assert result.exit_code == 0, result.output
    assert "QUY ƯỚC — CHƯA KIỂM CHỨNG" in result.output
    assert "NHẬN tin (UTC)" in result.output
    assert result.output.count("\n  https://") == 2


def test_recent_filters_by_symbol(config):
    runner.invoke(cli.app, ["poll", "--config", str(config)])
    result = runner.invoke(cli.app, ["recent", "--config", str(config), "--symbol", "ethusdt"])
    assert "Ethereum developers set date" in result.output
    assert "Spot Bitcoin ETF" not in result.output


def test_recent_on_an_empty_store_says_how_to_fill_it(config):
    result = runner.invoke(cli.app, ["recent", "--config", str(config)])
    assert result.exit_code == 0
    assert "cryptopred-news poll" in result.output


# -- watch ---------------------------------------------------------------------------------


def test_watch_keeps_going_after_a_pass_that_raises(tmp_path):
    """Each pass stands alone. A watch that dies on the first hiccup is a watch
    nobody can leave running."""
    store = NewsStore(tmp_path / "news.db")
    fetcher = FeedFetcher(
        [FeedConfig(name="Live", url="https://live.example.com/rss")],
        transport=httpx.MockTransport(_handler),
    )
    calls = {"n": 0}
    real_add = store.add

    def flaky_add(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("database is locked")
        return real_add(*args, **kwargs)

    store.add = flaky_add
    printed: list[str] = []
    passes = cli.run_watch(
        fetcher, store, every=60, echo=printed.append, sleep=lambda _s: None, max_passes=3
    )
    assert passes == 3
    assert any("Lượt quét lỗi" in line for line in printed)
    assert store.count() == 3


def test_watch_prints_new_headlines_but_not_the_backlog(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    store.add(
        [Entry(uid="old", title="Something from last week", link=None, published_at=None)],
        source="Seed", received_at=pd.Timestamp("2026-09-20T00:00Z"),
    )
    fetcher = FeedFetcher(
        [FeedConfig(name="Live", url="https://live.example.com/rss")],
        transport=httpx.MockTransport(_handler),
    )
    printed: list[str] = []
    cli.run_watch(fetcher, store, every=60, echo=printed.append, sleep=lambda _s: None,
                  max_passes=1)
    # Three backlog items: summarised in one line, not dumped.
    assert len(printed) == 1
    assert "nạp 3 tin tồn đọng" in printed[0]


def test_a_headline_line_marks_tags_and_impact_as_questions():
    row = {
        "received_at": "2026-09-30T12:34:56.000000+00:00",
        "source": "Example",
        "title": "SEC approves spot ether ETFs",
        "link": "https://news.example.com/a",
        "symbols": ["ETHUSDT"],
        "high_impact": True,
        "impact_terms": ["SEC", "approves", "ETFs"],
        "is_backlog": 0,
    }
    line = cli.format_headline(row)
    assert line.startswith("2026-09-30 12:34 UTC · Example · ETHUSDT · TÁC ĐỘNG CAO?")
