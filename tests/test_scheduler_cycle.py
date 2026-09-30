"""The scheduler cycle: what it refreshes, and what a failure at start-up does."""

import httpx

from cryptopred.config import Config


def test_a_failing_first_cycle_does_not_stop_the_scheduler_starting(monkeypatch):
    """Under `restart: unless-stopped`, an exception here was a crash loop that
    re-requested years of bars every few seconds."""
    from cryptopred.serve import cli

    def blocked(cfg, interval):
        raise httpx.HTTPStatusError(
            "451", request=httpx.Request("GET", "https://x"), response=httpx.Response(451)
        )

    monkeypatch.setattr(cli, "run_cycle", blocked)
    assert cli.first_cycle(Config(), "1h") is False


def test_every_cycle_refreshes_funding_as_well_as_bars(tmp_path, monkeypatch):
    """The model trains on funding features and the predictor reads them from
    the store. A cycle that fetched only bars left them frozen for weeks."""
    from cryptopred.serve import runner

    calls = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(runner, "BinanceClient", FakeClient)
    monkeypatch.setattr(runner, "run_klines_ingest", lambda *a: calls.append("klines") or 0)
    monkeypatch.setattr(runner, "run_funding_ingest", lambda *a: calls.append("funding") or 0)

    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = []  # nothing to predict; only the sync step is under test
    counts = runner.run_cycle(cfg, interval="1h")
    assert calls == ["klines", "funding"]
    assert counts["funding"] == 0
