"""Running unattended on a server: alerts that reach a phone, and a dead
scheduler that someone hears about."""

import json
import logging

import httpx
import pandas as pd
import pytest

from cryptopred.config import Config
from cryptopred.serve import alerts, doctor, heartbeat, telegram, watchdog

TOKEN = "123456:SECRET-token_value"


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch):
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "HEALTHCHECK_PING_URL"):
        monkeypatch.delenv(name, raising=False)


def _telegram_transport(sent, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, json={"ok": status == 200, "description": "chat not found"})

    return httpx.MockTransport(handler)


# -- telegram -----------------------------------------------------------------


def test_unconfigured_telegram_is_off_and_silent():
    assert telegram.configured() is None
    assert telegram.send("hello") is False


def test_a_message_goes_to_the_configured_chat():
    sent = []
    assert telegram.send("xin chào", TOKEN, "42", transport=_telegram_transport(sent))
    body = json.loads(sent[0].content)
    assert body["chat_id"] == "42" and body["text"] == "xin chào"


def test_a_message_too_long_for_telegram_is_cut_not_lost():
    sent = []
    telegram.send("x" * 10_000, TOKEN, "42", transport=_telegram_transport(sent))
    assert len(json.loads(sent[0].content)["text"]) == telegram.MAX_TEXT


def test_a_rejected_or_unreachable_send_never_raises():
    assert telegram.send("x", TOKEN, "42", transport=_telegram_transport([], 400)) is False

    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    assert telegram.send("x", TOKEN, "42", transport=httpx.MockTransport(boom)) is False


def test_the_bot_token_never_reaches_a_log(caplog):
    """httpx logs every request URL at INFO, and the token is in the URL. On a
    server, that log is what `docker compose logs` shows anyone who asks."""
    caplog.set_level(logging.DEBUG)

    def boom(request):
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    telegram.send("x", TOKEN, "42", transport=_telegram_transport([]))
    telegram.send("x", TOKEN, "42", transport=_telegram_transport([], 401))
    telegram.send("x", TOKEN, "42", transport=httpx.MockTransport(boom))
    assert "SECRET" not in caplog.text


def test_alerts_are_forwarded_whole_when_telegram_is_configured(tmp_path, monkeypatch):
    """The phone gets the caveat too - it is the part that stops an alert
    reading as advice."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    forwarded = []
    monkeypatch.setattr(alerts.telegram, "send", lambda text: forwarded.append(text) or True)
    monkeypatch.setattr(alerts.subprocess, "Popen", lambda *a, **k: None)

    message = "BTCUSDT — model bắn LONG\nHồ sơ: đúng 7/15\n\nKHÔNG phải khuyến nghị"
    alerts.notify(message, log_path=tmp_path / "signals.log")
    assert forwarded == [message]


# -- the dead-man's switch ----------------------------------------------------


def test_the_external_ping_fires_after_each_heartbeat(tmp_path, monkeypatch):
    hits = []
    monkeypatch.setenv("HEALTHCHECK_PING_URL", "https://hc-ping.example/abc")
    real_client = httpx.Client

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda r: hits.append(str(r.url)) or httpx.Response(200)
        )
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    heartbeat.write(tmp_path / "heartbeat.json", "1h", {"predictions": 1})
    assert hits == ["https://hc-ping.example/abc"]


def test_a_failing_external_ping_costs_nothing(monkeypatch):
    monkeypatch.setenv("HEALTHCHECK_PING_URL", "https://hc-ping.example/abc")

    def boom(request):
        raise httpx.ConnectTimeout("slow", request=request)

    assert heartbeat.ping_external(transport=httpx.MockTransport(boom)) is False


# -- the watchdog -------------------------------------------------------------


def _beat(path, minutes_ago):
    stamp = pd.Timestamp("2026-09-30T12:00Z") - pd.Timedelta(minutes=minutes_ago)
    path.write_text(json.dumps({"last_cycle": stamp.isoformat()}), encoding="utf-8")


def test_one_message_when_the_scheduler_dies_and_one_when_it_returns(tmp_path):
    """A message every ten minutes for a day is a message nobody reads."""
    path = tmp_path / "heartbeat.json"
    sent = []
    dog = watchdog.SchedulerWatchdog(path, send=lambda text: sent.append(text) or True)
    now = pd.Timestamp("2026-09-30T12:00Z")

    _beat(path, 30)
    assert dog.check(now=now) == "ok"
    _beat(path, 300)
    assert dog.check(now=now) == "down"
    assert dog.check(now=now) == "ok"  # still down, already said so
    _beat(path, 5)
    assert dog.check(now=now) == "recovered"

    assert len(sent) == 2
    assert "ĐÃ DỪNG" in sent[0] and "chạy lại" in sent[1]


def test_a_scheduler_that_never_ran_is_an_outage_too(tmp_path):
    sent = []
    dog = watchdog.SchedulerWatchdog(tmp_path / "missing.json", send=sent.append)
    assert dog.check() == "down"


def test_the_watchdog_does_not_start_without_telegram(tmp_path):
    assert watchdog.start(tmp_path / "heartbeat.json") is None


# -- doctor -------------------------------------------------------------------


def _cfg(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    return cfg


def _web_ok():
    return httpx.MockTransport(lambda r: httpx.Response(200, text="<rss/>"))


def test_doctor_names_a_geo_blocked_region_plainly(tmp_path):
    """The failure a new VPS is most likely to have, and the one with no code fix."""
    blocked = httpx.MockTransport(lambda r: httpx.Response(451))
    checks = doctor.run_checks(_cfg(tmp_path), binance_transport=blocked, web_transport=_web_ok())
    binance = next(c for c in checks if c.name.startswith("Binance"))
    assert binance.state == "fail" and "451" in binance.detail
    assert not doctor.healthy(checks)
    assert "KHÔNG SẴN SÀNG" in doctor.format_checks(checks)


def test_doctor_passes_a_reachable_machine_and_warns_about_a_missing_model(tmp_path):
    ok = httpx.MockTransport(
        lambda r: httpx.Response(200, json=[] if "klines" in r.url.path else {})
    )
    checks = doctor.run_checks(_cfg(tmp_path), binance_transport=ok, web_transport=_web_ok())
    by_name = {c.name: c for c in checks}
    assert by_name["Binance futures API"].state == "ok"
    assert by_name["model BTCUSDT"].state == "warn"
    assert doctor.healthy(checks)  # a missing model is a warning, not a blocker


def test_doctor_can_send_a_telegram_test(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    sent = []
    checks = doctor.run_checks(
        _cfg(tmp_path),
        send_test=True,
        binance_transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])),
        web_transport=_telegram_transport(sent),
    )
    assert next(c for c in checks if c.name == "Telegram").state == "ok"
    assert any("sendMessage" in str(r.url) for r in sent)


def test_health_exits_nonzero_without_a_recent_cycle(tmp_path):
    from typer.testing import CliRunner

    from cryptopred.serve.cli import app

    config = tmp_path / "cfg.yaml"
    config.write_text(f"data:\n  root: {tmp_path}\n", encoding="utf-8")
    runner = CliRunner()
    assert runner.invoke(app, ["health", "--config", str(config)]).exit_code == 1
    heartbeat.write(tmp_path / "heartbeat.json", "1h", {})
    assert runner.invoke(app, ["health", "--config", str(config)]).exit_code == 0

