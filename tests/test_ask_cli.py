"""With no credentials the CLI must explain itself, not raise."""

from typer.testing import CliRunner

from cryptopred.ask.cli import app
from cryptopred.ask.session import AskResult

runner = CliRunner()


def _result(answer, unmatched):
    return AskResult(
        answer=answer,
        audit={
            "ok": not unmatched,
            "matched": [],
            "unmatched": unmatched,
            "limitation": "chỉ kiểm được con số",
        },
        usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0},
        cost_usd=0.0038,
    )


def test_missing_credentials_explains_and_points_at_the_brief(monkeypatch):
    monkeypatch.setattr("cryptopred.ask.cli._credentials_available", lambda: False)
    result = runner.invoke(app, ["hỏi gì đó"])
    assert result.exit_code == 1
    assert "cryptopred-brief" in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_unmatched_numbers_are_flagged_with_an_instruction_not_to_trust_them(
    monkeypatch,
):
    monkeypatch.setattr("cryptopred.ask.cli._credentials_available", lambda: True)
    monkeypatch.setattr(
        "cryptopred.ask.cli.answer_question",
        lambda *a, **k: _result("Giá về 2.900.", [2900.0]),
    )
    result = runner.invoke(app, ["hỏi gì đó", "--no-fetch"])
    assert result.exit_code == 0
    assert "KHÔNG TRUY ĐƯỢC NGUỒN" in result.output
    assert "$0.0038" in result.output


def test_a_clean_answer_says_every_number_was_traced(monkeypatch):
    monkeypatch.setattr("cryptopred.ask.cli._credentials_available", lambda: True)
    monkeypatch.setattr(
        "cryptopred.ask.cli.answer_question", lambda *a, **k: _result("58,4%.", [])
    )
    result = runner.invoke(app, ["hỏi gì đó", "--no-fetch"])
    assert result.exit_code == 0
    assert "truy được" in result.output
    assert "KHÔNG TRUY ĐƯỢC NGUỒN" not in result.output


def test_the_audit_limitation_is_always_printed(monkeypatch):
    """The audit's silence on prose must be visible, not mistaken for approval."""
    monkeypatch.setattr("cryptopred.ask.cli._credentials_available", lambda: True)
    monkeypatch.setattr(
        "cryptopred.ask.cli.answer_question", lambda *a, **k: _result("ok", [])
    )
    result = runner.invoke(app, ["hỏi gì đó", "--no-fetch"])
    assert "chỉ kiểm được con số" in result.output


def test_json_output_is_machine_readable(monkeypatch):
    import json

    monkeypatch.setattr("cryptopred.ask.cli._credentials_available", lambda: True)
    monkeypatch.setattr(
        "cryptopred.ask.cli.answer_question", lambda *a, **k: _result("ok", [])
    )
    result = runner.invoke(app, ["hỏi gì đó", "--no-fetch", "--json"])
    payload = json.loads(result.output)
    assert payload["answer"] == "ok"
    assert payload["cost_usd"] == 0.0038


def test_a_failed_refresh_is_non_fatal_and_says_so(monkeypatch):
    """Answering silently on stale data is the failure this guards against."""
    monkeypatch.setattr("cryptopred.ask.cli._credentials_available", lambda: True)
    monkeypatch.setattr(
        "cryptopred.ask.cli.answer_question", lambda *a, **k: _result("ok", [])
    )

    def boom(*a, **k):
        raise RuntimeError("mạng hỏng")

    monkeypatch.setattr("cryptopred.ask.cli._refresh_bars", boom)
    result = runner.invoke(app, ["hỏi gì đó"])
    assert result.exit_code == 0
    assert "mạng hỏng" in result.output
    assert "đã lưu" in result.output


# -- review findings, 2026-10-01 -----------------------------------------------


def test_the_refresh_reaches_the_kline_ingest(monkeypatch):
    """It imported cryptopred.ingest.runner, which does not exist. The
    ImportError landed in the "stale answer beats no answer" handler, so every
    answer was computed on stored bars - and the test above, which replaces
    _refresh_bars wholesale, never ran the import."""
    from cryptopred.ask.cli import _refresh_bars
    from cryptopred.config import load_config

    calls = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("cryptopred.ingest.binance.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "cryptopred.ingest.cli.run_klines_ingest",
        lambda cfg, client, store: calls.append(list(cfg.data.intervals)),
    )
    _refresh_bars(load_config(None), "4h")
    assert calls == [["4h"]]


def test_a_cp1252_console_does_not_crash_the_answer(monkeypatch):
    """Every line is Vietnamese. On a Windows console, typer.echo raised on the
    first one it could not encode - after the answer had been paid for."""
    import typer

    printed = []
    real_echo = typer.echo

    def cp1252_console(message="", *args, **kwargs):
        str(message).encode("cp1252")
        printed.append(str(message))
        real_echo(message, *args, **kwargs)

    monkeypatch.setattr(typer, "echo", cp1252_console)
    monkeypatch.setattr("cryptopred.ask.cli._credentials_available", lambda: True)
    monkeypatch.setattr(
        "cryptopred.ask.cli.answer_question",
        lambda *a, **k: _result("Giá về 2.900.", [2900.0]),
    )
    result = runner.invoke(app, ["hỏi gì đó", "--no-fetch"])
    assert result.exit_code == 0, result.output
    assert any("2,900" in line for line in printed)
    assert any("$0.0038" in line for line in printed)
