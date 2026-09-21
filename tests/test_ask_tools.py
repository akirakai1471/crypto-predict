"""The tools are the surface the model sees.

Their schemas must be strict, and their descriptions must carry the warnings,
because the model reads the description before it reads the values.
"""

import pandas as pd
import pytest

from cryptopred.ask.tools import TOOL_SCHEMAS, BriefingTools
from tests.conftest import make_ohlcv


@pytest.fixture
def tools(tmp_path):
    from cryptopred.config import Config
    from cryptopred.ingest.storage import ParquetStore

    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT", "ETHUSDT"]
    ParquetStore(tmp_path / "raw").write(
        "klines", "BTCUSDT", "1h", make_ohlcv(n=3000, seed=61)
    )
    return BriefingTools(cfg)


def test_all_six_tools_are_declared():
    names = {s["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "market_snapshot",
        "indicators",
        "levels",
        "touch_probability",
        "model_signal",
        "track_record",
    }


def test_every_schema_is_strict_and_closed():
    for schema in TOOL_SCHEMAS:
        assert schema["strict"] is True, schema["name"]
        assert schema["input_schema"]["additionalProperties"] is False, schema["name"]
        assert "required" in schema["input_schema"], schema["name"]


def test_the_indicator_tool_description_carries_the_warning():
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "indicators")
    assert "chưa đo" in schema["description"].lower()


def test_target_is_a_discriminated_object_not_a_bare_number():
    """Passing -3 must not leave the tool guessing between 3% down and a price
    of minus three."""
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "touch_probability")
    target = schema["input_schema"]["properties"]["target"]
    assert target["properties"]["kind"]["enum"] == ["pct", "price"]


def test_touch_probability_accepts_an_absolute_price(tools):
    out = tools.touch_probability(
        symbol="BTCUSDT", target={"kind": "price", "value": 20_000.0}, horizon_hours=24
    )
    assert "conditional" in out and "unconditional" in out
    assert out["target_price"] == pytest.approx(20_000.0)


def test_an_absolute_price_above_the_market_tests_upward(tools):
    """The direction of the test follows from where the price sits, not from a
    separate flag the caller could set inconsistently."""
    last = tools.last_close("BTCUSDT")
    out = tools.touch_probability(
        symbol="BTCUSDT", target={"kind": "price", "value": last * 1.05}, horizon_hours=24
    )
    assert out["target_pct"] > 0


def test_a_symbol_outside_the_allowed_pair_is_refused(tools):
    out = tools.market_snapshot(symbol="SOLUSDT")
    assert out["source"] == "unavailable"
    assert "BTCUSDT" in out["reason"]


def test_model_signal_for_a_symbol_with_no_model_explains_why(tools):
    out = tools.model_signal(symbol="ETHUSDT")
    assert out["source"] == "unavailable"
    assert "model" in out["reason"].lower()


def test_track_record_with_an_empty_log_is_unavailable(tools):
    out = tools.track_record(symbol="BTCUSDT")
    assert out["source"] == "unavailable"


def test_track_record_with_predictions_but_none_scored_yet(tools, tmp_path):
    """0 correct out of 0 is 0.0, which reads as a measured failure rather than
    as no measurement. `Measured` rejects n <= 0 to stop exactly this, so this
    path must produce an Unavailable accuracy rather than raising."""
    from cryptopred.serve.store import PredictionStore

    store = PredictionStore(tmp_path / "predictions.db")
    store.record_prediction(
        symbol="BTCUSDT",
        interval="1h",
        bar_close_time=pd.Timestamp("2024-02-01", tz="UTC"),
        proba=(0.2, 0.3, 0.5),
        signal=0,
        close_price=42000.0,
        model_version="v1",
    )
    out = tools.track_record(symbol="BTCUSDT")
    assert out["n_scored"] == 0
    assert out["accuracy"]["source"] == "unavailable"


def test_every_tool_returns_json_serialisable_output(tools):
    """Tool results go over the wire as JSON. A numpy float or a Timestamp here
    would fail at the API boundary rather than in any test."""
    import json

    payloads = [
        tools.market_snapshot(symbol="BTCUSDT"),
        tools.indicators(symbol="BTCUSDT"),
        tools.levels(symbol="BTCUSDT"),
        tools.touch_probability(
            symbol="BTCUSDT", target={"kind": "pct", "value": -0.03}, horizon_hours=24
        ),
        tools.model_signal(symbol="BTCUSDT"),
        tools.track_record(symbol="BTCUSDT"),
    ]
    for payload in payloads:
        json.dumps(payload)
