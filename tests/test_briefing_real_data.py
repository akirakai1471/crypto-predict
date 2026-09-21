"""Run the measurement layer over the actual store, if it is present.

Synthetic bars cannot catch a column that the real parquet files name
differently, or a regime cell that turns out to be empty on seven years of ETH.
Skipped in a clean checkout so the suite stays portable.
"""

import pandas as pd
import pytest

from cryptopred.briefing.report import format_brief
from cryptopred.briefing.touch import MIN_CELL_BARS, touch_probability
from cryptopred.config import load_config
from cryptopred.ingest.storage import ParquetStore


@pytest.fixture
def eth_bars():
    cfg = load_config(None)
    bars = ParquetStore(cfg.data.root / "raw").read("klines", "ETHUSDT", "1h")
    if bars.empty:
        pytest.skip("no stored ETHUSDT bars in this checkout")
    return bars


def test_touch_probability_runs_on_seven_years_of_eth(eth_bars):
    result = touch_probability(eth_bars, target_pct=-0.03, horizon=72)
    assert result["unconditional"].n > 10_000
    assert 0.0 <= result["unconditional"].value <= 1.0


def test_the_conditional_cell_is_populated_on_real_data(eth_bars):
    """If nine cells leave the current one below the floor on 59,000 bars, the
    bucketing is too fine and the floor needs revisiting.

    Do not lower MIN_CELL_BARS to make this pass - the fix would be coarsening
    the trend dimension, and the finding belongs in docs/findings.md.
    """
    result = touch_probability(eth_bars, target_pct=-0.03, horizon=72)
    conditional = result["conditional"].to_dict()
    assert conditional["source"] == "measured", conditional.get("reason")
    assert conditional["n"] >= MIN_CELL_BARS


def test_conditioning_changes_the_answer_enough_to_be_worth_doing(eth_bars):
    """If the conditional and unconditional figures were always identical, nine
    cells of machinery would be buying nothing."""
    result = touch_probability(eth_bars, target_pct=-0.03, horizon=72)
    gap = abs(result["conditional"].value - result["unconditional"].value)
    assert gap > 0.01


def test_wait_times_come_from_the_cell_on_real_data(eth_bars):
    result = touch_probability(eth_bars, target_pct=-0.03, horizon=72)
    assert result["wait_source"] == "cell"
    assert result["wait_hours"]["median"] <= result["wait_hours"]["p90"]


def test_the_brief_renders_for_eth(eth_bars):
    text = format_brief("ETHUSDT", "1h", eth_bars, pd.DataFrame())
    assert "ĐO ĐƯỢC" in text
    assert "QUY ƯỚC" in text
    assert "95%" not in text
