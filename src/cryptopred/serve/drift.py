"""Is the deployed model still firing as often as it was built to?

This exists because of a failure it would have caught immediately. A model was
saved reporting that it traded 12.87% of bars against an 8% target, and then
produced zero signals across 336 logged bars over sixteen days. Nothing on the
dashboard said so: an empty signal list looks identical to a quiet market.

The check is deliberately crude. It does not ask whether the model is right — the
prediction log answers that, slowly, and only after each horizon elapses. It asks
whether the model is *doing anything*, which is knowable from the first day and
is the failure that wastes the most time when it goes unnoticed.

Only rows from the current model version count. An older version's probabilities
came from a different calibrator on a different scale, so measuring today's
cutoff against yesterday's margins compares two unrelated numbers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# How far a rate may stray before it counts as drift, measured rather than
# assumed. Signals come in runs, so a few weeks' rate swings far more than
# independent bars would: on out-of-fold margins from twenty symbols, with each
# fold's cutoff set at its own 8%, a model working exactly as built strayed past
# 2x in 43% of 500-bar windows, past 4x in 15% and past 8x in 5% (2026-10-01,
# docs/findings.md). The old 2.0 called drift on a sound model nearly half the
# time, and was read as evidence. 8.0 is the measured 95th percentile.
DRIFT_TOLERANCE = 8.0

# A rate is judged over this many bars (three weeks of hours) and no fewer.
MIN_ROWS_TO_JUDGE = 500

# Silence is judged sooner, because it is the failure this module exists for -
# but not from the first few bars: a sound model fires on nothing in 12.5% of
# 200-bar windows, 5.3% of 336-bar ones and 2.3% of 500-bar ones.
MIN_ROWS_FOR_SILENCE = 336


def coverage_drift(
    history: pd.DataFrame,
    model_version: str | None,
    cutoff: float | None,
    target: float | None,
    window: int = 500,
) -> dict[str, Any]:
    """Compare the model's live signal rate against the rate it was built for.

    `history` is the prediction log. Backfilled rows are included: the question
    is whether the rule fires on this market, and a row written late still
    records probabilities the model produced without seeing the future.
    """
    if cutoff is None or target is None or model_version is None:
        return {
            "state": "no_rule",
            "detail": "no saved model, or a model saved without a trading rule",
            "n_rows": 0,
        }

    if history.empty or "model_version" not in history.columns:
        return {"state": "no_data", "detail": "nothing logged yet", "n_rows": 0}

    rows = history[history["model_version"] == model_version]
    # The store hands the log back newest first. Without putting it in time
    # order, tail() keeps the OLDEST bars, and once a model has logged more than
    # `window` of them the check freezes on its first weeks and never sees the
    # present again.
    if "bar_close_time" in rows.columns:
        rows = rows.sort_values(
            "bar_close_time", key=lambda s: pd.to_datetime(s, utc=True), kind="stable"
        )
    rows = rows.tail(window)
    n = len(rows)

    margin = (rows["prob_up"] - rows["prob_down"]).abs()
    # A bar whose most likely class is FLAT is never traded, however wide the gap
    # between the two directional probabilities happens to be.
    directional = rows[["prob_down", "prob_flat", "prob_up"]].to_numpy().argmax(axis=1) != 1
    actual = float(((margin >= cutoff) & directional).mean()) if n else 0.0

    if actual <= 0 and n >= MIN_ROWS_FOR_SILENCE:
        return {
            "state": "silent",
            "detail": (
                f"0 of {n} bars cleared the cutoff {cutoff:.4f}. The rule was set "
                f"to fire on {target:.0%}. The model is not trading at all "
                "(a sound model does this in about 1 window in 20 this long)"
            ),
            "n_rows": n,
            "actual": 0.0,
            "target": target,
            "ratio": float("inf"),
        }
    if n < MIN_ROWS_TO_JUDGE:
        return {
            "state": "too_few",
            "detail": (
                f"{n} bars logged under this model, firing on {actual:.1%} so far; "
                f"{MIN_ROWS_TO_JUDGE} needed before a rate means anything - signals "
                "come in runs, and a few weeks of them swing widely"
            ),
            "n_rows": n,
            "actual": actual,
            "target": target,
        }

    ratio = max(actual / target, target / actual) if actual > 0 else float("inf")
    state = "ok" if ratio <= DRIFT_TOLERANCE else "drifted"
    return {
        "state": state,
        "detail": (
            f"fires on {actual:.1%} of the last {n} bars against a {target:.0%} "
            f"target ({ratio:.1f}x; a sound model strays past 2x in 4 windows of 10 "
            f"and past {DRIFT_TOLERANCE:.0f}x in 1 of 20)"
        ),
        "n_rows": n,
        "actual": actual,
        "target": target,
        "ratio": ratio,
    }


def format_drift(drift: dict[str, Any]) -> str:
    """One line for the status report."""
    label = {
        "ok": "OK",
        "drifted": "DRIFTED",
        "silent": "SILENT",
        "too_few": "too early to say",
        "no_rule": "no rule",
        "no_data": "no data",
    }.get(drift["state"], "unknown")
    return f"Signal rate: {label} — {drift.get('detail', '')}"
