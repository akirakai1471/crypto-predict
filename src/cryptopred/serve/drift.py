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

# Same tolerance as the save gate in models/train.py. A model firing at more
# than twice or less than half its planned rate is a different strategy from the
# one that was backtested, whichever direction it drifted.
DRIFT_TOLERANCE = 2.0

# Below this many rows the observed rate is noise. At an 8% target, 60 bars is
# fewer than five expected signals, and seeing none of those is unremarkable.
MIN_ROWS_TO_JUDGE = 60


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
    rows = rows.tail(window)
    n = len(rows)
    if n < MIN_ROWS_TO_JUDGE:
        return {
            "state": "too_few",
            "detail": (
                f"{n} bars logged under this model; "
                f"{MIN_ROWS_TO_JUDGE} needed before a rate means anything"
            ),
            "n_rows": n,
            "target": target,
        }

    margin = (rows["prob_up"] - rows["prob_down"]).abs()
    # A bar whose most likely class is FLAT is never traded, however wide the gap
    # between the two directional probabilities happens to be.
    directional = rows[["prob_down", "prob_flat", "prob_up"]].to_numpy().argmax(axis=1) != 1
    actual = float(((margin >= cutoff) & directional).mean())

    if actual <= 0:
        return {
            "state": "silent",
            "detail": (
                f"0 of {n} bars cleared the cutoff {cutoff:.4f}. The rule was set "
                f"to fire on {target:.0%}. The model is not trading at all"
            ),
            "n_rows": n,
            "actual": 0.0,
            "target": target,
            "ratio": float("inf"),
        }

    ratio = max(actual / target, target / actual)
    state = "ok" if ratio <= DRIFT_TOLERANCE else "drifted"
    return {
        "state": state,
        "detail": (
            f"fires on {actual:.1%} of the last {n} bars against a {target:.0%} "
            f"target ({ratio:.1f}x)"
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
