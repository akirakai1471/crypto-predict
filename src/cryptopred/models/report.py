"""Human-readable evaluation report, including the GO/NO-GO verdict.

The verdict is deliberately hard to pass and deliberately printed in full, with
the losing comparisons shown as well as the winning ones. A report that only
shows favourable numbers is how people end up trading a model that does not work.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cryptopred.briefing.touch import block_length
from cryptopred.models.metrics import accuracy_by_confidence, block_bootstrap_mean
from cryptopred.models.selection import signals_by_quantile_per_fold

DOWN, FLAT, UP = 0, 1, 2

# Out-of-sample directional accuracy above this on hourly crypto is not a
# triumph, it is a symptom. See spec section 5.
IMPLAUSIBLE_ACCURACY = 0.60

# The trading rule the gate scores, unless the caller passes the configured one.
DEFAULT_COVERAGE = 0.08

# Baselines that make no directional calls; they are compared on Brier over
# every bar, which asks whether the probabilities beat the class frequencies.
BRIER_BASELINES = ("prior",)


def _fold_ids(evaluation: dict[str, Any]) -> np.ndarray:
    sizes = [int(f["n_test"]) for f in evaluation["folds"]]
    ids = np.repeat(np.arange(len(sizes)), sizes)
    if len(ids) != len(evaluation["y_true"]):
        return np.zeros(len(evaluation["y_true"]), dtype=int)
    return ids


def _credit(proba: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """1 for a right directional call, 0 for a wrong one, per row.

    A tie between UP and DOWN is a coin flip, worth half a point when the
    outcome was a direction and nothing when it was FLAT - its expected score.
    Resolving the tie by argmax made the "coin_flip" baseline always DOWN.
    """
    up, down = proba[:, UP], proba[:, DOWN]
    credit = np.where(
        up > down,
        truth == UP,
        np.where(down > up, truth == DOWN, 0.5 * (truth != FLAT)),
    )
    return credit.astype(float)


def verdict(
    evaluation: dict[str, Any],
    coverage: float = DEFAULT_COVERAGE,
    horizon: int = 24,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Decide whether the model earned the right to be deployed.

    Scored on the rule that is traded: the top `coverage` of bars by
    directional margin, ranked within each fold. Passing requires beating every
    baseline on those same bars AND a block-bootstrap interval on the paired
    difference against the toughest one that excludes zero.

    Until 2026-10-01 this scored the withdrawn 0.50-threshold rule instead,
    with the baselines scored on every bar. On BTC that rule took 65% of fold 0
    and 1% of fold 4, so one fold supplied two thirds of the rows - the first
    CORRECTION in docs/findings.md, still alive in the gate - and the model's
    lead over always_up on the bars it chose was zero: the reported gap was
    which bars were counted, not what was predicted. `threshold` is accepted
    for compatibility and ignored.
    """
    y = np.asarray(evaluation["y_true"])
    proba = np.asarray(evaluation["proba"])
    fold_ids = _fold_ids(evaluation)
    signal = signals_by_quantile_per_fold(proba, fold_ids, coverage)
    chosen = np.flatnonzero(signal != 0)  # time order
    truth = y[chosen]
    model_credit = np.where(signal[chosen] > 0, truth == UP, truth == DOWN).astype(float)
    n = len(chosen)
    model_acc = float(model_credit.mean()) if n else float("nan")
    per_fold = {
        int(f): {
            "n": int(((signal != 0) & (fold_ids == f)).sum()),
            "accuracy": float(model_credit[fold_ids[chosen] == f].mean())
            if ((signal != 0) & (fold_ids == f)).any()
            else float("nan"),
        }
        for f in np.unique(fold_ids)
    }

    model_metrics = evaluation["model"]
    comparisons: dict[str, dict[str, Any]] = {}
    credits: dict[str, np.ndarray] = {}
    for name, b_proba in evaluation["baseline_proba"].items():
        if name in BRIER_BASELINES:
            b_brier = evaluation["baselines"][name]["brier"]
            comparisons[name] = {
                "baseline_accuracy": float("nan"),
                "gap": float("nan"),
                "basis": "brier",
                "model_brier": model_metrics["brier"],
                "baseline_brier": b_brier,
                "beaten": bool(model_metrics["brier"] < b_brier),
            }
            continue
        credits[name] = _credit(np.asarray(b_proba)[chosen], truth)
        b_acc = float(credits[name].mean()) if n else float("nan")
        gap = model_acc - b_acc
        comparisons[name] = {
            "baseline_accuracy": b_acc,
            "gap": gap,
            "basis": "directional_accuracy",
            "beaten": bool(np.isfinite(gap) and gap > 0),
        }

    beat_all = n > 0 and all(c["beaten"] for c in comparisons.values())
    finite = {k: v["baseline_accuracy"] for k, v in comparisons.items() if k in credits}
    toughest = max(finite, key=finite.get) if finite and n else None
    interval = None
    if toughest is not None:
        interval = block_bootstrap_mean(
            model_credit - credits[toughest], block=block_length(horizon, n)
        )

    significant = bool(interval and interval["ci_lower"] > 0)
    implausible = bool(np.isfinite(model_acc) and model_acc > IMPLAUSIBLE_ACCURACY)
    calls_up = float((signal[chosen] > 0).mean()) if n else float("nan")

    if n == 0:
        decision, reason = "NO-GO", "the rule selected no bars"
    elif implausible:
        decision = "INVESTIGATE"
        reason = (
            f"directional accuracy {model_acc:.2%} exceeds {IMPLAUSIBLE_ACCURACY:.0%}, "
            "which is implausible on this horizon — treat as a leakage bug until proven otherwise"
        )
    elif beat_all and significant:
        decision = "GO"
        reason = (
            "beats every baseline on the bars it trades, and the margin survives a "
            "block bootstrap"
        )
    elif beat_all:
        decision = "NO-GO"
        reason = (
            "beats the baselines on the point estimate, but the margin is inside the "
            "noise once overlapping labels are resampled in blocks"
        )
    else:
        losers = [k for k, c in comparisons.items() if not c["beaten"]]
        decision = "NO-GO"
        reason = f"fails to beat on the bars it trades: {', '.join(losers)}"

    return {
        "decision": decision,
        "reason": reason,
        "comparisons": comparisons,
        "toughest_baseline": toughest,
        "bootstrap": interval,
        "model_accuracy": model_acc,
        "n_selected": n,
        "coverage": coverage,
        "calls_up": calls_up,
        "per_fold": per_fold,
    }


def format_evaluation(
    evaluation: dict[str, Any],
    symbol: str,
    interval: str,
    threshold: float | None = None,
    coverage: float = DEFAULT_COVERAGE,
    horizon: int = 24,
) -> str:
    model = evaluation["model"]
    v = verdict(evaluation, coverage=coverage, horizon=horizon)

    lines = [
        "=" * 68,
        f"WALK-FORWARD EVALUATION — {symbol} {interval}",
        "=" * 68,
        f"Out-of-sample rows:     {evaluation['n_test_total']:,}",
        f"Folds:                  {len(evaluation['folds'])}",
        f"Rule scored:            top {coverage:.0%} by directional margin, per fold",
        "",
        "MODEL",
        f"  directional accuracy:   {_fmt(v['model_accuracy'])}"
        f"   on the {v['n_selected']:,} bars the rule trades",
        f"  calls UP on:            {_fmt_pct(v['calls_up'])} of those bars",
        f"  accuracy (all bars):    {model['accuracy']:.4f}",
        f"  Brier score:            {model['brier']:.4f}  (lower is better)",
        f"  log loss:               {model['log_loss']:.4f}",
        "",
        "BASELINES (directional accuracy on the same bars)",
    ]

    for name, comparison in v["comparisons"].items():
        mark = "beaten" if comparison["beaten"] else "NOT BEATEN"
        if comparison["basis"] == "brier":
            lines.append(
                f"  {name:<12} makes no directional calls; compared on Brier: "
                f"model {comparison['model_brier']:.4f} vs {comparison['baseline_brier']:.4f}"
                f"   {mark}"
            )
        else:
            lines.append(
                f"  {name:<12} {_fmt(comparison['baseline_accuracy'])}"
                f"   gap {_fmt_signed(comparison['gap'])}   {mark}"
            )

    if v["bootstrap"]:
        b = v["bootstrap"]
        lines += [
            "",
            f"BLOCK BOOTSTRAP vs toughest baseline ({v['toughest_baseline']}), "
            f"blocks of {b['block']} bars, {b['n_blocks']} blocks",
            f"  mean difference:  {_fmt_signed(b['mean_difference'])}",
            f"  95% CI:           [{_fmt_signed(b['ci_lower'])}, {_fmt_signed(b['ci_upper'])}]",
            f"  excludes zero:    {b['excludes_zero']}",
        ]

    lines += ["", "ACCURACY BY CONFIDENCE BUCKET"]
    table = accuracy_by_confidence(evaluation["y_true"], evaluation["proba"])
    for bucket, row in table.iterrows():
        acc = "n/a" if np.isnan(row["accuracy"]) else f"{row['accuracy']:.4f}"
        lines.append(f"  {str(bucket):<16} n={int(row['n']):>7,}   accuracy={acc}")

    lines += ["", "PER-FOLD (the rule's bars in each fold)"]
    for i, fold in enumerate(evaluation["folds"]):
        rule = v["per_fold"].get(i, {"n": 0, "accuracy": float("nan")})
        lines.append(
            f"  fold {fold['fold']}  {str(fold['test_start'])[:10]} -> {str(fold['test_end'])[:10]}"
            f"  n={fold['n_test']:,}  traded={rule['n']:,}  dir_acc={_fmt(rule['accuracy'])}"
        )

    lines += ["", "TOP FEATURES BY GAIN"]
    for name, gain in list(evaluation["feature_importance"].items())[:15]:
        lines.append(f"  {name:<28} {gain:,.0f}")

    lines += [
        "",
        "=" * 68,
        f"VERDICT: {v['decision']}",
        f"  {v['reason']}",
        "=" * 68,
    ]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.4f}"


def _fmt_pct(value: float) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.1%}"


def _fmt_signed(value: float) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:+.4f}"
