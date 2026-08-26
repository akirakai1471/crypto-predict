"""Human-readable evaluation report, including the GO/NO-GO verdict.

The verdict is deliberately hard to pass and deliberately printed in full, with
the losing comparisons shown as well as the winning ones. A report that only
shows favourable numbers is how people end up trading a model that does not work.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cryptopred.models.metrics import accuracy_by_confidence, bootstrap_difference

# Out-of-sample directional accuracy above this on hourly crypto is not a
# triumph, it is a symptom. See spec section 5.
IMPLAUSIBLE_ACCURACY = 0.60


def verdict(evaluation: dict[str, Any], threshold: float = 0.5) -> dict[str, Any]:
    """Decide whether the model earned the right to be deployed.

    Passing requires beating every baseline on directional accuracy AND a
    bootstrap confidence interval on the best comparison that excludes zero.
    """
    model = evaluation["model"]
    model_acc = model["directional_accuracy"]

    comparisons = {}
    for name, metrics in evaluation["baselines"].items():
        baseline_acc = metrics["directional_accuracy"]

        if not np.isfinite(baseline_acc):
            # This baseline never commits to a direction (the `prior` baseline
            # predicts the majority FLAT class on every bar), so directional
            # accuracy is undefined for it. Falling back to Brier keeps the
            # comparison meaningful and, being a proper scoring rule, strictly
            # harder to game than accuracy.
            comparisons[name] = {
                "baseline_accuracy": float("nan"),
                "gap": float("nan"),
                "basis": "brier",
                "model_brier": model["brier"],
                "baseline_brier": metrics["brier"],
                "beaten": bool(model["brier"] < metrics["brier"]),
            }
            continue

        gap = model_acc - baseline_acc
        comparisons[name] = {
            "baseline_accuracy": baseline_acc,
            "gap": gap,
            "basis": "directional_accuracy",
            "beaten": bool(np.isfinite(gap) and gap > 0),
        }

    beat_all = all(c["beaten"] for c in comparisons.values())

    # Bootstrap against the toughest baseline the model actually has to clear.
    finite = {
        name: c["baseline_accuracy"]
        for name, c in comparisons.items()
        if np.isfinite(c["baseline_accuracy"])
    }
    toughest = max(finite, key=finite.get) if finite else None
    interval = None
    if toughest is not None:
        interval = bootstrap_difference(
            evaluation["y_true"],
            evaluation["proba"],
            evaluation["baseline_proba"][toughest],
            threshold=threshold,
            n_boot=1000,
        )

    significant = bool(interval and interval["ci_lower"] > 0)
    implausible = bool(np.isfinite(model_acc) and model_acc > IMPLAUSIBLE_ACCURACY)

    if implausible:
        decision = "INVESTIGATE"
        reason = (
            f"directional accuracy {model_acc:.2%} exceeds {IMPLAUSIBLE_ACCURACY:.0%}, "
            "which is implausible on this horizon — treat as a leakage bug until proven otherwise"
        )
    elif beat_all and significant:
        decision = "GO"
        reason = "beats every baseline, and the margin survives bootstrap resampling"
    elif beat_all:
        decision = "NO-GO"
        reason = "beats the baselines on the point estimate, but the margin is inside the noise"
    else:
        losers = [n for n, c in comparisons.items() if not c["beaten"]]
        decision = "NO-GO"
        reason = f"fails to beat: {', '.join(losers)}"

    return {
        "decision": decision,
        "reason": reason,
        "comparisons": comparisons,
        "toughest_baseline": toughest,
        "bootstrap": interval,
        "model_accuracy": model_acc,
    }


def format_evaluation(
    evaluation: dict[str, Any], symbol: str, interval: str, threshold: float = 0.5
) -> str:
    model = evaluation["model"]
    v = verdict(evaluation, threshold=threshold)

    lines = [
        "=" * 68,
        f"WALK-FORWARD EVALUATION — {symbol} {interval}",
        "=" * 68,
        f"Out-of-sample rows:     {evaluation['n_test_total']:,}",
        f"Folds:                  {len(evaluation['folds'])}",
        f"Signal threshold:       {threshold:.2f}",
        "",
        "MODEL",
        f"  accuracy (all bars):    {model['accuracy']:.4f}",
        f"  directional accuracy:   {_fmt(model['directional_accuracy'])}",
        f"  signals taken:          {model['n_signals']:,} ({model['coverage']:.1%} of bars)",
        f"  sign accuracy:          {_fmt(model.get('sign_accuracy', float('nan')))}"
        "   (did price move the predicted way at all)",
        f"  Brier score:            {model['brier']:.4f}  (lower is better)",
        f"  log loss:               {model['log_loss']:.4f}",
        "",
        "BASELINES (directional accuracy)",
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
            f"BOOTSTRAP vs toughest baseline ({v['toughest_baseline']})",
            f"  mean difference:  {_fmt_signed(b['mean_difference'])}",
            f"  95% CI:           [{_fmt_signed(b['ci_lower'])}, {_fmt_signed(b['ci_upper'])}]",
            f"  excludes zero:    {b['excludes_zero']}",
        ]

    lines += ["", "ACCURACY BY CONFIDENCE BUCKET"]
    table = accuracy_by_confidence(evaluation["y_true"], evaluation["proba"])
    for bucket, row in table.iterrows():
        acc = "n/a" if np.isnan(row["accuracy"]) else f"{row['accuracy']:.4f}"
        lines.append(f"  {str(bucket):<16} n={int(row['n']):>7,}   accuracy={acc}")

    lines += ["", "PER-FOLD"]
    for fold in evaluation["folds"]:
        lines.append(
            f"  fold {fold['fold']}  {str(fold['test_start'])[:10]} -> {str(fold['test_end'])[:10]}"
            f"  n={fold['n_test']:,}  dir_acc={_fmt(fold['directional_accuracy'])}"
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


def _fmt_signed(value: float) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:+.4f}"
