"""Dataset sanity checks, run before any model sees the data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.dataset.builder import feature_columns

# Above this absolute correlation with the forward return, a feature is almost
# certainly leaking. See spec section 5.
SUSPICIOUS_CORRELATION = 0.30

_CLASS_NAMES = {-1.0: "down", 0.0: "flat", 1.0: "up"}


def quality_report(dataset: pd.DataFrame) -> dict[str, Any]:
    features = feature_columns(dataset)

    balance = dataset["label"].map(_CLASS_NAMES).value_counts(normalize=True).to_dict()

    stds = dataset[features].std()
    constant = sorted(stds[stds == 0].index.tolist())

    correlations = (
        dataset[features + ["forward_return"]]
        .corr()["forward_return"]
        .drop("forward_return")
        .abs()
    )
    suspicious = sorted(correlations[correlations > SUSPICIOUS_CORRELATION].index.tolist())

    return {
        "rows": len(dataset),
        "n_features": len(features),
        "start": dataset.index.min(),
        "end": dataset.index.max(),
        "class_balance": balance,
        "constant_features": constant,
        "suspicious_features": suspicious,
        "max_abs_correlation": float(correlations.max()),
        "top_correlations": correlations.sort_values(ascending=False).head(10).to_dict(),
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "DATASET QUALITY REPORT",
        "=" * 60,
        f"Rows:       {report['rows']:,}",
        f"Features:   {report['n_features']}",
        f"Period:     {report['start']} -> {report['end']}",
        "",
        "Class balance:",
    ]
    for name, share in sorted(report["class_balance"].items()):
        lines.append(f"  {name:<6} {share:6.2%}")

    lines.append("")
    lines.append(f"Max |correlation| with forward return: {report['max_abs_correlation']:.4f}")
    lines.append("Top correlations:")
    for name, value in report["top_correlations"].items():
        lines.append(f"  {name:<28} {value:.4f}")

    if report["constant_features"]:
        lines.append("")
        lines.append(f"WARNING constant features: {', '.join(report['constant_features'])}")

    if report["suspicious_features"]:
        lines.append("")
        lines.append("*** LEAKAGE SUSPECTED ***")
        lines.append(f"Features correlating >{SUSPICIOUS_CORRELATION} with the future:")
        for name in report["suspicious_features"]:
            lines.append(f"  {name}")
        lines.append("Do not train until this is explained.")

    lines.append("=" * 60)
    return "\n".join(lines)
