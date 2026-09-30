"""Run the pre-registered model changes against a baseline, and judge them.

Everything here is fixed by docs/preregistration-improvements.md, committed
before this code first touched real data: the arms and their single parameter
values, the symbols, the metric and the four-part decision rule. Nothing is
tunable from the command line except which of the registered arms to run and
how many cores to use.

Every arm runs in the same invocation on the same data, and the baseline is
re-run there. Comparing an arm with a number reported last month would compare
it with a different dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from cryptopred.config import Config
from cryptopred.dataset.builder import build_dataset
from cryptopred.features.context import context_symbol
from cryptopred.ingest.storage import ParquetStore
from cryptopred.models.validation import (
    FrozenConfig,
    SymbolResult,
    evaluate_symbol,
    evaluate_symbol_selection,
)

# The twenty from docs/preregistration-multisymbol.md, unchanged.
PREREGISTERED_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "FILUSDT", "INJUSDT", "TRXUSDT", "ETCUSDT",
)

# The decision rule, as written down before running.
WIN_SHARE = 0.75  # of tested symbols, rounded up: 15 of 20
MIN_MEAN_GAIN = 0.005  # +0.5 percentage points of sign accuracy
ALPHA = 0.025  # two arms, so half the usual budget each


@dataclass(frozen=True)
class Arm:
    name: str
    recency_half_life: float | None = None
    market_context: bool = False

    def train_overrides(self) -> dict[str, Any]:
        if self.recency_half_life is None:
            return {}
        return {"recency_half_life": self.recency_half_life}


ARMS = {
    "baseline": Arm("baseline"),
    "recency": Arm("recency", recency_half_life=8760),
    "market_context": Arm("market_context", market_context=True),
}


def run_arm_symbol(
    symbol: str,
    cfg: Config,
    interval: str,
    config: FrozenConfig,
    arm_name: str,
    threads: int = 0,
) -> SymbolResult:
    """One arm on one symbol, reading its data inside the worker."""
    arm = ARMS[arm_name]
    parquet = ParquetStore(cfg.data.root / "raw")
    bars = parquet.read("klines", symbol, interval)
    if bars.empty:
        return SymbolResult(symbol, status="skipped: no bars stored")

    context = None
    if arm.market_context:
        context = parquet.read("klines", context_symbol(symbol), interval)
        if context.empty:
            return SymbolResult(
                symbol, status=f"skipped: no {context_symbol(symbol)} bars for context"
            )

    funding = parquet.read("funding", symbol, "8h")
    dataset = build_dataset(
        bars,
        interval=interval,
        horizon=config.horizon,
        atr_period=cfg.labels.atr_period,
        band_k=cfg.labels.band_k,
        funding=funding if not funding.empty else None,
        symbol=symbol,
        feature_config=cfg.features,
        context_bars=context,
    )
    return evaluate_symbol(
        symbol, dataset, bars, config, threads=threads, train_overrides=arm.train_overrides()
    )


def _sign(result: SymbolResult) -> float | None:
    if result.status != "tested":
        return None
    return result.detail.get("sign_accuracy")


def sign_test_p(wins: int, n: int) -> float:
    """One-sided: the chance of at least `wins` of `n` coin flips landing heads."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def judge(baseline: list[SymbolResult], arm: list[SymbolResult]) -> dict[str, Any]:
    """Apply the pre-registered rule to one arm. Lists are aligned by symbol.

    A symbol counts only if both runs tested it and both produced a sign
    accuracy; a skip on either side counts toward neither.
    """
    pairs = [
        (b, a)
        for b, a in zip(baseline, arm, strict=True)
        if _sign(b) is not None and _sign(a) is not None
    ]
    n = len(pairs)
    gains = np.array([_sign(a) - _sign(b) for b, a in pairs]) if n else np.array([])
    wins = int((gains > 0).sum())
    needed = math.ceil(WIN_SHARE * n)
    mean_gain = float(gains.mean()) if n else 0.0

    compared = {b.symbol for b, _ in pairs}
    gates_base = sum(r.passed for r in baseline if r.symbol in compared)
    gates_arm = sum(r.passed for r in arm if r.symbol in compared)

    ll_base = [b.detail.get("log_loss") for b, _ in pairs]
    ll_arm = [a.detail.get("log_loss") for _, a in pairs]
    have_ll = n > 0 and None not in ll_base and None not in ll_arm
    mean_ll_base = float(np.mean(ll_base)) if have_ll else None
    mean_ll_arm = float(np.mean(ll_arm)) if have_ll else None

    p_value = sign_test_p(wins, n)
    criteria = {
        "wins": n > 0 and wins >= needed and p_value <= ALPHA,
        "margin": n > 0 and mean_gain >= MIN_MEAN_GAIN,
        "gates": gates_arm >= gates_base,
        "log_loss": have_ll and mean_ll_arm <= mean_ll_base,
    }
    return {
        "n": n,
        "wins": wins,
        "needed": needed,
        "p_value": p_value,
        "mean_gain": mean_gain,
        "gates_base": gates_base,
        "gates_arm": gates_arm,
        "log_loss_base": mean_ll_base,
        "log_loss_arm": mean_ll_arm,
        "criteria": criteria,
        "decision": "ADOPTED" if all(criteria.values()) else "REJECTED",
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def format_experiment(results: dict[str, list[SymbolResult]]) -> str:
    """Per-symbol table, then one verdict block per arm."""
    arms = list(results)
    symbols = [r.symbol for r in results[arms[0]]]
    col = max(10, max(len(a) for a in arms) + 2)
    width = 12 + col * len(arms)
    lines = [
        "=" * width,
        "PRE-REGISTERED EXPERIMENT — docs/preregistration-improvements.md",
        "=" * width,
        "Sign accuracy on the traded 8% of bars, per symbol and arm.",
        "",
        f"{'symbol':>10}  " + "".join(f"{a:>{col}}" for a in arms),
        "-" * width,
    ]
    for i, symbol in enumerate(symbols):
        cells = []
        for a in arms:
            r = results[a][i]
            cells.append(f"{_pct(_sign(r)) if _sign(r) is not None else 'skip':>{col}}")
        lines.append(f"{symbol:>10}  " + "".join(cells))

    if "baseline" not in results:
        lines += ["", "No baseline arm was run, so nothing can be judged."]
        return "\n".join(lines)

    for a in arms:
        if a == "baseline":
            continue
        v = judge(results["baseline"], results[a])
        c = v["criteria"]
        mark = {True: "pass", False: "FAIL"}
        ll = (
            f"{v['log_loss_arm']:.4f} vs {v['log_loss_base']:.4f}"
            if v["log_loss_base"] is not None
            else "unavailable"
        )
        lines += [
            "",
            "-" * width,
            f"{a}: {v['decision']}",
            f"  1. wins on most symbols   {mark[c['wins']]}  {v['wins']}/{v['n']} better, "
            f"{v['needed']} needed, sign-test p = {v['p_value']:.4f} (alpha {ALPHA})",
            f"  2. by enough to matter    {mark[c['margin']]}  mean {v['mean_gain']:+.2%} "
            f"(needs {MIN_MEAN_GAIN:+.1%})",
            f"  3. no gate passes lost    {mark[c['gates']]}  {v['gates_arm']} vs "
            f"{v['gates_base']} symbols passing all three gates",
            f"  4. log loss not worse     {mark[c['log_loss']]}  {ll}",
        ]

    lines += [
        "",
        "=" * width,
        "Twenty crypto symbols over one period are not twenty independent tests; the",
        "p-value is optimistic. An ADOPTED arm earns a place in the default config, not",
        "a proven edge. Record this result in docs/findings.md either way.",
        "=" * width,
    ]
    return "\n".join(lines)


# -- cost-aware selection -------------------------------------------------------
# docs/preregistration-cost-aware-selection.md: one arm, so the full alpha.

SELECTION_WIN_SHARE = 0.75
SELECTION_ALPHA = 0.05
SELECTION_ARMS = ("margin", "margin_x_vol")


def run_selection_symbol(
    symbol: str,
    cfg: Config,
    interval: str,
    config: FrozenConfig,
    threads: int = 0,
) -> dict[str, SymbolResult]:
    """Train one symbol once and select its trades both ways."""
    parquet = ParquetStore(cfg.data.root / "raw")
    bars = parquet.read("klines", symbol, interval)
    if bars.empty:
        skipped = SymbolResult(symbol, status="skipped: no bars stored")
        return dict.fromkeys(SELECTION_ARMS, skipped)

    funding = parquet.read("funding", symbol, "8h")
    dataset = build_dataset(
        bars,
        interval=interval,
        horizon=config.horizon,
        atr_period=cfg.labels.atr_period,
        band_k=cfg.labels.band_k,
        funding=funding if not funding.empty else None,
        symbol=symbol,
        feature_config=cfg.features,
    )
    return evaluate_symbol_selection(symbol, dataset, bars, config, threads=threads)


def judge_selection(baseline: list[SymbolResult], arm: list[SymbolResult]) -> dict[str, Any]:
    """The four-part rule, applied as written. Lists are aligned by symbol."""
    pairs = [
        (b, a)
        for b, a in zip(baseline, arm, strict=True)
        if b.status == "tested" and a.status == "tested"
    ]
    n = len(pairs)
    wins = sum(
        a.detail["doubled_cost_return"] > b.detail["doubled_cost_return"] for b, a in pairs
    )
    needed = math.ceil(SELECTION_WIN_SHARE * n)
    p_value = sign_test_p(wins, n)

    sharpe_base = float(np.mean([b.detail["sharpe"] for b, _ in pairs])) if n else 0.0
    sharpe_arm = float(np.mean([a.detail["sharpe"] for _, a in pairs])) if n else 0.0
    gates_base = sum(b.passed for b, _ in pairs)
    gates_arm = sum(a.passed for _, a in pairs)
    two_base = sum(b.detail["two_sided"] == "TWO-SIDED" for b, _ in pairs)
    two_arm = sum(a.detail["two_sided"] == "TWO-SIDED" for _, a in pairs)

    criteria = {
        "costs": n > 0 and wins >= needed and p_value <= SELECTION_ALPHA,
        "sharpe": n > 0 and sharpe_arm >= sharpe_base,
        "gates": gates_arm >= gates_base,
        "two_sided": two_arm >= two_base,
    }
    return {
        "n": n,
        "wins": wins,
        "needed": needed,
        "p_value": p_value,
        "sharpe_base": sharpe_base,
        "sharpe_arm": sharpe_arm,
        "gates_base": gates_base,
        "gates_arm": gates_arm,
        "two_sided_base": two_base,
        "two_sided_arm": two_arm,
        "criteria": criteria,
        "decision": "ADOPTED" if all(criteria.values()) else "REJECTED",
    }


def _cell(result: SymbolResult, key: str, fmt: str) -> str:
    if result.status != "tested" or result.detail.get(key) is None:
        return "skip"
    return format(result.detail[key], fmt)


def format_selection_experiment(results: dict[str, list[SymbolResult]]) -> str:
    """Per-symbol table for both rules, then the verdict."""
    base, arm = results["margin"], results["margin_x_vol"]
    width = 104
    header = (
        f"{'symbol':>10}  {'2x-cost return':>21}  {'sharpe':>13}  {'sign%':>15}  "
        f"{'|move|':>15}  {'two-sided?':>21}"
    )
    lines = [
        "=" * width,
        "PRE-REGISTERED EXPERIMENT — docs/preregistration-cost-aware-selection.md",
        "=" * width,
        "Same model, same probabilities; each cell is  margin  /  margin x vol72.",
        "",
        header,
        "-" * width,
    ]
    for b, a in zip(base, arm, strict=True):
        cells = [
            f"{_cell(b, 'doubled_cost_return', '+.1%'):>10} "
            f"{_cell(a, 'doubled_cost_return', '+.1%'):>10}",
            f"{_cell(b, 'sharpe', '.2f'):>6} {_cell(a, 'sharpe', '.2f'):>6}",
            f"{_cell(b, 'sign_accuracy', '.2%'):>7} {_cell(a, 'sign_accuracy', '.2%'):>7}",
            f"{_cell(b, 'mean_abs_move', '.2%'):>7} {_cell(a, 'mean_abs_move', '.2%'):>7}",
            f"{_cell(b, 'two_sided', ''):>10} {_cell(a, 'two_sided', ''):>10}",
        ]
        lines.append(f"{b.symbol:>10}  " + "  ".join(cells))

    v = judge_selection(base, arm)
    c = v["criteria"]
    mark = {True: "pass", False: "FAIL"}
    lines += [
        "",
        "-" * width,
        f"margin_x_vol: {v['decision']}",
        f"  1. survives costs better  {mark[c['costs']]}  {v['wins']}/{v['n']} symbols higher "
        f"at doubled costs, {v['needed']} needed, sign-test p = {v['p_value']:.4f} "
        f"(alpha {SELECTION_ALPHA})",
        f"  2. not just bigger bets   {mark[c['sharpe']]}  mean Sharpe {v['sharpe_arm']:.3f} "
        f"vs {v['sharpe_base']:.3f}",
        f"  3. no gate passes lost    {mark[c['gates']]}  {v['gates_arm']} vs "
        f"{v['gates_base']} symbols passing all three gates",
        f"  4. no long-bias trap      {mark[c['two_sided']]}  {v['two_sided_arm']} vs "
        f"{v['two_sided_base']} symbols TWO-SIDED",
        "",
        "=" * width,
        "Twenty correlated symbols over one period are not twenty independent tests.",
        "Record this result in docs/findings.md either way.",
        "=" * width,
    ]
    return "\n".join(lines)
