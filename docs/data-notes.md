# Data Notes

## Dataset build 2026-08-26

Source: Binance USDT-perpetual futures (`fapi.binance.com`), no API key.

| Symbol | Interval | Rows | Period | down/flat/up | Max abs corr |
|---|---|---|---|---|---|
| BTCUSDT | 1h | 60,294 | 2019-10-09 16:00 → 2026-08-25 21:00 | 28.47 / 41.36 / 30.16 % | 0.0456 |
| ETHUSDT | 1h | 58,384 | 2019-12-28 06:00 → 2026-08-25 21:00 | 28.79 / 40.31 / 30.90 % | 0.0584 |
| BTCUSDT | 1m | (pending) | | | |
| ETHUSDT | 1m | (pending) | | | |

85 features per row. Raw kline coverage before feature warm-up: BTCUSDT 61,041
bars from 2019-09-08, ETHUSDT 59,131 bars from 2019-11-27 — the later start is
when the ETHUSDT perpetual was listed, not a data problem.

## Known gaps

`cryptopred-ingest report` shows `gaps=0` for both symbols on 1h. No exchange
downtime holes in the hourly series.

## Leakage screen

Highest absolute correlation between any feature and the forward return is
0.0584 (`bar_range_norm`, ETHUSDT). The suspicion threshold is 0.30, so nothing
is flagged. The features that correlate most are all volatility measures, which
is expected and benign: bigger recent bars mean bigger forward moves in either
direction, which says nothing about direction.

All six tests in `tests/test_leakage.py` pass.

## Decisions

- `band_k = 0.5` kept unchanged. It produces a ~41% flat class, close to the
  target 30/40/30 split. No retune needed.
- Open interest and long/short ratio are **not** training features: Binance
  retains only ~30 days of `futures/data/*` history, far too little for a
  multi-year dataset. They are collected by `cryptopred-ingest stats` for
  dashboard display and slow local accumulation.
- Price series is USDT perpetual futures, not spot, so it matches the market
  that would actually be traded and aligns with funding rate data.
- Feature columns are stored as float32. The 1m datasets run to millions of
  rows and float64 doubles both memory and file size for no benefit to
  gradient-boosted trees.

## What the numbers mean for Plan 2

A near-zero feature/label correlation is the expected and healthy result. It is
not evidence that a model will fail — gradient boosting finds non-linear
interactions that pairwise correlation cannot see — but it does set the
expectation that any edge will be small. If Plan 2 reports out-of-sample
accuracy above 60% on the 1h horizon, treat it as a bug until proven otherwise.
