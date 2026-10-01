"""The six tools the model may call.

Descriptions matter as much as schemas: the model reads the description before
it reads the values, so the warning about conventional indicators lives there
rather than only in the payload.

Only BTCUSDT and ETHUSDT are allowed. The other eighteen symbols are refreshed
by hand and answering about two-week-old bars is the staleness failure this
design exists to avoid.

Nothing here writes. Every method is a read over stored data, and there is no
code path from this module to an exchange.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cryptopred.briefing.indicators import current_indicators
from cryptopred.briefing.levels import daily_pivots, swing_levels
from cryptopred.briefing.provenance import Measured, Unavailable
from cryptopred.briefing.snapshot import market_snapshot
from cryptopred.briefing.touch import MEASURED_COVERAGE, touch_probability
from cryptopred.config import Config
from cryptopred.ingest.storage import ParquetStore
from cryptopred.models.registry import ModelRegistry
from cryptopred.serve.drift import coverage_drift
from cryptopred.serve.status import overlap_interval
from cryptopred.serve.store import PredictionStore
from cryptopred.timeframes import interval_to_timedelta

ALLOWED_SYMBOLS = ("BTCUSDT", "ETHUSDT")

_SYMBOL_ONLY = {
    "type": "object",
    "properties": {"symbol": {"type": "string", "enum": list(ALLOWED_SYMBOLS)}},
    "required": ["symbol"],
    "additionalProperties": False,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "market_snapshot",
        "description": (
            "Giá đóng gần nhất, thay đổi 24h/7d/30d, khối lượng, funding, và "
            "ĐỘ CŨ CỦA DỮ LIỆU. Luôn gọi tool này trước; nếu is_stale là true thì "
            "phải nói rõ trong câu trả lời rằng số liệu mô tả thời điểm cũ."
        ),
        "strict": True,
        "input_schema": _SYMBOL_ONLY,
    },
    {
        "name": "indicators",
        "description": (
            "RSI, MACD, ATR, ADX, realized vol. CẢNH BÁO: dự án này CHƯA ĐO các "
            "chỉ báo đó có giá trị dự báo trên dữ liệu này hay không. Được phép "
            "đọc số ra, KHÔNG được suy ra dự báo từ chúng."
        ),
        "strict": True,
        "input_schema": _SYMBOL_ONLY,
    },
    {
        "name": "levels",
        "description": (
            "Pivot ngày và đỉnh/đáy xoay đã xác nhận. Là quy ước, chưa kiểm chứng. "
            "Dùng để lấy ra các mức giá cụ thể rồi đưa vào touch_probability — "
            "đó mới là chỗ có số đo thật."
        ),
        "strict": True,
        "input_schema": _SYMBOL_ONLY,
    },
    {
        "name": "touch_probability",
        "description": (
            "SỐ ĐO THẬT. Trong lịch sử, giá chạm mức này trong bao nhiêu phần trăm "
            "số lần, tính trên những giờ có cùng chế độ biến động và xu hướng, kèm "
            "cỡ mẫu, khoảng bootstrap, và phân phối thời gian chờ. Đây là tool trả "
            "lời câu 'khi nào'. Nếu conditional là unavailable thì nói rõ là ô chế "
            "độ không đủ mẫu và dùng con số unconditional. Khoảng tin cậy có độ phủ "
            f"đo được khoảng {MEASURED_COVERAGE:.0%}, KHÔNG phải 95% — đừng gọi nó là 95%."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "enum": list(ALLOWED_SYMBOLS)},
                "target": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["pct", "price"]},
                        "value": {"type": "number"},
                    },
                    "required": ["kind", "value"],
                    "additionalProperties": False,
                },
                "horizon_hours": {"type": "integer", "minimum": 1, "maximum": 720},
            },
            "required": ["symbol", "target", "horizon_hours"],
            "additionalProperties": False,
        },
    },
    {
        "name": "model_signal",
        "description": (
            "Dự báo hướng hiện tại của model đã lưu, margin so với cutoff, và "
            "trạng thái drift. ETHUSDT KHÔNG có model vì trượt cổng kiểm — tool sẽ "
            "trả unavailable kèm lý do, và đó là câu trả lời đúng, không phải lỗi."
        ),
        "strict": True,
        "input_schema": _SYMBOL_ONLY,
    },
    {
        "name": "track_record",
        "description": (
            "Độ chính xác thật đã ghi nhận (kèm n và khoảng tin cậy), và chi phí "
            "vòng lệnh. Luôn đọc hai số này cạnh nhau: 54% nghe hay cho tới khi "
            "đặt cạnh mức mà phí đòi hỏi."
        ),
        "strict": True,
        "input_schema": _SYMBOL_ONLY,
    },
]


class BriefingTools:
    """Executes the tools. Pure reads; nothing here writes or trades."""

    def __init__(self, cfg: Config, interval: str = "1h") -> None:
        self.cfg = cfg
        self.interval = interval
        self.parquet = ParquetStore(cfg.data.root / "raw")
        self.registry = ModelRegistry(cfg.data.root / "models")

    # -- helpers -----------------------------------------------------------

    def _refuse(self, symbol: str) -> Unavailable:
        return Unavailable(
            reason=(
                f"{symbol} không nằm trong phạm vi. Chỉ trả lời về "
                f"{' và '.join(ALLOWED_SYMBOLS)}, vì các coin khác không được "
                "cập nhật tự động và dữ liệu sẽ cũ."
            )
        )

    def _bars(self, symbol: str) -> pd.DataFrame | Unavailable:
        if symbol not in ALLOWED_SYMBOLS:
            return self._refuse(symbol)
        bars = self.parquet.read("klines", symbol, self.interval)
        if bars.empty:
            return Unavailable(
                reason=f"không có nến {symbol} {self.interval} trong kho"
            )
        return bars

    def _store(self) -> PredictionStore:
        return PredictionStore(self.cfg.data.root / "predictions.db")

    def last_close(self, symbol: str) -> float:
        """The reference price targets are measured against. Raises if absent —
        callers that might not have bars use `_bars` instead."""
        bars = self._bars(symbol)
        if isinstance(bars, Unavailable):
            raise ValueError(bars.reason)
        return float(bars["close"].iloc[-1])

    # -- tools -------------------------------------------------------------

    def market_snapshot(self, symbol: str) -> dict[str, Any]:
        bars = self._bars(symbol)
        if isinstance(bars, Unavailable):
            return bars.to_dict()
        funding = self.parquet.read("funding", symbol, "8h")
        return market_snapshot(bars, funding)

    def indicators(self, symbol: str) -> dict[str, Any]:
        bars = self._bars(symbol)
        if isinstance(bars, Unavailable):
            return bars.to_dict()
        return current_indicators(bars)

    def levels(self, symbol: str) -> dict[str, Any]:
        bars = self._bars(symbol)
        if isinstance(bars, Unavailable):
            return bars.to_dict()
        return {"pivots": daily_pivots(bars), **swing_levels(bars)}

    def touch_probability(
        self, symbol: str, target: dict[str, Any], horizon_hours: int
    ) -> dict[str, Any]:
        bars = self._bars(symbol)
        if isinstance(bars, Unavailable):
            return bars.to_dict()

        last = float(bars["close"].iloc[-1])
        if target["kind"] == "price":
            # The direction of the test follows from where the price sits
            # relative to the market, so a caller cannot ask for an upward test
            # against a level below the market.
            target_pct = float(target["value"]) / last - 1.0
        else:
            target_pct = float(target["value"])

        result = touch_probability(bars, target_pct=target_pct, horizon=horizon_hours)
        return {
            "target_pct": target_pct,
            "target_price": last * (1.0 + target_pct),
            "reference_close": last,
            "horizon_hours": horizon_hours,
            "cell_label": result["cell_label"],
            "conditional": result["conditional"].to_dict(),
            "unconditional": result["unconditional"].to_dict(),
            "wait_hours": result["wait_hours"],
            # Which sample the wait times came from. Without this the model
            # could attribute unconditional waits to the named regime.
            "wait_source": result["wait_source"],
        }

    def model_signal(self, symbol: str) -> dict[str, Any]:
        if symbol not in ALLOWED_SYMBOLS:
            return self._refuse(symbol).to_dict()

        version = self.registry.latest(symbol, self.interval)
        if version is None:
            return Unavailable(
                reason=(
                    f"{symbol} không có model nào được lưu. Model chỉ được lưu khi "
                    "vượt cổng kiểm; không có model là kết quả đúng của cổng, "
                    "không phải lỗi."
                )
            ).to_dict()

        meta = self.registry.load(version).metadata
        store = self._store()
        latest = store.latest_prediction(symbol, self.interval)
        if latest is None:
            return Unavailable(
                reason="chưa ghi nhận dự báo nào cho model này"
            ).to_dict()

        drift = coverage_drift(
            store.history(symbol, self.interval, limit=100_000),
            model_version=version,
            cutoff=meta.get("margin_cutoff"),
            target=meta.get("signal_coverage"),
        )
        margin = abs(float(latest["prob_up"]) - float(latest["prob_down"]))
        return {
            "model_version": version,
            "bar_close_time": str(latest["bar_close_time"]),
            "prob_down": float(latest["prob_down"]),
            "prob_flat": float(latest["prob_flat"]),
            "prob_up": float(latest["prob_up"]),
            "margin": margin,
            "margin_cutoff": meta.get("margin_cutoff"),
            "signal": int(latest["signal"]),
            "drift": drift,
        }

    def track_record(self, symbol: str) -> dict[str, Any]:
        if symbol not in ALLOWED_SYMBOLS:
            return self._refuse(symbol).to_dict()

        history = self._store().history(symbol, self.interval, limit=100_000)
        if history.empty:
            return Unavailable(reason="chưa có dự báo nào được ghi").to_dict()

        scored = history[history["actual_return"].notna()]
        # Backfilled rows were predicted after their bar had closed, by a model
        # that may postdate it; they prove nothing about foresight. The status
        # report and the dashboard already drop them - counted here, ask
        # reported a different record from the one beside it on the screen.
        if "was_backfilled" in scored.columns:
            scored = scored[scored["was_backfilled"].fillna(0).astype(int) == 0]
        correct = int(scored["is_correct"].sum()) if not scored.empty else 0
        n = int(len(scored))
        cost = self.cfg.strategy.taker_fee * 2 + self.cfg.strategy.slippage * 2

        # An accuracy computed from zero scored bars is 0.0, which reads as a
        # measured failure rather than as no measurement. `Measured` rejects
        # n <= 0 for exactly this reason.
        horizon = self.cfg.labels.horizon_bars.get(self.interval, 24) * interval_to_timedelta(
            self.interval
        )
        interval_95, n_eff = overlap_interval(
            scored["is_correct"], scored["bar_close_time"], horizon
        )
        accuracy: Measured | Unavailable = (
            Measured(
                value=correct / n,
                n=n,
                interval=interval_95,
                # Every scored bar shares 23 of its 24 hours with the next; as
                # independent trials the interval held the truth a third of the
                # time. Counted as non-overlapping windows instead.
                method=(
                    f"mọi nến đã chấm điểm; Wilson trên {n_eff} cửa sổ 24h không chồng "
                    "nhau, vì kết quả các nến liền nhau không độc lập"
                ),
            )
            if n > 0
            else Unavailable(
                reason="chưa có nến nào được chấm điểm, nên chưa có độ chính xác nào"
            )
        )
        return {
            "n_scored": n,
            "accuracy": accuracy.to_dict(),
            "round_trip_cost": cost,
            "note": (
                "Độ chính xác hoà vốn phụ thuộc biên độ di chuyển trung vị; "
                "xem `cryptopred-model breakeven` để có con số cho từng khung."
            ),
        }
