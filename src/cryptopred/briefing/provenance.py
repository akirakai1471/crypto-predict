"""Every value leaving this package carries its own epistemic status.

There is no path that returns a bare float. This project's history is a record
of confident figures evaporating — a compounding bug, a calibration artefact, a
gate that passed a model firing on nothing — and each one looked like a number
while behaving like a sentence. The distinction is enforced by the type rather
than by a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CONVENTION_WARNING = (
    "Chưa đo chỉ báo này có giá trị dự báo trên dữ liệu này. Đừng đặt lệnh dựa vào nó."
)


@dataclass(frozen=True)
class Measured:
    """A number computed from data, with the sample behind it."""

    value: float
    n: int
    ci95: tuple[float, float] | None = None
    method: str = ""

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(
                "Measured needs n > 0. A figure computed from no observations is "
                "not a measurement; return Unavailable instead."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "measured",
            "value": self.value,
            "n": self.n,
            "ci95": list(self.ci95) if self.ci95 is not None else None,
            "method": self.method,
        }


@dataclass(frozen=True)
class Convention:
    """A number computed by a conventional formula whose predictive value this
    project has not measured. RSI, MACD, pivot points.

    `validated` requires `evidence`: a pointer to where the measurement lives.
    Without it the flag would be a claim nobody has to back up, which is the
    exact failure this class exists to prevent.
    """

    value: float
    reading: str
    validated: bool = False
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.validated and not self.evidence:
            raise ValueError(
                "validated=True needs `evidence` naming where the measurement lives"
            )

    def to_dict(self) -> dict[str, Any]:
        if self.validated and not self.evidence:
            raise ValueError("validated Convention reached to_dict() without evidence")
        out: dict[str, Any] = {
            "source": "convention",
            "value": self.value,
            "reading": self.reading,
            "validated": self.validated,
        }
        if self.validated:
            out["evidence"] = self.evidence
        else:
            out["warning"] = CONVENTION_WARNING
        return out


@dataclass(frozen=True)
class Unavailable:
    """No number, and why. A first-class answer, not an exception.

    "ETHUSDT has no model because it fails the gates" is more useful than
    silence and more honest than a fallback figure.
    """

    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": "unavailable", "reason": self.reason}
