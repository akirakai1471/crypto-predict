"""Did every number in the answer come from a tool?

This is the layer that actually binds. The types make it hard to lose a warning
and the system prompt asks for care, but neither can stop a model from writing a
price nobody measured. This can.

What it cannot do is check "động lượng còn tích cực". That limit is reported in
every audit rather than left implied, so the audit's silence on prose is visible
rather than mistaken for approval.
"""

from __future__ import annotations

import re
from typing import Any

# Bare integers below this are structure — "trong 24 giờ", "3 mức" — not
# measurements. The bound is deliberately low: a price or a percentage worth
# auditing is almost never a small round integer, and every figure above it is
# checked regardless of shape.
TRIVIAL_BELOW = 100.0

# A figure matches a tool value if it is within this relative distance, which
# covers rounding and unit-of-percent differences.
REL_TOLERANCE = 0.02

LIMITATION = (
    "Chỉ kiểm được con số. Không kiểm được câu định tính như "
    "'động lượng còn tích cực' — loại câu đó không có gì để đối chiếu."
)

_NUMBER = re.compile(r"-?\d[\d.,]*\d|-?\d")

# A date is not a claim about the market. Without this the year in "ngày
# 21/09/2026" is flagged as unsourced, and an audit that cries wolf on a true
# statement spends the credibility of the one signal this feature rests on.
_DATE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2}")


def _parse(token: str) -> list[float]:
    """Both conventions, because the answer is Vietnamese and the tools are not.

    "2.500" is two thousand five hundred to a Vietnamese reader and two point
    five to a parser that assumes English. Rather than guess, return every
    plausible reading and let the match decide.

    Ordered largest first and deduplicated, deliberately: an unmatched figure is
    reported by its first reading, and a set here would make that report vary
    between runs. When nothing matches, the Vietnamese reading is the one to
    show the user.
    """
    candidates = [
        token.replace(".", "").replace(",", "."),  # Vietnamese: . groups, , decimal
        token.replace(",", ""),                    # English: , groups, . decimal
    ]
    out: list[float] = []
    for candidate in candidates:
        try:
            value = float(candidate)
        except ValueError:
            continue
        if value not in out:
            out.append(value)
    return sorted(out, key=abs, reverse=True)


def extract_numbers(text: str) -> list[float]:
    """Every plausible numeric reading in the text."""
    values: list[float] = []
    for match in _NUMBER.finditer(text):
        values.extend(_parse(match.group()))
    return values


def _matches(figure: float, tool_numbers: list[float]) -> bool:
    """Is this figure one of the tool's numbers, allowing for rounding and for
    a fraction being written as a percentage?"""
    for value in tool_numbers:
        for scaled in (value, value * 100.0, value / 100.0):
            if scaled == 0:
                if figure == 0:
                    return True
                continue
            if abs(figure - scaled) / abs(scaled) <= REL_TOLERANCE:
                return True
    return False


def _is_percentage(answer: str, end: int) -> bool:
    """Is the figure ending at `end` written as a percentage?

    Allows one space, because "31.2 %" is as common as "31.2%" in Vietnamese
    prose and the two must be treated alike.
    """
    return answer[end : end + 2].lstrip().startswith("%")


def audit_answer(answer: str, tool_numbers: list[float]) -> dict[str, Any]:
    """Match every figure in the answer against the numbers tools returned."""
    matched: list[float] = []
    unmatched: list[float] = []
    date_spans = [m.span() for m in _DATE.finditer(answer)]

    for match in _NUMBER.finditer(answer):
        if any(start <= match.start() < end for start, end in date_spans):
            continue

        readings = _parse(match.group())
        if not readings:
            continue

        hit = next((r for r in readings if _matches(r, tool_numbers)), None)
        if hit is not None:
            matched.append(hit)
            continue

        # Nothing matched. A small bare integer is sentence structure rather
        # than a claim; anything larger is a figure the reader could act on.
        #
        # A percentage is never structure, whatever its size. "60%" against a
        # measured 58.4% is wrong by more than the tolerance and is exactly the
        # kind of number this audit exists to catch — it was slipping through
        # only for want of a decimal point.
        trivial = all(
            abs(r) < TRIVIAL_BELOW and float(r).is_integer() for r in readings
        )
        if trivial and not _is_percentage(answer, match.end()):
            continue
        unmatched.append(readings[0])

    return {
        "ok": not unmatched,
        "matched": matched,
        "unmatched": unmatched,
        "limitation": LIMITATION,
    }


def collect_tool_numbers(payload: Any) -> list[float]:
    """Every numeric leaf in a tool result, at any depth.

    Booleans are excluded: `is_stale: True` is a fact about the data, not a
    number the answer may quote, and letting it through would make 1.0 and 0.0
    match anything.
    """
    out: list[float] = []
    if isinstance(payload, dict):
        for value in payload.values():
            out.extend(collect_tool_numbers(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            out.extend(collect_tool_numbers(value))
    elif isinstance(payload, bool):
        pass
    elif isinstance(payload, (int, float)):
        out.append(float(payload))
    return out
