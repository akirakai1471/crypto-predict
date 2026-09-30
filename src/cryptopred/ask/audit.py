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

# A bare integer below this can be sentence structure — "trong 24 giờ",
# "3 mức", the 14 in "RSI-14" — but only when the words around it say so
# (_STRUCTURE_AFTER, _PERIOD_BEFORE). "RSI đang ở 85" is a claim, and exempting
# every small integer let it through unchecked.
TRIVIAL_BELOW = 100.0

# A figure matches a tool value if it is within this relative distance, which
# covers rounding.
REL_TOLERANCE = 0.02

# Every tool reports rates as fractions (0.352, never 35.2). So a figure written
# as a percentage may only match a tool value times 100, and a bare figure only
# the value itself. Matching either way round against every number let "60%"
# pass on the strength of a 61-hour wait time.
_PERCENT_AFTER = re.compile(r"\s?(?:%|phần\s+trăm|percent)", re.IGNORECASE)

# Scale words after a figure: "90K", "70 nghìn", "1,2 triệu". Without them
# "90K" was audited as 90.
_SCALE_AFTER = re.compile(
    r"\s?(k|nghìn|ngàn|triệu|tr|m|tỷ|tỉ|b)(?![a-zà-ỹđ])", re.IGNORECASE
)
_SCALES = {
    "k": 1e3, "nghìn": 1e3, "ngàn": 1e3,
    "triệu": 1e6, "tr": 1e6, "m": 1e6,
    "tỷ": 1e9, "tỉ": 1e9, "b": 1e9,
}

# Units that make a small integer a count or a duration rather than a reading.
_STRUCTURE_AFTER = re.compile(
    r"\s?(?:giờ|tiếng|h\b|ngày|nến|tuần|tháng|năm|phút|mức|lần|coin|cặp|bước|"
    r"nguồn|kịch bản|trường hợp|chỉ báo|ô\b|nhóm)",
    re.IGNORECASE,
)
# An indicator's period: the 14 in "RSI-14", "RSI 14", "EMA20", "MACD(12".
# Upper-case only - indicator names are acronyms, and "khoảng 35" is a claim.
_PERIOD_BEFORE = re.compile(r"\b[A-Z]{2,}[-\s(]?$")
# A list number at the start of a line: "1." or "2)".
_LIST_ITEM = re.compile(r"(?:^|\n)\s*$")

LIMITATION = (
    "Chỉ kiểm được con số. Không kiểm được câu định tính như "
    "'động lượng còn tích cực' — loại câu đó không có gì để đối chiếu."
)

_NUMBER = re.compile(r"-?\d[\d.,]*\d|-?\d")
# Vietnamese: dots group thousands, a comma is the decimal ("2.500", "84.560,6").
_VI_GROUPED = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$")
# English: commas group thousands, a dot is the decimal ("2,500", "84,560.6").
_EN_GROUPED = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")

# A date is not a claim about the market. Without this the year in "ngày
# 21/09/2026" is flagged as unsourced, and an audit that cries wolf on a true
# statement spends the credibility of the one signal this feature rests on.
_DATE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2}|\b\d{1,2}:\d{2}\b")


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
    sign, digits = ("-", token[1:]) if token.startswith("-") else ("", token)
    candidates = []
    # A reading counts only if its thousands separators group digits in threes.
    # "0,61" has no English reading: taking it as 061 once matched a 61-hour
    # wait time.
    if "." not in digits or _VI_GROUPED.match(digits):
        candidates.append(sign + digits.replace(".", "").replace(",", "."))
    if "," not in digits or _EN_GROUPED.match(digits):
        candidates.append(sign + digits.replace(",", ""))
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


def _matches(figure: float, tool_numbers: list[float], percent: bool = False) -> bool:
    """Is this figure one of the tool's numbers, allowing for rounding?

    A percentage is compared with tool values times 100 and nothing else; a
    bare figure with tool values as they are and nothing else.
    """
    for value in tool_numbers:
        target = value * 100.0 if percent else value
        if target == 0:
            if figure == 0:
                return True
            continue
        if abs(figure - target) / abs(target) <= REL_TOLERANCE:
            return True
    return False


def _is_percentage(answer: str, end: int) -> bool:
    """Is the figure ending at `end` written as a percentage - "%", "phần trăm",
    with or without a space before it?"""
    return _PERCENT_AFTER.match(answer, end) is not None


def _scale(answer: str, end: int) -> tuple[float, int]:
    """(multiplier, end of the scale word) for "90K", "70 nghìn"; (1, end) if none."""
    found = _SCALE_AFTER.match(answer, end)
    if found is None:
        return 1.0, end
    return _SCALES[found.group(1).lower()], found.end()


def _is_structure(answer: str, start: int, end: int) -> bool:
    """Does the text around a small integer mark it as a count, a duration,
    an indicator period or a list number rather than a reading?"""
    before = answer[max(0, start - 12) : start]
    return (
        _STRUCTURE_AFTER.match(answer, end) is not None
        or _PERIOD_BEFORE.search(before) is not None
        or (
            _LIST_ITEM.search(before) is not None
            and answer[end : end + 1] in (".", ")")
        )
    )


def audit_answer(answer: str, tool_numbers: list[float]) -> dict[str, Any]:
    """Match every figure in the answer against the numbers tools returned."""
    matched: list[float] = []
    unmatched: list[float] = []
    # U+2212 is how a typeset minus arrives; without this "−1,18%" read as +1.18.
    answer = answer.replace("−", "-")
    date_spans = [m.span() for m in _DATE.finditer(answer)]

    for match in _NUMBER.finditer(answer):
        if any(start <= match.start() < end for start, end in date_spans):
            continue

        base = _parse(match.group())
        if not base:
            continue
        multiplier, after = _scale(answer, match.end())
        percent = multiplier == 1.0 and _is_percentage(answer, match.end())
        readings = [r * multiplier for r in base]

        hit = next((r for r in readings if _matches(r, tool_numbers, percent)), None)
        if hit is not None:
            matched.append(hit)
            continue

        # Nothing matched. A small bare integer may be sentence structure - but
        # only when the words around it say so. A percentage, a scaled figure or
        # an unexplained integer is a claim the reader could act on.
        small_integer = multiplier == 1.0 and all(
            abs(r) < TRIVIAL_BELOW and float(r).is_integer() for r in readings
        )
        if (
            small_integer
            and not percent
            and _is_structure(answer, match.start(), after)
        ):
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
