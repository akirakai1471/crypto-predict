"""Keyword tags for headlines: which coin it names, and whether it sounds big.

CONVENTION, UNVALIDATED - "QUY ƯỚC — CHƯA KIỂM CHỨNG" wherever a person sees
it. Nobody has measured that a headline containing "ETF" moves BTC more than
one that does not, or that a count of tagged headlines predicts anything. The
tags exist so a person can scan the feed faster and so that, once months of
point-in-time headlines exist, their value can be measured instead of assumed.

Matching is on whole words and ignores case: "Ethan" is not ETH, "SECurity" is
not the SEC, "Tether" is not ether. One term is case-sensitive: "fed" is the
past tense of "feed", so only "Fed" and "FED" count as the Federal Reserve.

Tags are computed when a headline is read, never stored. Editing a keyword list
then re-tags the whole history under one definition, instead of leaving rows
tagged under two definitions mixed in one table, which would quietly turn any
later measurement into a comparison between vocabularies.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# Extend by adding a symbol and the words that name it. A trailing "*" matches
# any continuation of the word ("liquidat*" -> liquidation, liquidated).
SYMBOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "BTCUSDT": ("bitcoin", "bitcoins", "btc", "xbt", "btcusdt"),
    "ETHUSDT": ("ethereum", "ether", "eth", "ethusdt"),
}

HIGH_IMPACT_KEYWORDS: tuple[str, ...] = (
    # regulators and courts
    "sec", "cftc", "lawsuit", "lawsuits", "sues", "sued",
    "ban", "bans", "banned",
    "approval", "approve", "approves", "approved",
    "etf", "etfs",
    # losses
    "hack", "hacks", "hacked", "hacker", "hackers",
    "exploit", "exploits", "exploited",
    "liquidat*", "bankrupt*", "insolven*",
    # monetary policy
    "fomc", "rate cut", "rate cuts", "rate hike", "rate hikes",
    # market structure
    "halving", "delist*",
)

# Terms whose lowercase form is an ordinary English word.
CASE_SENSITIVE_KEYWORDS: tuple[str, ...] = ("Fed", "FED")

# Shown next to every tag, in every place a person can see one.
TAG_CAVEAT = (
    "Nhãn coin và “tác động cao” là QUY ƯỚC — CHƯA KIỂM CHỨNG: chỉ là khớp từ "
    "khoá, dự án chưa đo tin nào dự báo được giá. Model không đọc tin."
)


@dataclass(frozen=True)
class Tags:
    symbols: tuple[str, ...]
    # The words that made it "high impact", so the reader can judge the match
    # instead of trusting a flag.
    impact_terms: tuple[str, ...]

    @property
    def high_impact(self) -> bool:
        return bool(self.impact_terms)


def _term_pattern(term: str) -> str:
    wildcard = term.endswith("*")
    words = term.rstrip("*").split()
    # "rate cut", "rate-cut" and "rate  cut" are the same phrase.
    body = r"[\s\-]+".join(re.escape(word) for word in words)
    return body + (r"\w*" if wildcard else "")


def _compile(terms: Iterable[str], flags: int = 0) -> re.Pattern[str] | None:
    terms = sorted(set(terms), key=len, reverse=True)
    if not terms:
        return None
    return re.compile(r"\b(?:" + "|".join(_term_pattern(t) for t in terms) + r")\b", flags)


def _unique(matches: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for match in matches:
        seen.setdefault(" ".join(match.casefold().split()), match)
    return tuple(seen.values())


class Tagger:
    """Keyword matching compiled once. The module-level instance uses the lists
    above; pass other lists to extend or test them."""

    def __init__(
        self,
        symbol_keywords: Mapping[str, Iterable[str]] = SYMBOL_KEYWORDS,
        impact_keywords: Iterable[str] = HIGH_IMPACT_KEYWORDS,
        case_sensitive_keywords: Iterable[str] = CASE_SENSITIVE_KEYWORDS,
    ) -> None:
        self._symbols = {
            symbol: pattern
            for symbol, words in symbol_keywords.items()
            if (pattern := _compile(words, re.IGNORECASE)) is not None
        }
        self._impact = [
            pattern
            for pattern in (
                _compile(impact_keywords, re.IGNORECASE),
                _compile(case_sensitive_keywords),
            )
            if pattern is not None
        ]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._symbols)

    def tag(self, title: str) -> Tags:
        symbols = tuple(s for s, pattern in self._symbols.items() if pattern.search(title))
        hits = sorted(
            (m.start(), m.group(0)) for pattern in self._impact for m in pattern.finditer(title)
        )
        return Tags(symbols=symbols, impact_terms=_unique(text for _, text in hits))


DEFAULT_TAGGER = Tagger()


def tag_title(title: str) -> Tags:
    return DEFAULT_TAGGER.tag(title)
