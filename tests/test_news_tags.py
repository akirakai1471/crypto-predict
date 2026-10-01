"""Keyword tags. CONVENTION, UNVALIDATED.

These tests pin what the matcher does, not whether it is useful - nobody has
measured that. What they guard against is the matcher being wrong on its own
terms: a substring match tags "Ethan" as ETH and "SECurity" as the regulator,
and every count built on the tags then counts the wrong thing.
"""

import pytest

from cryptopred.news.tags import TAG_CAVEAT, Tagger, tag_title

# -- symbols -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "symbols"),
    [
        ("Bitcoin tops $120K", ("BTCUSDT",)),
        ("BTC dominance climbs", ("BTCUSDT",)),
        ("Bitcoin's hashrate hits record", ("BTCUSDT",)),
        ("$BTC and $ETH diverge", ("BTCUSDT", "ETHUSDT")),
        ("ETH/BTC ratio falls to 2020 lows", ("BTCUSDT", "ETHUSDT")),
        ("Ether slides as Ethereum fees drop", ("ETHUSDT",)),
        ("bitcoin", ("BTCUSDT",)),
    ],
)
def test_coins_are_tagged_on_whole_words_in_any_case(title, symbols):
    assert tag_title(title).symbols == symbols


@pytest.mark.parametrize(
    "title",
    [
        "Ethan Park joins payments firm",   # not ETH
        "Tether mints another $1B USDT",    # "ether" inside a word
        "Ethernet standard turns 50",       # "ether" as a prefix
        "Ethena stablecoin supply grows",   # a different project
        "wBTC supply grows on L2s",         # wrapped BTC is a different token
        "Methodology update",               # "eth" inside a word
    ],
)
def test_words_that_merely_contain_a_ticker_are_not_tagged(title):
    assert tag_title(title).symbols == ()


# -- high impact ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "terms"),
    [
        ("SEC approves spot ether ETFs", ("SEC", "approves", "ETFs")),
        ("SEC's case against exchange dropped", ("SEC",)),
        ("Exchange hacked for $40M", ("hacked",)),
        ("Protocol exploit drains pool", ("exploit",)),
        ("Liquidations top $1B as bitcoin slides", ("Liquidations",)),
        ("Lender files for bankruptcy", ("bankruptcy",)),
        ("Fed holds rates; FOMC minutes due", ("Fed", "FOMC")),
        ("FED CUTS RATES", ("FED",)),
        ("Markets price a rate-cut in December", ("rate-cut",)),
        ("Country moves to ban crypto mining", ("ban",)),
        ("Exchange to delist privacy coins", ("delist",)),
        ("Bitcoin halving is 200 days away", ("halving",)),
        ("Investors file lawsuit over token sale", ("lawsuit",)),
    ],
)
def test_high_impact_terms_are_found_and_named(title, terms):
    tags = tag_title(title)
    assert tags.high_impact
    assert tags.impact_terms == terms


@pytest.mark.parametrize(
    "title",
    [
        "Wallet maker adds SECurity key support",   # SEC inside a word
        "Secure enclave wallets explained",
        "Traders fed up with sideways market",      # "fed", the verb
        "Whales fed the rally",
        "Hackathon winners announced",              # not a hack
        "Urban miners and banana republics",        # "ban" inside words
        "Bankman trial documentary released",       # "bank", not "bankrupt"
        "Federal holiday closes US markets",        # "Fed" as a prefix
        "Pirate radio: rate cutter tested",         # "rate cut" as a prefix
    ],
)
def test_near_misses_are_not_high_impact(title):
    assert not tag_title(title).high_impact, tag_title(title)


def test_a_repeated_term_is_reported_once():
    assert tag_title("ETF inflows: the ETF week").impact_terms == ("ETF",)


# -- extension ------------------------------------------------------------------------


def test_the_symbol_map_is_extensible():
    tagger = Tagger(symbol_keywords={"SOLUSDT": ("solana", "sol")})
    assert tagger.tag("Solana outage ends").symbols == ("SOLUSDT",)
    assert tagger.tag("Solar stocks rally").symbols == ()


def test_the_caveat_says_unvalidated_and_that_the_model_does_not_read_news():
    assert "QUY ƯỚC — CHƯA KIỂM CHỨNG" in TAG_CAVEAT
    assert "Model không đọc tin" in TAG_CAVEAT


# -- review findings, 2026-10-01 --------------------------------------------------


@pytest.mark.parametrize(
    ("title", "symbols"),
    [
        ("Bitcoin Cash jumps 12%", ()),
        ("Bitcoin SV miners exit", ()),
        ("Ethereum Classic hit by reorg", ()),
        ("Bitcoin Cash lags as Bitcoin rallies", ("BTCUSDT",)),
        ("Ethereum Classic and Ethereum diverge", ("ETHUSDT",)),
    ],
)
def test_another_coin_named_after_one_is_not_that_coin(title, symbols):
    assert tag_title(title).symbols == symbols


@pytest.mark.parametrize(
    ("title", "high_impact"),
    [
        ("Treasury Sec. Bessent speaks on dollar", False),   # a secretary
        ("SEC sues exchange", True),
        ("Federal Reserve holds rates steady", True),
        ("Country moves toward banning crypto mining", True),
        ("Bank regulator prohibits crypto custody", True),
    ],
)
def test_impact_terms_reviewed_on_2026_10_01(title, high_impact):
    assert tag_title(title).high_impact is high_impact, tag_title(title)
