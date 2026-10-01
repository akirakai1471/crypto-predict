"""Feed parsing.

Publishers get feeds wrong constantly. A parser that raises on the first odd
field loses every other headline in the feed along with it; one that silently
mangles a field puts garbage on the dashboard. These tests pin both halves:
tolerate what is merely sloppy, refuse what is not a feed.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from cryptopred.news.parse import (
    FeedParseError,
    clean_title,
    normalize_url,
    parse_date,
    parse_feed,
    safe_link,
    stable_uid,
)

FIXTURES = Path(__file__).parent / "fixtures" / "news"


def _rss(items: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>T</title>'
        f"{items}</channel></rss>"
    ).encode()


# -- whole feeds ---------------------------------------------------------------


def test_an_rss2_feed_yields_one_entry_per_item():
    entries = parse_feed(
        (FIXTURES / "rss2.xml").read_bytes(), base_url="https://news.example.com/rss/",
        source="Example",
    )
    assert [e.title for e in entries] == [
        "Spot Bitcoin ETF sees record inflows as SEC reviews filings",
        "Ethereum developers set date for next upgrade",
        "Ethan Park named chief security officer at payments firm",
    ]
    first = entries[0]
    assert first.link.startswith("https://news.example.com/markets/2026/09/29/")
    assert first.published_at == datetime(2026, 9, 29, 14, 5, tzinfo=UTC)
    # A non-URL guid is namespaced by source: two publishers can mint "0001".
    assert first.uid == "Example#a1b2c3d4-0001"
    # -0400 is converted, not dropped.
    assert entries[2].published_at == datetime(2026, 9, 29, 16, 0, tzinfo=UTC)


def test_the_channel_title_and_media_titles_are_not_headlines():
    """<channel><title> and <media:title> share the local name "title". Only
    items produce entries, and an item's own <title> wins over an extension's."""
    entries = parse_feed((FIXTURES / "rss2.xml").read_bytes(), source="Example")
    titles = [e.title for e in entries]
    assert "Example Crypto News" not in titles
    assert "photo caption" not in titles


def test_an_atom_feed_uses_the_alternate_link_resolved_against_the_feed():
    entries = parse_feed(
        (FIXTURES / "atom.xml").read_bytes(),
        base_url="https://atom.example.com/feed.atom",
        source="AtomEx",
    )
    assert [e.title for e in entries] == [
        "Exchange halts withdrawals after $40M exploit",
        "Fed signals rate cut as ether slides",
    ]
    exploit, fed = entries
    # rel="alternate", not the enclosure; relative href made absolute.
    assert exploit.link == "https://atom.example.com/2026/09/29/exchange-exploit"
    # <published> before <updated>: the edit stamp moves every time.
    assert exploit.published_at == datetime(2026, 9, 29, 12, 50, tzinfo=UTC)
    assert fed.published_at == datetime(2026, 9, 29, 15, 0, tzinfo=UTC)
    assert exploit.uid == "AtomEx#tag:atom.example.com,2026:post-9001"
    assert fed.uid == "atom.example.com/2026/09/29/fed-rate-cut"


def test_an_rss1_rdf_feed_is_read():
    data = b"""<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel rdf:about="https://rdf.example.com/"><title>R</title></channel>
      <item rdf:about="https://rdf.example.com/1">
        <title>Bitcoin miners sell</title><link>https://rdf.example.com/1</link>
        <dc:date>2026-09-29T10:00:00+00:00</dc:date>
      </item>
    </rdf:RDF>"""
    (entry,) = parse_feed(data, source="R")
    assert entry.title == "Bitcoin miners sell"
    assert entry.uid == "rdf.example.com/1"
    assert entry.published_at == datetime(2026, 9, 29, 10, tzinfo=UTC)


# -- malformed input -------------------------------------------------------------


def test_html_entities_that_xml_does_not_define_do_not_lose_the_feed():
    """&nbsp; and &rsquo; are fatal to a conforming XML parser and common in
    real feeds. Losing a whole feed to one apostrophe is not an option."""
    data = _rss("<item><title>Bitcoin&rsquo;s&nbsp;rally</title><link>https://x.io/a</link></item>")
    (entry,) = parse_feed(data, source="X")
    assert entry.title == "Bitcoin’s rally"


def test_whitespace_and_a_byte_order_mark_before_the_declaration_are_tolerated():
    data = b"\xef\xbb\xbf\n\n  " + _rss("<item><title>ETH up</title><link>https://x.io/a</link></item>")
    assert [e.title for e in parse_feed(data)] == ["ETH up"]


def test_items_without_a_title_are_dropped_and_the_rest_survive():
    data = _rss(
        "<item><link>https://x.io/no-title</link></item>"
        "<item><title>   </title><link>https://x.io/blank</link></item>"
        "<item><title>Kept</title><link>https://x.io/kept</link></item>"
    )
    assert [e.title for e in parse_feed(data)] == ["Kept"]


def test_a_bad_date_becomes_none_instead_of_losing_the_item():
    data = _rss(
        "<item><title>A</title><link>https://x.io/a</link><pubDate>yesterday-ish</pubDate></item>"
        "<item><title>B</title><link>https://x.io/b</link><pubDate></pubDate></item>"
    )
    entries = parse_feed(data)
    assert [e.title for e in entries] == ["A", "B"]
    assert all(e.published_at is None for e in entries)


def test_a_document_that_is_not_xml_raises():
    with pytest.raises(FeedParseError):
        parse_feed(b"<rss><channel><item><title>unclosed")


def test_an_html_page_served_with_200_is_not_an_empty_feed():
    """A Cloudflare challenge or a captive portal must count as a failed fetch.
    Read as a feed with no items, the outage would be invisible."""
    with pytest.raises(FeedParseError, match="not an RSS or Atom"):
        parse_feed(b"<html><head><title>Just a moment...</title></head><body/></html>")


def test_an_empty_body_raises():
    with pytest.raises(FeedParseError):
        parse_feed(b"   ")


# -- titles ----------------------------------------------------------------------


def test_titles_lose_their_markup_and_entities():
    assert clean_title("<b>Bitcoin</b> &amp; <i>Ether</i>") == "Bitcoin & Ether"


def test_double_escaped_markup_is_stripped_too():
    """&amp;lt;b&amp;gt; survives one unescape as a tag."""
    assert clean_title("&lt;b&gt;SEC&lt;/b&gt; sues exchange") == "SEC sues exchange"


def test_a_less_than_sign_that_is_not_a_tag_is_kept():
    assert clean_title("BTC <3 ETH > SOL") == "BTC <3 ETH > SOL"


def test_an_absurdly_long_title_is_capped():
    assert len(clean_title("word " * 500)) <= 300


# -- dates -----------------------------------------------------------------------


def test_rfc822_and_iso_dates_both_parse_to_utc():
    assert parse_date("Tue, 29 Sep 2026 14:05:00 +0700") == datetime(2026, 9, 29, 7, 5, tzinfo=UTC)
    assert parse_date("2026-09-29T14:05:00Z") == datetime(2026, 9, 29, 14, 5, tzinfo=UTC)


def test_a_date_without_a_zone_is_refused_not_guessed():
    """It could be off by up to fourteen hours. A wrong claim is worse than none
    for measuring how late publishers are."""
    assert parse_date("2026-09-29T14:05:00") is None
    assert parse_date("Tue, 29 Sep 2026 14:05:00 -0000") is None


# -- links and identity ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "javascript:alert(1)",
        "  JavaScript:alert(1)",
        "java\tscript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox",
        "//no-scheme.example.com/a",
        "",
    ],
)
def test_only_absolute_http_links_survive(raw):
    """Nothing downstream should ever have to decide whether a stranger's link
    is safe to put in an href."""
    assert safe_link(raw) is None


def test_a_javascript_link_in_a_feed_is_dropped_but_the_headline_kept():
    data = _rss("<item><title>Click me</title><link>javascript:alert(1)</link></item>")
    (entry,) = parse_feed(data, base_url="https://x.io/feed", source="X")
    assert entry.link is None
    assert entry.title == "Click me"


def test_relative_links_are_resolved_against_the_feed_url():
    assert safe_link("/a/b", "https://x.io/feed/") == "https://x.io/a/b"


def test_the_same_article_behind_tracking_parameters_gets_one_key():
    """Otherwise a feed that rotates utm_ tags re-issues its backlog as new."""
    a = normalize_url("https://WWW.x.io/a/?utm_source=rss&id=7#top")
    b = normalize_url("http://www.x.io/a?id=7&utm_medium=feed")
    assert a == b == "www.x.io/a?id=7"


def test_uid_prefers_the_guid_then_the_link_then_the_title():
    assert stable_uid("S", "post-1", "https://x.io/a", "T") == "S#post-1"
    assert stable_uid("S", None, "https://x.io/a", "T") == "x.io/a"
    assert stable_uid("S", None, None, "Big  News") == "S#title:big news"


# -- review findings, 2026-10-01: one bad item cost the whole feed --------------


@pytest.mark.parametrize(
    "bad_item",
    [
        "<item><title>Bad link</title><link>http://[oops/x</link></item>",
        "<item><title>Bad port</title><guid>http://a.com:99999999/x</guid></item>",
        "<item><title>Bad date</title><link>https://a.com/2</link>"
        "<pubDate>0001-01-01T00:00:00+05:00</pubDate></item>",
    ],
)
def test_one_malformed_item_does_not_cost_the_other_headlines(bad_item):
    """Each raised out of parse_feed - ValueError from urlsplit, OverflowError
    from astimezone - and the poll recorded a failed fetch for a feed with one
    odd item in it."""
    good = "<item><title>Good one</title><link>https://a.com/1</link></item>"
    doc = f"<rss><channel>{good}{bad_item}</channel></rss>".encode()
    titles = [e.title for e in parse_feed(doc, source="s")]
    assert "Good one" in titles
    # The bad field becomes None; the headline itself is still worth keeping.
    assert len(titles) == 2


def test_an_unreadable_date_is_none_not_an_error():
    assert parse_date("0001-01-01T00:00:00+05:00") is None


def test_an_unfollowable_link_is_none_not_an_error():
    assert safe_link("http://[oops/x") is None
    assert safe_link("http://a.com:99999999/x") is None
