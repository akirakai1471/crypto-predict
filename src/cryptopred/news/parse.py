"""Turn RSS and Atom bytes into headlines.

Publishers get feeds wrong in every way the formats allow: HTML inside titles,
entities XML has never heard of, dates in three formats or none, whitespace
before the XML declaration. None of that is a reason to lose the other items
in the feed, so each item is read on its own and a bad field becomes None
rather than an exception. Only a document that is not a feed at all raises.

Standard library only. `xml.etree.ElementTree` on this interpreter's expat
(2.4+) refuses entity-expansion bombs and never fetches external entities, so a
hostile feed can waste one poll but cannot reach the filesystem or the network.
"""

from __future__ import annotations

import email.utils
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from html.entities import name2codepoint
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


class FeedParseError(ValueError):
    """The bytes are not a readable RSS or Atom document."""


@dataclass(frozen=True)
class Entry:
    """One headline as the feed stated it.

    `published_at` is the publisher's claim and nothing more. It is kept because
    the gap between it and our own clock is worth measuring, but nothing that
    has to be point-in-time reads it. The timestamp that counts is
    `received_at`, stamped by our clock when the row is first stored - see
    news/store.py for why.
    """

    uid: str
    title: str
    link: str | None
    published_at: datetime | None


# A feed stuffing the article body into <title> should not stuff the dashboard.
MAX_TITLE_CHARS = 300

_FEED_ROOTS = {"rss", "feed", "RDF"}
_OWN_NAMESPACES = {"", "http://www.w3.org/2005/Atom", "http://purl.org/rss/1.0/"}
_ITEM_TAGS = {"item", "entry"}
# Preference order: the first-publication date before any "updated" stamp,
# which moves on every edit.
_DATE_TAGS = ("pubDate", "published", "issued", "date", "updated", "modified")

# Tags start with a letter or "/" so "BTC <3 ETH >" is left alone.
_TAG = re.compile(r"<!--.*?-->|</?[A-Za-z][^<>]*>", re.DOTALL)
_XML_ENTITIES = {b"amp", b"lt", b"gt", b"quot", b"apos"}
_NAMED_ENTITY = re.compile(rb"&([A-Za-z][A-Za-z0-9]{1,31});")
# Query parameters that identify the click, not the article.
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")


def parse_feed(data: bytes, base_url: str = "", source: str = "") -> list[Entry]:
    """Every readable item in an RSS 0.9x/1.0/2.0 or Atom document.

    Items without a usable title are dropped: a headline with nothing to read
    is nothing to show and nothing to count. Raises FeedParseError only when
    the document itself is unreadable or is not a feed - an HTML error page
    served with status 200 must count as a failed fetch, not as a feed that
    happens to be empty.
    """
    root = _parse_xml(data)
    if _local(root.tag) not in _FEED_ROOTS:
        raise FeedParseError(f"not an RSS or Atom document: <{_local(root.tag)}>")

    entries = []
    for element in root.iter():
        if _local(element.tag) not in _ITEM_TAGS:
            continue
        entry = _entry(element, base_url=base_url, source=source)
        if entry is not None:
            entries.append(entry)
    return entries


def clean_title(raw: str) -> str:
    """Plain text from a title that may carry markup and entities.

    Tags are stripped before and after unescaping because feeds encode HTML at
    two depths: CDATA holding `<b>` arrives as a tag, `&amp;lt;b&amp;gt;`
    arrives as `&lt;b&gt;` and only becomes a tag once unescaped.
    """
    text = _TAG.sub(" ", raw)
    text = html.unescape(text)
    text = _TAG.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > MAX_TITLE_CHARS:
        text = text[: MAX_TITLE_CHARS - 1].rstrip() + "…"
    return text


def title_key(title: str) -> str:
    """The form two titles must share to count as the same headline."""
    return " ".join(title.casefold().split())


def parse_date(text: str | None) -> datetime | None:
    """A publisher date as UTC, or None when it cannot be read unambiguously.

    A date with no time zone is refused rather than assumed to be UTC. It could
    be off by up to fourteen hours, and a wrong claim is worse than none for
    the one thing this field is for: measuring how late publishers are.
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    parsed: datetime | None
    try:
        parsed = email.utils.parsedate_to_datetime(text)  # RSS: RFC 822
    except (TypeError, ValueError, IndexError, OverflowError):
        parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text)  # Atom, dc:date: RFC 3339
        except ValueError:
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def normalize_url(url: str) -> str:
    """A dedupe key for a URL: same article, same key.

    Drops the scheme (a feed that moves from http to https would otherwise
    re-issue its whole backlog as new), the fragment, tracking parameters and a
    trailing slash; lowercases the host. This is a key, never a link to follow.
    """
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith(_TRACKING_PARAMS)
    )
    path = parts.path.rstrip("/") or ""
    return urlunsplit(("", host, path, urlencode(query), "")).lstrip("/")


def safe_link(raw: str | None, base_url: str = "") -> str | None:
    """An absolute http(s) link, or None.

    Anything else - `javascript:`, `data:`, a bare path with no base - is
    dropped here so that nothing downstream ever has to decide whether a link
    from a stranger's feed is safe to put in an href.
    """
    if not raw or not raw.strip():
        return None
    absolute = urljoin(base_url, raw.strip()) if base_url else raw.strip()
    parts = urlsplit(absolute)
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    # Reassembled rather than returned raw: urlsplit has already dropped the
    # tabs and newlines a browser would also ignore, so what is checked here is
    # what ends up in the href.
    return urlunsplit(parts)


def stable_uid(source: str, guid: str | None, link: str | None, title: str) -> str:
    """The identity a headline keeps across polls.

    The feed's own id first - that is what it is for - then the link. A guid
    that is not a URL ("post-12345") is namespaced by source, since two
    publishers can mint the same number. The title is the last resort, for
    items with neither.
    """
    if guid and guid.strip():
        guid = guid.strip()
        if urlsplit(guid).scheme.lower() in ("http", "https"):
            return normalize_url(guid)
        return f"{source}#{guid}"
    if link:
        return normalize_url(link)
    return f"{source}#title:{title_key(title)}"


# -- internals ----------------------------------------------------------------


def _parse_xml(data: bytes) -> ET.Element:
    # A byte-order mark or a blank line before <?xml is a common publisher
    # slip and fatal to a conforming parser.
    data = data.lstrip(b"\xef\xbb\xbf").lstrip()
    if not data:
        raise FeedParseError("empty document")
    try:
        return ET.fromstring(data)
    except ET.ParseError as first:
        # HTML entities (&nbsp;, &rsquo;) are undefined in XML. Rewriting them
        # as numeric references is lossless, and far more common than any
        # other malformation. Byte-level, so any ASCII-compatible encoding the
        # document declares still applies.
        repaired = _NAMED_ENTITY.sub(_numeric_entity, data)
        if repaired == data:
            raise FeedParseError(f"malformed XML: {first}") from first
        try:
            return ET.fromstring(repaired)
        except ET.ParseError as second:
            raise FeedParseError(f"malformed XML: {second}") from second


def _numeric_entity(match: re.Match[bytes]) -> bytes:
    name = match.group(1)
    if name in _XML_ENTITIES:
        return match.group(0)
    codepoint = name2codepoint.get(name.decode("ascii"))
    return f"&#{codepoint};".encode("ascii") if codepoint else match.group(0)


def _local(tag: object) -> str:
    """Tag name without its namespace. Comments and PIs have non-string tags."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    """Direct children with this local name, the feed's own vocabulary first.

    An RSS item can carry `media:title` beside `title`, or `atom:link` beside
    `link`; the extension version is often empty or points somewhere else.
    """
    matches = [child for child in element if _local(child.tag) == name]
    return sorted(matches, key=lambda child: _namespace(child.tag) not in _OWN_NAMESPACES)


def _namespace(tag: object) -> str:
    if isinstance(tag, str) and tag.startswith("{"):
        return tag[1:].split("}", 1)[0]
    return ""


def _text(element: ET.Element | None) -> str:
    # itertext, not .text: Atom's type="xhtml" puts the title inside a <div>.
    return "".join(element.itertext()) if element is not None else ""


def _first_text(element: ET.Element, *names: str) -> str:
    for name in names:
        for child in _children(element, name):
            value = _text(child).strip()
            if value:
                return value
    return ""


def _link(element: ET.Element) -> str:
    """RSS puts the URL in <link>'s text; Atom puts it in href on a rel=alternate
    link (rel absent means alternate). Some RSS feeds carry an atom:link too."""
    hrefs: list[tuple[str, str]] = []
    for child in _children(element, "link"):
        text = _text(child).strip()
        if text:
            return text
        href = (child.get("href") or "").strip()
        if href:
            hrefs.append((child.get("rel") or "alternate", href))
    for rel, href in hrefs:
        if rel == "alternate":
            return href
    return hrefs[0][1] if hrefs else ""


def _entry(element: ET.Element, base_url: str, source: str) -> Entry | None:
    title = clean_title(_first_text(element, "title"))
    if not title:
        return None

    link = safe_link(_link(element), base_url)
    guid_el = next(iter(_children(element, "guid")), None)
    guid = _text(guid_el).strip() or _first_text(element, "id")
    if not guid:
        # RSS 1.0 identifies an item by its rdf:about attribute.
        guid = next(
            (v for k, v in element.attrib.items() if _local(k) == "about"), ""
        ).strip()
    if link is None and guid_el is not None and guid_el.get("isPermaLink", "true") != "false":
        link = safe_link(guid, base_url)

    published = None
    for name in _DATE_TAGS:
        published = parse_date(_first_text(element, name))
        if published is not None:
            break

    return Entry(
        uid=stable_uid(source, guid, link, title),
        title=title,
        link=link,
        published_at=published,
    )
