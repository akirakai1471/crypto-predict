"""GET /api/news, and the dashboard section that renders it.

The payload carries its own caveat for the same reason /api/metrics does: a
consumer that can render the tags without the "unvalidated" next to them will,
sooner or later. And both the title and the link come from strangers' feeds,
so nothing in the payload may be a usable script.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cryptopred.config import Config
from cryptopred.news.parse import Entry
from cryptopred.news.store import NewsStore
from cryptopred.serve.api import create_app

T0 = pd.Timestamp("2026-09-30T12:00:00Z")


@pytest.fixture
def cfg(tmp_path):
    cfg = Config()
    cfg.data.root = tmp_path
    cfg.data.symbols = ["BTCUSDT"]
    cfg.data.intervals = ["1h"]
    return cfg


def _seed(cfg):
    store = NewsStore(cfg.data.root / "news.db")
    rows = [
        ("a", "SEC approves spot bitcoin ETF options", "https://news.example.com/a"),
        ("b", "Ether staking climbs", "https://news.example.com/b"),
        ("c", "Market wrap", None),
    ]
    for i, (uid, title, link) in enumerate(rows):
        store.add(
            [Entry(uid=uid, title=title, link=link, published_at=None)],
            source="Example", received_at=T0 + pd.Timedelta(minutes=i),
        )
    store.record_poll(
        feeds_ok=4, feeds_failed=1, inserted=3, polled_at=T0 + pd.Timedelta(minutes=3)
    )
    return store


def test_news_is_newest_first_with_tags_and_the_caveat(cfg):
    _seed(cfg)
    body = TestClient(create_app(cfg)).get("/api/news").json()

    assert [i["title"] for i in body["items"]] == [
        "Market wrap", "Ether staking climbs", "SEC approves spot bitcoin ETF options",
    ]
    sec = body["items"][2]
    assert sec["symbols"] == ["BTCUSDT"]
    assert sec["high_impact"] is True
    assert sec["impact_terms"] == ["SEC", "approves", "ETF"]
    assert sec["received_at"].startswith("2026-09-30T12:00:00")
    assert "QUY ƯỚC — CHƯA KIỂM CHỨNG" in body["caveat"]
    assert body["tagging"] == "convention_unvalidated"
    assert body["last_poll"]["feeds_ok"] == 4


def test_news_filters_by_symbol_case_insensitively(cfg):
    _seed(cfg)
    body = TestClient(create_app(cfg)).get("/api/news?symbol=ethusdt").json()
    assert [i["title"] for i in body["items"]] == ["Ether staking climbs"]


def test_news_limit_is_honoured_and_capped(cfg):
    _seed(cfg)
    client = TestClient(create_app(cfg))
    assert len(client.get("/api/news?limit=1").json()["items"]) == 1
    assert len(client.get("/api/news?limit=100000").json()["items"]) == 3
    assert len(client.get("/api/news?limit=-5").json()["items"]) == 1


def test_an_empty_store_is_an_empty_list_and_no_poll(cfg):
    body = TestClient(create_app(cfg)).get("/api/news").json()
    assert body["items"] == []
    assert body["last_poll"] is None
    assert body["n_feeds"] == 5


def test_a_javascript_link_never_leaves_the_api(cfg):
    """The parser drops these on the way in; the API drops them again on the way
    out, so a row written by anything else cannot become a clickable script."""
    store = NewsStore(cfg.data.root / "news.db")
    store.add(
        [Entry(uid="x", title="<img src=x onerror=alert(1)>", link="javascript:alert(1)",
               published_at=None)],
        source="Evil", received_at=T0,
    )
    (item,) = TestClient(create_app(cfg)).get("/api/news").json()["items"]
    assert item["link"] is None
    # The title is served as data; the page escapes it before it touches the DOM.
    assert item["title"] == "<img src=x onerror=alert(1)>"


def test_the_dashboard_has_the_news_section_and_escapes_it(cfg):
    page = TestClient(create_app(cfg)).get("/").text
    assert "Tin mới" in page
    assert "chưa kiểm chứng" in page
    assert page.index("Thông báo gần đây") < page.index("Tin mới")
    # Titles and links are escaped and links opened without an opener handle.
    assert 'rel="noopener noreferrer"' in page
    assert 'target="_blank"' in page
    assert "escapeHtml(n.title)" in page
    assert "safeHref(n.link)" in page


def test_headlines_old_on_arrival_are_hidden_unless_asked_for(cfg):
    store = NewsStore(cfg.data.root / "news.db")
    store.add(
        [Entry(uid="old", title="Old video", link=None,
               published_at=(T0 - pd.Timedelta(days=9)).to_pydatetime())],
        source="Decrypt", received_at=T0,
    )
    client = TestClient(create_app(cfg))
    assert client.get("/api/news").json()["items"] == []
    (item,) = client.get("/api/news?include_old=true").json()["items"]
    assert item["old_on_arrival"] is True
