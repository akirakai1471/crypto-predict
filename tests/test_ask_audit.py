"""The audit is the layer with teeth, so its test is written first.

It catches invented price levels and invented percentages - the exact failure in
the answer that prompted this feature. It does not catch qualitative claims, and
that limit is reported rather than implied.
"""

from cryptopred.ask.audit import audit_answer, collect_tool_numbers, extract_numbers


def test_vietnamese_decimals_and_thousands_are_both_parsed():
    """2.500 means two thousand five hundred; 58,4 means fifty-eight point four."""
    assert 2500.0 in extract_numbers("giá về 2.500 USD")
    assert 58.4 in extract_numbers("xác suất 58,4%")
    assert 2500.0 in extract_numbers("giá về 2,500 USD")
    assert 58.4 in extract_numbers("xác suất 58.4%")


def test_an_invented_number_is_flagged():
    report = audit_answer(
        answer="Giá sẽ về 2.500 rồi bật lên 2.900.",
        tool_numbers=[2500.0],
    )
    assert report["unmatched"] == [2900.0]
    assert report["ok"] is False


def test_rounding_is_tolerated():
    report = audit_answer(answer="xác suất 58,4%", tool_numbers=[0.584231])
    assert report["unmatched"] == []
    assert report["ok"] is True


def test_percentages_match_their_fractional_tool_value():
    report = audit_answer(answer="chạm 31,2% số lần", tool_numbers=[0.312])
    assert report["ok"] is True


def test_small_ordinals_are_not_treated_as_claims():
    """'trong 24 giờ' and 'top 3' are structure, not measurements."""
    report = audit_answer(answer="trong 24 giờ, xem 3 mức", tool_numbers=[])
    assert report["ok"] is True


def test_a_large_invented_integer_is_still_caught():
    """The small-ordinal exemption must not become a hole big enough to drive a
    price through."""
    report = audit_answer(answer="giá về 2900", tool_numbers=[])
    assert report["ok"] is False


def test_the_report_states_what_it_did_not_check():
    report = audit_answer(answer="động lượng còn tích cực", tool_numbers=[])
    assert "định tính" in report["limitation"]


def test_the_source_table_lists_matched_figures():
    report = audit_answer(answer="xác suất 58,4%", tool_numbers=[0.584231])
    assert report["matched"] == [58.4]


def test_unmatched_figures_are_reported_deterministically():
    """A figure has several plausible readings; the report must pick the same
    one every run, or the audit's own output becomes flaky."""
    first = audit_answer(answer="giá về 2.900", tool_numbers=[])
    for _ in range(5):
        assert audit_answer(answer="giá về 2.900", tool_numbers=[]) == first


def test_tool_numbers_are_collected_from_nested_payloads():
    payload = {
        "conditional": {"value": 0.584, "n": 2401, "interval": [0.569, 0.599]},
        "levels": [{"value": 2500.0}, {"value": 2900.0}],
        "is_stale": True,
        "label": "biến động cao",
    }
    found = collect_tool_numbers(payload)
    assert 0.584 in found
    assert 2900.0 in found
    assert True not in found  # a bool is not a measurement
