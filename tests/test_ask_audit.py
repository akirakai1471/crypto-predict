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


def test_a_wrong_percentage_under_100_is_flagged_not_exempted():
    """The critical hole review found: a stated 60% against a measured 58.4%
    is wrong by more than the tolerance, and was slipping through the
    small-integer exemption purely for want of a decimal point."""
    report = audit_answer("xác suất là 60%.", tool_numbers=[0.584231])
    assert report["ok"] is False
    assert 60.0 in report["unmatched"]


def test_a_correct_whole_percentage_still_matches():
    """Tightening the exemption must not start flagging true statements."""
    report = audit_answer("xác suất là 58%.", tool_numbers=[0.5812])
    assert report["ok"] is True


def test_a_percentage_written_with_a_space_is_treated_the_same():
    assert audit_answer("60 % số lần", tool_numbers=[0.584231])["ok"] is False
    assert audit_answer("31.2 % số lần", tool_numbers=[0.312])["ok"] is True


def test_small_integers_that_are_not_percentages_stay_exempt():
    report = audit_answer("trong 24 giờ, xem 3 mức", tool_numbers=[])
    assert report["ok"] is True


def test_a_date_is_not_treated_as_an_unsourced_claim():
    """An audit that cries wolf on a true statement spends the credibility of
    the one signal this feature rests on."""
    assert audit_answer("Hôm nay là ngày 21/09/2026.", tool_numbers=[])["ok"] is True
    assert audit_answer("nến đóng 2026-09-11 07:00", tool_numbers=[])["ok"] is True


def test_a_price_beside_a_date_is_still_checked():
    """Skipping dates must not create a shadow where a real figure can hide."""
    report = audit_answer("Ngày 21/09/2026 giá về 2900.", tool_numbers=[])
    assert report["ok"] is False
    assert 2900.0 in report["unmatched"]


# -- review findings, 2026-09-30: figures that passed without a source ----------


TOOLS = [0.352, 0.461, 61.0, 24.0, 72.0, 84560.6, -0.0118, 66.1, 2401.0]


def test_a_percentage_cannot_borrow_a_match_from_a_wait_time():
    """"khoảng 60%" passed because 60 is within 2% of a 61-hour wait. Rates
    are fractions in every tool, so a percentage matches only value x 100."""
    report = audit_answer("xác suất khoảng 60%", TOOLS)
    assert report["ok"] is False and 60.0 in report["unmatched"]
    assert audit_answer("xác suất 35,2%", TOOLS)["ok"] is True


def test_a_bare_figure_cannot_borrow_a_match_from_a_fraction():
    """35 is not 0.352, and 0.61 is not the 61-hour wait."""
    report = audit_answer("chỉ số khoảng 35, hệ số 0,61", [0.352, 61.0])
    assert sorted(report["unmatched"]) == [0.61, 35.0]


def test_phan_tram_is_a_percent_sign():
    assert audit_answer("60 phần trăm số lần", TOOLS)["ok"] is False
    assert audit_answer("46,1 phần trăm số lần", TOOLS)["ok"] is True


def test_thousands_words_scale_the_figure():
    assert audit_answer("giá có thể về 90K", TOOLS)["ok"] is False
    assert audit_answer("giá có thể về 70 nghìn USD", TOOLS)["ok"] is False
    assert audit_answer("giá đang ở 84,56K", TOOLS)["ok"] is True


def test_a_typeset_minus_is_a_minus():
    assert audit_answer("đổi 24h: −1,18%", TOOLS)["ok"] is True
    assert audit_answer("đổi 24h: −5%", TOOLS)["ok"] is False


def test_a_small_integer_reading_is_checked_but_its_period_is_not():
    """"RSI-14 đang ở 85": 14 is the indicator's period, 85 is a claim."""
    report = audit_answer("RSI-14 đang ở 85", TOOLS)
    assert report["unmatched"] == [85.0]
    assert audit_answer("RSI-14 đang ở 66,1", TOOLS)["ok"] is True


def test_counts_durations_and_list_numbers_stay_exempt():
    text = "1. Trong 72 giờ, 2 kịch bản.\n2) Xem 3 mức và 5 nguồn trong 2 tuần."
    assert audit_answer(text, TOOLS)["ok"] is True


def test_a_clock_time_is_not_a_claim():
    assert audit_answer("nến đóng lúc 07:00 UTC", [])["ok"] is True


def test_a_decimal_comma_has_no_thousands_reading():
    """"0,61" is 0.61. Read the English way it became 061 = 61."""
    assert extract_numbers("0,61") == [0.61]
    assert extract_numbers("58,4") == [58.4]
    assert sorted(extract_numbers("2.500")) == [2.5, 2500.0]  # genuinely ambiguous


def test_a_lower_case_word_before_a_number_is_not_an_indicator_name():
    assert audit_answer("chỉ số khoảng 35", [])["unmatched"] == [35.0]
    assert audit_answer("EMA20 và RSI 14", [])["ok"] is True
