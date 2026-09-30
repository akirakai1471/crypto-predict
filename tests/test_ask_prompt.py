"""The prompt tells the model this project's specific history of overconfidence,
not a generic instruction to be careful."""

from cryptopred.ask.prompt import SYSTEM_PROMPT
from cryptopred.briefing.touch import MEASURED_COVERAGE


def test_it_forbids_numbers_that_did_not_come_from_a_tool():
    assert "không được" in SYSTEM_PROMPT.lower()
    assert "tool" in SYSTEM_PROMPT.lower()


def test_it_names_the_projects_own_failures_concretely():
    """Generic caution does not survive contact with a confident-sounding table.
    Specific history does."""
    for marker in ("0 tín hiệu", "99.4%", "findings.md"):
        assert marker in SYSTEM_PROMPT


def test_it_forbids_recommendations():
    assert "khuyến nghị" in SYSTEM_PROMPT


def test_it_requires_saying_khong_do_duoc():
    assert "không đo được" in SYSTEM_PROMPT


def test_it_tells_the_model_the_interval_is_not_ninety_five_percent():
    """The model will otherwise reach for the number every reader assumes."""
    assert "95%" in SYSTEM_PROMPT
    assert f"{MEASURED_COVERAGE:.0%}" in SYSTEM_PROMPT


def test_it_is_not_so_long_it_dominates_the_cache():
    assert len(SYSTEM_PROMPT) < 6000
