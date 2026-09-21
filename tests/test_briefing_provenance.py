"""A conventional indicator must not be able to escape without its warning.

RSI is not useless. But this project has never measured whether RSI predicts
anything on this data, and the type system should say so rather than a comment
nobody reads.
"""

import pytest

from cryptopred.briefing.provenance import Convention, Measured, Unavailable


def test_measured_carries_its_sample_size():
    m = Measured(value=0.584, n=2401, ci95=(0.569, 0.599), method="block bootstrap")
    assert m.to_dict() == {
        "source": "measured",
        "value": 0.584,
        "n": 2401,
        "ci95": [0.569, 0.599],
        "method": "block bootstrap",
    }


def test_convention_is_unvalidated_by_default_and_warns():
    c = Convention(value=66.1, reading="quy ước >70 là quá mua")
    d = c.to_dict()
    assert d["source"] == "convention"
    assert d["validated"] is False
    assert "chưa đo" in d["warning"].lower()


def test_convention_cannot_claim_validation_without_evidence():
    """validated=True is a claim about measurement, so it needs a citation."""
    with pytest.raises(ValueError, match="evidence"):
        Convention(value=66.1, reading="x", validated=True)

    ok = Convention(value=66.1, reading="x", validated=True, evidence="docs/findings.md#rsi")
    assert ok.to_dict()["validated"] is True
    assert "warning" not in ok.to_dict()


def test_unavailable_gives_a_reason_not_a_number():
    u = Unavailable(reason="ETHUSDT trượt cổng kiểm (−6.83% sau phí) nên không có model")
    assert u.to_dict() == {"source": "unavailable", "reason": u.reason}
    assert not hasattr(u, "value")
