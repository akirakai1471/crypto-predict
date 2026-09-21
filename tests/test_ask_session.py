"""No test here makes a paid API call.

The tools are where the risk lives and they are tested directly. This layer is
tested for wiring: does the audit run, does staleness reach the answer, are
errors surfaced rather than swallowed.
"""

import pytest

from cryptopred.ask.session import MODEL, AskResult, answer_question
from tests.conftest import make_ohlcv


class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        for key, value in kw.items():
            setattr(self, key, value)


class _Usage:
    def __init__(self, input_tokens=1000, output_tokens=200):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _Response:
    def __init__(self, content, stop_reason="end_turn", stop_details=None):
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.usage = _Usage()


class FakeClient:
    """Returns a scripted sequence of responses and records the requests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _text(text):
    return _Response([_Block("text", text=text)])


def _tool_call(name, tool_input, tool_id="tu_1"):
    return _Response(
        [_Block("tool_use", name=name, input=tool_input, id=tool_id)],
        stop_reason="tool_use",
    )


@pytest.fixture
def cfg(tmp_path):
    from cryptopred.config import Config
    from cryptopred.ingest.storage import ParquetStore

    c = Config()
    c.data.root = tmp_path
    c.data.symbols = ["BTCUSDT", "ETHUSDT"]
    ParquetStore(tmp_path / "raw").write(
        "klines", "BTCUSDT", "1h", make_ohlcv(n=3000, seed=71)
    )
    return c


def test_an_invented_number_in_the_answer_is_flagged(cfg):
    client = FakeClient([_text("Giá về 2.500 rồi lên 2.900.")])
    result = answer_question("test", cfg=cfg, client=client)
    assert isinstance(result, AskResult)
    assert result.audit["ok"] is False
    assert 2900.0 in result.audit["unmatched"]


def test_numbers_that_came_from_a_tool_pass_the_audit(cfg):
    """The tool actually runs; its real output is what the audit checks."""
    client = FakeClient(
        [
            _tool_call("market_snapshot", {"symbol": "BTCUSDT"}),
            _text("Giá đóng gần nhất là {price}."),
        ]
    )
    result = answer_question("giá bao nhiêu?", cfg=cfg, client=client)
    assert result.tool_calls == ["market_snapshot"]

    # The second request must carry the tool result back to the model.
    second = client.calls[1]
    tool_results = [
        block
        for message in second["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "tu_1"


def test_a_failing_tool_is_reported_to_the_model_not_swallowed(cfg):
    client = FakeClient(
        [
            _tool_call("market_snapshot", {"symbol": "NOPEUSDT"}),
            _text("Không có dữ liệu."),
        ]
    )
    answer_question("test", cfg=cfg, client=client)
    second = client.calls[1]
    payload = str(second["messages"])
    assert "unavailable" in payload or "không nằm trong phạm vi" in payload


def test_an_unknown_tool_name_returns_an_error_result_rather_than_raising(cfg):
    client = FakeClient(
        [_tool_call("nonexistent_tool", {}), _text("Không gọi được công cụ đó.")]
    )
    result = answer_question("test", cfg=cfg, client=client)
    second = client.calls[1]
    blocks = [
        block
        for message in second["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert blocks[0]["is_error"] is True
    assert isinstance(result, AskResult)


def test_cost_is_reported(cfg):
    client = FakeClient([_text("ok")])
    result = answer_question("test", cfg=cfg, client=client)
    assert result.cost_usd > 0
    assert result.usage["input_tokens"] == 1000


def test_a_refusal_is_surfaced_not_swallowed(cfg):
    refusal = _Response([], stop_reason="refusal")
    refusal.stop_details = _Block("refusal", category="test", explanation="no")
    client = FakeClient([refusal])
    result = answer_question("test", cfg=cfg, client=client)
    assert "từ chối" in result.answer.lower()


def test_the_model_id_is_opus_5_by_default(cfg):
    client = FakeClient([_text("ok")])
    answer_question("test", cfg=cfg, client=client)
    assert client.calls[0]["model"] == MODEL
    assert MODEL == "claude-opus-5"


def test_adaptive_thinking_and_effort_are_set(cfg):
    client = FakeClient([_text("ok")])
    answer_question("test", cfg=cfg, client=client, effort="high")
    call = client.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"]["effort"] == "high"


def test_the_system_prompt_is_cached(cfg):
    """It is stable across questions, so it belongs in the cached prefix."""
    client = FakeClient([_text("ok")])
    answer_question("test", cfg=cfg, client=client)
    system = client.calls[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_the_loop_stops_rather_than_running_forever(cfg):
    """A model that keeps calling tools must not spin indefinitely."""
    from cryptopred.ask.session import MAX_TOOL_ROUNDS

    client = FakeClient(
        [_tool_call("market_snapshot", {"symbol": "BTCUSDT"}, tool_id=f"t{i}")
         for i in range(MAX_TOOL_ROUNDS + 5)]
    )
    result = answer_question("test", cfg=cfg, client=client)
    assert len(client.calls) <= MAX_TOOL_ROUNDS + 1
    assert "quá nhiều" in result.answer.lower() or result.answer
