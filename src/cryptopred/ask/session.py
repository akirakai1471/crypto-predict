"""The Claude Opus 5 tool-calling loop.

**Why a manual loop rather than the SDK's tool runner**, which its own docs
recommend by default: the runner keeps its message history internally and does
not expose it. This feature's binding honesty mechanism is an audit that traces
every number in the answer back to a tool result, so the tool results are not
incidental — they are the point. A loop that hides them would leave the audit
with nothing to check.

Two lesser reasons point the same way. The tool schemas here are hand-written
because their Vietnamese descriptions carry the warnings the model reads before
it reads any value, and `@beta_tool` derives schemas from Python signatures
instead. And the runner is beta, while `messages.create` is not.

`client` is injectable, so every test in this layer runs with no network call
and no spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cryptopred.ask.audit import audit_answer, collect_tool_numbers
from cryptopred.ask.prompt import SYSTEM_PROMPT
from cryptopred.ask.tools import TOOL_SCHEMAS, BriefingTools
from cryptopred.config import Config

MODEL = "claude-opus-5"

# Anthropic list price, dollars per million tokens, cached 2026-09-21.
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 25.00

# A question needs a handful of tool calls. Well past that the model is looping,
# and every extra round costs real money on a request the user cannot see.
MAX_TOOL_ROUNDS = 12


@dataclass
class AskResult:
    answer: str
    audit: dict[str, Any]
    usage: dict[str, int]
    cost_usd: float
    tool_calls: list[str] = field(default_factory=list)


def _default_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


def answer_question(
    question: str,
    cfg: Config,
    interval: str = "1h",
    effort: str = "medium",
    model: str = MODEL,
    client: Any | None = None,
) -> AskResult:
    """Answer one question, then audit the answer against what the tools said."""
    client = client or _default_client()
    tools = BriefingTools(cfg, interval=interval)

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    tool_payloads: list[Any] = []
    called: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}

    response = None
    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Stable across every question, so it belongs in the cached
                    # prefix; the question sits after it and varies.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOL_SCHEMAS,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=messages,
        )
        _accumulate(usage, response)

        if getattr(response, "stop_reason", None) == "refusal":
            detail = getattr(response, "stop_details", None)
            reason = getattr(detail, "explanation", "") if detail else ""
            return AskResult(
                answer=f"Model từ chối trả lời câu này. Lý do hệ thống báo: {reason}",
                audit=_empty_audit(),
                usage=usage,
                cost_usd=_cost(usage),
                tool_calls=called,
            )

        calls = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
        if not calls:
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for call in calls:
            called.append(call.name)
            payload, is_error = _run_tool(tools, call.name, call.input)
            if not is_error:
                tool_payloads.append(payload)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": str(payload),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})
    else:
        return AskResult(
            answer=(
                f"Dừng lại sau {MAX_TOOL_ROUNDS} vòng gọi công cụ mà chưa có câu trả "
                "lời. Model đang lặp — hãy hỏi lại cụ thể hơn."
            ),
            audit=_empty_audit(),
            usage=usage,
            cost_usd=_cost(usage),
            tool_calls=called,
        )

    text = "\n".join(
        b.text for b in response.content if getattr(b, "type", "") == "text"
    )
    return AskResult(
        answer=text,
        audit=audit_answer(text, collect_tool_numbers(tool_payloads)),
        usage=usage,
        cost_usd=_cost(usage),
        tool_calls=called,
    )


def _run_tool(
    tools: BriefingTools, name: str, arguments: dict[str, Any]
) -> tuple[Any, bool]:
    """Run one tool. A failure becomes a result the model can read and report,
    not an exception that ends the turn with nothing to show the user."""
    method = getattr(tools, name, None)
    if method is None or name not in {s["name"] for s in TOOL_SCHEMAS}:
        return ({"error": f"không có công cụ tên '{name}'"}, True)
    try:
        return (method(**arguments), False)
    except Exception as exc:  # noqa: BLE001 - the model gets to see any failure
        return ({"error": f"{type(exc).__name__}: {exc}"}, True)


def _accumulate(usage: dict[str, int], response: Any) -> None:
    u = getattr(response, "usage", None)
    if u is None:
        return
    usage["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0)
    usage["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
    usage["cache_read_input_tokens"] += int(
        getattr(u, "cache_read_input_tokens", 0) or 0
    )


def _empty_audit() -> dict[str, Any]:
    return {"ok": True, "matched": [], "unmatched": [], "limitation": ""}


def _cost(usage: dict[str, int]) -> float:
    return (
        usage["input_tokens"] / 1e6 * INPUT_PER_MTOK
        + usage["output_tokens"] / 1e6 * OUTPUT_PER_MTOK
    )
