"""The Claude Opus 5 tool-calling loop.

**Why a manual loop rather than the SDK's tool runner**, which its own docs
recommend by default: schema control. The tools here declare `strict: true`,
`additionalProperties: false`, an enum-closed symbol, and a discriminated
`target: {kind, value}` object — and the runner's documented path for client
tools is the `@beta_tool` decorator, which derives its schema from the Python
signature. Raw tool dicts are documented as accepted for *server* tools. The
descriptions matter as much as the shapes: they carry the warnings the model
reads before it reads any value.

An earlier version of this docstring led with a different reason — that the
runner hides its message history, leaving the audit nothing to trace. Review
found that overstated. The runner does keep its own history, but
`generate_tool_call_response()` is a documented affordance that returns each
round's tool results, and mirroring the history as the docs already demonstrate
would have given the audit what it needs. The claim is corrected rather than
quietly dropped, because this module exists to stop overstated claims reaching a
reader and the standard applies to its own comments first.

A third reason, minor: the runner is beta and `messages.create` is not.

`client` is injectable, so every test in this layer runs with no network call
and no spend.
"""

from __future__ import annotations

import json
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
# Prompt-cache prices as multiples of the input price: a 5-minute cache write
# (the system prompt's ephemeral marker) costs 1.25x, a cache read 0.1x.
# `input_tokens` counts only the uncached remainder, so leaving these out
# under-reported every question after the first by the whole cached prefix.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

# Requests the model may make for one question. A question needs a handful of
# tool calls; well past that the model is looping, and every extra round costs
# real money on a request the user cannot see. Counted as requests rather than
# tool rounds so the number means what its name says.
MAX_REQUESTS = 13


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
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    response = None
    for _ in range(MAX_REQUESTS):
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
                    # JSON, not str(): a Python repr sends True/False and single
                    # quotes, and would render a stray nan/inf as a bare word.
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})
    else:
        return AskResult(
            answer=(
                f"Dừng lại sau {MAX_REQUESTS} lượt gọi model mà chưa có câu trả "
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
    # The loop breaks on "no tool calls", which is true of every terminal stop
    # reason — including the ones that mean the answer was cut off. Without this
    # a truncated answer is returned looking complete, which is the same failure
    # the SDK docs warn about for an unhandled pause_turn.
    stop = getattr(response, "stop_reason", None)
    if stop in ("max_tokens", "stop_sequence"):
        text += (
            f"\n\n[Câu trả lời bị cắt giữa chừng: stop_reason = {stop}. "
            "Phần trên có thể thiếu.]"
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
    usage["cache_creation_input_tokens"] += int(
        getattr(u, "cache_creation_input_tokens", 0) or 0
    )
    usage["cache_read_input_tokens"] += int(
        getattr(u, "cache_read_input_tokens", 0) or 0
    )


def _empty_audit() -> dict[str, Any]:
    return {"ok": True, "matched": [], "unmatched": [], "limitation": ""}


def _cost(usage: dict[str, int]) -> float:
    input_equivalent = (
        usage["input_tokens"]
        + usage.get("cache_creation_input_tokens", 0) * CACHE_WRITE_MULTIPLIER
        + usage.get("cache_read_input_tokens", 0) * CACHE_READ_MULTIPLIER
    )
    return (
        input_equivalent / 1e6 * INPUT_PER_MTOK
        + usage["output_tokens"] / 1e6 * OUTPUT_PER_MTOK
    )
