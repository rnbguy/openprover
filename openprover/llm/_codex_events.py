"""Typed Codex turn-stream normalization."""

import re
from dataclasses import dataclass, field
from typing import Final, Protocol

from openai_codex import TurnHandle, TurnResult
from openai_codex.models import (
    AgentMessageDeltaNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    JsonObject,
    Notification,
    ReasoningSummaryTextDeltaNotification,
    ReasoningTextDeltaNotification,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)
from openai_codex.types import ThreadItem, ThreadTokenUsage, TurnStatus

_TOOL_STATUSES: Final = {
    "completed": "ok",
    "failed": "error",
    "inProgress": "running",
}


class StreamCallback(Protocol):
    def __call__(self, text: str, kind: str) -> None: ...


class ToolStartCallback(Protocol):
    def __call__(self, name: str, arguments: JsonObject) -> None: ...


class ToolCallback(Protocol):
    def __call__(
        self,
        name: str,
        arguments: JsonObject,
        result: str,
        status: str,
        duration_ms: int,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StreamCallbacks:
    stream: StreamCallback | None
    tool: ToolCallback | None
    tool_start: ToolStartCallback | None


@dataclass(slots=True)
class _StreamState:
    """Mutable accumulation is required while one SDK turn is consumed."""

    result_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    streamed_text: dict[str, str] = field(default_factory=dict)
    reasoning_text_ids: set[str] = field(default_factory=set)
    emitted_reasoning_ids: set[str] = field(default_factory=set)
    reasoning_summaries: dict[str, list[str]] = field(default_factory=dict)
    started_tools: set[str] = field(default_factory=set)
    items: list[ThreadItem] = field(default_factory=list)
    usage: ThreadTokenUsage | None = None
    completed: TurnCompletedNotification | None = None


def run_turn(turn: TurnHandle, callbacks: StreamCallbacks) -> tuple[TurnResult, str, str]:
    """Consume a typed turn stream while forwarding OpenProver callbacks."""
    if callbacks.stream is None and callbacks.tool is None and callbacks.tool_start is None:
        return turn.run(), "", ""

    state = _StreamState()
    stream = turn.stream()
    try:
        for event in stream:
            _handle_event(event, state, callbacks)
    finally:
        stream.close()
    if state.completed is None:
        raise RuntimeError("Codex turn completed without a typed completion event")
    completed = state.completed.turn
    _emit_buffered_summaries(state, callbacks)
    final_response = _final_response(state.items)
    if final_response is None and completed.status is TurnStatus.interrupted:
        final_response = "".join(state.result_parts) or None
    return (
        TurnResult(
            id=completed.id,
            status=completed.status,
            error=completed.error,
            started_at=completed.started_at,
            completed_at=completed.completed_at,
            duration_ms=completed.duration_ms,
            final_response=final_response,
            items=state.items,
            usage=state.usage,
        ),
        "".join(state.result_parts),
        "".join(state.thinking_parts),
    )


def _handle_event(event: Notification, state: _StreamState, callbacks: StreamCallbacks) -> None:
    match event.payload:
        case AgentMessageDeltaNotification(delta=delta, item_id=item_id):
            state.result_parts.append(delta)
            state.streamed_text[item_id] = state.streamed_text.get(item_id, "") + delta
            if callbacks.stream is not None:
                callbacks.stream(delta, "text")
        case ReasoningTextDeltaNotification(delta=delta, item_id=item_id):
            state.reasoning_text_ids.add(item_id)
            _emit_thinking(delta, state, callbacks)
        case ReasoningSummaryTextDeltaNotification(delta=delta, item_id=item_id):
            state.reasoning_summaries.setdefault(item_id, []).append(delta)
        case ItemStartedNotification(item=item):
            _emit_tool_start(item, state, callbacks)
        case ItemCompletedNotification(item=item):
            state.items.append(item)
            _emit_completed_item(item, state, callbacks)
        case ThreadTokenUsageUpdatedNotification(token_usage=usage):
            state.usage = usage
        case TurnCompletedNotification() as completed:
            state.completed = completed
        case _:
            return


def _emit_thinking(text: str, state: _StreamState, callbacks: StreamCallbacks) -> None:
    state.thinking_parts.append(text)
    if callbacks.stream is not None:
        callbacks.stream(text, "thinking")


def _emit_tool_start(item: ThreadItem, state: _StreamState, callbacks: StreamCallbacks) -> None:
    root = item.root
    if root.type != "mcpToolCall" or root.id in state.started_tools:
        return
    state.started_tools.add(root.id)
    if callbacks.tool_start is not None:
        arguments = root.arguments if isinstance(root.arguments, dict) else {}
        callbacks.tool_start(root.tool, arguments)


def _emit_completed_item(
    item: ThreadItem, state: _StreamState, callbacks: StreamCallbacks
) -> None:
    root = item.root
    if root.type == "agentMessage":
        missing = root.text.removeprefix(state.streamed_text.get(root.id, ""))
        if missing:
            state.result_parts.append(missing)
            if callbacks.stream is not None:
                callbacks.stream(missing, "text")
        return
    if root.type == "reasoning":
        if root.id not in state.reasoning_text_ids:
            thinking = "\n".join(root.content or root.summary or [])
            if thinking:
                state.emitted_reasoning_ids.add(root.id)
                _emit_thinking(thinking, state, callbacks)
        return
    if root.type != "mcpToolCall":
        return
    _emit_tool_start(item, state, callbacks)
    if callbacks.tool is not None:
        arguments = root.arguments if isinstance(root.arguments, dict) else {}
        result = "" if root.result is None else "\n".join(
            part["text"] for part in root.result.content if part.get("type") == "text"
        )
        if root.error is not None:
            result = root.error.message
        callbacks.tool(
            root.tool,
            arguments,
            result,
            _tool_status(root.tool, root.status.value, root.error is not None, result),
            root.duration_ms or 0,
        )


def _emit_buffered_summaries(state: _StreamState, callbacks: StreamCallbacks) -> None:
    for item_id, parts in state.reasoning_summaries.items():
        if item_id not in state.reasoning_text_ids and item_id not in state.emitted_reasoning_ids:
            _emit_thinking("".join(parts), state, callbacks)


def _final_response(items: list[ThreadItem]) -> str | None:
    phase_less: str | None = None
    for item in reversed(items):
        root = item.root
        if root.type != "agentMessage":
            continue
        phase = root.phase
        if phase is None:
            if phase_less is None:
                phase_less = root.text
        elif phase.value == "final_answer":
            return root.text
    return phase_less


def _tool_status(tool: str, status: str, has_error: bool, result: str) -> str:
    if has_error:
        return "error"
    if tool == "lean_verify" and status == "completed":
        if re.search(r"^\d+:\d+: error", result, re.MULTILINE):
            return "error"
        if re.search(r"(?i)\bsorry\b", result):
            return "partial"
    return _TOOL_STATUSES[status]


def thinking_from_items(items: list[ThreadItem]) -> str:
    """Extract stable reasoning text from completed typed items."""
    parts = [
        "\n".join(root.content or root.summary or [])
        for item in items
        if (root := item.root).type == "reasoning" and (root.content or root.summary)
    ]
    return "\n\n".join(parts)


def usage_from_result(turn_result: TurnResult) -> JsonObject:
    """Normalize the SDK's latest token breakdown into OpenProver's shape."""
    if turn_result.usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
        }
    usage = turn_result.usage.last
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
        "total_tokens": usage.total_tokens,
    }
