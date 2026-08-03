from dataclasses import replace

import pytest
from openai_codex.models import (
    AgentMessageDeltaNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    Notification,
    ReasoningSummaryTextDeltaNotification,
    ReasoningTextDeltaNotification,
)
from openai_codex.types import ThreadItem, ThreadTokenUsage, TurnStatus

from openprover.llm._codex_events import usage_from_result
from tests.codex_fakes import FakeTurn, client, completed_event, result


def _item_event(method: str, item: ThreadItem) -> Notification:
    if method == "item/started":
        payload = ItemStartedNotification(
            item=item, started_at_ms=1, thread_id="thread-1", turn_id="turn-1"
        )
    else:
        payload = ItemCompletedNotification(
            item=item, completed_at_ms=2, thread_id="thread-1", turn_id="turn-1"
        )
    return Notification(method=method, payload=payload)


def test_completed_final_answer_overrides_streamed_commentary(monkeypatch: pytest.MonkeyPatch, tmp_path):
    commentary = ThreadItem.model_validate(
        {"type": "agentMessage", "id": "commentary", "phase": "commentary", "text": "work"}
    )
    final = ThreadItem.model_validate(
        {"type": "agentMessage", "id": "final", "phase": "final_answer", "text": "proof"}
    )
    events = [
        Notification(
            method="item/agentMessage/delta",
            payload=AgentMessageDeltaNotification(
                delta="work", item_id="commentary", thread_id="thread-1", turn_id="turn-1"
            ),
        ),
        _item_event("item/completed", commentary),
        Notification(
            method="item/agentMessage/delta",
            payload=AgentMessageDeltaNotification(
                delta="proof", item_id="final", thread_id="thread-1", turn_id="turn-1"
            ),
        ),
        _item_event("item/completed", final),
        completed_event(TurnStatus.completed),
    ]
    codex, _ = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result(), events)])
    streamed: list[str] = []

    response = codex.call("prompt", "system", stream_callback=lambda text, kind: streamed.append(text))

    assert streamed == ["work", "proof"]
    assert response["result"] == "proof"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"type": "agentMessage", "id": "final", "text": "fallback"}, "fallback"),
        ({"type": "agentMessage", "id": "commentary", "phase": "commentary", "text": "work"}, ""),
    ],
)
def test_completed_phase_less_message_is_only_used_without_final_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path, payload: dict, expected: str
):
    item = ThreadItem.model_validate(payload)
    events = [_item_event("item/completed", item), completed_event(TurnStatus.completed)]
    codex, _ = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result(), events)])

    response = codex.call("prompt", "system", stream_callback=lambda text, kind: None)

    assert response["result"] == expected


def test_completed_uses_last_phase_less_message(monkeypatch: pytest.MonkeyPatch, tmp_path):
    first = ThreadItem.model_validate({"type": "agentMessage", "id": "first", "text": "first"})
    last = ThreadItem.model_validate({"type": "agentMessage", "id": "last", "text": "last"})
    events = [
        _item_event("item/completed", first),
        _item_event("item/completed", last),
        completed_event(TurnStatus.completed),
    ]
    codex, _ = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result(), events)])

    response = codex.call("prompt", "system", stream_callback=lambda text, kind: None)

    assert response["result"] == "last"


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (
            [
                Notification(
                    method="item/reasoning/summaryTextDelta",
                    payload=ReasoningSummaryTextDeltaNotification(
                        delta="summary", item_id="reasoning", summary_index=0,
                        thread_id="thread-1", turn_id="turn-1",
                    ),
                ),
                Notification(
                    method="item/reasoning/textDelta",
                    payload=ReasoningTextDeltaNotification(
                        content_index=0, delta="text", item_id="reasoning",
                        thread_id="thread-1", turn_id="turn-1",
                    ),
                ),
                completed_event(TurnStatus.completed),
            ],
            "text",
        ),
        (
            [
                Notification(
                    method="item/reasoning/summaryTextDelta",
                    payload=ReasoningSummaryTextDeltaNotification(
                        delta="summary", item_id="reasoning", summary_index=0,
                        thread_id="thread-1", turn_id="turn-1",
                    ),
                ),
                completed_event(TurnStatus.completed),
            ],
            "summary",
        ),
    ],
)
def test_reasoning_summary_is_only_used_without_text_delta(
    monkeypatch: pytest.MonkeyPatch, tmp_path, events: list[Notification], expected: str
):
    codex, _ = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result(), events)])
    streamed: list[tuple[str, str]] = []

    response = codex.call(
        "prompt", "system", stream_callback=lambda text, kind: streamed.append((text, kind))
    )

    assert streamed == [(expected, "thinking")]
    assert response["thinking"] == expected


@pytest.mark.parametrize(
    ("content", "status", "error", "expected_status"),
    [
        ("OK", "completed", None, "ok"),
        ("1:2: error: invalid proof", "completed", None, "error"),
        ("no error: proof compiled", "completed", None, "ok"),
        ("warning: sorry remains", "completed", None, "partial"),
        ("ignored", "failed", {"message": "transaction failed"}, "error"),
    ],
)
def test_lean_verify_callback_extracts_text_and_classifies_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    content: str,
    status: str,
    error: dict | None,
    expected_status: str,
):
    payload = {
        "type": "mcpToolCall",
        "id": "tool-1",
        "arguments": {"code": "example : True := by trivial"},
        "server": "lean_tools",
        "status": status,
        "tool": "lean_verify",
        "result": {"content": [{"type": "text", "text": content}]},
    }
    if error is not None:
        payload["error"] = error
    item = ThreadItem.model_validate(payload)
    events = [
        _item_event("item/started", item),
        _item_event("item/completed", item),
        completed_event(TurnStatus.completed),
    ]
    codex, _ = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result(), events)])
    callbacks: list[tuple[str, str]] = []

    codex.call(
        "prompt",
        "system",
        tool_callback=lambda name, arguments, result, status, duration_ms: callbacks.append(
            (result, status)
        ),
    )

    expected_text = "transaction failed" if error is not None else content
    assert callbacks == [(expected_text, expected_status)]


def test_archive_includes_json_schema(monkeypatch: pytest.MonkeyPatch, tmp_path):
    archive_path = tmp_path / "call.md"
    schema = {"type": "object", "properties": {"proof": {"type": "string"}}}
    codex, _ = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result())])

    codex.call("prompt", "system", json_schema=schema, archive_path=archive_path)

    assert "======== JSON SCHEMA ========" in archive_path.read_text()


def test_usage_normalization_uses_cumulative_total():
    turn_result = replace(
        result(),
        usage=ThreadTokenUsage.model_validate(
            {
                "last": {
                    "inputTokens": 3,
                    "cachedInputTokens": 1,
                    "outputTokens": 2,
                    "reasoningOutputTokens": 1,
                    "totalTokens": 5,
                },
                "total": {
                    "inputTokens": 11,
                    "cachedInputTokens": 2,
                    "outputTokens": 7,
                    "reasoningOutputTokens": 5,
                    "totalTokens": 18,
                },
            }
        ),
    )

    assert usage_from_result(turn_result) == {
        "input_tokens": 11,
        "cached_input_tokens": 2,
        "output_tokens": 7,
        "reasoning_output_tokens": 5,
        "total_tokens": 18,
    }
