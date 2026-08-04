import inspect
import json
import os
from threading import Event, Thread

import pytest
from openai_codex import ApprovalMode, Sandbox
from openai_codex.models import (
    AgentMessageDeltaNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    JsonObject,
    Notification,
    ReasoningTextDeltaNotification,
    ThreadTokenUsageUpdatedNotification,
)
from openai_codex.types import ReasoningEffort, ThreadItem, TurnError, TurnStatus

import openprover.llm.codex as codex_module
from openprover.llm import Interrupted
from openprover.llm.codex import CodexTurnError
from tests.codex_fakes import (
    FakeCatalog,
    FakeTurn,
    FakeTurnRequest,
    catalog,
    client,
    completed_event,
    result,
    usage,
)


def test_sdk_adapter_is_lazy_has_expected_context_and_uses_no_manual_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    codex, fake = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result())])

    assert codex.context_length == 1_050_000
    assert fake.configs == []
    assert "subprocess" not in inspect.getsource(codex_module)
    assert "_rpc_request" not in inspect.getsource(codex_module)

    response = codex.call("prompt", "system")

    assert len(fake.configs) == 1
    assert response["result"] == "final"


@pytest.mark.parametrize(
    ("catalog_response", "error"),
    [(FakeCatalog([]), "does not advertise"), (catalog(ReasoningEffort.medium), "high reasoning")],
)
def test_catalog_requires_executable_model_and_high_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path, catalog_response: FakeCatalog, error: str
):
    codex, fake = client(monkeypatch, tmp_path, [], catalog_response)

    with pytest.raises(RuntimeError, match=error):
        codex.call("prompt", "system")

    assert fake.close_calls == 1
    codex.cleanup()
    assert fake.close_calls == 1


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-terra"])
def test_exact_codex_models_reach_thread_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path, model: str
):
    codex, fake = client(
        monkeypatch,
        tmp_path,
        [FakeTurn("turn-1", result())],
        catalog(ReasoningEffort.high, model=model),
        model=model,
    )

    codex.call("prompt", "system")

    assert fake.thread_models == [model]


def test_call_sets_restrictive_policy_schema_and_normalizes_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    codex, fake = client(
        monkeypatch, tmp_path, [FakeTurn("turn-1", result()), FakeTurn("turn-2", result())]
    )
    schema: JsonObject = {"type": "object", "properties": {"proof": {"type": "string"}}}
    mcp_config: JsonObject = {
        "mcp_servers": {
            "lean_tools": {
                "command": "lean-mcp",
                "args": ["--stdio"],
                "env": {"LEAN_PROJECT_DIR": "/tmp/lean"},
                "enabled_tools": ["lean_verify", "lean_search"],
                "required": True,
                "startup_timeout_ms": 30000,
            }
        }
    }
    codex.mcp_config = mcp_config

    response = codex.call("prompt", "system", json_schema=schema)
    codex.call("prompt", "system", web_search=True)

    assert fake.turn_requests == [
        FakeTurnRequest(ApprovalMode.deny_all, ReasoningEffort.high, Sandbox.read_only, schema),
        FakeTurnRequest(ApprovalMode.deny_all, ReasoningEffort.high, Sandbox.read_only, None),
    ]
    assert fake.configs[0].config_overrides == (
        "features.shell_tool=false",
        "features.multi_agent=false",
        "features.plugins=false",
        "features.apps=false",
    )
    assert fake.thread_configs[0]["web_search"] == "disabled"
    assert fake.thread_configs[0]["mcp_servers"] == mcp_config["mcp_servers"]
    assert fake.thread_configs[1]["web_search"] == "live"
    assert "mcp_servers" not in fake.thread_configs[1]
    assert response["raw"]["usage"] == {
        "input_tokens": 11,
        "cached_input_tokens": 2,
        "output_tokens": 7,
        "reasoning_output_tokens": 5,
        "total_tokens": 18,
    }


def test_streams_typed_text_reasoning_and_mcp_callbacks(monkeypatch: pytest.MonkeyPatch, tmp_path):
    tool_item = ThreadItem.model_validate(
        {
            "type": "mcpToolCall",
            "id": "tool-1",
            "arguments": {"code": "example : True := by trivial"},
            "server": "lean_tools",
            "status": "completed",
            "tool": "lean_verify",
            "result": {"content": [{"type": "text", "text": "OK"}]},
        }
    )
    events = [
        Notification(
            method="item/agentMessage/delta",
            payload=AgentMessageDeltaNotification(
                delta="answer", item_id="message-1", thread_id="thread-1", turn_id="turn-1"
            ),
        ),
        Notification(
            method="item/reasoning/textDelta",
            payload=ReasoningTextDeltaNotification(
                content_index=0,
                delta="think",
                item_id="reasoning-1",
                thread_id="thread-1",
                turn_id="turn-1",
            ),
        ),
        Notification(
            method="item/started",
            payload=ItemStartedNotification(
                item=tool_item, started_at_ms=1, thread_id="thread-1", turn_id="turn-1"
            ),
        ),
        Notification(
            method="item/completed",
            payload=ItemCompletedNotification(
                item=tool_item, completed_at_ms=2, thread_id="thread-1", turn_id="turn-1"
            ),
        ),
        completed_event(TurnStatus.completed),
    ]
    codex, _ = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result(), events)])
    stream: list[tuple[str, str]] = []
    started: list[tuple[str, JsonObject]] = []
    completed_tools: list[tuple[str, JsonObject, str, str, int]] = []

    response = codex.call(
        "prompt",
        "system",
        stream_callback=lambda text, kind: stream.append((text, kind)),
        tool_start_callback=lambda name, arguments: started.append((name, arguments)),
        tool_callback=lambda name, arguments, result, status, duration_ms: completed_tools.append(
            (name, arguments, result, status, duration_ms)
        ),
    )

    assert stream == [("answer", "text"), ("think", "thinking")]
    assert started == [("lean_verify", {"code": "example : True := by trivial"})]
    assert completed_tools[0][:2] == started[0]
    assert completed_tools[0][2] == "OK"
    assert completed_tools[0][3:] == ("ok", 0)
    assert response["result"] == ""
    assert response["thinking"] == "think"


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        (TurnStatus.failed, None, "Codex turn ended with unexpected status failed"),
        (TurnStatus.in_progress, None, "Codex turn ended with unexpected status inProgress"),
        (
            TurnStatus.completed,
            TurnError(message="SDK turn error"),
            "Codex turn error: SDK turn error",
        ),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_failed_or_unfinished_turn_never_returns_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    status: TurnStatus,
    error: TurnError | None,
    expected: str,
    streaming: bool,
):
    turn_result = result(status=status, error=error)
    events = [
        Notification(
            method="thread/tokenUsage/updated",
            payload=ThreadTokenUsageUpdatedNotification(
                thread_id="thread-1", token_usage=usage(), turn_id="turn-1"
            ),
        ),
        completed_event(status, error),
    ]
    codex, _ = client(monkeypatch, tmp_path, [FakeTurn("turn-1", turn_result, events)])
    archive_path = tmp_path / f"{status.value}.md"

    with pytest.raises(RuntimeError, match=expected) as raised:
        if streaming:
            codex.call(
                "prompt",
                "system",
                archive_path=archive_path,
                stream_callback=lambda text, kind: None,
            )
        else:
            codex.call("prompt", "system", archive_path=archive_path)

    assert isinstance(raised.value, CodexTurnError)
    assert raised.value.response["raw"]["usage"]["output_tokens"] == 7
    archived = archive_path.read_text()
    assert f"error: {expected}\n" in archived
    assert "output_tokens: 7\n" in archived
    assert json.loads(archive_path.with_suffix(".raw.json").read_text()) == (
        raised.value.response["raw"]
    )


def test_soft_interrupt_returns_partial_output(monkeypatch: pytest.MonkeyPatch, tmp_path):
    release = Event()
    handle = FakeTurn("turn-1", result(text="partial", status=TurnStatus.interrupted), block=release)
    codex, _ = client(monkeypatch, tmp_path, [handle])
    outcome: list[dict] = []
    thread = Thread(target=lambda: outcome.append(codex.call("prompt", "system")))
    thread.start()
    assert handle.started.wait(timeout=1)

    codex.soft_interrupt()
    release.set()
    thread.join(timeout=1)

    assert handle.interrupts == 1
    assert outcome[0]["result"] == "partial"
    assert outcome[0]["finish_reason"] == "soft_interrupted"


def test_hard_interrupts_every_concurrent_active_turn(monkeypatch: pytest.MonkeyPatch, tmp_path):
    release = Event()
    first = FakeTurn("turn-1", result(status=TurnStatus.interrupted), block=release)
    second = FakeTurn("turn-2", result(turn_id="turn-2", status=TurnStatus.interrupted), block=release)
    codex, _ = client(monkeypatch, tmp_path, [first, second])
    outcomes: list[Interrupted] = []

    def call() -> None:
        try:
            codex.call("prompt", "system")
        except Interrupted as exc:
            outcomes.append(exc)

    threads = [Thread(target=call), Thread(target=call)]
    for thread in threads:
        thread.start()
    assert first.started.wait(timeout=1)
    assert second.started.wait(timeout=1)

    codex.interrupt()
    release.set()
    for thread in threads:
        thread.join(timeout=1)

    assert first.interrupts == 1
    assert second.interrupts == 1
    assert len(outcomes) == 2
    assert outcomes[0].response is not None
    assert outcomes[0].response["raw"]["usage"]["output_tokens"] == 7


def test_sdk_config_inherits_ambient_codex_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    source_home = tmp_path / "source-codex"
    source_home.mkdir()
    auth = source_home / "auth.json"
    config_file = source_home / "config.toml"
    auth.write_text('{"token":"secret"}')
    config_file.write_text("untrusted = true\n")
    original_auth = auth.read_bytes()
    original_config = config_file.read_bytes()
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    codex, fake = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result())])

    # Given: a valid ambient Codex home with user configuration.
    # When: the Codex client starts a turn.
    codex.call("prompt", "system")

    # Then: the SDK inherits the ambient environment and OpenProver keeps its
    # restrictive feature overrides authoritative.
    config = fake.configs[0]
    assert config.env is None
    assert config.cwd is None
    assert os.environ["CODEX_HOME"] == str(source_home)
    assert auth.read_bytes() == original_auth
    assert config_file.read_bytes() == original_config
    assert fake.configs[0].config_overrides == (
        "features.shell_tool=false",
        "features.multi_agent=false",
        "features.plugins=false",
        "features.apps=false",
    )


def test_cleanup_closes_started_sdk_once(monkeypatch: pytest.MonkeyPatch, tmp_path):
    codex, fake = client(monkeypatch, tmp_path, [FakeTurn("turn-1", result())])
    codex.call("prompt", "system")

    codex.cleanup()
    codex.cleanup()

    assert fake.close_calls == 1
