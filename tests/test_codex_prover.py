import threading
from pathlib import Path

import pytest

import openprover.prover as prover_module
from openprover import prompts
from openprover.budget import Budget
from openprover.llm import Interrupted
from openprover.llm.codex import MODEL, CodexClient, CodexTurnError
from openprover.prover import Prover
from openprover.tui import TUI


class FakeTUI(TUI):
    def __init__(self) -> None:
        pass

    def stream_start(self, label: str = "thinking", tab: str = "planner") -> None:
        del label, tab

    def stream_end(self, tab: str = "planner") -> None:
        del tab

    def tab_log(
        self,
        tab_id: str,
        text: str,
        color: str = "",
        dim: bool = False,
    ) -> None:
        del tab_id, text, color, dim

    def log(
        self,
        text: str,
        color: str = "",
        bold: bool = False,
        dim: bool = False,
    ) -> None:
        del text, color, bold, dim


class RejectingCodex:
    def __init__(self) -> None:
        self.calls = 0
        self.soft_interrupt_cleared = 0

    def clear_soft_interrupt(self) -> None:
        self.soft_interrupt_cleared += 1

    def call(
        self,
        *,
        prompt: str,
        system_prompt: str,
        label: str,
        stream_callback,
        archive_path: Path | None,
        tool_callback=None,
        tool_start_callback=None,
        max_tokens: int | None = None,
    ) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "result": "partial",
                "thinking": "",
                "cost": 0.0,
                "duration_ms": 1,
                "raw": {},
                "finish_reason": "soft_interrupted",
            }
        return {
            "result": "final",
            "thinking": "",
            "cost": 0.0,
            "duration_ms": 1,
            "raw": {},
            "finish_reason": "stop",
        }


def test_worker_soft_interrupt_phase_two_omits_claude_only_keyword():
    prover = Prover.__new__(Prover)
    prover.worker_llm = RejectingCodex()
    prover.__dict__["tui"] = FakeTUI()
    prover.__dict__["_stream_cb"] = lambda _tab, output_only=False: None
    prover.__dict__["_check_error_policy"] = lambda _error: "stop"

    response = prover._run_worker_single_turn("prompt", "system", "worker", None)

    assert response["result"] == "final"
    assert prover.worker_llm.calls == 2
    assert prover.worker_llm.soft_interrupt_cleared == 1


@pytest.mark.parametrize(
    "error",
    [
        Interrupted({"raw": {"usage": {"output_tokens": 7}}}),
        CodexTurnError(
            "Codex turn error: failed", {"raw": {"usage": {"output_tokens": 7}}}
        ),
    ],
)
def test_budget_accounts_response_carried_usage_once_before_reraising(error):
    prover = Prover.__new__(Prover)
    prover.budget = Budget("tokens", 100)
    prover._budget_persistence_lock = threading.Lock()
    prover._backoff_delay = 16
    prover._rate_limit_backoff = 32
    prover._transient_backoff = 8
    checkpoints: list[int] = []
    tracked: list[dict] = []

    def checkpoint_locked(output_tokens=0):
        del output_tokens

    def checkpoint(output_tokens=0):
        checkpoints.append(output_tokens)

    def track(resp):
        tracked.append(resp)

    prover._checkpoint_budget_state_locked = checkpoint_locked
    prover._checkpoint_budget_state = checkpoint
    prover._track_output_tokens = track

    def failing_call():
        raise error

    with pytest.raises(type(error)) as raised:
        prover._call_with_budget(failing_call)

    assert raised.value is error
    assert checkpoints == [7]
    assert tracked == []
    assert (
        prover._backoff_delay,
        prover._rate_limit_backoff,
        prover._transient_backoff,
    ) == (16, 32, 8)


def test_exception_accounting_failure_preserves_original_error():
    prover = Prover.__new__(Prover)
    prover.budget = Budget("tokens", 100)
    prover._budget_persistence_lock = threading.Lock()
    error = CodexTurnError(
        "Codex turn error: failed",
        {"raw": {"usage": {"output_tokens": 7}}},
    )
    prover._checkpoint_budget_state_locked = lambda output_tokens=0: None

    def fail_checkpoint(output_tokens=0):
        del output_tokens
        raise RuntimeError("checkpoint failed")

    prover._checkpoint_budget_state = fail_checkpoint

    def fail_tracking(resp) -> None:
        del resp
        pytest.fail("failed responses must not use success accounting")

    prover._track_output_tokens = fail_tracking

    def failing_call():
        raise error

    with pytest.raises(CodexTurnError) as raised:
        prover._call_with_budget(failing_call)

    assert raised.value is error


def test_codex_worker_exposes_only_lean_search(tmp_path, monkeypatch):
    codex = CodexClient(MODEL, tmp_path / "archive")
    prover = Prover(
        tmp_path / "run",
        "theorem",
        "prove",
        lambda _work_dir: codex,
        MODEL,
        Budget("tokens", 100),
        True,
        False,
        FakeTUI(),
        lean_project_dir=tmp_path,
        lean_worker_tools=True,
    )
    captured_prompts = []

    def capture_worker_prompt(
        _prompt,
        system_prompt,
        _worker_id,
        _archive_path,
        *,
        use_mcp_tools=False,
    ):
        assert use_mcp_tools
        captured_prompts.append(system_prompt)
        return {}

    monkeypatch.setattr(prover, "_run_worker_single_turn", capture_worker_prompt)

    prover._run_worker({"description": "prove the theorem"}, "worker")

    assert prover.worker_llm.mcp_config["mcp_servers"]["lean_tools"][
        "enabled_tools"
    ] == ["lean_search"]
    assert "lean_search" in captured_prompts[0]
    assert "lean_verify" not in captured_prompts[0]
    assert "lean_store" not in captured_prompts[0]


@pytest.mark.parametrize(
    ("lean_store_available", "available_tools"),
    [
        (False, ("lean_verify", "lean_search")),
        (True, ("lean_verify", "lean_store", "lean_search")),
    ],
)
def test_claude_and_native_worker_prompts_retain_tools(
    lean_store_available,
    available_tools,
):
    prompt = prompts.worker_system_prompt(
        lean_worker_tools=True,
        lean_store_available=lean_store_available,
    )

    assert all(tool in prompt for tool in available_tools)


def test_response_carried_transient_failures_escalate_backoff(monkeypatch):
    failures = [
        CodexTurnError(
            "Codex turn error: 502 gateway", {"raw": {"usage": {"output_tokens": 7}}}
        ),
        CodexTurnError(
            "Codex turn error: 502 gateway", {"raw": {"usage": {"output_tokens": 7}}}
        ),
    ]
    responses = iter(
        [
            *failures,
            {
                "result": "final",
                "thinking": "",
                "cost": 0.0,
                "duration_ms": 1,
                "raw": {"usage": {"output_tokens": 1}},
                "finish_reason": "stop",
            },
        ]
    )

    def call(**_kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    prover = Prover.__new__(Prover)
    prover.worker_llm = type("Worker", (), {"call": staticmethod(call)})()
    prover.__dict__["tui"] = FakeTUI()
    prover.budget = Budget("tokens", 100)
    prover._budget_persistence_lock = threading.Lock()
    prover._checkpoint_budget_state_locked = lambda output_tokens=0: None
    prover._checkpoint_budget_state = lambda output_tokens=0: None
    prover._backoff_delay = 4
    prover._rate_limit_backoff = 4
    prover._transient_backoff = 4
    prover.on_budget_out = None
    prover.on_rate_limited = None
    prover.__dict__["_stream_cb"] = lambda _tab, output_only=False: None
    sleeps = []
    monkeypatch.setattr(prover_module.time, "sleep", sleeps.append)

    result = prover._run_worker_single_turn("prompt", "system", "worker", None)

    assert result["result"] == "final"
    assert sleeps == [4, 8]
