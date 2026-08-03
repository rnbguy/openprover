import threading

import pytest

from openprover import cli
from openprover.budget import Budget
from openprover.prover import Prover


class FakeTUI:
    def update_budget(self, _status):
        pass

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class FakeLLM:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.calls = 0

    def call(self, **_kwargs):
        self.calls += 1
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result

    def chat(self, **_kwargs):
        self.calls += 1
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def make_prover(tmp_path, budget, llm):
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    cli._save_run_config(
        work_dir, planner_model="sonnet", worker_model="sonnet",
        budget_mode="tokens", budget_limit=2, conclude_after=0.99,
        max_workers=1, isolation=True, autonomous=True, mode="prove",
        lean_project_dir=None, lean_items=False, lean_worker_tools=False,
        provider_url="http://127.0.0.1:8000", answer_reserve=4096,
        history_budget=0, verifier=True,
    )
    prover = Prover.__new__(Prover)
    prover.work_dir = work_dir
    prover.budget = budget
    prover.tui = FakeTUI()
    prover._budget_persistence_lock = threading.Lock()
    prover._backoff_delay = prover._rate_limit_backoff = prover._transient_backoff = 4
    prover.worker_llm = llm
    prover.step_num = 1
    prover._stream_cb = lambda *_args, **_kwargs: None
    prover.lean_work_dir = prover.lean_project_dir = None
    return prover


def response(result, finish_reason="stop", **extra):
    return {
        "result": result, "thinking": "", "cost": 1.0, "duration_ms": 1,
        "raw": {"usage": {"output_tokens": 2}}, "finish_reason": finish_reason,
        **extra,
    }


def skipped(result):
    assert result == {
        "result": "(skipped: budget exhausted)", "error": "budget_exhausted",
        "cost": 0.0, "duration_ms": 0, "raw": {},
    }


def test_single_turn_and_verifier_denial_are_deterministic_and_do_not_call_model(tmp_path):
    llm = FakeLLM()
    prover = make_prover(tmp_path, Budget("tokens", 2, initial_output_tokens=2), llm)

    skipped(prover._run_worker_single_turn("prompt", "system", "worker", None))
    skipped(prover._run_worker_multi_turn("prompt", "system", "worker", None))
    skipped(prover._run_verifier("task", "worker output", "verifier", None))

    assert llm.calls == 0


def test_single_turn_phase_two_denial_preserves_completed_accounting(tmp_path):
    llm = FakeLLM([response("partial", "length")])
    prover = make_prover(tmp_path, Budget("tokens", 2), llm)

    result = prover._run_worker_single_turn("prompt", "system", "worker", None)

    assert result["result"] == "(skipped: budget exhausted)"
    assert result["error"] == "budget_exhausted"
    assert result["cost"] == 1.0
    assert result["duration_ms"] == 1
    assert result["raw"] == {"usage": {"output_tokens": 2}}
    assert result["thinking"] == ""
    assert prover._run_verifiers([{"description": "task"}], [result], prover.work_dir) == {}
    assert llm.calls == 1


def test_verifier_phase_two_denial_preserves_completed_accounting(tmp_path):
    llm = FakeLLM([response("partial", "length")])
    prover = make_prover(tmp_path, Budget("tokens", 2), llm)

    result = prover._run_verifier("task", "worker output", "verifier", None)

    assert result["result"] == "(skipped: budget exhausted)"
    assert result["error"] == "budget_exhausted"
    assert result["cost"] == 1.0
    assert result["duration_ms"] == 1
    assert result["raw"] == {"usage": {"output_tokens": 2}}
    assert result["thinking"] == ""
    assert llm.calls == 1


def test_multi_turn_denial_after_response_preserves_composed_output(tmp_path):
    llm = FakeLLM([response("partial", "length")])
    prover = make_prover(tmp_path, Budget("tokens", 2), llm)

    result = prover._run_worker_multi_turn("prompt", "system", "worker", None)

    assert llm.calls == 1
    assert result["result"] == "partial"
    assert result["cost"] == 1.0
    assert result["duration_ms"] == 1
    assert result["error"] == "budget_exhausted"


def test_multi_turn_retry_to_exhaustion_keeps_accounting_without_tool_log_keyerror(
    monkeypatch, tmp_path,
):
    tool_call = {"id": "tool", "function": {"name": "lean_verify", "arguments": "{}"}}
    llm = FakeLLM([
        response("partial", "tool_calls", tool_calls=[tool_call]),
        RuntimeError("retry"),
    ])
    prover = make_prover(tmp_path, Budget("tokens", 4), llm)
    monkeypatch.setattr("openprover.prover.execute_worker_tool", lambda *_args: ("ok", "ok"))
    prover._check_error_policy = lambda _error: prover.budget.add_output_tokens(2) or "retry"

    result = prover._run_worker_multi_turn("prompt", "system", "worker", None)

    assert result["result"] == "(skipped: budget exhausted)"
    assert result["error"] == "budget_exhausted"
    assert result["cost"] == 1.0
    assert result["duration_ms"] == 1
    assert result["raw"]["usage"] == {"output_tokens": 2}
    assert "tool_calls_log" not in result


@pytest.mark.parametrize("error", ["interrupted", "budget_exhausted", "worker failed"])
def test_workers_with_any_error_are_not_verifier_eligible(tmp_path, error):
    llm = FakeLLM()
    prover = make_prover(tmp_path, Budget("tokens", 2), llm)

    result = prover._run_verifiers(
        [{"description": "task"}],
        [{"result": "worker output", "error": error}],
        prover.work_dir,
    )

    assert result == {}
    assert llm.calls == 0
