from types import SimpleNamespace

import pytest

import openprover.prover as prover_module
from openprover.prover import Prover


class FakeTUI:
    autonomous = False

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return self._next()

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self._next()

    def _next(self):
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def response(result, cost, duration_ms, usage, finish_reason, thinking="", **extra):
    raw = {"provider": "final"}
    if usage is not None:
        raw["usage"] = usage
    return {
        "result": result,
        "thinking": thinking,
        "cost": cost,
        "duration_ms": duration_ms,
        "raw": raw,
        "finish_reason": finish_reason,
        **extra,
    }


def planner(monkeypatch, tmp_path, responses):
    prover = Prover.__new__(Prover)
    monkeypatch.setattr(prover, "planner_llm", FakeLLM(responses), raising=False)
    monkeypatch.setattr(prover, "tui", FakeTUI(), raising=False)
    monkeypatch.setattr(prover, "work_dir", tmp_path, raising=False)
    monkeypatch.setattr(prover, "repo", SimpleNamespace(list_summaries=lambda: ""), raising=False)
    monkeypatch.setattr(prover, "budget", SimpleNamespace(
        status_str=lambda: "ok", fraction_spent=lambda: 0.0, summary_str=lambda: "ok",
    ), raising=False)
    prover.whiteboard = ""
    prover.step_history = []
    prover.step_num = 1
    prover.max_workers = 1
    prover.theorem_text = "theorem"
    prover.lean_theorem_text = ""
    prover.history_budget = 0
    prover.isolation = True
    prover.mode = "informal"
    prover.lean_items = False
    prover._steps_since_productive = 0
    prover._respawn_plan = None
    prover._stream_cb = lambda *_args, **_kwargs: None
    monkeypatch.setattr(prover_module.prompts, "format_planner_prompt", lambda **_kwargs: "prompt")
    monkeypatch.setattr(prover_module.prompts, "planner_system_prompt", lambda **_kwargs: "system")
    monkeypatch.setattr(prover_module.prompts, "format_planner_truncated", lambda *_args: "phase two")
    return prover


def test_planner_parse_retry_composes_completed_attempts(monkeypatch, tmp_path):
    first = response("invalid", 1.0, 10, {"output_tokens": 2}, "stop")
    final = response("final", 2.0, 20, {"output_tokens": 3}, "stop")
    prover = planner(monkeypatch, tmp_path, [first, final])
    tracked = []
    captured = {}
    monkeypatch.setattr(prover, "_track_output_tokens", tracked.append, raising=False)
    monkeypatch.setattr(prover, "_save_step", lambda step_dir, plan: None, raising=False)
    monkeypatch.setattr(prover, "_confirm_action", lambda plans, step_dir, planner_resp=None: captured.update(response=planner_resp) or "stop", raising=False)
    monkeypatch.setattr(prover_module.prompts, "format_planner_retry", lambda **kwargs: f"retry:{kwargs['raw_output']}")
    monkeypatch.setattr(prover_module.prompts, "parse_planner_toml", lambda text: None if text == "invalid" else [{"action": "write_whiteboard", "summary": "done"}])

    assert prover._do_step() == "stop"
    assert tracked == [first, final]
    assert prover.planner_llm.calls[1]["prompt"] == "retry:invalid"
    assert captured["response"]["raw"]["usage"] == {"output_tokens": 5}
    assert captured["response"]["cost"] == 3.0
    assert captured["response"]["duration_ms"] == 30
    assert captured["response"]["result"] == "final"


@pytest.mark.parametrize("failure", [prover_module.Interrupted(), RuntimeError("retry failed")])
def test_planner_retry_exit_preserves_completed_parse_accounting(monkeypatch, tmp_path, failure):
    first = response("invalid", 1.0, 10, {"output_tokens": 2}, "stop")
    prover = planner(monkeypatch, tmp_path, [first, failure])
    captured = {}
    monkeypatch.setattr(prover, "_track_output_tokens", lambda response: None, raising=False)
    monkeypatch.setattr(prover, "_check_error_policy", lambda error: "stop", raising=False)
    monkeypatch.setattr(prover, "_handle_interrupt", lambda step_dir, resp=None: captured.update(response=resp) or "stop", raising=False)
    monkeypatch.setattr(prover, "_save_step_meta", lambda step_dir, **kwargs: captured.update(response=kwargs.get("resp")), raising=False)
    monkeypatch.setattr(prover_module.prompts, "format_planner_retry", lambda **kwargs: "retry")
    monkeypatch.setattr(prover_module.prompts, "parse_planner_toml", lambda _text: None)

    assert prover._do_step() == "stop"
    assert captured["response"]["raw"]["usage"] == {"output_tokens": 2}
    assert captured["response"]["cost"] == 1.0
    assert captured["response"]["duration_ms"] == 10


@pytest.mark.parametrize("phase_two_error", [RuntimeError("phase two failed"), prover_module.Interrupted()])
def test_planner_phase_two_failure_preserves_completed_response(monkeypatch, tmp_path, phase_two_error):
    phase_one = response("partial", 1.0, 10, {"output_tokens": 2}, "length")
    prover = planner(monkeypatch, tmp_path, [phase_one, phase_two_error])
    captured = {}
    prover._check_error_policy = lambda error: "stop"
    monkeypatch.setattr(prover, "_track_output_tokens", lambda response: None, raising=False)
    prover._save_step_meta = lambda step_dir, **kwargs: captured.update(response=kwargs.get("resp"))
    prover._handle_interrupt = lambda step_dir, resp=None: captured.update(response=resp) or "stop"
    monkeypatch.setattr(prover_module.prompts, "parse_planner_toml", lambda _text: None)

    assert prover._do_step() == "stop"
    assert captured["response"]["raw"]["usage"] == {"output_tokens": 2}
    assert captured["response"]["cost"] == 1.0
    assert captured["response"]["duration_ms"] == 10


@pytest.mark.parametrize(
    ("runner", "args", "failure", "result_text", "error"),
    [
        (Prover._run_worker_single_turn, ("prompt", "system", "worker", None), RuntimeError("worker failed"), "Worker error: worker failed", "worker failed"),
        (Prover._run_worker_single_turn, ("prompt", "system", "worker", None), prover_module.Interrupted(), "(terminated by user)", "interrupted"),
        (Prover._run_verifier, ("task", "worker output", "verifier", None), RuntimeError("verifier failed"), "Verifier error: verifier failed", "verifier failed"),
        (Prover._run_verifier, ("task", "worker output", "verifier", None), prover_module.Interrupted(), "(terminated by user)", "interrupted"),
    ],
)
def test_worker_and_verifier_phase_two_failure_preserves_completed_response(monkeypatch, runner, args, failure, result_text, error):
    phase_one = response("partial", 1.0, 10, {"output_tokens": 2}, "length")
    prover = Prover.__new__(Prover)
    prover.worker_llm = FakeLLM([phase_one, failure])
    monkeypatch.setattr(prover, "tui", FakeTUI(), raising=False)
    prover._stream_cb = lambda *_args, **_kwargs: None
    monkeypatch.setattr(prover, "_check_error_policy", lambda exception: "stop", raising=False)

    result = runner(prover, *args)

    assert result["result"] == result_text
    assert result["error"] == error
    assert result["raw"]["usage"] == {"output_tokens": 2}
    assert result["cost"] == 1.0
    assert result["duration_ms"] == 10


@pytest.mark.parametrize("has_prior_response", [False, True])
def test_multi_turn_retry_keeps_completed_response_accounting(monkeypatch, has_prior_response):
    final = response("final", 3.0, 30, {"output_tokens": 300}, "stop")
    responses = [RuntimeError("retry"), final]
    if has_prior_response:
        tool_call = {"id": "tool", "function": {"name": "lean_verify", "arguments": "{}"}}
        responses.insert(0, response("partial", 2.0, 20, {"output_tokens": 20}, "tool_calls", tool_calls=[tool_call]))
    prover = Prover.__new__(Prover)
    prover.worker_llm = FakeLLM(responses)
    monkeypatch.setattr(prover, "tui", FakeTUI(), raising=False)
    prover._stream_cb = lambda *_args, **_kwargs: None
    prover.lean_work_dir = None
    prover.lean_project_dir = None
    prover.lean_explore_service = None
    monkeypatch.setattr(prover, "_check_error_policy", lambda error: "retry", raising=False)
    monkeypatch.setattr(prover_module, "execute_worker_tool", lambda *_args: ("ok", "ok"))

    result = prover._run_worker_multi_turn("prompt", "system", "worker", None)

    assert result["raw"]["usage"] == {"output_tokens": 320 if has_prior_response else 300}
    assert result["cost"] == (5.0 if has_prior_response else 3.0)
    assert result["duration_ms"] == (50 if has_prior_response else 30)
    assert result["result"] == "final"
    assert result["tool_calls_log"] == []


@pytest.mark.parametrize(
    ("runner", "args"),
    [
        (Prover._run_worker_single_turn, ("prompt", "system", "worker", None)),
        (Prover._run_verifier, ("task", "worker output", "verifier", None)),
    ],
)
def test_worker_and_verifier_retry_compose_prior_phase_one(monkeypatch, runner, args):
    phase_one = response("partial", 1.0, 10, {"output_tokens": 2}, "length")
    final = response("final", 3.0, 30, {"output_tokens": 4}, "stop", thinking="final thinking")
    prover = Prover.__new__(Prover)
    prover.worker_llm = FakeLLM([phase_one, RuntimeError("retry"), final])
    monkeypatch.setattr(prover, "tui", FakeTUI(), raising=False)
    prover._stream_cb = lambda *_args, **_kwargs: None
    monkeypatch.setattr(prover, "_check_error_policy", lambda error: "retry", raising=False)

    result = runner(prover, *args)

    assert result["raw"]["usage"] == {"output_tokens": 6}
    assert result["cost"] == 4.0
    assert result["duration_ms"] == 40
    assert result["result"] == "final"
    assert result["thinking"] == "final thinking"
