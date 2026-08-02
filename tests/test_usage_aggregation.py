from types import SimpleNamespace

import openprover.prover as prover_module
from openprover.prover import Prover


class FakeTUI:
    autonomous = False

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def call(self, **_kwargs):
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result

    def chat(self, **_kwargs):
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def response(result, thinking, cost, duration_ms, usage, finish_reason, **extra):
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


def test_planner_phase_two_composes_metadata_without_double_tracking(monkeypatch, tmp_path):
    phase_one = response(
        "partial", "first thinking", 1.25, 101,
        {"input_tokens": 2, "output_tokens": 3}, "length",
    )
    phase_two = response(
        "final", "second thinking", 2.5, 202,
        {"input_tokens": 20, "output_tokens": 30, "reasoning_output_tokens": 40,
         "prompt_tokens_details": {"cached_tokens": 9}, "server_tool_use": {"calls": 1}}, "stop",
        conversation_id="final-conversation", tool_calls=[{"id": "final-tool"}],
    )
    prover = planner(monkeypatch, tmp_path, [phase_one, phase_two])
    tracked = []
    monkeypatch.setattr(prover, "_track_output_tokens", lambda resp: tracked.append(resp), raising=False)
    monkeypatch.setattr(prover, "_save_step", lambda step_dir, plan: None, raising=False)
    captured = {}

    def confirm(plans, step_dir, planner_resp=None):
        captured["response"] = planner_resp
        return "stop"

    monkeypatch.setattr(prover, "_confirm_action", confirm, raising=False)
    monkeypatch.setattr(prover_module.prompts, "format_planner_prompt", lambda **_kwargs: "prompt")
    monkeypatch.setattr(prover_module.prompts, "planner_system_prompt", lambda **_kwargs: "system")
    monkeypatch.setattr(prover_module.prompts, "format_planner_truncated", lambda *_args: "phase two")
    monkeypatch.setattr(
        prover_module.prompts,
        "parse_planner_toml",
        lambda text: None if text == "partial" else [{"action": "write_whiteboard", "summary": "done"}],
    )

    assert prover._do_step() == "stop"
    assert tracked == [phase_one, phase_two]
    assert captured["response"]["raw"]["usage"] == {
        "input_tokens": 22,
        "output_tokens": 33,
        "reasoning_output_tokens": 40,
        "prompt_tokens_details": {"cached_tokens": 9},
        "server_tool_use": {"calls": 1},
    }
    assert captured["response"]["cost"] == 3.75
    assert captured["response"]["duration_ms"] == 303
    assert captured["response"]["result"] == "final"
    assert captured["response"]["thinking"] == "second thinking"
    assert captured["response"]["conversation_id"] == "final-conversation"
    assert captured["response"]["tool_calls"] == [{"id": "final-tool"}]


def test_single_turn_phase_two_composes_usage_and_keeps_existing_thinking_join(monkeypatch):
    phase_one = response(
        "partial", "first thinking", 1.5, 111,
        {"input_tokens": 3, "cache_creation_input_tokens": 5, "output_tokens": 7}, "length",
    )
    phase_two = response(
        "final", "second thinking", 2.5, 222,
        {"input_tokens": 30, "cached_input_tokens": 50, "output_tokens": 70, "provider_tokens": 90}, "stop",
        conversation_id="final-conversation", tool_calls=[{"id": "final-tool"}],
    )
    prover = Prover.__new__(Prover)
    prover.worker_llm = FakeLLM([phase_one, phase_two])
    monkeypatch.setattr(prover, "tui", FakeTUI(), raising=False)
    prover._stream_cb = lambda *_args, **_kwargs: None
    monkeypatch.setattr(prover, "_check_error_policy", lambda error: "stop", raising=False)

    result = prover._run_worker_single_turn("prompt", "system", "worker", None)

    assert result["raw"]["usage"] == {
        "input_tokens": 33,
        "cache_creation_input_tokens": 5,
        "output_tokens": 77,
        "cached_input_tokens": 50,
        "provider_tokens": 90,
    }
    assert result["cost"] == 4.0
    assert result["duration_ms"] == 333
    assert result["result"] == "final"
    assert result["thinking"] == "first thinkingsecond thinking"
    assert result["finish_reason"] == "stop"
    assert result["conversation_id"] == "final-conversation"
    assert result["tool_calls"] == [{"id": "final-tool"}]


def test_multi_turn_worker_sums_three_responses_once(monkeypatch):
    tool_call = {"id": "tool", "function": {"name": "lean_verify", "arguments": "{}"}}
    responses = [
        response("first", "first thinking", 1.0, 10, {"input_tokens": 1, "output_tokens": 2}, "tool_calls", tool_calls=[tool_call]),
        response("second", "second thinking", 2.0, 20, {"input_tokens": 10, "cache_read_input_tokens": 20, "output_tokens": 30}, "tool_calls", tool_calls=[tool_call]),
        response("final", "final thinking", 3.0, 30, {"input_tokens": 100, "cached_input_tokens": 200, "output_tokens": 300, "reasoning_output_tokens": 400}, "stop", conversation_id="final-conversation", tool_calls=[{"id": "final-tool"}]),
    ]
    prover = Prover.__new__(Prover)
    prover.worker_llm = FakeLLM(responses)
    monkeypatch.setattr(prover, "tui", FakeTUI(), raising=False)
    prover._stream_cb = lambda *_args, **_kwargs: None
    prover.lean_work_dir = None
    prover.lean_project_dir = None
    prover.lean_explore_service = None
    monkeypatch.setattr(prover_module, "execute_worker_tool", lambda *_args: ("ok", "ok"))

    result = prover._run_worker_multi_turn("prompt", "system", "worker", None)

    assert result["raw"]["usage"] == {
        "input_tokens": 111,
        "output_tokens": 332,
        "cache_read_input_tokens": 20,
        "cached_input_tokens": 200,
        "reasoning_output_tokens": 400,
    }
    assert result["cost"] == 6.0
    assert result["duration_ms"] == 60
    assert result["result"] == "final"
    assert result["thinking"] == "final thinking"
    assert result["finish_reason"] == "stop"
    assert result["conversation_id"] == "final-conversation"
    assert result["tool_calls"] == [{"id": "final-tool"}]


def test_verifier_phase_two_preserves_usage_when_final_response_has_none(monkeypatch):
    phase_one = response(
        "analysis", "first thinking", 4.0, 444,
        {"input_tokens": 4, "cache_read_input_tokens": 8, "output_tokens": 12}, "length",
    )
    phase_two = response(
        "VERDICT: CORRECT", "final thinking", 5.0, 555, None, "stop",
        conversation_id="final-conversation", tool_calls=[{"id": "final-tool"}],
    )
    prover = Prover.__new__(Prover)
    prover.worker_llm = FakeLLM([phase_one, phase_two])
    monkeypatch.setattr(prover, "tui", FakeTUI(), raising=False)
    prover._stream_cb = lambda *_args, **_kwargs: None
    monkeypatch.setattr(prover, "_check_error_policy", lambda error: "stop", raising=False)

    result = prover._run_verifier("task", "worker output", "verifier", None)

    assert result["raw"]["usage"] == {
        "input_tokens": 4,
        "cache_read_input_tokens": 8,
        "output_tokens": 12,
    }
    assert result["cost"] == 9.0
    assert result["duration_ms"] == 999
    assert result["result"] == "analysis\n\nVERDICT: CORRECT"
    assert result["thinking"] == "final thinking"
    assert result["finish_reason"] == "stop"
    assert result["conversation_id"] == "final-conversation"
    assert result["tool_calls"] == [{"id": "final-tool"}]
