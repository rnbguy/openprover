import itertools
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
        self.calls = 0

    def call(self, **_kwargs):
        self.calls += 1
        return self._next()

    def chat(self, **_kwargs):
        self.calls += 1
        return self._next()

    def _next(self):
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def response(result="partial", finish_reason="length", cost=1.0):
    return {
        "result": result,
        "thinking": "",
        "cost": cost,
        "duration_ms": 1,
        "raw": {"usage": {"output_tokens": 1}},
        "finish_reason": finish_reason,
    }


def retry_policy(prover):
    calls = []
    prover._check_error_policy = lambda error: calls.append(error) or (
        "retry" if len(calls) < 10 else "stop"
    )
    return calls


def planner(monkeypatch, tmp_path, responses):
    prover = Prover.__new__(Prover)
    prover.planner_llm = FakeLLM(responses)
    prover.__dict__["tui"] = FakeTUI()
    prover.__dict__["budget"] = SimpleNamespace(
        mode="tokens", status_str=lambda: "ok", fraction_spent=lambda: 0.0,
        summary_str=lambda: "ok",
    )
    prover.work_dir = tmp_path
    prover.__dict__["repo"] = SimpleNamespace(list_summaries=lambda: "")
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
    prover._last_error_msg = ""
    prover._stream_cb = lambda *_args, **_kwargs: None
    prover._save_step_meta = lambda *_args, **_kwargs: None
    monkeypatch.setattr(prover_module.prompts, "format_planner_prompt", lambda **_kwargs: "prompt")
    monkeypatch.setattr(prover_module.prompts, "planner_system_prompt", lambda **_kwargs: "system")
    return prover


def test_initial_planner_retry_stops_before_tenth_policy_call(monkeypatch, tmp_path):
    prover = planner(monkeypatch, tmp_path, itertools.repeat(RuntimeError("retry")))
    policy_calls = retry_policy(prover)

    assert prover._do_step() == "stop"
    assert prover.planner_llm.calls == 10
    assert len(policy_calls) == 9


def test_planner_phase_two_retry_stops_before_tenth_policy_call(monkeypatch, tmp_path):
    prover = planner(
        monkeypatch, tmp_path,
        itertools.chain([response()], itertools.repeat(RuntimeError("retry"))),
    )
    policy_calls = retry_policy(prover)

    assert prover._do_step() == "stop"
    assert prover.planner_llm.calls == 11
    assert len(policy_calls) == 9


def test_literature_retry_stops_before_tenth_policy_call(tmp_path):
    prover = Prover.__new__(Prover)
    prover.worker_llm = FakeLLM(itertools.repeat(RuntimeError("retry")))
    prover.__dict__["tui"] = FakeTUI()
    prover.__dict__["budget"] = SimpleNamespace(mode="tokens")
    prover.work_dir = tmp_path
    prover.isolation = False
    prover.step_num = 1
    prover._step_idx = 0
    prover.autonomous = False
    prover._stream_cb = lambda *_args, **_kwargs: None
    prover._push_output = lambda text: None
    prover._save_step_meta = lambda *_args, **_kwargs: None
    policy_calls = retry_policy(prover)
    step_dir = tmp_path / "step"
    step_dir.mkdir()

    assert prover._handle_literature_search(
        {"search_query": "query", "search_context": ""}, step_dir,
    ) == "continue"
    assert prover.worker_llm.calls == 10
    assert len(policy_calls) == 9


def test_multi_turn_retry_preserves_prior_response_and_stops_at_bound():
    prover = Prover.__new__(Prover)
    prover.worker_llm = FakeLLM(itertools.chain(
        [response(cost=7.0)], itertools.repeat(RuntimeError("retry")),
    ))
    prover.__dict__["tui"] = FakeTUI()
    prover.__dict__["budget"] = SimpleNamespace(mode="tokens")
    prover._stream_cb = lambda *_args, **_kwargs: None
    prover.lean_work_dir = None
    prover.lean_project_dir = None
    policy_calls = retry_policy(prover)

    result = prover._run_worker_multi_turn("prompt", "system", "worker", None)

    assert prover.worker_llm.calls == 11
    assert len(policy_calls) == 9
    assert result["cost"] == 7.0
    assert result["raw"]["usage"] == {"output_tokens": 1}
    assert result["error"] == "retry"
