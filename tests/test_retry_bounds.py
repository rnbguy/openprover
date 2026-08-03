import itertools

import pytest

import openprover.prover as prover_module
from openprover.budget import Budget
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


def response(result="final", finish_reason="stop", **extra):
    return {
        "result": result,
        "thinking": "",
        "cost": 1.0,
        "duration_ms": 1,
        "raw": {"usage": {"output_tokens": 1}},
        "finish_reason": finish_reason,
        **extra,
    }


def worker(responses):
    prover = Prover.__new__(Prover)
    prover.worker_llm = FakeLLM(responses)
    prover.__dict__["tui"] = FakeTUI()
    prover.__dict__["budget"] = Budget("tokens", 100)
    prover._backoff_delay = 4
    prover._rate_limit_backoff = 4
    prover._transient_backoff = 4
    prover._stream_cb = lambda *_args, **_kwargs: None
    prover.lean_work_dir = None
    prover.lean_project_dir = None
    prover.lean_explore_service = None
    return prover


@pytest.mark.parametrize(
    ("error", "on_budget_out", "on_rate_limited", "delays"),
    [
        (RuntimeError("502 gateway"), None, None, [4, 8, 16, 32, 60, 60, 60, 60, 60]),
        (RuntimeError("429 rate limit"), None, "backoff", [4, 16, 64, 64, 64, 64, 64, 64, 64]),
        (RuntimeError("spending limit"), "backoff", None, [4, 16, 64, 64, 64, 64, 64, 64, 64]),
    ],
)
def test_token_backoffs_stop_before_tenth_sleep(
    monkeypatch, error, on_budget_out, on_rate_limited, delays,
):
    prover = worker(itertools.repeat(error))
    prover.on_budget_out = on_budget_out
    prover.on_rate_limited = on_rate_limited
    sleeps = []
    monkeypatch.setattr(prover_module.time, "sleep", sleeps.append)

    def policy(error):
        action = Prover._check_error_policy(prover, error)
        return "stop" if len(sleeps) == 10 else action

    prover._check_error_policy = policy

    result = prover._run_worker_single_turn("prompt", "system", "worker", None)

    assert prover.worker_llm.calls == 10
    assert sleeps == delays
    assert result["error"] == str(error)


@pytest.mark.parametrize(
    ("error", "on_budget_out", "on_rate_limited"),
    [
        (RuntimeError("429 rate limit"), None, "exit"),
        (RuntimeError("spending limit"), "exit", None),
    ],
)
def test_exit_policies_do_not_retry(monkeypatch, error, on_budget_out, on_rate_limited):
    prover = worker([error])
    prover.on_budget_out = on_budget_out
    prover.on_rate_limited = on_rate_limited
    sleeps = []
    monkeypatch.setattr(prover_module.time, "sleep", sleeps.append)

    result = prover._run_worker_single_turn("prompt", "system", "worker", None)

    assert prover.worker_llm.calls == 1
    assert sleeps == []
    assert result["error"] == str(error)


@pytest.mark.parametrize(
    ("runner", "args"),
    [
        (Prover._run_worker_single_turn, ("prompt", "system", "worker", None)),
        (Prover._run_verifier, ("task", "worker output", "verifier", None)),
    ],
)
def test_phase_two_retries_share_the_model_failure_bound(monkeypatch, runner, args):
    prover = worker(itertools.chain(
        [response("partial", "length")], itertools.repeat(RuntimeError("retry")),
    ))
    policy_calls = []
    prover._check_error_policy = lambda error: policy_calls.append(error) or (
        "retry" if len(policy_calls) < 10 else "stop"
    )

    result = runner(prover, *args)

    assert prover.worker_llm.calls == 11
    assert len(policy_calls) == 9
    assert result["error"] == "retry"


def test_success_resets_single_turn_failure_count():
    prover = worker([
        *[RuntimeError("retry") for _ in range(9)],
        response("partial", "length"),
        RuntimeError("retry"),
        response(),
    ])
    policy_calls = []
    prover._check_error_policy = lambda error: policy_calls.append(error) or "retry"

    result = prover._run_worker_single_turn("prompt", "system", "worker", None)

    assert prover.worker_llm.calls == 12
    assert len(policy_calls) == 10
    assert result["result"] == "final"
