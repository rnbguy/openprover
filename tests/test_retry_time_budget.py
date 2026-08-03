import threading

import pytest

import openprover.prover as prover_module
from openprover.budget import Budget, BudgetExhausted
from openprover.prover import Prover


class FakeTUI:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class FakeMonotonic:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def prover_with_time_budget(monkeypatch, method, error):
    clock = FakeMonotonic()
    monkeypatch.setattr("openprover.budget.time.monotonic", clock)
    prover = Prover.__new__(Prover)
    prover.budget = Budget("time", 10)
    prover.__dict__["tui"] = FakeTUI()
    prover._budget_persistence_lock = threading.Lock()
    prover._checkpoint_budget_state_locked = lambda output_tokens=0: None
    prover._backoff_delay = 4
    prover._rate_limit_backoff = 4
    prover._transient_backoff = 4
    prover.on_budget_out = "backoff" if method == "_check_spending_limit" else None
    prover.on_rate_limited = "backoff" if method == "_check_rate_limited" else None
    return prover, clock, error


@pytest.mark.parametrize(
    ("method", "error"),
    [
        ("_check_spending_limit", RuntimeError("spending limit")),
        ("_check_rate_limited", RuntimeError("429 rate limit")),
        ("_check_transient", RuntimeError("502 gateway")),
    ],
)
def test_time_backoff_clamps_sleep_then_budget_admission_denies(monkeypatch, method, error):
    prover, clock, error = prover_with_time_budget(monkeypatch, method, error)
    clock.now = 8.0
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(prover_module.time, "sleep", sleep)

    assert getattr(prover, method)(error) == "retry"
    assert sleeps == [2.0]
    with pytest.raises(BudgetExhausted):
        prover._call_with_budget(lambda: pytest.fail("model call was admitted"))


@pytest.mark.parametrize(
    ("method", "error"),
    [
        ("_check_spending_limit", RuntimeError("spending limit")),
        ("_check_rate_limited", RuntimeError("429 rate limit")),
        ("_check_transient", RuntimeError("502 gateway")),
    ],
)
def test_time_backoff_at_deadline_stops_without_sleep(monkeypatch, method, error):
    prover, clock, error = prover_with_time_budget(monkeypatch, method, error)
    clock.now = 10.0
    sleeps = []
    monkeypatch.setattr(prover_module.time, "sleep", sleeps.append)

    assert getattr(prover, method)(error) == "stop"
    assert sleeps == []
