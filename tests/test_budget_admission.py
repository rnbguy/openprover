import logging
import threading

import pytest

from openprover import cli
from openprover.budget import Budget, BudgetExhausted
from openprover.prover import Prover


class FakeTUI:
    def __init__(self):
        self.updates = []

    def update_budget(self, status):
        self.updates.append(status)


class FakeMonotonic:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def save_run_config(work_dir, budget_mode="tokens"):
    work_dir.mkdir()
    cli._save_run_config(
        work_dir, planner_model="sonnet", worker_model="sonnet",
        budget_mode=budget_mode, budget_limit=10, conclude_after=0.99,
        max_workers=1, isolation=True, autonomous=True, mode="prove",
        lean_project_dir=None, lean_items=False, lean_worker_tools=False,
        provider_url="http://127.0.0.1:8000", answer_reserve=4096,
        history_budget=0, verifier=True,
    )


def make_prover(tmp_path, budget):
    work_dir = tmp_path / "run"
    save_run_config(work_dir, budget.mode)
    prover = Prover.__new__(Prover)
    prover.work_dir = work_dir
    prover.budget = budget
    prover.tui = FakeTUI()
    prover._budget_persistence_lock = threading.Lock()
    prover._backoff_delay = 4
    prover._rate_limit_backoff = 4
    prover._transient_backoff = 4
    return prover


def response(tokens):
    return {"raw": {"usage": {"output_tokens": tokens}}}


def test_admission_tracks_each_success_once_and_denies_after_recorded_exhaustion(tmp_path):
    prover = make_prover(tmp_path, Budget("tokens", 10))
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    results = []

    def call(tokens, entered):
        entered.set()
        assert release.wait(timeout=2)
        return response(tokens)

    def run(tokens, entered):
        results.append(prover._call_with_budget(call, tokens, entered))

    first = threading.Thread(target=run, args=(6, first_entered))
    second = threading.Thread(target=run, args=(6, second_entered))
    first.start()
    second.start()
    try:
        assert first_entered.wait(timeout=2)
        assert second_entered.wait(timeout=2)
    finally:
        release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(item["raw"]["usage"]["output_tokens"] for item in results) == [6, 6]
    assert prover.budget.total_output_tokens == 12
    denied_calls = []

    with pytest.raises(BudgetExhausted):
        prover._call_with_budget(lambda: denied_calls.append("called"))

    assert denied_calls == []
    assert "budget_output_tokens = 12\n" in (prover.work_dir / cli.RUN_CONFIG_FILE).read_text()


def test_failed_call_checkpoints_elapsed_and_preserves_original_exception(monkeypatch, tmp_path):
    clock = FakeMonotonic()
    monkeypatch.setattr("openprover.budget.time.monotonic", clock)
    prover = make_prover(tmp_path, Budget("time", 10))
    original = RuntimeError("model failed")

    def fail():
        clock.now = 7.0
        raise original

    with pytest.raises(RuntimeError) as caught:
        prover._call_with_budget(fail)

    assert caught.value is original
    assert "budget_elapsed_seconds = 7.0\n" in (prover.work_dir / cli.RUN_CONFIG_FILE).read_text()


def test_failed_checkpoint_does_not_mask_model_exception(caplog, tmp_path):
    prover = make_prover(tmp_path, Budget("tokens", 10))
    config_path = prover.work_dir / cli.RUN_CONFIG_FILE
    original = RuntimeError("model failed")

    def fail():
        config_path.write_text(config_path.read_text().replace(
            "budget_elapsed_seconds = 0.0", "budget_elapsed_seconds = malformed",
        ))
        raise original

    with caplog.at_level(logging.ERROR, logger="openprover"), pytest.raises(RuntimeError) as caught:
        prover._call_with_budget(fail)

    assert caught.value is original
    assert "Failed to checkpoint budget state" in caplog.text
