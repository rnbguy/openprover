import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from openprover import cli
from openprover.budget import Budget
from openprover.prover import Prover


class FakeMonotonic:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeTUI:
    def __init__(self):
        self.updates = []
        self.step_entries = []

    def update_budget(self, status):
        self.updates.append(status)


@pytest.fixture
def monotonic(monkeypatch):
    clock = FakeMonotonic()
    monkeypatch.setattr("openprover.budget.time.monotonic", clock)
    return clock


def save_run_config(work_dir, budget_mode="time"):
    work_dir.mkdir(exist_ok=True)
    cli._save_run_config(
        work_dir,
        planner_model="sonnet",
        worker_model="sonnet",
        budget_mode=budget_mode,
        budget_limit=100,
        conclude_after=0.99,
        max_workers=1,
        isolation=True,
        autonomous=True,
        mode="prove",
        lean_project_dir=None,
        lean_items=False,
        lean_worker_tools=False,
        provider_url="http://127.0.0.1:8000",
        answer_reserve=4096,
        history_budget=0,
        verifier=True,
    )


def prepare_resume(work_dir, elapsed_seconds, budget_mode="time"):
    save_run_config(work_dir, budget_mode)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config_path.write_text(
        config_path.read_text().replace(
            "budget_elapsed_seconds = 0.0",
            f"budget_elapsed_seconds = {elapsed_seconds}",
        )
    )
    (work_dir / "THEOREM.md").write_text("# theorem\n")
    (work_dir / "WHITEBOARD.md").write_text("whiteboard\n")


def make_prover(work_dir, budget):
    prover = Prover.__new__(Prover)
    prover.work_dir = work_dir
    prover.budget = budget
    tui = FakeTUI()
    prover.__dict__["tui"] = tui
    prover._budget_persistence_lock = threading.Lock()
    return prover, tui


def test_fresh_run_config_emits_zero_elapsed_state(tmp_path):
    save_run_config(tmp_path / "run")

    config = (tmp_path / "run" / cli.RUN_CONFIG_FILE).read_text()

    assert "budget_elapsed_seconds = 0.0\n" in config


def test_elapsed_time_grows_from_persisted_offset(monotonic):
    budget = Budget("time", 100, initial_elapsed_seconds=20.0)

    monotonic.now += 15.0

    assert budget.elapsed_seconds() == 35.0
    assert budget.fraction_spent() == 0.35
    assert budget.status_str() == "35s/1m40s"
    assert budget.summary_str() == "35s/1m40s elapsed (35%)"


def test_resume_does_not_charge_process_downtime(monotonic):
    first_process = Budget("time", 100)
    monotonic.now += 12.0
    persisted_elapsed = first_process.elapsed_seconds()

    monotonic.now += 10000.0
    resumed = Budget("time", 100, initial_elapsed_seconds=persisted_elapsed)

    assert resumed.elapsed_seconds() == 12.0
    assert resumed.fraction_spent() == 0.12


def test_resumed_time_budget_preserves_remaining_fraction_and_status(monotonic):
    budget = Budget("time", 100, initial_elapsed_seconds=95.0)

    monotonic.now += 4.0

    assert budget.fraction_spent() == 0.99
    assert 1.0 - budget.fraction_spent() == pytest.approx(0.01)
    assert budget.should_conclude()
    assert not budget.is_exhausted()
    assert budget.status_str() == "1m39s/1m40s"


@pytest.mark.parametrize("budget_mode", ["time", "tokens"])
def test_resume_restores_elapsed_state_for_every_budget_mode(
    monkeypatch, monotonic, tmp_path, budget_mode,
):
    work_dir = tmp_path / "run"
    prepare_resume(work_dir, 17.5, budget_mode)
    captured = {}

    def capture_prover(*_args, **kwargs):
        captured["budget"] = kwargs["budget"]
        return SimpleNamespace(inspect=lambda: None, work_dir=work_dir)

    monkeypatch.setattr(cli, "Prover", capture_prover)
    monkeypatch.setattr(
        sys, "argv", ["openprover", str(work_dir), "--headless", "--read-only"],
    )

    cli._cmd_prove()

    assert captured["budget"].mode == budget_mode
    assert captured["budget"].elapsed_seconds() == 17.5


def test_zero_token_response_checkpoints_elapsed(monkeypatch, monotonic, tmp_path):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    prover, tui = make_prover(work_dir, Budget("time", 100))
    monotonic.now += 7.0

    prover._track_output_tokens({"raw": {"usage": {"output_tokens": 0}}})

    config = (work_dir / cli.RUN_CONFIG_FILE).read_text()
    assert "budget_output_tokens = 0\n" in config
    assert "budget_elapsed_seconds = 7.0\n" in config
    assert tui.updates == ["7s/1m40s"]


def test_response_checkpoints_tokens_and_elapsed(monotonic, tmp_path):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    prover, _ = make_prover(work_dir, Budget("time", 100))
    monotonic.now += 8.0

    prover._track_output_tokens({"raw": {"usage": {"output_tokens": 5}}})

    config = (work_dir / cli.RUN_CONFIG_FILE).read_text()
    assert "budget_output_tokens = 5\n" in config
    assert "budget_elapsed_seconds = 8.0\n" in config
    assert prover.budget.total_output_tokens == 5


@pytest.mark.parametrize("elapsed_seconds", ["+1", "1.0e-3"])
def test_checkpoint_accepts_loader_compatible_elapsed_grammar(
    monotonic, tmp_path, elapsed_seconds,
):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config_path.write_text(
        config_path.read_text().replace(
            "budget_elapsed_seconds = 0.0", f"budget_elapsed_seconds = {elapsed_seconds}",
        )
    )
    prover, _ = make_prover(work_dir, Budget("time", 100))

    prover._checkpoint_budget_state()

    assert "budget_elapsed_seconds = 0.0\n" in config_path.read_text()


def test_tiny_elapsed_value_survives_two_checkpoints_and_resume_parsing(monotonic, tmp_path):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    prover, _ = make_prover(work_dir, Budget("time", 100))
    monotonic.now += 0.00005

    prover._checkpoint_budget_state()
    monotonic.now += 0.00005
    prover._checkpoint_budget_state()

    config = cli._load_run_config(work_dir)
    assert config is not None
    elapsed_seconds = config["budget_elapsed_seconds"]
    assert elapsed_seconds == pytest.approx(0.0001)


def test_atomic_replace_failure_keeps_token_runtime_state(monkeypatch, tmp_path):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    prover, tui = make_prover(work_dir, Budget("tokens", 100, initial_output_tokens=4))
    config_path = work_dir / cli.RUN_CONFIG_FILE
    original = config_path.read_text()

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        prover._track_output_tokens({"raw": {"usage": {"output_tokens": 3}}})

    assert config_path.read_text() == original
    assert prover.budget.total_output_tokens == 4
    assert tui.updates == []


def test_run_exit_checkpoints_elapsed(monkeypatch, monotonic, tmp_path):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    prover, _ = make_prover(work_dir, Budget("time", 100))
    prover.resumed = False
    prover.autonomous = True
    prover.shutting_down = False
    prover._spending_limit_hit = False
    prover.step_num = 0
    prover._consecutive_errors = 0
    prover._setup_tui = lambda **_kwargs: None
    prover._setup_tui_logging = lambda: None
    prover._do_step = lambda: "stop"
    prover._save_step_history = lambda: None
    monotonic.now += 11.0

    prover.run()

    assert "budget_elapsed_seconds = 11.0\n" in (
        work_dir / cli.RUN_CONFIG_FILE
    ).read_text()


def test_run_error_preserves_original_exception_when_checkpoint_fails(
    caplog, monotonic, tmp_path,
):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config_path.write_text(
        config_path.read_text().replace(
            "budget_elapsed_seconds = 0.0", "budget_elapsed_seconds = malformed",
        )
    )
    prover, _ = make_prover(work_dir, Budget("time", 100))
    prover.resumed = False
    prover.autonomous = True
    prover.shutting_down = False
    prover._spending_limit_hit = False
    prover.step_num = 0
    prover._setup_tui = lambda **_kwargs: None
    prover._setup_tui_logging = lambda: None
    original = RuntimeError("boom")
    prover._do_step = lambda: (_ for _ in ()).throw(original)
    monotonic.now += 13.0

    with caplog.at_level(logging.ERROR, logger="openprover"), pytest.raises(RuntimeError) as caught:
        prover.run()

    assert caught.value is original
    assert "Failed to checkpoint budget state" in caplog.text


def test_run_raises_checkpoint_failure_without_an_active_exception(tmp_path):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config_path.write_text(
        config_path.read_text().replace(
            "budget_elapsed_seconds = 0.0", "budget_elapsed_seconds = malformed",
        )
    )
    prover, _ = make_prover(work_dir, Budget("time", 100))
    prover.resumed = False
    prover.autonomous = True
    prover.shutting_down = False
    prover._spending_limit_hit = False
    prover.step_num = 0
    prover._consecutive_errors = 0
    prover._setup_tui = lambda **_kwargs: None
    prover._setup_tui_logging = lambda: None
    prover._do_step = lambda: "stop"
    prover._save_step_history = lambda: None

    with pytest.raises(RuntimeError, match="budget_elapsed_seconds"):
        prover.run()
