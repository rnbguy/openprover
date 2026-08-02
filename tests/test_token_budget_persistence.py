import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from openprover import cli
from openprover.budget import Budget
from openprover.prover import Prover


class FakeTUI:
    def __init__(self):
        self.updates = []

    def update_budget(self, status):
        self.updates.append(status)


def save_run_config(work_dir, budget_mode="tokens"):
    work_dir.mkdir(exist_ok=True)
    cli._save_run_config(
        work_dir,
        planner_model="sonnet",
        worker_model="sonnet",
        budget_mode=budget_mode,
        budget_limit=100 if budget_mode == "tokens" else 3600,
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


def prepare_resume(work_dir, output_tokens=None, budget_mode="tokens"):
    save_run_config(work_dir, budget_mode)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config = config_path.read_text()
    if output_tokens is None:
        config = config.replace("budget_output_tokens = 0\n", "")
    else:
        line = f"budget_output_tokens = {output_tokens}\n"
        if "budget_output_tokens" in config:
            config = config.replace("budget_output_tokens = 0\n", line)
        else:
            config += line
    config_path.write_text(config)
    (work_dir / "THEOREM.md").write_text("# theorem\n")
    (work_dir / "WHITEBOARD.md").write_text("whiteboard\n")


def test_fresh_run_config_emits_zero_output_token_state(tmp_path):
    save_run_config(tmp_path / "run")

    assert "budget_output_tokens = 0\n" in (tmp_path / "run" / cli.RUN_CONFIG_FILE).read_text()


def test_concurrent_token_tracking_persists_the_combined_total(monkeypatch, tmp_path):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    prover = Prover.__new__(Prover)
    prover.work_dir = work_dir
    prover.budget = Budget("tokens", 100)
    tui = FakeTUI()
    monkeypatch.setattr(prover, "tui", tui, raising=False)
    prover._budget_persistence_lock = threading.Lock()
    start = threading.Barrier(3)

    def track(tokens):
        start.wait()
        prover._track_output_tokens({"raw": {"usage": {"output_tokens": tokens}}})

    first = threading.Thread(target=track, args=(5,))
    second = threading.Thread(target=track, args=(7,))
    first.start()
    second.start()
    start.wait()
    first.join()
    second.join()

    config = (work_dir / cli.RUN_CONFIG_FILE).read_text()
    assert prover.budget.total_output_tokens == 12
    assert "budget_output_tokens = 12\n" in config
    assert 'planner_model = "sonnet"\n' in config
    assert any(update.startswith("12/100 tok") for update in tui.updates)


def test_token_tracking_replaces_loader_compatible_whitespace(monkeypatch, tmp_path):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config_path.write_text(
        config_path.read_text().replace(
            "budget_output_tokens = 0", "budget_output_tokens\t=\t17",
        )
    )
    prover = Prover.__new__(Prover)
    prover.work_dir = work_dir
    prover.budget = Budget("tokens", 100, initial_output_tokens=17)
    monkeypatch.setattr(prover, "tui", FakeTUI(), raising=False)
    prover._budget_persistence_lock = threading.Lock()

    prover._track_output_tokens({"raw": {"usage": {"output_tokens": 2}}})

    assert "budget_output_tokens = 19\n" in config_path.read_text()


@pytest.mark.parametrize(
    "token_lines",
    ["", "budget_output_tokens = 0\nbudget_output_tokens = 1\n"],
)
def test_token_tracking_rejects_missing_or_duplicate_token_state(
    monkeypatch, tmp_path, token_lines,
):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    original = config_path.read_text().replace("budget_output_tokens = 0\n", token_lines)
    config_path.write_text(original)
    prover = Prover.__new__(Prover)
    prover.work_dir = work_dir
    prover.budget = Budget("tokens", 100)
    tui = FakeTUI()
    monkeypatch.setattr(prover, "tui", tui, raising=False)
    prover._budget_persistence_lock = threading.Lock()

    with pytest.raises(RuntimeError, match="budget_output_tokens"):
        prover._track_output_tokens({"raw": {"usage": {"output_tokens": 2}}})

    assert config_path.read_text() == original
    assert not config_path.with_name(f"{config_path.name}.tmp").exists()
    assert prover.budget.total_output_tokens == 0
    assert tui.updates == []


def test_token_tracking_replace_failure_keeps_persisted_and_runtime_state(
    monkeypatch, tmp_path,
):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    original = config_path.read_text()
    prover = Prover.__new__(Prover)
    prover.work_dir = work_dir
    prover.budget = Budget("tokens", 100)
    tui = FakeTUI()
    monkeypatch.setattr(prover, "tui", tui, raising=False)
    prover._budget_persistence_lock = threading.Lock()

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        prover._track_output_tokens({"raw": {"usage": {"output_tokens": 2}}})

    assert config_path.read_text() == original
    assert prover.budget.total_output_tokens == 0
    assert tui.updates == []


def test_token_resume_initializes_budget_from_run_config_ignoring_step_meta(
    monkeypatch, tmp_path,
):
    work_dir = tmp_path / "run"
    prepare_resume(work_dir, output_tokens=17)
    step_dir = work_dir / "steps" / "step_001"
    step_dir.mkdir(parents=True)
    (step_dir / "meta.toml").write_text("output_tokens = 999\n")
    captured = {}

    def capture_prover(*_args, **kwargs):
        captured["budget"] = kwargs["budget"]
        return SimpleNamespace(inspect=lambda: None, work_dir=work_dir)

    monkeypatch.setattr(cli, "Prover", capture_prover)
    monkeypatch.setattr(
        sys, "argv", ["openprover", str(work_dir), "--headless", "--read-only"],
    )

    cli._cmd_prove()

    assert captured["budget"].total_output_tokens == 17


@pytest.mark.parametrize(
    ("override_args", "expected_mode"),
    [([], "time"), (["--max-tokens", "50"], "tokens")],
)
def test_time_resume_restores_output_tokens_with_or_without_token_override(
    monkeypatch, tmp_path, override_args, expected_mode,
):
    work_dir = tmp_path / "run"
    prepare_resume(work_dir, output_tokens=17, budget_mode="time")
    captured = {}

    def capture_prover(*_args, **kwargs):
        captured["budget"] = kwargs["budget"]
        return SimpleNamespace(inspect=lambda: None, work_dir=work_dir)

    monkeypatch.setattr(cli, "Prover", capture_prover)
    monkeypatch.setattr(
        sys,
        "argv",
        ["openprover", str(work_dir), *override_args, "--headless", "--read-only"],
    )

    cli._cmd_prove()

    assert captured["budget"].mode == expected_mode
    assert captured["budget"].total_output_tokens == 17


@pytest.mark.parametrize(
    ("config_state", "error_message"),
    [
        ("missing", "run_config.toml is missing"),
        ("empty", "run_config.toml is empty"),
        ("duplicate", "exactly one budget_output_tokens"),
    ],
)
def test_resume_rejects_invalid_authoritative_config_before_prover_or_client(
    monkeypatch, tmp_path, capsys, config_state, error_message,
):
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    config_path = work_dir / cli.RUN_CONFIG_FILE
    if config_state == "empty":
        config_path.write_text("")
    elif config_state == "duplicate":
        save_run_config(work_dir)
        config_path.write_text(config_path.read_text() + "budget_output_tokens = 1\n")
    (work_dir / "THEOREM.md").write_text("# theorem\n")
    (work_dir / "WHITEBOARD.md").write_text("whiteboard\n")
    created = {"prover": False, "client": False}

    def unexpected_prover(*_args, **_kwargs):
        created["prover"] = True
        raise AssertionError("Prover construction must not occur")

    def unexpected_client(*_args, **_kwargs):
        created["client"] = True
        raise AssertionError("Client construction must not occur")

    monkeypatch.setattr(cli, "Prover", unexpected_prover)
    monkeypatch.setattr(cli, "LLMClient", unexpected_client)
    monkeypatch.setattr(sys, "argv", ["openprover", str(work_dir), "--headless"])

    with pytest.raises(SystemExit) as error:
        cli._cmd_prove()

    assert error.value.code == 2
    assert created == {"prover": False, "client": False}
    assert error_message in capsys.readouterr().err


@pytest.mark.parametrize("budget_mode", ["tokens", "time"])
def test_resume_without_output_token_state_fails_before_prover_or_client(
    monkeypatch, tmp_path, capsys, budget_mode,
):
    work_dir = tmp_path / "run"
    prepare_resume(work_dir, budget_mode=budget_mode)
    created = {"prover": False, "client": False}

    def unexpected_prover(*_args, **_kwargs):
        created["prover"] = True
        raise AssertionError("Prover construction must not occur")

    def unexpected_client(*_args, **_kwargs):
        created["client"] = True
        raise AssertionError("Client construction must not occur")

    monkeypatch.setattr(cli, "Prover", unexpected_prover)
    monkeypatch.setattr(cli, "LLMClient", unexpected_client)
    monkeypatch.setattr(sys, "argv", ["openprover", str(work_dir), "--headless"])

    with pytest.raises(SystemExit) as error:
        cli._cmd_prove()

    assert error.value.code == 2
    assert created == {"prover": False, "client": False}
    assert "budget_output_tokens" in capsys.readouterr().err
