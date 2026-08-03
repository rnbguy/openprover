import sys
import threading

import pytest

from openprover import cli
from openprover.budget import Budget
from openprover.prover import Prover


class FakeTUI:
    def __init__(self):
        self.updates = []

    def update_budget(self, status):
        self.updates.append(status)


def write_resume_config(work_dir, token_lines):
    work_dir.mkdir()
    (work_dir / cli.RUN_CONFIG_FILE).write_text(
        'budget_mode = "tokens"\n'
        "budget_limit = 100\n"
        + token_lines
    )
    (work_dir / "THEOREM.md").write_text("# theorem\n")
    (work_dir / "WHITEBOARD.md").write_text("whiteboard\n")


@pytest.mark.parametrize(
    ("token_lines", "error_message"),
    [
        ("budget_output_tokens = -1\n", "nonnegative integer"),
        ("budget_output_tokens = 1.5\n", "nonnegative integer"),
        (
            "budget_output_tokens = 17\nbudget_output_tokens = -1\n",
            "exactly one budget_output_tokens",
        ),
    ],
)
def test_resume_rejects_invalid_token_values_before_prover_or_client(
    monkeypatch, tmp_path, capsys, token_lines, error_message,
):
    work_dir = tmp_path / "run"
    write_resume_config(work_dir, token_lines)
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


@pytest.mark.parametrize(
    ("token_lines", "error_message"),
    [
        ("budget_output_tokens = -1\n", "nonnegative integer"),
        ("budget_output_tokens = 1.5\n", "nonnegative integer"),
        (
            "budget_output_tokens = 17\nbudget_output_tokens = -1\n",
            "exactly one budget_output_tokens",
        ),
    ],
)
def test_checkpoint_rejects_invalid_token_values_without_runtime_mutation(
    monkeypatch, tmp_path, token_lines, error_message,
):
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    config_path = work_dir / cli.RUN_CONFIG_FILE
    original = 'planner_model = "sonnet"\n' + token_lines
    config_path.write_text(original)
    prover = Prover.__new__(Prover)
    prover.work_dir = work_dir
    prover.budget = Budget("tokens", 100)
    tui = FakeTUI()
    monkeypatch.setattr(prover, "tui", tui, raising=False)
    prover._budget_persistence_lock = threading.Lock()

    with pytest.raises(RuntimeError, match=error_message):
        prover._track_output_tokens({"raw": {"usage": {"output_tokens": 2}}})

    assert config_path.read_text() == original
    assert prover.budget.total_output_tokens == 0
    assert tui.updates == []


@pytest.mark.parametrize("value", ["0", "-1"])
def test_fresh_run_rejects_non_positive_max_tokens_before_prover_construction(
    monkeypatch, value, capsys,
):
    created = False

    def unexpected_prover(*_args, **_kwargs):
        nonlocal created
        created = True
        raise AssertionError("Prover construction must not occur")

    monkeypatch.setattr(cli, "Prover", unexpected_prover)
    monkeypatch.setattr(sys, "argv", ["openprover", "--max-tokens", value])

    with pytest.raises(SystemExit) as error:
        cli._cmd_prove()

    assert error.value.code == 2
    assert created is False
    assert "invalid positive integer value" in capsys.readouterr().err
