import sys
from types import SimpleNamespace

import pytest

from openprover import cli
from openprover.budget import Budget, format_elapsed_seconds


def save_run_config(work_dir):
    work_dir.mkdir()
    cli._save_run_config(
        work_dir,
        planner_model="sonnet",
        worker_model="sonnet",
        budget_mode="time",
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
    (work_dir / "THEOREM.md").write_text("# theorem\n")
    (work_dir / "WHITEBOARD.md").write_text("whiteboard\n")


@pytest.mark.parametrize(
    ("elapsed_lines", "error_message"),
    [
        ("", "exactly one budget_elapsed_seconds"),
        (
            "budget_elapsed_seconds = 1\nbudget_elapsed_seconds = not-a-number\n",
            "exactly one budget_elapsed_seconds",
        ),
        ("budget_elapsed_seconds = \n", "saved run_config.toml is invalid"),
        ("budget_elapsed_seconds = -1\n", "nonnegative number"),
        ("budget_elapsed_seconds = true\n", "nonnegative number"),
        ("budget_elapsed_seconds = 1.0e999\n", "nonnegative number"),
    ],
)
def test_resume_rejects_invalid_elapsed_state_before_prover_or_client(
    monkeypatch, tmp_path, capsys, elapsed_lines, error_message,
):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config_path.write_text(
        config_path.read_text().replace("budget_elapsed_seconds = 0.0\n", elapsed_lines)
    )
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
    ("elapsed_seconds", "expected"),
    [("12.5", 12.5), ("+1", 1.0), ("1.0e-3", 0.001)],
)
def test_resume_accepts_loader_compatible_elapsed_state(
    monkeypatch, tmp_path, elapsed_seconds, expected,
):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config_path.write_text(
        config_path.read_text().replace(
            "budget_elapsed_seconds = 0.0", f"budget_elapsed_seconds = {elapsed_seconds}",
        )
    )
    captured = {}

    def capture_prover(*_args, **kwargs):
        captured["budget"] = kwargs["budget"]
        return SimpleNamespace(inspect=lambda: None, work_dir=work_dir)

    monkeypatch.setattr(cli, "Prover", capture_prover)
    monkeypatch.setattr(
        sys, "argv", ["openprover", str(work_dir), "--headless", "--read-only"],
    )

    cli._cmd_prove()

    assert captured["budget"].initial_elapsed_seconds == expected


@pytest.mark.parametrize("elapsed_seconds", [float("inf"), -float("inf"), float("nan")])
def test_budget_rejects_nonfinite_initial_elapsed_seconds(elapsed_seconds):
    with pytest.raises(ValueError, match="finite"):
        Budget("time", 100, initial_elapsed_seconds=elapsed_seconds)


@pytest.mark.parametrize("elapsed_seconds", [float("inf"), -float("inf"), float("nan")])
def test_elapsed_serializer_rejects_nonfinite_values(elapsed_seconds):
    with pytest.raises(ValueError, match="finite"):
        format_elapsed_seconds(elapsed_seconds)
