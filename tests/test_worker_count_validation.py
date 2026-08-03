import sys
from pathlib import Path

import pytest

from openprover import cli


def save_run_config(work_dir: Path) -> None:
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


@pytest.mark.parametrize("value", ["0", "-1"])
def test_openprover_rejects_non_positive_max_workers(monkeypatch, value, capsys):
    monkeypatch.setattr(sys, "argv", ["openprover", "--max-workers", value])

    with pytest.raises(SystemExit) as error:
        cli._cmd_prove()

    assert error.value.code == 2
    assert "invalid positive integer value" in capsys.readouterr().err


def test_positive_worker_count_is_parsed():
    assert cli.positive_int("3") == 3


def test_resume_rejects_non_positive_saved_max_workers(
    monkeypatch, tmp_path, capsys,
):
    work_dir = tmp_path / "run"
    save_run_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config_path.write_text(
        config_path.read_text().replace("max_workers = 1", "max_workers = 0"),
    )
    monkeypatch.setattr(sys, "argv", ["openprover", str(work_dir), "--headless"])

    with pytest.raises(SystemExit) as error:
        cli._cmd_prove()

    assert error.value.code == 2
    assert "saved max_workers must be a positive integer" in capsys.readouterr().err
