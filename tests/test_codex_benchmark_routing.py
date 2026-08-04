import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["run_minif2f", "run_proofnet"])
def test_benchmark_rejects_baseline_codex_before_fresh_cli_work(
    monkeypatch,
    tmp_path,
    capsys,
    script,
):
    module = load_script(script)
    monkeypatch.setattr(
        module,
        "_check_tool",
        lambda *_args: pytest.fail("ran preflight before rejecting baseline Codex"),
    )
    monkeypatch.setattr(
        module,
        "_run_all",
        lambda *_args, **_kwargs: pytest.fail(
            "started benchmark work before rejecting baseline Codex"
        ),
    )
    data_loaders = (
        ("_fetch_lean_file", "_parse_theorems")
        if script == "run_minif2f"
        else ("_load_proofnet_problems",)
    )
    for data_loader in data_loaders:
        monkeypatch.setattr(
            module,
            data_loader,
            lambda *_args: pytest.fail(
                "loaded benchmark data before rejecting baseline Codex"
            ),
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script,
            "--method",
            "baseline",
            "--model",
            "codex",
            "--repo-path",
            str(tmp_path / "missing-repo"),
            "--informal",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        module.main()

    assert raised.value.code == 2
    assert "--method baseline does not support --model codex" in capsys.readouterr().err


@pytest.mark.parametrize("script", ["run_minif2f", "run_proofnet"])
def test_benchmark_resume_rejects_baseline_codex_before_work(
    monkeypatch,
    tmp_path,
    capsys,
    script,
):
    module = load_script(script)
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    (resume_dir / "config.json").write_text(
        json.dumps(
            {
                "method": "baseline",
                "model": "codex",
                "repo_path": str(tmp_path / "missing-repo"),
                "informal": True,
            }
        )
    )
    monkeypatch.setattr(
        module,
        "_check_tool",
        lambda *_args: pytest.fail("ran preflight before validating resume"),
    )
    monkeypatch.setattr(
        module,
        "_run_all",
        lambda *_args, **_kwargs: pytest.fail(
            "started benchmark work before validating resume"
        ),
    )
    data_loaders = (
        ("_fetch_lean_file", "_parse_theorems")
        if script == "run_minif2f"
        else ("_load_proofnet_problems",)
    )
    for data_loader in data_loaders:
        monkeypatch.setattr(
            module,
            data_loader,
            lambda *_args: pytest.fail("loaded benchmark data before validating resume"),
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "--resume", str(resume_dir)])

    with pytest.raises(SystemExit) as raised:
        module.main()

    assert raised.value.code == 2
    assert "--method baseline does not support --model codex" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("script", "saved_field", "saved_model"),
    [
        (script, saved_field, saved_model)
        for script in ("run_minif2f", "run_proofnet")
        for saved_field in ("model", "planner_model", "worker_model")
        for saved_model in ("gpt", "gpt-5.6-sol")
    ],
)
def test_benchmark_resume_rejects_stale_saved_model_before_work(
    monkeypatch,
    tmp_path,
    capsys,
    script,
    saved_field,
    saved_model,
):
    module = load_script(script)
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    (resume_dir / "config.json").write_text(
        json.dumps(
            {
                "repo_path": str(tmp_path / "missing-repo"),
                "informal": True,
                saved_field: saved_model,
            }
        )
    )
    monkeypatch.setattr(
        module,
        "_check_tool",
        lambda *_args: pytest.fail("ran preflight before validating resume"),
    )
    monkeypatch.setattr(
        module,
        "_run_all",
        lambda *_args, **_kwargs: pytest.fail(
            "started subprocess work before validating resume"
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "--resume", str(resume_dir)])

    with pytest.raises(SystemExit) as raised:
        module.main()

    error = capsys.readouterr().err
    assert raised.value.code == 2
    assert saved_model in error
    assert f"--{saved_field.replace('_', '-')}" in error
