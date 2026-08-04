import sys
from types import SimpleNamespace

import pytest

from openprover import cli


def save_resume_config(work_dir):
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


@pytest.mark.parametrize("field", ["planner_model", "worker_model"])
@pytest.mark.parametrize("value", ["gpt", "gpt-5.6-sol"])
def test_resume_rejects_stale_saved_model_before_prover_or_backend(
    monkeypatch, tmp_path, capsys, field, value,
):
    work_dir = tmp_path / "run"
    save_resume_config(work_dir)
    config_path = work_dir / cli.RUN_CONFIG_FILE
    config_path.write_text(
        config_path.read_text().replace(
            f'{field} = "sonnet"', f'{field} = "{value}"',
        ),
    )
    created = {"prover": False, "backend": False}

    def unexpected_prover(*_args, **_kwargs):
        created["prover"] = True
        raise AssertionError("Prover construction must not occur")

    def unexpected_backend(*_args, **_kwargs):
        created["backend"] = True
        raise AssertionError("Backend construction must not occur")

    monkeypatch.delenv("CODEX_MODEL", raising=False)
    monkeypatch.setattr(cli, "Prover", unexpected_prover)
    monkeypatch.setattr(cli, "LLMClient", unexpected_backend)
    monkeypatch.setattr(cli, "CodexClient", unexpected_backend)
    monkeypatch.setattr(sys, "argv", ["openprover", str(work_dir), "--headless"])

    with pytest.raises(SystemExit) as error:
        cli._cmd_prove()

    stderr = capsys.readouterr().err
    assert error.value.code == 2
    assert field in stderr
    assert value in stderr
    assert created == {"prover": False, "backend": False}


@pytest.mark.parametrize(
    (
        "model",
        "planner_model",
        "worker_model",
        "codex_model",
        "expected_codex",
        "expected_other",
    ),
    [
        ("codex", None, None, None, ["gpt-5.6-sol", "gpt-5.6-sol"], []),
        ("sonnet", "codex", "sonnet", "gpt-5.6-luna", ["gpt-5.6-luna"], ["sonnet"]),
        ("sonnet", "sonnet", "codex", "gpt-5.6-terra", ["gpt-5.6-terra"], ["sonnet"]),
    ],
)
def test_planner_and_worker_route_codex_to_configured_model(
    monkeypatch,
    tmp_path,
    model,
    planner_model,
    worker_model,
    codex_model,
    expected_codex,
    expected_other,
):
    codex_models = []
    other_models = []
    theorem = tmp_path / "theorem.md"
    theorem.write_text("theorem")

    if codex_model is None:
        monkeypatch.delenv("CODEX_MODEL", raising=False)
    else:
        monkeypatch.setenv("CODEX_MODEL", codex_model)

    def make_client(models, model_name):
        models.append(model_name)
        return SimpleNamespace(
            model=model_name,
            total_cost=0.0,
            call_count=0,
            cleanup=lambda: None,
        )

    def make_codex(model_name, _archive_dir):
        return make_client(codex_models, model_name)

    def make_other(model_name, _archive_dir, **_kwargs):
        return make_client(other_models, model_name)

    class ProverSpy:
        def __init__(self, **kwargs):
            self.planner_llm = kwargs["make_llm"](kwargs["work_dir"])
            self.worker_llm = kwargs["make_worker_llm"](kwargs["work_dir"])
            self.lean_work_dir = None
            self.work_dir = kwargs["work_dir"]
            self.budget = SimpleNamespace(total_output_tokens=0)
            self.mode = "prove"
            self._spending_limit_hit = False
            self._llm_error_exit = False

        def run(self):
            pass

        def request_interrupt(self):
            pass

    class HeadlessTUISpy:
        def cleanup(self):
            pass

    argv = [
        "openprover",
        str(tmp_path / "run"),
        "--theorem",
        str(theorem),
        "--model",
        model,
        "--headless",
    ]
    if planner_model is not None:
        argv.extend(["--planner-model", planner_model])
    if worker_model is not None:
        argv.extend(["--worker-model", worker_model])
    monkeypatch.setattr(cli, "CodexClient", make_codex)
    monkeypatch.setattr(cli, "LLMClient", make_other)
    monkeypatch.setattr(cli, "Prover", ProverSpy)
    monkeypatch.setattr(cli, "HeadlessTUI", HeadlessTUISpy)
    monkeypatch.setattr(cli.atexit, "register", lambda _callback: None)
    monkeypatch.setattr(sys, "argv", argv)

    cli._cmd_prove()

    assert codex_models == expected_codex
    assert other_models == expected_other
