import argparse
import sys
from types import SimpleNamespace

import pytest

from openprover import cli


class _ParserCaptured(Exception):
    pass


def test_openprover_model_choices_preserve_explicit_models(monkeypatch):
    captured = []
    expected_models = (
        "sonnet",
        "opus",
        "codex",
        "minimax-m2.5",
        "leanstral",
        "glm-5",
        "kimi-k2.5",
        "minimax-m2.7",
    )
    original_parse_args = cli.argparse.ArgumentParser.parse_args

    def capture_parser(parser, *_args, **_kwargs):
        captured.extend(
            action
            for action in parser._actions
            if action.dest in ("model", "planner_model", "worker_model")
        )
        raise _ParserCaptured

    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", capture_parser)

    with pytest.raises(_ParserCaptured):
        cli._cmd_prove()

    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", original_parse_args)
    assert [action.dest for action in captured] == ["model", "planner_model", "worker_model"]
    for action in captured:
        assert tuple(action.choices) == expected_models
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", choices=action.choices)
        assert parser.parse_args(["--model", "codex"]).model == "codex"
        for rejected in ("gpt", "gpt-5.6-terra", "gptish", "claude-4.6"):
            with pytest.raises(SystemExit):
                parser.parse_args(["--model", rejected])


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
