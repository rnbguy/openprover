import importlib.util
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


class FakeCodexClient:
    instances = []

    def __init__(self, model, archive_dir, **kwargs):
        self.model = model
        self.archive_dir = archive_dir
        self.constructor_kwargs = kwargs
        self.call_kwargs = None
        self.cleaned_up = False
        self.instances.append(self)

    def call(self, **kwargs):
        self.call_kwargs = kwargs
        return {
            "finish_reason": "stop",
            "duration_ms": 1,
            "result": "pong",
            "thinking": "",
            "raw": {
                "usage": {
                    "input_tokens": 11,
                    "cached_input_tokens": 2,
                    "output_tokens": 7,
                    "reasoning_output_tokens": 3,
                    "total_tokens": 18,
                }
            },
        }

    def cleanup(self):
        self.cleaned_up = True


@pytest.fixture
def ping_harness(monkeypatch, tmp_path):
    module = load_script("ping_codex")
    FakeCodexClient.instances = []
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(module, "CodexClient", FakeCodexClient)
    monkeypatch.setattr(
        module.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(tmp_path / "archive"),
    )
    return module


def test_ping_help_omits_model_and_token_limit_options(ping_harness):
    help_options = set(ping_harness.build_parser().format_help().split())

    assert help_options.isdisjoint({"--model", "--max-tokens"})


@pytest.mark.parametrize("option", ["--model", "--max-tokens"])
def test_ping_rejects_removed_options(ping_harness, option):
    with pytest.raises(SystemExit) as raised:
        ping_harness.build_parser().parse_args([option, "1"])

    assert raised.value.code == 2


def test_ping_reaches_client_without_external_codex_probe(ping_harness):
    assert not hasattr(ping_harness, "subprocess")

    ping_harness.main([])

    assert len(FakeCodexClient.instances) == 1


def test_ping_uses_supported_model(ping_harness):
    ping_harness.main([])

    assert FakeCodexClient.instances[0].model == "gpt-5.6-sol"


def test_ping_does_not_forward_token_limits(ping_harness):
    ping_harness.main([])
    client = FakeCodexClient.instances[0]

    assert (client.constructor_kwargs, "max_tokens" in client.call_kwargs) == ({}, False)


def test_ping_cleans_up_client(ping_harness):
    ping_harness.main([])

    assert FakeCodexClient.instances[0].cleaned_up is True


def test_ping_exposes_only_lean_search_to_codex(tmp_path):
    module = load_script("ping_codex")

    enabled_tools = module._build_mcp_config(str(tmp_path))["mcp_servers"]["lean_tools"][
        "enabled_tools"
    ]

    assert enabled_tools == ["lean_search"]


def test_ping_prints_codex_usage_fields(ping_harness, capsys):
    ping_harness.main([])

    output = capsys.readouterr().out
    assert all(
        f"{label}: {value}" in output
        for label, value in (
            ("input_tokens", 11),
            ("cached_input_tokens", 2),
            ("output_tokens", 7),
            ("reasoning_output_tokens", 3),
            ("total_tokens", 18),
        )
    )


@pytest.mark.parametrize(
    "model_and_checks",
    [("gpt", []), ("sonnet", ["claude"])],
    ids=["gpt", "sonnet"],
)
def test_putnam_preflight_checks_only_external_cli_models(
    monkeypatch,
    tmp_path,
    model_and_checks,
):
    model, expected_checks = model_and_checks
    module = load_script("run_putnam")
    repo = tmp_path / "PutnamBench"
    informal = repo / "informal"
    informal.mkdir(parents=True)
    (informal / "putnam.json").write_text(
        '[{"problem_name": "example", "informal_statement": "statement"}]',
        encoding="utf-8",
    )
    checks = []
    parallel_runs = []
    monkeypatch.setattr(module, "_check_tool", checks.append)
    monkeypatch.setattr(
        module,
        "_run_parallel",
        lambda *_args: parallel_runs.append(True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_putnam",
            "--repo-path",
            str(repo),
            "--model",
            model,
            "--informal",
            "--parallelism",
            "2",
        ],
    )

    module.main()

    assert (checks, parallel_runs) == (expected_checks, [True])
