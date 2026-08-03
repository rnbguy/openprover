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


@pytest.mark.parametrize(
    ("script", "option"),
    [
        ("run_minif2f", "--max-workers"),
        ("run_minif2f", "--parallelism"),
        ("run_putnam", "--max-workers"),
        ("run_putnam", "--parallelism"),
        ("run_proofnet", "--max-workers"),
        ("run_proofnet", "--parallelism"),
    ],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_benchmark_rejects_non_positive_worker_counts(
    monkeypatch, script, option, value, capsys,
):
    module = load_script(script)
    monkeypatch.setattr(sys, "argv", [script, option, value])

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    assert "invalid positive integer value" in capsys.readouterr().err
