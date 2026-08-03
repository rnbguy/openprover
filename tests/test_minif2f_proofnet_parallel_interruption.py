import importlib.util
import signal
import sys
from argparse import Namespace
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


class FakeFuture:
    def __init__(self, *, result=None, error=None):
        self._result = result
        self._error = error
        self.cancelled = False

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        self.cancelled = True
        return True


class FakeExecutor:
    def __init__(self, futures):
        self.futures = futures

    def submit(self, _runner, name, *_args):
        return self.futures[name]

    def shutdown(self, wait=True, *, cancel_futures=False):
        if cancel_futures:
            for future in self.futures.values():
                future.cancel()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.shutdown()


@pytest.fixture(params=["run_minif2f", "run_proofnet"])
def benchmark(request):
    return load_script(request.param)


def benchmark_args(*, parallelism=1):
    return Namespace(
        method="openprover",
        split="valid",
        model="leanstral",
        planner_model=None,
        worker_model=None,
        max_workers=1,
        max_tokens=None,
        max_time="5s",
        informal=True,
        provider_url="http://127.0.0.1:8000",
        isolation=True,
        verifier=True,
        parallelism=parallelism,
    )


def test_parallel_interrupt_persists_completed_results_and_cancels_pending_work(
    monkeypatch, tmp_path, benchmark,
):
    # Given
    completed_entry = {
        "name": "done",
        "status": "error",
        "elapsed": 125,
        "error": "hard timeout (125s)",
    }
    interrupt = KeyboardInterrupt()
    completed = FakeFuture(result=completed_entry)
    interrupted = FakeFuture(error=interrupt)
    pending = FakeFuture()
    executor = FakeExecutor({"done": completed, "interrupted": interrupted, "pending": pending})
    snapshots = []
    problems = {
        "done": {"informal": "", "formal": "", "header": ""},
        "interrupted": {"informal": "", "formal": "", "header": ""},
        "pending": {"informal": "", "formal": "", "header": ""},
    }
    monkeypatch.setattr(benchmark, "ThreadPoolExecutor", lambda max_workers: executor)
    monkeypatch.setattr(benchmark, "as_completed", lambda _futures: [completed, interrupted])
    monkeypatch.setattr(benchmark, "_save_results", lambda _path, results: snapshots.append(list(results)))

    # When
    with pytest.raises(KeyboardInterrupt) as raised:
        benchmark._run_all(problems, None, tmp_path, benchmark_args(parallelism=3))

    # Then
    assert raised.value is interrupt
    assert snapshots == [[], [completed_entry]]
    assert pending.cancelled is True


def test_main_maps_sigterm_to_exit_and_restores_handler(monkeypatch, tmp_path, benchmark):
    # Given
    future = FakeFuture()
    previous_handler = signal.getsignal(signal.SIGTERM)

    def raise_sigterm():
        signal.raise_signal(signal.SIGTERM)

    def unexpected_sigterm(_signum, _frame):
        raise AssertionError("SIGTERM handler was not installed")

    future.result = raise_sigterm
    executor = FakeExecutor({"example": future})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "ThreadPoolExecutor", lambda max_workers: executor)
    monkeypatch.setattr(benchmark, "as_completed", lambda _futures: [future])
    signal.signal(signal.SIGTERM, unexpected_sigterm)
    if benchmark.__name__ == "run_minif2f":
        monkeypatch.setattr(
            benchmark,
            "_fetch_lean_file",
            lambda _split: "theorem example : True := by\n  sorry",
        )
        monkeypatch.setattr(sys, "argv", ["run_minif2f", "valid", "--model", "leanstral"])
    else:
        repo_path = tmp_path / "proofnet"
        data_dir = repo_path / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "proofnet.jsonl").write_text(
            '{"split":"valid","name":"example","informal_prefix":"/-- statement -/",'
            '"formal_statement":"theorem example : True :=","header":"import Mathlib\\n"}\n'
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_proofnet", "valid", "--repo-path", str(repo_path), "--informal", "--model", "leanstral"],
        )
    # When
    try:
        with pytest.raises(SystemExit) as raised:
            benchmark.main()

        # Then
        assert raised.value.code == 143
        assert future.cancelled is True
        assert signal.getsignal(signal.SIGTERM) is unexpected_sigterm
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
