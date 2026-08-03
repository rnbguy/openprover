import contextlib
import importlib.util
import os
import signal
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    cli = ModuleType("openprover.cli")
    cli.__dict__["positive_int"] = int
    previous_cli = sys.modules.get("openprover.cli")
    sys.modules["openprover.cli"] = cli
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_cli is None:
            del sys.modules["openprover.cli"]
        else:
            sys.modules["openprover.cli"] = previous_cli
    return module


class FakeProcess:
    def __init__(self, *, stdout="", stderr="", returncode=0, error=None, pid=4242):
        self.pid = pid
        self.args = []
        self.returncode = None
        self.stdout = stdout
        self.stderr = stderr
        self._completed_returncode = returncode
        self.error = error
        self.timeouts = []
        self.kill_calls = 0
        self.wait_calls = 0

    def communicate(self, input=None, timeout=None):
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        self.returncode = self._completed_returncode
        return self.stdout, self.stderr

    def poll(self):
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = -signal.SIGKILL

    def wait(self):
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = self._completed_returncode
        return self.returncode

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.wait()


class PopenFactory:
    def __init__(self, process):
        self.process = process
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(kwargs)
        self.process.args = command
        return self.process


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


def putnam_args(*, max_time="5s", parallelism=2):
    return Namespace(
        model="leanstral", planner_model=None, worker_model=None, max_workers=1,
        max_time=max_time, informal=True, provider_url="http://127.0.0.1:8000",
        isolation=True, parallelism=parallelism,
    )


def invoke_putnam_parallel(module, monkeypatch, tmp_path):
    return module._run_problem("example", "statement", tmp_path, putnam_args())


def invoke_putnam_sequential(module, monkeypatch, tmp_path):
    repo = tmp_path / "PutnamBench"
    data_dir = repo / "informal"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "putnam.json").write_text(
        '[{"problem_name": "example", "informal_statement": "statement"}]'
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_putnam", "--repo-path", str(repo), "--problem", "example",
            "--informal", "--model", "leanstral", "--max-time", "5s",
        ],
    )
    return module.main()


def invoke_proofbench(module, monkeypatch, _tmp_path):
    monkeypatch.setattr(
        module,
        "load_problems",
        lambda _path: {"PB-Test": {"Problem ID": "PB-Test", "Problem": "statement"}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_proofbench", "PB-Test", "--model", "minimax-m2.5", "--max-time", "5s"],
    )
    return module.main()


LAUNCH_CASES = [
    pytest.param("run_putnam", invoke_putnam_parallel, id="putnam-parallel"),
    pytest.param("run_putnam", invoke_putnam_sequential, id="putnam-sequential"),
    pytest.param("run_proofbench", invoke_proofbench, id="proofbench"),
]
CLI_CASES = LAUNCH_CASES[1:]


def install_process(monkeypatch, module, process, *, group_kill_fails=False):
    popen = PopenFactory(process)
    group_kills = []

    def killpg(pid, sig):
        group_kills.append((pid, sig))
        if group_kill_fails:
            raise ProcessLookupError(pid)
        process.returncode = -sig

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    clock = SimpleNamespace(monotonic=lambda: 100.0)
    monkeypatch.setattr(module, "time", clock, raising=False)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(module, "killpg", killpg, raising=False)
    return popen, group_kills


@pytest.mark.parametrize(("script", "invoke"), LAUNCH_CASES)
def test_launch_uses_outer_deadline_and_new_session(
    monkeypatch, tmp_path, script, invoke,
):
    # Given
    module = load_script(script)
    process = FakeProcess(stdout="[result] proved")
    popen, _ = install_process(monkeypatch, module, process)

    # When
    invoke(module, monkeypatch, tmp_path)

    # Then
    assert (process.timeouts, popen.calls[0].get("start_new_session")) == ([125], True)


@pytest.mark.parametrize(("script", "invoke"), LAUNCH_CASES)
@pytest.mark.parametrize("group_kill_fails", [False, True])
def test_timeout_kills_process_group_with_fallback_and_wait(
    monkeypatch, tmp_path, script, invoke, group_kill_fails,
):
    # Given
    module = load_script(script)
    timeout = subprocess.TimeoutExpired("openprover", 125)
    process = FakeProcess(error=timeout)
    _, group_kills = install_process(
        monkeypatch, module, process, group_kill_fails=group_kill_fails
    )

    # When
    with contextlib.suppress(subprocess.TimeoutExpired):
        invoke(module, monkeypatch, tmp_path)

    # Then
    assert group_kills == [(process.pid, signal.SIGKILL)]
    assert process.kill_calls == int(group_kill_fails)
    assert process.wait_calls == 1


@pytest.mark.parametrize(("script", "invoke"), LAUNCH_CASES)
@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(7)])
def test_interrupt_cleans_process_group_and_propagates(
    monkeypatch, tmp_path, script, invoke, error,
):
    # Given
    module = load_script(script)
    process = FakeProcess(error=error)
    _, group_kills = install_process(monkeypatch, module, process)

    # When
    with pytest.raises(type(error)) as raised:
        invoke(module, monkeypatch, tmp_path)

    # Then
    assert raised.value is error
    assert group_kills == [(process.pid, signal.SIGKILL)]
    assert process.kill_calls == 0
    assert process.wait_calls == 1


def test_putnam_parallel_timeout_returns_error_tuple(monkeypatch, tmp_path):
    # Given
    module = load_script("run_putnam")
    timeout = subprocess.TimeoutExpired("openprover", 125)
    process = FakeProcess(error=timeout)
    install_process(monkeypatch, module, process)

    # When
    result = invoke_putnam_parallel(module, monkeypatch, tmp_path)

    # Then
    assert result == ("example", "error", 0.0, "hard timeout (125s)")


def test_putnam_parallel_reports_timeout_as_error_status(monkeypatch, tmp_path, capsys):
    # Given
    module = load_script("run_putnam")
    timeout_result = ("example", "error", 125.0, "hard timeout (125s)")
    future = FakeFuture(result=timeout_result)
    executor = FakeExecutor({"example": future})
    monkeypatch.setattr(module, "ThreadPoolExecutor", lambda max_workers: executor)
    monkeypatch.setattr(module, "as_completed", lambda _futures: [future])

    # When
    module._run_parallel({"example": "statement"}, tmp_path, putnam_args())

    # Then
    output = capsys.readouterr()
    assert "1 errors (of 1)" in output.out
    assert "hard timeout (125s)" in output.err


def test_putnam_parallel_interrupt_cancels_pending_work(monkeypatch, tmp_path):
    # Given
    module = load_script("run_putnam")
    completed = FakeFuture(
        result=("done", "error", 125.0, "hard timeout (125s)")
    )
    interrupt = KeyboardInterrupt()
    interrupted = FakeFuture(error=interrupt)
    pending = FakeFuture()
    executor = FakeExecutor(
        {"done": completed, "interrupted": interrupted, "pending": pending}
    )
    monkeypatch.setattr(module, "ThreadPoolExecutor", lambda max_workers: executor)
    monkeypatch.setattr(
        module, "as_completed", lambda _futures: [completed, interrupted]
    )

    # When
    with pytest.raises(KeyboardInterrupt) as raised:
        module._run_parallel(
            {"done": "", "interrupted": "", "pending": ""},
            tmp_path,
            putnam_args(parallelism=3),
        )

    # Then
    assert raised.value is interrupt
    assert pending.cancelled is True


@pytest.mark.parametrize(("script", "invoke"), CLI_CASES)
def test_cli_timeout_cleans_up_and_propagates(monkeypatch, tmp_path, script, invoke):
    # Given
    module = load_script(script)
    timeout = subprocess.TimeoutExpired("openprover", 125)
    process = FakeProcess(error=timeout)
    install_process(monkeypatch, module, process)

    # When
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        invoke(module, monkeypatch, tmp_path)

    # Then
    assert raised.value is timeout


@pytest.mark.parametrize(("script", "invoke"), CLI_CASES)
def test_cli_propagates_nonzero_child_status(monkeypatch, tmp_path, script, invoke):
    # Given
    module = load_script(script)
    process = FakeProcess(returncode=9, stderr="fatal error")
    install_process(monkeypatch, module, process)

    # When
    with pytest.raises((subprocess.CalledProcessError, SystemExit)) as raised:
        invoke(module, monkeypatch, tmp_path)

    # Then
    error = raised.value
    status = error.returncode if isinstance(error, subprocess.CalledProcessError) else error.code
    assert status == 9
