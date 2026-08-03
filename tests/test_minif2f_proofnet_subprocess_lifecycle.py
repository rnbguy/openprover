import importlib.util
import os
import signal
import subprocess
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


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeProcess:
    def __init__(self, *, stdout="", error=None, pid=4242):
        self.pid = pid
        self.returncode = None
        self.stdout = stdout
        self.stderr = ""
        self.error = error
        self.timeouts = []
        self.kill_calls = 0
        self.wait_calls = 0

    def communicate(self, input=None, timeout=None):
        self.timeouts.append(timeout)
        if self.error is not None:
            error = self.error
            self.error = None
            raise error
        self.returncode = 0
        return self.stdout, self.stderr

    def poll(self):
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = -signal.SIGKILL

    def wait(self):
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.wait()


class PopenFactory:
    def __init__(self, processes, clock, launch_delays=()):
        self.processes = processes
        self.clock = clock
        self.launch_delays = launch_delays
        self.calls = []

    def __call__(self, *_args, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if index < len(self.launch_delays):
            self.clock.now += self.launch_delays[index]
        if index >= len(self.processes):
            raise AssertionError("openprover relaunched after its deadline")
        process = self.processes[index]
        process.args = _args[0]
        return process


@pytest.fixture(params=["run_minif2f", "run_proofnet"])
def benchmark(request):
    return load_script(request.param)


def benchmark_args(*, max_time="5s", parallelism=1):
    return Namespace(
        method="openprover",
        split="valid",
        model="leanstral",
        planner_model=None,
        worker_model=None,
        max_workers=1,
        max_tokens=None,
        max_time=max_time,
        informal=True,
        provider_url="http://127.0.0.1:8000",
        isolation=True,
        verifier=True,
        parallelism=parallelism,
    )


def run_openprover(benchmark, bench_dir, args):
    info = {"informal": "statement", "formal": "theorem example : True := by\n  sorry", "header": ""}
    return benchmark._run_openprover("example", info, None, bench_dir, args)


def install_clock(monkeypatch, benchmark, clock):
    monkeypatch.setattr(benchmark.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(benchmark.time, "sleep", clock.sleep)


def test_openprover_starts_in_a_new_session(monkeypatch, tmp_path, benchmark):
    # Given
    clock = FakeClock()
    process = FakeProcess(stdout="[result] proved")
    popen = PopenFactory([process], clock)
    install_clock(monkeypatch, benchmark, clock)
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)

    # When
    result = run_openprover(benchmark, tmp_path, benchmark_args())

    # Then
    assert result["status"] == "proved"
    assert popen.calls[0].get("start_new_session") is True


def test_communicate_uses_only_the_remaining_deadline(monkeypatch, tmp_path, benchmark):
    # Given
    clock = FakeClock()
    process = FakeProcess(stdout="[result] proved")
    popen = PopenFactory([process], clock, launch_delays=[4])
    install_clock(monkeypatch, benchmark, clock)
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)

    # When
    result = run_openprover(benchmark, tmp_path, benchmark_args())

    # Then
    assert result["status"] == "proved"
    assert process.timeouts == [121]


@pytest.mark.parametrize("group_kill_fails", [False, True])
def test_timeout_kills_the_process_group_with_fallback_and_wait(
    monkeypatch, tmp_path, benchmark, group_kill_fails,
):
    # Given
    clock = FakeClock()
    process = FakeProcess(error=subprocess.TimeoutExpired("openprover", 125))
    popen = PopenFactory([process], clock)
    group_kills = []

    def killpg(pid, sig):
        group_kills.append((pid, sig))
        if group_kill_fails:
            raise ProcessLookupError(pid)
        process.returncode = -sig

    install_clock(monkeypatch, benchmark, clock)
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(benchmark, "killpg", killpg, raising=False)

    # When
    result = run_openprover(benchmark, tmp_path, benchmark_args())

    # Then
    assert result["status"] == "error"
    assert result["error"] == "hard timeout (125s)"
    assert group_kills == [(process.pid, signal.SIGKILL)]
    assert process.kill_calls == int(group_kill_fails)
    assert process.wait_calls == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(7)])
def test_base_interrupt_cleans_process_group_and_propagates(
    monkeypatch, tmp_path, benchmark, error,
):
    # Given
    clock = FakeClock()
    process = FakeProcess(error=error)
    popen = PopenFactory([process], clock)
    group_kills = []

    def killpg(pid, sig):
        group_kills.append((pid, sig))
        process.returncode = -sig

    install_clock(monkeypatch, benchmark, clock)
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(benchmark, "killpg", killpg, raising=False)

    # When
    with pytest.raises(type(error)) as raised:
        run_openprover(benchmark, tmp_path, benchmark_args())

    # Then
    assert raised.value is error
    assert group_kills == [(process.pid, signal.SIGKILL)]
    assert process.kill_calls == 0
    assert process.wait_calls == 1


def test_rate_limit_retry_shares_one_absolute_deadline(monkeypatch, tmp_path, benchmark):
    # Given
    clock = FakeClock()
    first = FakeProcess(stdout="[result] rate_limited", pid=4242)
    second = FakeProcess(stdout="[result] proved", pid=4243)
    popen = PopenFactory([first, second], clock, launch_delays=[10, 0])
    install_clock(monkeypatch, benchmark, clock)
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)

    # When
    result = run_openprover(benchmark, tmp_path, benchmark_args(max_time="10m"))

    # Then
    assert result["status"] == "proved"
    assert clock.sleeps == [benchmark.RATE_LIMIT_WAIT]
    assert first.timeouts == [710]
    assert second.timeouts == [110]


def test_exhausted_rate_limit_wait_returns_timeout_without_relaunch(
    monkeypatch, tmp_path, benchmark,
):
    # Given
    clock = FakeClock()
    process = FakeProcess(stdout="[result] rate_limited")
    popen = PopenFactory([process], clock, launch_delays=[10])
    install_clock(monkeypatch, benchmark, clock)
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)

    # When
    result = run_openprover(benchmark, tmp_path, benchmark_args())

    # Then
    assert result["name"] == "example"
    assert result["status"] == "error"
    assert result["error"] == "hard timeout (125s)"
    assert sum(clock.sleeps) <= 115
    assert len(popen.calls) == 1


def test_timeout_is_returned_as_a_retryable_error_result(monkeypatch, tmp_path, benchmark):
    # Given
    clock = FakeClock()
    process = FakeProcess(error=subprocess.TimeoutExpired("openprover", 125))
    install_clock(monkeypatch, benchmark, clock)
    monkeypatch.setattr(benchmark.subprocess, "Popen", PopenFactory([process], clock))

    # When
    result = run_openprover(benchmark, tmp_path, benchmark_args())

    # Then
    assert result["name"] == "example"
    assert result["status"] == "error"
    assert result["error"] == "hard timeout (125s)"
