import os
import signal
import subprocess
from types import SimpleNamespace

import pytest

from openprover.lean import core as lean_core
from openprover.lean.core import LeanWorkDir


def test_run_lean_check_kills_process_group_and_reaps_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    # Given
    lean_file = tmp_path / "Proof.lean"
    lean_file.write_text("example : True := by trivial")
    process_group_kills = []
    fallback_kills = []
    waits = []
    popen_calls = []

    def communicate(*, timeout):
        raise subprocess.TimeoutExpired("lake", timeout)

    process = SimpleNamespace(
        pid=4242,
        communicate=communicate,
        kill=lambda: fallback_kills.append(None),
        wait=lambda: waits.append(None),
    )

    def popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    monkeypatch.setattr(lean_core.subprocess, "Popen", popen)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda process_group, sig: process_group_kills.append((process_group, sig)),
    )

    # When
    result = lean_core.run_lean_check(lean_file, tmp_path, timeout=7)

    # Then
    assert result[0] is False
    assert result[1] == "Lean verification timed out after 7s"
    assert callable(popen_calls[0][1]["preexec_fn"])
    assert process_group_kills == [(process.pid, signal.SIGKILL)]
    assert fallback_kills == []
    assert waits == [None]


def test_cleanup_removes_only_owned_openprover_directory(tmp_path):
    # Given
    work_dir = LeanWorkDir(tmp_path)
    work_dir.write_proof("example : True := by trivial")
    unowned_dir = tmp_path / "OpenProver-unowned"
    unowned_dir.mkdir()
    unowned_file = unowned_dir / "keep.txt"
    unowned_file.write_text("keep")

    # When
    getattr(work_dir, "cleanup")()

    # Then
    assert not work_dir.dir.exists()
    assert unowned_file.read_text() == "keep"


def test_cleanup_is_idempotent(tmp_path):
    # Given
    work_dir = LeanWorkDir(tmp_path)

    # When
    getattr(work_dir, "cleanup")()
    getattr(work_dir, "cleanup")()

    # Then
    assert not work_dir.dir.exists()
