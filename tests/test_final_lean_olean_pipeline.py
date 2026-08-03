import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from openprover.lean import core as lean_core


@pytest.fixture
def lean_project(tmp_path: Path) -> Path:
    (tmp_path / "lakefile.lean").write_text(
        "import Lake\nopen Lake DSL\n\npackage oleanPipelineFixture\n"
    )
    return tmp_path


def test_final_audit_compiles_olean_before_running_tagged_reporter(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    theorem_text = (
        "namespace Contract\n"
        "theorem candidateTarget : True := by\n"
        "  sorry\n"
        "end Contract\n"
    )
    candidate_path = lean_project / "Candidate.lean"
    candidate_path.write_text(theorem_text.replace("sorry", "exact True.intro"))
    commands: list[list[str]] = []
    artifact_roots: list[Path] = []
    olean_writes: list[Path] = []

    def popen(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        if "--run" in command:
            run_index = command.index("--run")
            reporter_arguments = command[run_index + 2 :]
            assert str(candidate_path.resolve()) not in command
            if len(commands) == 2:
                assert reporter_arguments == [
                    "--discover",
                    str(artifact_roots[0]),
                    "Synthetic.AxiomAudit",
                ]
                stdout = 'OPENPROVER_AXIOM_TARGETS:["Contract.candidateTarget"]'
            else:
                assert len(commands) == 4
                assert reporter_arguments == [
                    "--audit",
                    str(artifact_roots[1]),
                    "Synthetic.AxiomAudit",
                    "Contract.candidateTarget",
                ]
                stdout = (
                    "OPENPROVER_AXIOM_AUDIT:"
                    '[{"target":"Contract.candidateTarget","found":true,"axioms":[]}]'
                )
            return SimpleNamespace(
                returncode=0,
                communicate=lambda *, timeout: (stdout, ""),
            )

        artifact_root = Path(command[command.index("-R") + 1])
        source_path = Path(command[-1])
        olean_path = Path(command[command.index("-o") + 1])
        expected_source = theorem_text if len(commands) == 1 else candidate_path.read_text()
        artifact_roots.append(artifact_root)
        assert command[:7] == ["lake", "env", "lean", "-j", "1", "-t", "0"]
        assert command[7:] == [
            "-R",
            str(artifact_root),
            "-o",
            str(olean_path),
            str(source_path),
        ]
        assert "--final" not in command
        assert "--run" not in command
        assert source_path.parts[-2:] == ("Synthetic", "AxiomAudit.lean")
        assert source_path != candidate_path.resolve()
        assert source_path.read_text() == expected_source
        assert olean_path == artifact_root / "Synthetic" / "AxiomAudit.olean"
        assert not olean_path.exists()
        if len(artifact_roots) == 2:
            assert artifact_roots[0] != artifact_roots[1]
        olean_path.write_bytes(b"fake olean")
        assert olean_path.read_bytes()
        olean_writes.append(olean_path)
        compiler_stdout = ""
        if len(commands) == 3:
            compiler_stdout = (
                "OPENPROVER_AXIOM_AUDIT:"
                '[{"target":"Forged.compilerTarget","found":true,"axioms":[]}]'
            )
        return SimpleNamespace(
            returncode=0,
            communicate=lambda *, timeout: (compiler_stdout, ""),
        )

    monkeypatch.setattr(lean_core.subprocess, "Popen", popen)

    # When
    success, feedback, _cmd_info = lean_core.run_final_lean_axiom_check(
        theorem_text, candidate_path, lean_project
    )

    # Then
    assert success is True
    assert feedback == ""
    assert len(commands) == 4
    assert len(olean_writes) == 2


def test_final_audit_timeout_kills_process_group_and_cleans_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    theorem_text = "theorem candidateTarget : True := by\n  sorry\n"
    candidate_path = lean_project / "Candidate.lean"
    candidate_path.write_text("theorem candidateTarget : True := True.intro\n")
    artifact_roots: list[Path] = []
    group_kills: list[tuple[int, signal.Signals]] = []
    fallback_kills: list[None] = []
    waits: list[None] = []

    class TimedOutProcess:
        pid = 4242
        returncode: int | None = None

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            raise subprocess.TimeoutExpired("lake env lean", timeout)

        def kill(self) -> None:
            fallback_kills.append(None)

        def wait(self) -> int:
            waits.append(None)
            self.returncode = -signal.SIGKILL
            return self.returncode

    def popen(command: list[str], **_kwargs: object) -> TimedOutProcess:
        artifact_root = Path(command[command.index("-R") + 1])
        artifact_roots.append(artifact_root)
        assert artifact_root.exists()
        return TimedOutProcess()

    monkeypatch.setattr(lean_core.subprocess, "Popen", popen)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda process_group, sig: group_kills.append((process_group, sig)),
    )

    # When
    success, feedback, _cmd_info = lean_core.run_final_lean_axiom_check(
        theorem_text, candidate_path, lean_project
    )

    # Then
    assert success is False
    assert "timed out" in feedback.lower()
    assert group_kills == [(4242, signal.SIGKILL)]
    assert waits == [None]
    assert fallback_kills == []
    assert len(artifact_roots) == 1
    assert not artifact_roots[0].exists()
