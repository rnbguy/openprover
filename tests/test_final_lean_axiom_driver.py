import subprocess
import textwrap
from types import SimpleNamespace

import pytest

from openprover.lean import core as lean_core
from openprover.lean.core import LeanWorkDir


@pytest.fixture
def lean_project(tmp_path):
    (tmp_path / "lakefile.lean").write_text(
        "import Lake\nopen Lake DSL\n\npackage axiomAuditFixture\n\n"
        "lean_lib FixtureDependency\n"
    )
    return tmp_path


def write_proof(project_dir, source):
    work_dir = LeanWorkDir(project_dir)
    return work_dir.make_file("axiom-audit", textwrap.dedent(source).strip() + "\n")


def run_axiom_check(proof_path, project_dir, target_names):
    check = getattr(lean_core, "run_lean_axiom_check", None)
    if check is None:
        pytest.fail("expected openprover.lean.core.run_lean_axiom_check API is not implemented")
    return check(proof_path, project_dir, target_names)


def test_axiom_probe_runs_trusted_driver_with_candidate_and_targets_as_arguments(
    monkeypatch,
    lean_project,
):
    # Given
    proof_path = write_proof(
        lean_project,
        """
        theorem candidateTarget : True := True.intro
        """,
    )
    audit_dirs = []

    def popen(command, **kwargs):
        if "--run" not in command:
            root = type(proof_path)(command[command.index("-R") + 1])
            source_path = type(proof_path)(command[-1])
            olean_path = type(proof_path)(command[command.index("-o") + 1])
            assert command[:7] == ["lake", "env", "lean", "-j", "1", "-t", "0"]
            assert source_path == root / "Synthetic" / "AxiomAudit.lean"
            assert source_path.read_text() == proof_path.read_text()
            olean_path.write_bytes(b"olean")
            return SimpleNamespace(
                returncode=0, communicate=lambda *, timeout: ("", "")
            )
        run_index = command.index("--run")
        driver_path = type(proof_path)(command[run_index + 1])
        audit_dirs.append(driver_path.parent)
        assert command[:7] == ["lake", "env", "lean", "-j", "1", "-t", "0"]
        assert command[run_index + 2 :] == [
            "--audit", str(driver_path.parent / "candidate"), "Synthetic.AxiomAudit", "candidateTarget"
        ]
        assert str(proof_path) not in command
        assert driver_path != proof_path
        assert driver_path.parent != proof_path.parent
        assert driver_path.is_file()
        assert "theorem candidateTarget" not in driver_path.read_text()
        return SimpleNamespace(
            returncode=0,
            communicate=lambda *, timeout: (
                'OPENPROVER_AXIOM_AUDIT:[{"target":"candidateTarget","found":true,"axioms":[]}]',
                "",
            ),
        )

    monkeypatch.setattr(lean_core.subprocess, "Popen", popen)

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("candidateTarget",)
    )

    # Then
    assert success is True
    assert feedback == ""
    assert proof_path.is_file()
    assert not audit_dirs[0].exists()


def test_empty_target_tuple_is_rejected_before_starting_probe(
    monkeypatch,
    lean_project,
):
    # Given
    proof_path = write_proof(
        lean_project,
        """
        theorem candidateTarget : True := True.intro
        """,
    )

    def unexpected_popen(command, **kwargs):
        pytest.fail("empty targets must be rejected before starting the axiom probe")

    monkeypatch.setattr(lean_core.subprocess, "Popen", unexpected_popen)

    # When
    success, feedback, _cmd_info = run_axiom_check(proof_path, lean_project, ())

    # Then
    assert success is False
    assert "target" in feedback.lower()
    assert proof_path.is_file()


@pytest.mark.parametrize(
    ("probe_stdout", "target_names", "expected_fragment"),
    [
        (
            'OPENPROVER_AXIOM_AUDIT:[{"target":"probeTarget","found":true}]',
            ("probeTarget",),
            "malformed",
        ),
        (
            'OPENPROVER_AXIOM_AUDIT:[{"target":"probeTarget","found":true,"axioms":[]}]',
            ("probeTarget", "missingResultTarget"),
            "missing",
        ),
    ],
)
def test_malformed_or_incomplete_probe_results_are_rejected(
    monkeypatch,
    lean_project,
    probe_stdout,
    target_names,
    expected_fragment,
):
    # Given
    proof_path = write_proof(
        lean_project,
        """
        theorem probeTarget : True := True.intro
        theorem missingResultTarget : True := True.intro
        """,
    )
    def popen(command, **kwargs):
        if "--run" not in command:
            type(proof_path)(command[command.index("-o") + 1]).write_bytes(b"olean")
            stdout = ""
        else:
            stdout = probe_stdout
        return SimpleNamespace(
            returncode=0,
            communicate=lambda *, timeout: (stdout, ""),
        )

    monkeypatch.setattr(lean_core.subprocess, "Popen", popen)

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, target_names
    )

    # Then
    assert success is False
    assert expected_fragment in feedback.lower()


def test_axiom_probe_cleans_audit_directory_when_popen_is_missing(
    monkeypatch,
    lean_project,
):
    # Given
    proof_path = write_proof(
        lean_project,
        """
        theorem candidateTarget : True := True.intro
        """,
    )
    audit_dirs = []

    def missing_popen(command, **kwargs):
        root = type(proof_path)(command[command.index("-R") + 1])
        audit_dirs.append(root.parent)
        assert (root / "Synthetic" / "AxiomAudit.lean").is_file()
        raise FileNotFoundError("lake")

    monkeypatch.setattr(lean_core.subprocess, "Popen", missing_popen)

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("candidateTarget",)
    )

    # Then
    assert success is False
    assert "lake" in feedback.lower()
    assert proof_path.is_file()
    assert not audit_dirs[0].exists()


def test_axiom_probe_kills_and_waits_then_cleans_audit_directory_on_timeout(
    monkeypatch,
    lean_project,
):
    # Given
    proof_path = write_proof(
        lean_project,
        """
        theorem candidateTarget : True := True.intro
        """,
    )
    audit_dirs = []
    process_events = []

    class TimedOutProcess:
        returncode = None

        def communicate(self, *, timeout):
            raise subprocess.TimeoutExpired(["lake", "env", "lean"], timeout)

        def kill(self):
            process_events.append("kill")

        def wait(self):
            process_events.append("wait")
            self.returncode = -9

    def popen(command, **kwargs):
        root = type(proof_path)(command[command.index("-R") + 1])
        audit_dirs.append(root.parent)
        assert (root / "Synthetic" / "AxiomAudit.lean").is_file()
        return TimedOutProcess()

    monkeypatch.setattr(lean_core.subprocess, "Popen", popen)

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("candidateTarget",)
    )

    # Then
    assert success is False
    assert "timed out" in feedback.lower()
    assert process_events == ["kill", "wait"]
    assert proof_path.is_file()
    assert not audit_dirs[0].exists()


_CLEAN_PROOF = "theorem candidateTarget : True := True.intro"


def test_clean_final_audit_accepts_relative_project_and_candidate_paths(
    monkeypatch,
    lean_project,
):
    # Given
    theorem_text = "theorem candidateTarget : True := by\n  sorry\n"
    proof_path = write_proof(lean_project, _CLEAN_PROOF)
    parent = lean_project.parent
    relative_project = lean_project.relative_to(parent)
    relative_proof = proof_path.relative_to(parent)
    monkeypatch.chdir(parent)

    # When
    success, feedback, _cmd_info = lean_core.run_final_lean_axiom_check(
        theorem_text, relative_proof, relative_project
    )

    # Then
    assert success is True
    assert feedback == ""
    assert proof_path.is_file()


def test_axiom_probe_cleans_audit_directory_when_popen_raises_subprocess_error(
    monkeypatch,
    lean_project,
):
    # Given
    proof_path = write_proof(lean_project, _CLEAN_PROOF)
    audit_dirs = []

    def failing_popen(command, **kwargs):
        root = type(proof_path)(command[command.index("-R") + 1])
        audit_dirs.append(root.parent)
        assert (root / "Synthetic" / "AxiomAudit.lean").is_file()
        raise subprocess.SubprocessError("process launch failed")

    monkeypatch.setattr(lean_core.subprocess, "Popen", failing_popen)

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("candidateTarget",)
    )

    # Then
    assert success is False
    assert "could not start" in feedback.lower()
    assert proof_path.is_file()
    assert not audit_dirs[0].exists()


def test_axiom_probe_cleans_process_and_audit_directory_on_keyboard_interrupt(
    monkeypatch,
    lean_project,
):
    # Given
    proof_path = write_proof(lean_project, _CLEAN_PROOF)
    audit_dirs = []
    process_events = []

    class InterruptedProcess:
        returncode = None

        def communicate(self, *, timeout):
            raise KeyboardInterrupt

        def kill(self):
            process_events.append("kill")

        def wait(self):
            process_events.append("wait")
            self.returncode = -9

    def popen(command, **kwargs):
        root = type(proof_path)(command[command.index("-R") + 1])
        audit_dirs.append(root.parent)
        assert (root / "Synthetic" / "AxiomAudit.lean").is_file()
        return InterruptedProcess()

    monkeypatch.setattr(lean_core.subprocess, "Popen", popen)

    # When
    with pytest.raises(KeyboardInterrupt):
        run_axiom_check(proof_path, lean_project, ("candidateTarget",))

    # Then
    assert process_events == ["kill", "wait"]
    assert proof_path.is_file()
    assert not audit_dirs[0].exists()
