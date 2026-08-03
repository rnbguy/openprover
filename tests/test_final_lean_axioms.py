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


def write_checked_proof(project_dir, source):
    proof_path = write_proof(project_dir, source)
    success, feedback, _cmd_info = lean_core.run_lean_check(proof_path, project_dir)
    assert success, feedback
    return proof_path


def write_dependency(project_dir):
    dependency = project_dir / "FixtureDependency.lean"
    dependency.write_text(
        "axiom importedAxiom : True\n\n"
        "theorem importedBridge : True := importedAxiom\n"
    )
    result = subprocess.run(
        ["lake", "build", "FixtureDependency"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def run_axiom_check(proof_path, project_dir, target_names):
    check = getattr(lean_core, "run_lean_axiom_check", None)
    if check is None:
        pytest.fail("expected openprover.lean.core.run_lean_axiom_check API is not implemented")
    return check(proof_path, project_dir, target_names)


def test_axiom_free_target_is_accepted(lean_project):
    # Given
    proof_path = write_checked_proof(
        lean_project,
        """
        theorem cleanTarget : True := True.intro
        """,
    )

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("cleanTarget",)
    )

    # Then
    assert success is True
    assert feedback == ""


def test_all_approved_standard_axioms_are_accepted(lean_project):
    # Given
    proof_path = write_checked_proof(
        lean_project,
        r"""
        theorem approvedTarget
            (p q : Prop) (equiv : p <-> q) (inhabited : Nonempty p)
            (r : Nat -> Nat -> Prop) (a b : Nat) (related : r a b) :
            p /\ p = q /\ Quot.mk r a = Quot.mk r b := by
          exact And.intro (Classical.choice inhabited)
            (And.intro (propext equiv) (Quot.sound related))
        """,
    )

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("approvedTarget",)
    )

    # Then
    assert success is True
    assert feedback == ""


def test_direct_custom_axiom_is_rejected(lean_project):
    # Given
    proof_path = write_checked_proof(
        lean_project,
        """
        axiom directAxiom : True

        theorem directTarget : True := directAxiom
        """,
    )

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("directTarget",)
    )

    # Then
    assert success is False
    assert "directAxiom" in feedback


def test_private_custom_axiom_is_rejected(lean_project):
    # Given
    proof_path = write_checked_proof(
        lean_project,
        """
        private axiom hiddenAxiom : True

        theorem privateTarget : True := hiddenAxiom
        """,
    )

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("privateTarget",)
    )

    # Then
    assert success is False
    assert "hiddenAxiom" in feedback


def test_imported_transitive_custom_axiom_is_rejected(lean_project):
    # Given
    write_dependency(lean_project)
    proof_path = write_checked_proof(
        lean_project,
        """
        import FixtureDependency

        theorem importedTarget : True := importedBridge
        """,
    )

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("importedTarget",)
    )

    # Then
    assert success is False
    assert "importedAxiom" in feedback


def test_unused_imported_custom_axiom_is_ignored(lean_project):
    # Given
    write_dependency(lean_project)
    proof_path = write_checked_proof(
        lean_project,
        """
        import FixtureDependency

        theorem independentTarget : True := True.intro
        """,
    )

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("independentTarget",)
    )

    # Then
    assert success is True
    assert feedback == ""


def test_mixed_targets_report_the_failing_target_and_axiom(lean_project):
    # Given
    proof_path = write_checked_proof(
        lean_project,
        """
        axiom mixedAxiom : True

        theorem cleanMixedTarget : True := True.intro
        theorem failingMixedTarget : True := mixedAxiom
        """,
    )

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path,
        lean_project,
        ("cleanMixedTarget", "failingMixedTarget"),
    )

    # Then
    assert success is False
    assert "failingMixedTarget" in feedback
    assert "mixedAxiom" in feedback


def test_missing_target_is_rejected(lean_project):
    # Given
    proof_path = write_checked_proof(
        lean_project,
        """
        theorem presentTarget : True := True.intro
        """,
    )

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("missingTarget",)
    )

    # Then
    assert success is False
    assert "missingTarget" in feedback


def test_retained_sorry_axiom_is_rejected(lean_project):
    # Given
    proof_path = write_proof(
        lean_project,
        """
        theorem unfinishedTarget : True := sorryAx True true
        """,
    )
    lean_success, lean_feedback, _cmd_info = lean_core.run_lean_check(proof_path, lean_project)
    assert lean_success is False
    assert "declaration uses `sorry`" in lean_feedback

    # When
    success, feedback, _cmd_info = run_axiom_check(
        proof_path, lean_project, ("unfinishedTarget",)
    )

    # Then
    assert success is False
    assert "sorryAx" in feedback


@pytest.mark.parametrize(
    ("probe_stdout", "expected_fragment"),
    [
        ("", "missing"),
        ("not a structured axiom audit record", "malformed"),
        (
            '[{"target":"probeTarget","found":true,"axioms":[]}]',
            "malformed",
        ),
    ],
)
def test_missing_or_malformed_probe_output_is_rejected(
    monkeypatch,
    lean_project,
    probe_stdout,
    expected_fragment,
):
    # Given
    proof_path = write_checked_proof(
        lean_project,
        """
        theorem probeTarget : True := True.intro
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
        proof_path, lean_project, ("probeTarget",)
    )

    # Then
    assert success is False
    assert expected_fragment in feedback.lower()


def test_axiom_probe_uses_single_lean_thread(monkeypatch, lean_project):
    # Given
    proof_path = write_checked_proof(
        lean_project,
        """
        theorem probeTarget : True := True.intro
        """,
    )
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        if "--run" not in command:
            type(proof_path)(command[command.index("-o") + 1]).write_bytes(b"olean")
            stdout = ""
        else:
            stdout = 'OPENPROVER_AXIOM_AUDIT:[{"target":"probeTarget","found":true,"axioms":[]}]'
        return SimpleNamespace(
            returncode=0,
            communicate=lambda *, timeout: (stdout, ""),
        )

    monkeypatch.setattr(lean_core.subprocess, "Popen", popen)

    # When
    run_axiom_check(proof_path, lean_project, ("probeTarget",))

    # Then
    assert commands[0][3:5] == ["-j", "1"]
