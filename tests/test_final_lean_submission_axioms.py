import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import pytest

from openprover.lean import core as lean_core
from openprover.lean.core import LeanWorkDir
from openprover.prover import Prover, Repo

MULTI_TARGET_THEOREM: Final = textwrap.dedent(
    """
    namespace AuditCase

    theorem cleanTarget : True := by
      sorry

    theorem forgedTarget : True := by
      sorry

    end AuditCase
    """
).strip() + "\n"

SUBMISSION_THEOREM: Final = "theorem submissionTarget : True := by\n  sorry\n"


@pytest.fixture
def lean_project(tmp_path: Path) -> Path:
    (tmp_path / "lakefile.lean").write_text(
        "import Lake\nopen Lake DSL\n\n"
        "package finalAxiomFixture\n\n"
        "lean_lib FixtureDependency\n"
    )
    return tmp_path


def write_checked_source(project_dir: Path, source: str) -> Path:
    proof_path = LeanWorkDir(project_dir).make_file(
        "final-axiom", textwrap.dedent(source).strip() + "\n"
    )
    success, feedback, _cmd_info = lean_core.run_lean_check(proof_path, project_dir)
    assert success, feedback
    return proof_path


def run_final_axiom_check(
    theorem_text: str,
    proof_path: Path,
    project_dir: Path,
) -> tuple[bool, str, str]:
    check = getattr(lean_core, "run_final_lean_axiom_check", None)
    if check is None:
        pytest.fail("expected lean_core.run_final_lean_axiom_check API is not implemented")
    return check(theorem_text, proof_path, project_dir)


def build_sorry_dependency(project_dir: Path) -> None:
    (project_dir / "FixtureDependency.lean").write_text(
        "theorem importedSorryTarget : True := by\n  sorry\n"
    )
    result = subprocess.run(
        ["lake", "build", "FixtureDependency"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def make_submission_prover(
    monkeypatch: pytest.MonkeyPatch,
    project_dir: Path,
    proof_text: str,
) -> tuple[Prover, Path, list[str]]:
    run_dir = project_dir / "run"
    run_dir.mkdir()
    step_dir = run_dir / "step"
    step_dir.mkdir()
    repo = Repo(run_dir / "repo")
    repo.write_item("candidate", proof_text, fmt="lean")

    prover = Prover.__new__(Prover)
    prover.work_dir = run_dir
    prover.repo = repo
    prover.lean_project_dir = project_dir
    prover.lean_work_dir = LeanWorkDir(project_dir)
    prover.lean_theorem_text = SUBMISSION_THEOREM
    prover.__dict__["tui"] = SimpleNamespace(log=lambda *_args, **_kwargs: None)
    outputs: list[str] = []
    monkeypatch.setattr(prover, "_push_output", outputs.append)
    monkeypatch.setattr(prover, "_check_completion", lambda _feedback: "stop")
    return prover, step_dir, outputs


def test_final_check_accepts_namespace_qualified_multiple_targets(
    lean_project: Path,
) -> None:
    # Given
    proof_path = write_checked_source(
        lean_project,
        """
        namespace AuditCase
        theorem cleanTarget : True := True.intro
        theorem forgedTarget : True := True.intro
        end AuditCase
        """,
    )

    # When
    success, feedback, _cmd_info = run_final_axiom_check(
        MULTI_TARGET_THEOREM, proof_path, lean_project
    )

    # Then
    assert success is True, feedback
    assert feedback == ""


def test_final_check_rejects_only_forged_namespace_target(
    lean_project: Path,
) -> None:
    # Given
    proof_path = write_checked_source(
        lean_project,
        """
        namespace AuditCase
        axiom forgedSource : True
        theorem cleanTarget : True := True.intro
        theorem forgedTarget : True := forgedSource
        end AuditCase
        """,
    )

    # When
    success, feedback, _cmd_info = run_final_axiom_check(
        MULTI_TARGET_THEOREM, proof_path, lean_project
    )

    # Then
    assert success is False
    assert "AuditCase.forgedTarget" in feedback
    assert "AuditCase.forgedSource" in feedback
    assert "AuditCase.cleanTarget" not in feedback


def test_final_check_ignores_imported_sorry_backed_declarations(
    lean_project: Path,
) -> None:
    # Given
    build_sorry_dependency(lean_project)
    theorem_text = "import FixtureDependency\n\ntheorem localTarget : True := by\n  sorry\n"
    proof_path = write_checked_source(
        lean_project,
        """
        import FixtureDependency
        theorem localTarget : True := True.intro
        """,
    )

    # When
    success, feedback, _cmd_info = run_final_axiom_check(
        theorem_text, proof_path, lean_project
    )

    # Then
    assert success is True
    assert feedback == ""


def test_final_check_accepts_private_sorry_target_replacement(
    lean_project: Path,
) -> None:
    # Given
    theorem_text = "private theorem hiddenTarget : True := by\n  sorry\n"
    proof_path = write_checked_source(
        lean_project,
        "private theorem hiddenTarget : True := True.intro",
    )

    # When
    success, feedback, _cmd_info = run_final_axiom_check(
        theorem_text, proof_path, lean_project
    )

    # Then
    assert success is True, feedback
    assert feedback == ""


def test_final_check_fails_closed_when_discovery_does_not_elaborate(
    lean_project: Path,
) -> None:
    # Given
    theorem_text = (
        "theorem discoveryFailure : True := by\n"
        "  exact missingDiscoveryWitness\n"
    )
    proof_path = write_checked_source(
        lean_project, "theorem discoveryFailure : True := True.intro"
    )

    # When
    success, feedback, cmd_info = run_final_axiom_check(
        theorem_text, proof_path, lean_project
    )

    # Then
    assert success is False
    assert feedback
    assert "lake env lean" in cmd_info


def test_submit_keeps_structural_rejection_before_final_audit(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    proof_text = (
        "axiom renamedSource : True\n\n"
        "theorem renamedTarget : True := renamedSource\n"
    )
    write_checked_source(lean_project, proof_text)
    prover, step_dir, outputs = make_submission_prover(monkeypatch, lean_project, proof_text)

    # When
    result = prover._handle_submit_lean_proof({"lean_proof_slug": "candidate"}, step_dir)

    # Then
    assert result == "continue"
    assert outputs
    assert "the theorem header" in outputs[0].lower()
    assert not (prover.work_dir / "PROOF.lean").exists()


def test_submit_rejects_compiling_structurally_valid_custom_axiom_proof(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    proof_text = (
        "axiom forgedSubmission : True\n\n"
        "theorem submissionTarget : True := by\n"
        "  exact forgedSubmission\n"
    )
    write_checked_source(lean_project, proof_text)
    assert Prover._check_proof_preserves_theorem(SUBMISSION_THEOREM, proof_text) is None
    prover, step_dir, _outputs = make_submission_prover(monkeypatch, lean_project, proof_text)

    # When
    result = prover._handle_submit_lean_proof({"lean_proof_slug": "candidate"}, step_dir)

    # Then
    assert prover.lean_work_dir is not None
    assert (
        result,
        (prover.work_dir / "PROOF.lean").exists(),
        (prover.lean_work_dir.dir / "PROOF.lean").exists(),
    ) == ("continue", False, False)


def test_submit_accepts_compiling_axiom_free_proof(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    proof_text = "theorem submissionTarget : True := by\n  exact True.intro\n"
    write_checked_source(lean_project, proof_text)
    assert Prover._check_proof_preserves_theorem(SUBMISSION_THEOREM, proof_text) is None
    prover, step_dir, _outputs = make_submission_prover(monkeypatch, lean_project, proof_text)

    # When
    result = prover._handle_submit_lean_proof({"lean_proof_slug": "candidate"}, step_dir)

    # Then
    assert result == "stop"
    assert prover.lean_work_dir is not None
    assert (prover.work_dir / "PROOF.lean").read_text() == proof_text
    assert (prover.lean_work_dir.dir / "PROOF.lean").read_text() == proof_text
