import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import pytest

from openprover.lean import core as lean_core
from openprover.lean.core import LeanWorkDir
from openprover.prover import Prover, Repo

SUBMISSION_THEOREM: Final = "theorem submissionTarget : True := by\n  sorry\n"
MISSING_ARTIFACT_PROOF: Final = (
    "import Lean.Elab.Command\n"
    "open Lean\n\n"
    "run_cmd do\n"
    "  let env <- getEnv\n"
    "  if env.mainModule == `Synthetic.AxiomAudit then\n"
    '    IO.println "OPENPROVER_AXIOM_AUDIT:'
    '[{\\"target\\":\\"submissionTarget\\",\\"found\\":true,\\"axioms\\":[]}]"\n'
    "    IO.Process.exit 0\n\n"
    "theorem submissionTarget : True := by\n"
    "  exact True.intro\n"
)


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


def test_final_check_rejects_custom_axiom_despite_candidate_name_serializer(
    lean_project: Path,
) -> None:
    # Given
    proof_path = write_checked_source(
        lean_project,
        """
        import Lean.Data.Json
        open Lean

        local instance : ToJson Name where
          toJson _name := Json.str "propext"

        axiom customSource : True
        theorem serializedTarget : True := customSource
        """,
    )
    theorem_text = "theorem serializedTarget : True := by\n  sorry\n"

    # When
    success, feedback, _cmd_info = run_final_axiom_check(
        theorem_text, proof_path, lean_project
    )

    # Then
    assert success is False
    assert "customSource" in feedback


def test_final_check_ignores_candidate_run_cmd_macro(
    lean_project: Path,
) -> None:
    # Given
    proof_path = write_checked_source(
        lean_project,
        """
        import Lean.Elab.Command

        macro_rules (kind := Lean.runCmd)
          | `(run_cmd $_seq:doSeq) =>
              `(command| theorem shadowedAuditCommand : True := True.intro)

        theorem macroTarget : True := True.intro
        """,
    )
    theorem_text = "theorem macroTarget : True := by\n  sorry\n"

    # When
    success, feedback, _cmd_info = run_final_axiom_check(
        theorem_text, proof_path, lean_project
    )

    # Then
    assert success is True, feedback
    assert feedback == ""


def test_final_check_rejects_original_without_sorry_backed_declarations(
    lean_project: Path,
) -> None:
    # Given
    theorem_text = "theorem alreadyProved : True := True.intro\n"
    proof_path = write_checked_source(lean_project, theorem_text)

    # When
    success, feedback, _cmd_info = run_final_axiom_check(
        theorem_text, proof_path, lean_project
    )

    # Then
    assert success is False
    assert feedback


def test_submit_rejects_forged_axiom_despite_candidate_name_serializer(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    proof_text = textwrap.dedent(
        """
        import Lean.Data.Json
        open Lean

        local instance : ToJson Name where
          toJson _name := Json.str "propext"

        axiom customSource : True
        theorem submissionTarget : True := by
          exact customSource
        """
    ).strip() + "\n"
    write_checked_source(lean_project, proof_text)
    assert Prover._check_proof_preserves_theorem(SUBMISSION_THEOREM, proof_text) is None
    prover, step_dir, _outputs = make_submission_prover(
        monkeypatch, lean_project, proof_text
    )

    # When
    result = prover._handle_submit_lean_proof({"lean_proof_slug": "candidate"}, step_dir)

    # Then
    assert prover.lean_work_dir is not None
    axiom_result = step_dir / "lean" / "proof_axiom_result.txt"
    assert axiom_result.exists()
    archived_result = axiom_result.read_text().strip()
    assert archived_result
    assert archived_result != "OK"
    assert (
        result,
        (prover.work_dir / "PROOF.lean").exists(),
        (prover.lean_work_dir.dir / "PROOF.lean").exists(),
    ) == ("continue", False, False)


def test_submit_rejects_original_without_sorry_backed_declarations(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    proof_text = "theorem submissionTarget : True := True.intro\n"
    write_checked_source(lean_project, proof_text)
    assert Prover._check_proof_preserves_theorem(proof_text, proof_text) is None
    prover, step_dir, _outputs = make_submission_prover(
        monkeypatch, lean_project, proof_text
    )
    prover.lean_theorem_text = proof_text

    # When
    result = prover._handle_submit_lean_proof({"lean_proof_slug": "candidate"}, step_dir)

    # Then
    assert prover.lean_work_dir is not None
    axiom_result = step_dir / "lean" / "proof_axiom_result.txt"
    assert axiom_result.exists()
    archived_result = axiom_result.read_text().strip()
    assert archived_result
    assert archived_result != "OK"
    assert (
        result,
        (prover.work_dir / "PROOF.lean").exists(),
        (prover.lean_work_dir.dir / "PROOF.lean").exists(),
    ) == ("continue", False, False)


def test_final_check_rejects_compiler_report_when_olean_is_missing(
    lean_project: Path,
) -> None:
    # Given
    proof_path = write_checked_source(lean_project, MISSING_ARTIFACT_PROOF)

    # When
    success, feedback, _cmd_info = run_final_axiom_check(
        SUBMISSION_THEOREM, proof_path, lean_project
    )

    # Then
    assert success is False
    assert "artifact" in feedback.lower()
    assert proof_path.is_file()
    assert not tuple(proof_path.parent.glob("axiom_probe_*"))


def test_submit_rejects_compiler_report_when_olean_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    write_checked_source(lean_project, MISSING_ARTIFACT_PROOF)
    assert (
        Prover._check_proof_preserves_theorem(
            SUBMISSION_THEOREM, MISSING_ARTIFACT_PROOF
        )
        is None
    )
    prover, step_dir, _outputs = make_submission_prover(
        monkeypatch, lean_project, MISSING_ARTIFACT_PROOF
    )

    # When
    result = prover._handle_submit_lean_proof({"lean_proof_slug": "candidate"}, step_dir)

    # Then
    assert prover.lean_work_dir is not None
    lean_dir = step_dir / "lean"
    assert (lean_dir / "proof_result.txt").read_text().strip() == "OK"
    archived_result = (lean_dir / "proof_axiom_result.txt").read_text().strip()
    assert archived_result != "OK"
    assert "artifact" in archived_result.lower()
    assert (
        result,
        (prover.work_dir / "PROOF.lean").exists(),
        (prover.lean_work_dir.dir / "PROOF.lean").exists(),
    ) == ("continue", False, False)
    assert not tuple(prover.lean_work_dir.dir.glob("axiom_probe_*"))
