import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import pytest

from openprover.inspect import _load_lean_pages
from openprover.lean.core import LeanWorkDir
from openprover.prover import Prover, Repo

SUBMISSION_THEOREM: Final = (
    "import FixtureDependency\n\n"
    "theorem submissionTarget : True := by\n"
    "  sorry\n"
)
SUBMISSION_PROOF: Final = (
    "import FixtureDependency\n\n"
    "theorem submissionTarget : True := by\n"
    "  exact importedBridge\n"
)


@pytest.fixture
def lean_project(tmp_path: Path) -> Path:
    (tmp_path / "lakefile.lean").write_text(
        "import Lake\nopen Lake DSL\n\n"
        "package finalSubmissionInspectFixture\n\n"
        "lean_lib FixtureDependency\n"
    )
    return tmp_path


def build_transitive_axiom_dependency(project_dir: Path) -> None:
    (project_dir / "FixtureDependency.lean").write_text(
        "axiom importedTransitiveAxiom : True\n\n"
        "theorem importedBridge : True := importedTransitiveAxiom\n"
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
) -> tuple[Prover, Path, list[str]]:
    run_dir = project_dir / "run"
    step_dir = run_dir / "steps" / "step_001"
    step_dir.mkdir(parents=True)
    repo = Repo(run_dir / "repo")
    repo.write_item("candidate", SUBMISSION_PROOF, fmt="lean")

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


def test_inspect_marks_imported_transitive_axiom_submission_failed(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    build_transitive_axiom_dependency(lean_project)
    assert Prover._check_proof_preserves_theorem(
        SUBMISSION_THEOREM, SUBMISSION_PROOF
    ) is None
    prover, step_dir, outputs = make_submission_prover(monkeypatch, lean_project)

    # When
    result = prover._handle_submit_lean_proof(
        {"lean_proof_slug": "candidate"}, step_dir
    )

    # Then
    assert prover.lean_work_dir is not None
    assert result == "continue"
    assert not (prover.work_dir / "PROOF.lean").exists()
    assert not (prover.lean_work_dir.dir / "PROOF.lean").exists()
    assert "importedTransitiveAxiom" in "\n".join(outputs)

    lean_dir = step_dir / "lean"
    assert (lean_dir / "proof_attempt.lean").read_text() == SUBMISSION_PROOF
    assert (lean_dir / "proof_result.txt").read_text().strip() == "OK"
    axiom_result = (lean_dir / "proof_axiom_result.txt").read_text()
    assert "importedTransitiveAxiom" in axiom_result

    pages = _load_lean_pages(step_dir, 1)

    assert len(pages) == 1
    assert pages[0]["type"] == "lean_err"
    assert pages[0]["metadata"] == "FAILED"
    rendered = "".join(text for _style, text in pages[0]["segments"])
    assert "importedTransitiveAxiom" in rendered


@pytest.mark.parametrize(
    "axiom_result",
    [None, "OK\n"],
    ids=["absent-audit", "successful-audit"],
)
def test_inspect_keeps_successful_compile_archives_lean_ok(
    tmp_path: Path,
    axiom_result: str | None,
) -> None:
    # Given
    step_dir = tmp_path / "steps" / "step_007"
    lean_dir = step_dir / "lean"
    lean_dir.mkdir(parents=True)
    (lean_dir / "proof_attempt.lean").write_text(SUBMISSION_PROOF)
    (lean_dir / "proof_result.txt").write_text("OK\n")
    if axiom_result is not None:
        (lean_dir / "proof_axiom_result.txt").write_text(axiom_result)

    # When
    pages = _load_lean_pages(step_dir, 7)

    # Then
    assert len(pages) == 1
    assert pages[0]["type"] == "lean_ok"
    assert pages[0]["metadata"] == "OK"


def test_inspect_marks_interrupted_final_axiom_audit_failed(
    monkeypatch: pytest.MonkeyPatch,
    lean_project: Path,
) -> None:
    # Given
    prover, step_dir, _outputs = make_submission_prover(monkeypatch, lean_project)
    prover.repo.write_item(
        "candidate",
        "theorem submissionTarget : True := by\n  exact True.intro\n",
        fmt="lean",
    )
    marker = "Lean axiom audit incomplete"
    interruption = KeyboardInterrupt(marker)

    def interrupt_axiom_check(*_args: object, **_kwargs: object) -> None:
        raise interruption

    monkeypatch.setattr(
        "openprover.prover.run_final_lean_axiom_check",
        interrupt_axiom_check,
    )

    # When
    with pytest.raises(KeyboardInterrupt) as raised:
        prover._handle_submit_lean_proof(
            {"lean_proof_slug": "candidate"}, step_dir
        )

    # Then
    assert raised.value is interruption
    assert prover.lean_work_dir is not None
    assert not (prover.work_dir / "PROOF.lean").exists()
    assert not (prover.lean_work_dir.dir / "PROOF.lean").exists()

    lean_dir = step_dir / "lean"
    assert (lean_dir / "proof_result.txt").read_text() == "OK"
    assert (lean_dir / "proof_axiom_result.txt").read_text() == marker
    assert not (lean_dir / "proof_axiom_cmd.txt").exists()

    pages = _load_lean_pages(step_dir, 1)

    assert len(pages) == 1
    assert pages[0]["type"] == "lean_err"
    assert pages[0]["metadata"] == "FAILED"
    rendered = "".join(text for _style, text in pages[0]["segments"])
    assert marker in rendered
