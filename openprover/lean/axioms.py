from pathlib import Path
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from .axiom_driver import run_direct, run_final

_APPROVED_AXIOMS: Final = frozenset({"propext", "Classical.choice", "Quot.sound"})


class _ProbeResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)

    target: str
    found: bool
    axioms: tuple[str, ...]


_RESULTS_ADAPTER: Final = TypeAdapter(tuple[_ProbeResult, ...])


def _audit_results(
    payload: str, cmd_info: str, target_names: tuple[str, ...]
) -> tuple[bool, str, str]:
    try:
        results = _RESULTS_ADAPTER.validate_json(payload, strict=True)
    except ValidationError:
        return False, "Lean axiom audit probe output malformed", cmd_info
    if not results:
        return False, "Lean target discovery produced no targets", cmd_info
    if tuple(result.target for result in results) != target_names:
        return False, "Lean axiom audit probe output missing or malformed", cmd_info
    if len({result.target for result in results}) != len(results):
        return False, "Lean axiom audit probe output malformed", cmd_info
    if any(not result.found and result.axioms for result in results):
        return False, "Lean axiom audit probe output malformed", cmd_info
    if any(len(set(result.axioms)) != len(result.axioms) for result in results):
        return False, "Lean axiom audit probe output malformed", cmd_info
    failures: list[str] = []
    for result in results:
        if not result.found:
            failures.append(f"{result.target}: missing Lean target")
            continue
        disallowed = sorted(set(result.axioms) - _APPROVED_AXIOMS)
        if disallowed:
            failures.append(
                f"{result.target}: disallowed axioms: {', '.join(disallowed)}"
            )
    if failures:
        return False, "\n".join(sorted(failures)), cmd_info
    return True, "", cmd_info


def run_lean_axiom_check(
    lean_file: Path, project_dir: Path, target_names: tuple[str, ...]
) -> tuple[bool, str, str]:
    """Audit the transitive axioms of exact Lean declaration names."""
    result = run_direct(lean_file, project_dir, target_names)
    if not result.success:
        return False, result.payload, result.cmd_info
    return _audit_results(result.payload, result.cmd_info, result.targets)


def run_final_lean_axiom_check(
    theorem_text: str, proof_path: Path, project_dir: Path
) -> tuple[bool, str, str]:
    """Discover original proof targets and audit them in a candidate module."""
    result = run_final(theorem_text, proof_path, project_dir)
    if not result.success:
        return False, result.payload, result.cmd_info
    return _audit_results(result.payload, result.cmd_info, result.targets)
