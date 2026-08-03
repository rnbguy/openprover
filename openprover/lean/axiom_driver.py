import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter, ValidationError

_AUDIT_SENTINEL: Final = "OPENPROVER_AXIOM_AUDIT:"
_TARGETS_SENTINEL: Final = "OPENPROVER_AXIOM_TARGETS:"
_TARGETS_ADAPTER: Final = TypeAdapter(tuple[str, ...])
_MAX_BYTES: Final = 16384 * 1024 * 1024
_TIMEOUT_SECONDS: Final = 300
_REPORTER: Final = r'''import Lean.Environment
import Lean.CoreM
import Lean.Util.CollectAxioms
import Lean.Util.Path
import Lean.Data.Json

open Lean

def loadArtifact (root : System.FilePath) : IO Environment := do
  let previous <- searchPathRef.get
  searchPathRef.set (root :: previous)
  try
    let expected := (root / "Synthetic" / "AxiomAudit.olean").normalize
    let actual <- findOLean `Synthetic.AxiomAudit
    unless actual.normalize == expected do
      throw <| IO.userError s!"artifact identity mismatch: {actual} != {expected}"
    importModules #[{ module := `Synthetic.AxiomAudit }] {} 0 #[] false false .private
  finally
    searchPathRef.set previous

def inEnvironment {alpha : Type} (env : Environment) (action : CoreM alpha) : IO alpha :=
  action.toIO' { fileName := "AxiomReporter.lean", fileMap := default } { env }

def moduleDeclarations (env : Environment) : Array Name :=
  match env.getModuleIdx? `Synthetic.AxiomAudit with
  | none => #[]
  | some index => env.header.moduleData[index]!.constNames

def discover (env : Environment) : IO (Array String) :=
  inEnvironment env do
    let targets <- (moduleDeclarations (← getEnv)).filterM fun name =>
      return (← collectAxioms name).contains ``sorryAx
    let sorted := targets.toList.mergeSort fun left right => left.toString < right.toString
    pure <| (List.map (fun name => name.toString) sorted).toArray

def result (target : String) (found : Bool) (axioms : Array Name) : Json :=
  Json.mkObj [("target", Json.str target), ("found", Json.bool found), ("axioms", Json.arr (axioms.map fun name => Json.str name.toString))]

def audit (env : Environment) (targets : Array String) : IO (Array Json) :=
  inEnvironment env do
    targets.mapM fun target => do
      let name := target.toName
      let env <- getEnv
      match (env.setExporting false).find? name with
      | none => pure <| result target false #[]
      | some _ =>
        let axioms <- collectAxioms name
        pure <| result target true axioms

def main (args : List String) : IO Unit := do
  match args with
  | "--discover" :: root :: "Synthetic.AxiomAudit" :: [] =>
    let targets <- discover (← loadArtifact root)
    IO.println ("OPENPROVER_AXIOM_TARGETS:" ++ (Json.arr (targets.map Json.str)).compress)
  | "--audit" :: root :: "Synthetic.AxiomAudit" :: targets =>
    let results <- audit (← loadArtifact root) targets.toArray
    IO.println ("OPENPROVER_AXIOM_AUDIT:" ++ (Json.arr results).compress)
  | _ => throw <| IO.userError "usage: AxiomReporter.lean --discover|--audit ROOT Synthetic.AxiomAudit [TARGET...]"
'''


@dataclass(frozen=True, slots=True)
class DriverResult:
    success: bool
    payload: str
    cmd_info: str
    targets: tuple[str, ...] = ()


def _kill_and_reap(process: subprocess.Popen[str]) -> bool:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, OSError):
        try:
            process.kill()
        except OSError:
            killed = False
        else:
            killed = True
    else:
        killed = True
    try:
        _ = process.wait()
    except OSError:
        return False
    return killed or process.returncode is not None


def _setup_process() -> None:
    import resource

    os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (_MAX_BYTES, _MAX_BYTES))


def _run_process(
    label: str, command: list[str], project_dir: Path
) -> tuple[bool, str, str]:
    cwd = str(project_dir)
    cmd_info = f"{label}\ncwd: {cwd}\ncmd: {' '.join(command)}"
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_setup_process,  # noqa: PLW1509
        )
    except FileNotFoundError:
        return False, "lake command not found - is Lean/Lake installed and on PATH?", cmd_info
    except (OSError, subprocess.SubprocessError):
        return False, "Lean axiom audit process could not start", cmd_info
    cleaned = True
    try:
        stdout, stderr = process.communicate(timeout=_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        cleaned = _kill_and_reap(process)
        feedback = "Lean axiom audit timed out after 300s"
        if not cleaned:
            feedback += "; process cleanup failed"
        return False, feedback, cmd_info
    except (OSError, subprocess.SubprocessError):
        cleaned = _kill_and_reap(process)
        feedback = "Lean axiom audit communication failed"
        if not cleaned:
            feedback += "; process cleanup failed"
        return False, feedback, cmd_info
    finally:
        if process.returncode is None:
            _ = _kill_and_reap(process)
    if process.returncode != 0:
        feedback = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
        return False, feedback or "Lean axiom audit process failed", cmd_info
    return True, stdout, cmd_info


def _compile(source: str, root: Path, project_dir: Path) -> tuple[bool, str, str]:
    source_path = root / "Synthetic" / "AxiomAudit.lean"
    olean_path = root / "Synthetic" / "AxiomAudit.olean"
    source_path.parent.mkdir(parents=True)
    _ = source_path.write_text(source)
    command = ["lake", "env", "lean", "-j", "1", "-t", "0", "-R", str(root),
               "-o", str(olean_path), str(source_path)]
    success, feedback, cmd_info = _run_process("Lean axiom compiler", command, project_dir)
    if success and (not olean_path.is_file() or olean_path.stat().st_size == 0):
        return False, "Lean axiom compiler artifact missing or empty", cmd_info
    return success, feedback, cmd_info


def _report(
    mode: str, root: Path, project_dir: Path, targets: tuple[str, ...] = ()
) -> tuple[bool, str, str]:
    reporter_path = root.parent / "AxiomReporter.lean"
    _ = reporter_path.write_text(_REPORTER)
    command = [
        "lake",
        "env",
        "lean",
        "-j",
        "1",
        "-t",
        "0",
        "--run",
        str(reporter_path),
        mode,
        str(root),
        "Synthetic.AxiomAudit",
        *targets,
    ]
    return _run_process("Lean axiom reporter", command, project_dir)


def _tagged_payload(stdout: str, sentinel: str) -> tuple[bool, str]:
    payloads = [
        line.strip().removeprefix(sentinel)
        for line in stdout.splitlines()
        if line.strip().startswith(sentinel)
    ]
    if not payloads:
        feedback = "Lean axiom probe output missing" if not stdout.strip() else "Lean axiom audit probe output malformed"
        return False, feedback
    if len(payloads) != 1:
        return False, "Lean axiom audit probe output malformed"
    return True, payloads[0]


def _discovered_targets(stdout: str, cmd_info: str) -> DriverResult:
    tagged, payload = _tagged_payload(stdout, _TARGETS_SENTINEL)
    if not tagged:
        return DriverResult(False, payload, cmd_info)
    try:
        targets = _TARGETS_ADAPTER.validate_json(payload, strict=True)
    except ValidationError:
        return DriverResult(False, "Lean target discovery output malformed", cmd_info)
    if not targets:
        return DriverResult(False, "Lean target discovery produced no targets", cmd_info)
    if len(set(targets)) != len(targets):
        return DriverResult(False, "Lean target discovery output malformed", cmd_info)
    return DriverResult(True, "", cmd_info, targets)


def run_direct(
    source_path: Path, project_dir: Path, target_names: tuple[str, ...]
) -> DriverResult:
    if not target_names:
        return DriverResult(False, "Lean axiom audit requires at least one target", "")
    return _run_pipeline(source_path, project_dir, target_names, None)


def run_final(source: str, source_path: Path, project_dir: Path) -> DriverResult:
    return _run_pipeline(source_path, project_dir, (), source)


def _run_pipeline(
    source_path: Path,
    project_dir: Path,
    target_names: tuple[str, ...],
    original_source: str | None,
) -> DriverResult:
    cmd_infos: list[str] = []
    try:
        resolved_project_dir = project_dir.resolve()
        resolved_source_path = source_path.resolve()
        base_parent = resolved_source_path.parent
        candidate_source = resolved_source_path.read_text()
        with tempfile.TemporaryDirectory(prefix="axiom_probe_", dir=base_parent) as directory:
            work_dir = Path(directory).resolve()
            if original_source is not None:
                original_root = work_dir / "original"
                success, feedback, cmd_info = _compile(
                    original_source, original_root, resolved_project_dir
                )
                cmd_infos.append(cmd_info)
                if not success:
                    return DriverResult(False, feedback, "\n\n".join(cmd_infos))
                success, stdout, cmd_info = _report(
                    "--discover", original_root, resolved_project_dir
                )
                cmd_infos.append(cmd_info)
                if not success:
                    return DriverResult(False, stdout, "\n\n".join(cmd_infos))
                discovery = _discovered_targets(stdout, "\n\n".join(cmd_infos))
                if not discovery.success:
                    return discovery
                target_names = discovery.targets
            candidate_root = work_dir / "candidate"
            success, feedback, cmd_info = _compile(
                candidate_source, candidate_root, resolved_project_dir
            )
            cmd_infos.append(cmd_info)
            if not success:
                return DriverResult(False, feedback, "\n\n".join(cmd_infos))
            success, stdout, cmd_info = _report(
                "--audit", candidate_root, resolved_project_dir, target_names
            )
            cmd_infos.append(cmd_info)
            if not success:
                return DriverResult(False, stdout, "\n\n".join(cmd_infos))
            tagged, payload = _tagged_payload(stdout, _AUDIT_SENTINEL)
            if not tagged:
                return DriverResult(False, payload, "\n\n".join(cmd_infos))
            return DriverResult(True, payload, "\n\n".join(cmd_infos), target_names)
    except (OSError, UnicodeError):
        return DriverResult(False, "Lean axiom audit temporary artifact error", "\n\n".join(cmd_infos))
