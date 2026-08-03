import pytest

from openprover import cli
from openprover.prover import Prover


@pytest.mark.parametrize(
    ("mode", "artifacts", "finished"),
    [
        ("prove", (), False),
        ("prove", ("PROOF.md",), True),
        ("formalize_only", (), False),
        ("formalize_only", ("PROOF.lean",), True),
        ("prove_and_formalize", ("PROOF.md",), False),
        ("prove_and_formalize", ("PROOF.lean",), False),
        ("prove_and_formalize", ("PROOF.md", "PROOF.lean"), True),
    ],
)
def test_is_finished_uses_mode_proof_artifacts(tmp_path, mode, artifacts, finished):
    for artifact in artifacts:
        (tmp_path / artifact).touch()
    (tmp_path / "DISCUSSION.md").touch()

    assert cli._is_finished(tmp_path, mode) is finished


class _Budget:
    def __init__(self, *, exhausted=False, conclude=False):
        self.exhausted = exhausted
        self.conclude = conclude
        self.conclude_after = 0.99

    def limit_str(self):
        return "test"

    def is_exhausted(self):
        return self.exhausted

    def should_conclude(self):
        return self.conclude


class _TUI:
    def __init__(self):
        self.step_entries = [{}]

    def setup(self, **_kwargs):
        pass

    def log(self, *_args, **_kwargs):
        pass


def _prover(tmp_path, *, budget=None):
    prover = Prover.__new__(Prover)
    prover.work_dir = tmp_path
    prover.__dict__["budget"] = budget or _Budget()
    prover.__dict__["tui"] = _TUI()
    prover.autonomous = True
    prover.theorem_name = "theorem"
    prover.model = "fake"
    prover.whiteboard = "whiteboard"
    prover.max_workers = 1
    prover.isolation = False
    prover.mode = "prove"
    prover.resumed = False
    prover.shutting_down = False
    prover._spending_limit_hit = False
    prover._llm_error_exit = False
    prover.step_num = 0
    prover.step_history = []
    prover._current_planner_result = "planner"
    prover._current_step_action = "submit_proof"
    prover._current_step_summary = ""
    prover._current_action_outputs = []
    prover._consecutive_errors = 0
    prover._steps_since_productive = 0
    prover._setup_tui_logging = lambda: None
    prover._save_step_history = lambda: None
    prover._maybe_respawn_interrupted_workers = lambda: None
    prover._load_history = lambda: None
    discussion_calls = []
    prover._write_discussion = lambda: discussion_calls.append(None)
    return prover, discussion_calls


def test_run_writes_discussion_after_completed_proof(tmp_path):
    prover, discussion_calls = _prover(tmp_path)

    def do_step():
        prover._current_planner_result = "planner"
        (tmp_path / "PROOF.md").touch()
        return "stop"

    prover._do_step = do_step
    prover._run()

    assert len(discussion_calls) == 1


@pytest.mark.parametrize(
    "reason",
    ["budget_conclusion", "budget_exhaustion", "spending_limit", "llm_error", "shutdown"],
)
def test_run_skips_discussion_for_unsuccessful_or_stopped_run(tmp_path, reason):
    budget = _Budget(
        exhausted=reason == "budget_exhaustion",
        conclude=reason == "budget_conclusion",
    )
    prover, discussion_calls = _prover(tmp_path, budget=budget)
    prover._spending_limit_hit = reason == "spending_limit"
    prover._llm_error_exit = reason == "llm_error"
    prover.shutting_down = reason == "shutdown"

    def do_step():
        prover._current_planner_result = "planner"
        if reason == "budget_conclusion":
            (tmp_path / "PROOF.md").touch()
            return "continue"
        return "stop"

    prover._do_step = do_step
    prover._run()

    assert discussion_calls == []


def test_discussion_only_run_is_resumable_in_all_modes(tmp_path):
    (tmp_path / "DISCUSSION.md").touch()

    assert all(not cli._is_finished(tmp_path, mode) for mode in (
        "prove", "formalize_only", "prove_and_formalize",
    ))


@pytest.mark.parametrize(
    ("mode", "artifacts", "finished"),
    [
        ("prove", (), False),
        ("prove", ("PROOF.md",), True),
        ("formalize_only", (), False),
        ("formalize_only", ("PROOF.lean",), True),
        ("prove_and_formalize", (), False),
        ("prove_and_formalize", ("PROOF.md",), False),
        ("prove_and_formalize", ("PROOF.lean",), False),
        ("prove_and_formalize", ("PROOF.md", "PROOF.lean"), True),
    ],
)
def test_prover_is_finished_matches_cli(tmp_path, mode, artifacts, finished):
    for artifact in artifacts:
        (tmp_path / artifact).touch()
    (tmp_path / "DISCUSSION.md").touch()
    prover = Prover.__new__(Prover)
    prover.work_dir = tmp_path
    prover.mode = mode

    assert prover.is_finished is finished
    assert prover.is_finished is cli._is_finished(tmp_path, mode)
