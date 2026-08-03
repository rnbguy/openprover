from types import SimpleNamespace

import pytest

from openprover.prover import Prover


class FakeTUI:
    autonomous = True

    def __init__(self):
        self.step_entries = [{}]

    def add_worker_tab(self, *_args, **_kwargs):
        return SimpleNamespace(task_summary="")

    def mark_worker_done(self, *_args):
        pass

    def set_waiting_status(self, *_args):
        pass

    def snapshot_worker_tabs(self, *_args):
        pass

    def tab_log(self, *_args, **_kwargs):
        pass

    def update_step_status(self, *_args, **_kwargs):
        pass

    def _sync_step_log_line(self, *_args):
        pass

    def log(self, *_args, **_kwargs):
        pass


def make_prover(tmp_path, mode):
    prover = Prover.__new__(Prover)
    prover.autonomous = True
    prover._current_action_outputs = []
    prover._interrupt_count = 0
    prover._step_idx = 0
    prover._workers_active = False
    prover.max_workers = 1
    prover.planner_llm = SimpleNamespace(clear_interrupt=lambda: None)
    prover.worker_llm = SimpleNamespace(clear_interrupt=lambda: None)
    prover.shutting_down = False
    prover.step_num = 1
    prover.verifier = True
    prover.mode = mode
    prover.work_dir = tmp_path
    prover.__dict__["tui"] = FakeTUI()
    prover.__dict__["_push_output"] = lambda _text: None
    prover._llm_error_exit = False
    prover._last_error_msg = ""
    prover.__dict__["_run_worker"] = lambda *_args: {
        "result": "worker result", "cost": 0.0,
        "duration_ms": 0, "raw": {}, "error": "",
    }
    return prover


@pytest.mark.parametrize("mode", ["formalize_only", "prove_and_formalize"])
def test_formal_modes_skip_automatic_verifiers(tmp_path, mode):
    prover = make_prover(tmp_path, mode)

    def fail_if_called(*_args):
        pytest.fail("formal mode must not call automatic verifiers")

    prover.__dict__["_run_verifiers"] = fail_if_called

    prover._handle_spawn({"tasks": [{"description": "formal task"}]}, tmp_path)

    assert prover.verifier is True


def test_informal_prove_mode_runs_automatic_verifiers(tmp_path):
    prover = make_prover(tmp_path, "prove")
    calls = []
    prover.__dict__["_run_verifiers"] = lambda *_args: calls.append(None) or {}

    prover._handle_spawn({"tasks": [{"description": "informal task"}]}, tmp_path)

    assert len(calls) == 1
