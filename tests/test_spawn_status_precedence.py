from types import SimpleNamespace

import pytest

from openprover.prover import MAX_CONSECUTIVE_ERRORS, Prover


class _TUI:
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


def _prover(tmp_path):
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
    prover.work_dir = tmp_path
    prover.__dict__["tui"] = _TUI()
    prover.__dict__["_push_output"] = lambda text: None
    prover._llm_error_exit = False
    prover._last_error_msg = "before"
    return prover


def _side_effect_counts(prover):
    counts = {"planner": 0, "worker": 0, "update": 0, "log": 0}
    prover.planner_llm.clear_interrupt = lambda: counts.__setitem__("planner", counts["planner"] + 1)
    prover.worker_llm.clear_interrupt = lambda: counts.__setitem__("worker", counts["worker"] + 1)
    prover.tui.update_step_status = lambda *_args, **_kwargs: counts.__setitem__("update", counts["update"] + 1)
    prover.tui.log = lambda *_args, **_kwargs: counts.__setitem__("log", counts["log"] + 1)
    return counts


def test_spawn_verifier_interruption_takes_precedence(tmp_path):
    prover = _prover(tmp_path)
    counts = _side_effect_counts(prover)
    prover.__dict__["_run_worker"] = lambda *args: {
        "result": "worker", "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": "",
    }
    prover.__dict__["_run_verifier"] = lambda *args: {
        "result": "interrupted", "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": "interrupted",
    }

    prover._handle_spawn({"tasks": [{"description": "task"}]}, tmp_path)

    assert prover._read_step_meta(tmp_path)["status"] == "interrupted"
    assert counts == {"planner": 1, "worker": 1, "update": 1, "log": 1}


def test_spawn_verifier_counter_interrupt_marks_successful_response_interrupted(tmp_path):
    prover = _prover(tmp_path)
    counts = _side_effect_counts(prover)
    prover.__dict__["_run_worker"] = lambda *args: {
        "result": "worker", "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": "",
    }

    def run_verifier(*args):
        prover._interrupt_count += 1
        return {"result": "verified", "cost": 0.0, "duration_ms": 0,
                "raw": {}, "error": ""}

    prover.__dict__["_run_verifier"] = run_verifier

    prover._handle_spawn({"tasks": [{"description": "task"}]}, tmp_path)

    assert prover._read_step_meta(tmp_path)["status"] == "interrupted"
    assert counts == {"planner": 1, "worker": 1, "update": 1, "log": 1}


def test_spawn_worker_interruption_skips_paid_verifiers(tmp_path):
    prover = _prover(tmp_path)
    prover.max_workers = 2
    verifier_calls = []

    def run_worker(task, *args):
        error = "interrupted" if task["description"] == "interrupted" else ""
        return {
            "result": error or "success", "cost": 0.0, "duration_ms": 0,
            "raw": {}, "error": error,
        }

    prover.__dict__["_run_worker"] = run_worker
    prover.__dict__["_run_verifier"] = lambda *args: verifier_calls.append(args)

    prover._handle_spawn(
        {"tasks": [{"description": "interrupted"}, {"description": "skipped"}]},
        tmp_path,
    )

    assert verifier_calls == []
    assert prover._read_step_meta(tmp_path)["status"] == "interrupted"


def test_spawn_shutdown_synthetic_interruption_has_no_continuation_side_effects(tmp_path):
    prover = _prover(tmp_path)
    counts = _side_effect_counts(prover)

    def run_worker(*args):
        prover.shutting_down = True
        return {
            "result": "worker", "cost": 0.0, "duration_ms": 0,
            "raw": {}, "error": "",
        }

    prover.__dict__["_run_worker"] = run_worker

    prover._handle_spawn(
        {"tasks": [{"description": "first"}, {"description": "synthetic"}]},
        tmp_path,
    )

    assert prover._read_step_meta(tmp_path)["status"] == "interrupted"
    assert counts == {"planner": 0, "worker": 0, "update": 0, "log": 0}
    assert prover.autonomous is True


@pytest.mark.parametrize("winning_error", ["interrupted", "budget_exhausted"])
def test_spawn_retry_side_effect_does_not_override_higher_precedence(
    tmp_path, winning_error,
):
    prover = _prover(tmp_path)

    def run_worker(*args):
        assert prover._retry_action(RuntimeError("retry ceiling"), MAX_CONSECUTIVE_ERRORS) == "stop"
        return {
            "result": "worker", "cost": 0.0, "duration_ms": 0,
            "raw": {}, "error": "",
        }

    prover.__dict__["_run_worker"] = run_worker
    prover.__dict__["_run_verifier"] = lambda *args: {
        "result": winning_error, "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": winning_error,
    }

    prover._handle_spawn({"tasks": [{"description": "task"}]}, tmp_path)

    assert prover._read_step_meta(tmp_path)["status"] == winning_error
    assert prover._llm_error_exit is False
    assert prover._last_error_msg == "before"
