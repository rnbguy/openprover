from types import SimpleNamespace

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


def make_prover(tmp_path, *, verifier=False):
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
    prover.verifier = verifier
    prover.work_dir = tmp_path
    prover.tui = FakeTUI()
    prover._push_output = lambda _text: None
    prover._llm_error_exit = False
    prover._last_error_msg = ""
    return prover


def test_spawn_worker_response_error_is_persisted_and_reported(tmp_path):
    prover = make_prover(tmp_path)
    prover._run_worker = lambda *_args: {
        "result": "worker failed", "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": "worker failed",
    }

    prover._handle_spawn({"tasks": [{"description": "task"}]}, tmp_path)

    meta = prover._read_step_meta(tmp_path)
    assert meta["status"] == "error"
    assert prover._llm_error_exit is True
    assert prover._last_error_msg == "worker failed"


def test_spawn_worker_future_error_is_persisted(tmp_path):
    prover = make_prover(tmp_path)

    def fail(*_args):
        raise RuntimeError("future failed")

    prover._run_worker = fail

    prover._handle_spawn({"tasks": [{"description": "task"}]}, tmp_path)

    assert prover._read_step_meta(tmp_path)["status"] == "error"
    assert prover._last_error_msg == "future failed"


def test_spawn_verifier_error_is_persisted(tmp_path):
    prover = make_prover(tmp_path, verifier=True)
    prover._run_worker = lambda *_args: {
        "result": "worker", "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": "",
    }
    prover._run_verifier = lambda *_args: {
        "result": "verifier failed", "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": "verifier failed",
    }

    prover._handle_spawn({"tasks": [{"description": "task"}]}, tmp_path)

    assert prover._read_step_meta(tmp_path)["status"] == "error"
    assert prover._last_error_msg == "verifier failed"


def test_spawn_verifier_future_error_is_persisted(tmp_path):
    prover = make_prover(tmp_path, verifier=True)
    prover._run_worker = lambda *_args: {
        "result": "worker", "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": "",
    }

    def fail(*_args):
        raise RuntimeError("verifier future failed")

    prover._run_verifier = fail

    prover._handle_spawn({"tasks": [{"description": "task"}]}, tmp_path)

    assert prover._read_step_meta(tmp_path)["status"] == "error"
    assert prover._last_error_msg == "verifier future failed"


def test_spawn_interruption_takes_precedence_over_errors(tmp_path):
    prover = make_prover(tmp_path)
    prover._run_worker = lambda *_args: {
        "result": "interrupted", "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": "interrupted",
    }

    prover._handle_spawn({"tasks": [{"description": "task"}]}, tmp_path)

    assert prover._read_step_meta(tmp_path)["status"] == "interrupted"
    assert prover._llm_error_exit is False


def test_spawn_budget_error_keeps_budget_semantics(tmp_path):
    prover = make_prover(tmp_path)
    prover._run_worker = lambda *_args: {
        "result": "budget", "cost": 0.0, "duration_ms": 0,
        "raw": {}, "error": "budget_exhausted",
    }

    prover._handle_spawn({"tasks": [{"description": "task"}]}, tmp_path)

    assert prover._read_step_meta(tmp_path)["status"] == "budget_exhausted"
    assert prover._llm_error_exit is False


def test_spawn_interrupted_status_does_not_set_error_indicator_for_mixed_responses(
    tmp_path,
):
    prover = make_prover(tmp_path, verifier=False)
    prover.max_workers = 2

    def run_worker(task, *_args):
        error = "interrupted" if task["description"] == "interrupt" else "worker failed"
        return {
            "result": error, "cost": 0.0, "duration_ms": 0,
            "raw": {}, "error": error,
        }

    prover._run_worker = run_worker

    prover._handle_spawn(
        {"tasks": [{"description": "interrupt"}, {"description": "ordinary"}]},
        tmp_path,
    )

    assert prover._read_step_meta(tmp_path)["status"] == "interrupted"
    assert prover._llm_error_exit is False
    assert prover._last_error_msg == ""


def test_spawn_budget_status_does_not_set_error_indicator_for_mixed_responses(
    tmp_path,
):
    prover = make_prover(tmp_path, verifier=False)
    prover.max_workers = 2

    def run_worker(task, *_args):
        error = (
            "budget_exhausted"
            if task["description"] == "budget"
            else "worker failed"
        )
        return {
            "result": error, "cost": 0.0, "duration_ms": 0,
            "raw": {}, "error": error,
        }

    prover._run_worker = run_worker

    prover._handle_spawn(
        {"tasks": [{"description": "budget"}, {"description": "ordinary"}]},
        tmp_path,
    )

    assert prover._read_step_meta(tmp_path)["status"] == "budget_exhausted"
    assert prover._llm_error_exit is False
    assert prover._last_error_msg == ""
