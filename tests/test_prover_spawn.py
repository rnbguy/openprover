from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest

import openprover.prover as prover_module
from openprover.prover import Prover


class FakeTUI:
    autonomous = True

    def __init__(self):
        self.done = []
        self.done_events = {}
        self.step_entries = [{}]
        self.waiting = []
        self.worker_tabs = []

    def add_worker_tab(self, tab_id, _label, task_description=""):
        del task_description
        self.done_events[tab_id] = Event()
        self.worker_tabs.append((tab_id, _label))
        return SimpleNamespace(task_summary="")

    def mark_worker_done(self, tab_id):
        self.done.append(tab_id)
        self.done_events[tab_id].set()

    def set_waiting_status(self, text):
        self.waiting.append(text)

    def snapshot_worker_tabs(self, _step_num):
        pass

    def tab_log(self, *_args, **_kwargs):
        pass

    def update_step_status(self, *_args, **_kwargs):
        pass

    def _sync_step_log_line(self, _step_idx):
        pass

    def log(self, *_args, **_kwargs):
        pass


def make_prover(tmp_path, max_workers=2):
    prover = Prover.__new__(Prover)
    tui = FakeTUI()
    output = []
    prover.autonomous = True
    prover._current_action_outputs = []
    prover._interrupt_count = 0
    prover._step_idx = 0
    prover._workers_active = False
    prover.max_workers = max_workers
    prover.planner_llm = SimpleNamespace(clear_interrupt=lambda: None)
    prover.shutting_down = False
    prover.step_num = 1
    prover.verifier = False
    prover.worker_llm = SimpleNamespace(
        _soft_generation=0,
        clear_interrupt=lambda: None,
    )
    prover.work_dir = tmp_path
    prover.__dict__["tui"] = tui
    prover.__dict__["_push_output"] = output.append
    return prover, tui, output


def test_spawn_rolls_all_tasks_with_bounded_outstanding_workers(tmp_path):
    prover, tui, _ = make_prover(tmp_path)
    started = [Event() for _ in range(5)]
    release = [Event() for _ in range(5)]
    active = 0
    peak = 0
    lock = Lock()
    workers_dir = tmp_path / "workers"

    def run_worker(task, _worker_id, _archive):
        nonlocal active, peak
        index = int(task["description"].removeprefix("task "))
        assert all((workers_dir / f"task_{task_index}.md").exists()
                   for task_index in range(5))
        with lock:
            active += 1
            peak = max(peak, active)
        started[index].set()
        assert release[index].wait(timeout=1)
        with lock:
            active -= 1
        return {"result": f"result {index}", "cost": 0.0,
                "duration_ms": 0, "raw": {}, "error": ""}

    prover.__dict__["_run_worker"] = run_worker
    plan = {"tasks": [{"description": f"task {index}"} for index in range(5)]}
    handler = Thread(target=prover._handle_spawn, args=(plan, tmp_path))
    handler.start()
    try:
        assert started[0].wait(timeout=1)
        assert started[1].wait(timeout=1)
        release[0].set()
        assert started[2].wait(timeout=1)
        release[2].set()
        assert started[3].wait(timeout=1)
        release[3].set()
        assert started[4].wait(timeout=1)
        release[4].set()
        release[1].set()
    finally:
        for event in release:
            event.set()
        handler.join(timeout=1)

    assert not handler.is_alive()
    assert peak <= 2
    assert all(event.is_set() for event in started)
    assert sorted(tui.done) == [f"worker_1_{index}" for index in range(5)]
    assert [
        (workers_dir / f"result_{index}.md").read_text()
        for index in range(5)
    ] == [f"result {index}" for index in range(5)]


def test_spawn_keeps_result_slots_in_task_order_after_reverse_completion_and_error(tmp_path):
    prover, _, output = make_prover(tmp_path, max_workers=3)
    started = [Event() for _ in range(3)]
    release = [Event() for _ in range(3)]

    def run_worker(task, _worker_id, _archive):
        index = int(task["description"].removeprefix("task "))
        started[index].set()
        assert release[index].wait(timeout=1)
        if index == 1:
            raise RuntimeError("worker failed")
        return {"result": f"result {index}", "cost": 0.0,
                "duration_ms": 0, "raw": {}, "error": ""}

    prover.__dict__["_run_worker"] = run_worker
    plan = {"tasks": [{"description": f"task {index}"} for index in range(3)]}
    handler = Thread(target=prover._handle_spawn, args=(plan, tmp_path))
    handler.start()
    try:
        assert all(event.wait(timeout=1) for event in started)
        release[2].set()
        release[1].set()
        release[0].set()
    finally:
        for event in release:
            event.set()
        handler.join(timeout=1)

    assert not handler.is_alive()
    workers_dir = tmp_path / "workers"
    assert (workers_dir / "result_0.md").read_text() == "result 0"
    assert (workers_dir / "result_1.md").read_text() == "Worker error: worker failed"
    assert (workers_dir / "result_2.md").read_text() == "result 2"
    assert output[0].index("## Worker 0") < output[0].index("## Worker 1")
    assert output[0].index("## Worker 1") < output[0].index("## Worker 2")


@pytest.mark.parametrize("stop_kind", ["interrupt_count", "shutdown", "response"])
def test_spawn_interruption_stops_admission_and_records_original_indices(
    tmp_path, stop_kind,
):
    prover, tui, _ = make_prover(tmp_path)
    first_started = Event()
    second_started = Event()
    release_first = Event()
    release_second = Event()
    called = []

    def run_worker(task, _worker_id, _archive):
        index = int(task["description"].removeprefix("task "))
        called.append(index)
        if index == 0:
            first_started.set()
            assert release_first.wait(timeout=1)
            if stop_kind == "interrupt_count":
                prover._interrupt_count += 1
            elif stop_kind == "shutdown":
                prover.shutting_down = True
            else:
                return {"result": "(terminated by user)", "cost": 0.0,
                        "duration_ms": 0, "raw": {}, "error": "interrupted"}
        else:
            second_started.set()
            assert release_second.wait(timeout=1)
        return {"result": f"result {index}", "cost": 0.0,
                "duration_ms": 0, "raw": {}, "error": ""}

    prover.__dict__["_run_worker"] = run_worker
    plan = {"tasks": [{"description": f"task {index}"} for index in range(5)]}
    handler = Thread(target=prover._handle_spawn, args=(plan, tmp_path))
    handler.start()
    try:
        assert first_started.wait(timeout=1)
        assert second_started.wait(timeout=1)
        release_first.set()
        assert tui.done_events["worker_1_0"].wait(timeout=1)
        release_second.set()
    finally:
        release_first.set()
        release_second.set()
        handler.join(timeout=1)

    assert not handler.is_alive()
    assert sorted(called) == [0, 1]
    assert tui.waiting[-1] == ""
    assert tui.done.count("worker_1_0") == 1
    assert tui.done.count("worker_1_1") == 1
    assert tui.done.count("worker_1_2") == 1
    assert tui.done.count("worker_1_3") == 1
    assert tui.done.count("worker_1_4") == 1
    workers_dir = tmp_path / "workers"
    assert [
        (workers_dir / f"result_{index}.md").read_text()
        for index in range(5)
    ] == [("(terminated by user)" if stop_kind == "response" else "result 0"),
          "result 1", "(terminated by user)",
          "(terminated by user)", "(terminated by user)"]
    meta = (tmp_path / "meta.toml").read_text()
    assert 'index = 2\n' in meta
    assert 'index = 3\n' in meta
    assert 'index = 4\n' in meta
    assert meta.count('error = "interrupted"') == (4 if stop_kind == "response" else 3)


def test_resume_respawn_uses_numeric_task_indices(tmp_path):
    prover, _, _ = make_prover(tmp_path)
    step_dir = tmp_path / "steps" / "step_001"
    workers_dir = step_dir / "workers"
    workers_dir.mkdir(parents=True)
    (step_dir / "planner.toml").write_text('action = "spawn"\nsummary = ""\n')
    (step_dir / "meta.toml").write_text(
        'status = "interrupted"\n\n[[workers]]\nindex = 2\nerror = "interrupted"\n'
    )
    for index in (0, 2, 10):
        (workers_dir / f"task_{index}.md").write_text(f"task {index}")
    (workers_dir / "result_0.md").write_text("result 0")
    (workers_dir / "result_10.md").write_text("result 10")

    prover.__dict__["_respawn_plan"] = None
    prover._maybe_respawn_interrupted_workers()

    respawn_plan = prover.__dict__["_respawn_plan"]
    assert respawn_plan is not None
    assert respawn_plan["tasks"] == [
        {"description": "task 2", "_original_index": 2}
    ]
    assert respawn_plan["completed_workers"] == {
        0: {"description": "task 0", "result": "result 0"},
        10: {"description": "task 10", "result": "result 10"},
    }


def test_respawn_keeps_original_worker_tab_and_verdict_indices(monkeypatch, tmp_path):
    prover, tui, _ = make_prover(tmp_path)
    prover.verifier = True
    prover.__dict__["_run_worker"] = lambda *_args: {
        "result": "fresh worker", "cost": 0.0,
        "duration_ms": 0, "raw": {}, "error": "",
    }
    prover.__dict__["_run_verifiers"] = lambda *_args: {
        0: {"result": "fresh verdict", "error": ""}
    }
    monkeypatch.setattr(prover_module.prompts, "extract_verdict", lambda result: result)
    plan = {
        "tasks": [{"description": "task 10", "_original_index": 10}],
        "completed_workers": {
            2: {
                "description": "task 2",
                "result": "completed worker",
                "verifier_result": "carried verdict",
            },
        },
    }

    prover._handle_spawn(plan, tmp_path)

    assert ("worker_1_10", "Worker 10") in tui.worker_tabs
    assert tui.step_entries[0]["verdicts"] == {
        2: "carried verdict",
        10: "fresh verdict",
    }
