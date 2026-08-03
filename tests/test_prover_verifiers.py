from concurrent.futures import Future
from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest

import openprover.prover as prover_module
from openprover.prover import Prover


class FakeTUI:
    def __init__(self):
        self.done = []
        self.done_events = {}
        self.step_entries = [{}]
        self.waiting = []
        self.worker_tabs = []

    def add_worker_tab(self, tab_id, label, task_description=""):
        self.done_events[tab_id] = Event()
        self.worker_tabs.append((tab_id, label, task_description))
        return SimpleNamespace()

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
    prover.autonomous = True
    prover._current_action_outputs = []
    prover._interrupt_count = 0
    prover._step_idx = 0
    prover._workers_active = False
    prover.max_workers = max_workers
    prover.planner_llm = SimpleNamespace(clear_interrupt=lambda: None)
    prover.shutting_down = False
    prover.step_num = 1
    prover.tui = tui
    prover.verifier = False
    prover.worker_llm = SimpleNamespace(clear_interrupt=lambda: None)
    prover.work_dir = tmp_path
    prover._push_output = lambda _text: None
    return prover, tui


def test_verifiers_roll_eligible_workers_with_bounded_local_results(tmp_path):
    prover, tui = make_prover(tmp_path)
    tasks = [{"description": f"task {index}"} for index in range(6)]
    worker_resps = [
        {"result": f"worker {index}", "error": ""}
        if index != 2 else {"result": "worker 2", "error": "worker failed"}
        for index in range(6)
    ]
    started = [Event() for _ in range(6)]
    release = [Event() for _ in range(6)]
    active = 0
    peak = 0
    lock = Lock()

    def run_verifier(task_desc, _worker_output, _verifier_id, _archive):
        nonlocal active, peak
        index = int(task_desc.removeprefix("task "))
        with lock:
            active += 1
            peak = max(peak, active)
        started[index].set()
        assert release[index].wait(timeout=1)
        with lock:
            active -= 1
        if index == 3:
            raise RuntimeError("verifier failed")
        return {"result": f"verdict {index}", "cost": 0.0,
                "duration_ms": 0, "raw": {}, "error": ""}

    prover.__dict__["_run_verifier"] = run_verifier
    outcome = []
    handler = Thread(target=lambda: outcome.append(
        prover._run_verifiers(tasks, worker_resps, tmp_path)))
    handler.start()
    try:
        assert started[0].wait(timeout=1)
        assert started[1].wait(timeout=1)
        release[0].set()
        assert started[3].wait(timeout=1)
        release[3].set()
        assert started[4].wait(timeout=1)
        release[4].set()
        assert started[5].wait(timeout=1)
        release[5].set()
        release[1].set()
    finally:
        for event in release:
            event.set()
        handler.join(timeout=1)

    assert not handler.is_alive()
    assert peak <= 2
    assert not started[2].is_set()
    assert outcome[0][0]["result"] == "verdict 0"
    assert outcome[0][1]["result"] == "verdict 1"
    assert outcome[0][3] == {
        "result": "Verifier error: verifier failed", "cost": 0.0,
        "duration_ms": 0, "raw": {}, "error": "verifier failed",
    }
    assert outcome[0][4]["result"] == "verdict 4"
    assert outcome[0][5]["result"] == "verdict 5"
    assert sorted(tui.done) == [f"verifier_1_{index}" for index in (0, 1, 3, 4, 5)]


@pytest.mark.parametrize("stop_kind", ["interrupt_count", "shutdown", "response"])
def test_verifier_interruption_stops_admission_and_completes_tabs(tmp_path, stop_kind):
    prover, tui = make_prover(tmp_path)
    tasks = [{"description": f"task {index}"} for index in range(5)]
    worker_resps = [{"result": f"worker {index}", "error": ""} for index in range(5)]
    first_started = Event()
    second_started = Event()
    release_first = Event()
    release_second = Event()
    called = []

    def run_verifier(task_desc, _worker_output, _verifier_id, _archive):
        index = int(task_desc.removeprefix("task "))
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
        return {"result": f"verdict {index}", "cost": 0.0,
                "duration_ms": 0, "raw": {}, "error": ""}

    prover.__dict__["_run_verifier"] = run_verifier
    outcome = []
    handler = Thread(target=lambda: outcome.append(
        prover._run_verifiers(tasks, worker_resps, tmp_path)))
    handler.start()
    try:
        assert first_started.wait(timeout=1)
        assert second_started.wait(timeout=1)
        release_first.set()
        assert tui.done_events["verifier_1_0"].wait(timeout=1)
        release_second.set()
    finally:
        release_first.set()
        release_second.set()
        handler.join(timeout=1)

    assert not handler.is_alive()
    assert sorted(called) == [0, 1]
    assert tui.waiting[-1] == ""
    assert all(tui.done.count(f"verifier_1_{index}") == 1 for index in range(5))
    assert outcome[0][2] == {
        "result": "(terminated by user)", "cost": 0.0,
        "duration_ms": 0, "raw": {}, "error": "interrupted",
    }


def test_verifier_uses_original_task_identity_and_archive(tmp_path):
    prover, tui = make_prover(tmp_path)
    calls = []

    def run_verifier(_task_desc, _worker_output, verifier_id, archive):
        calls.append((verifier_id, archive.name))
        return {"result": "verdict", "cost": 0.0,
                "duration_ms": 0, "raw": {}, "error": ""}

    prover.__dict__["_run_verifier"] = run_verifier
    result = prover._run_verifiers(
        [{"description": "task 10", "_original_index": 10}],
        [{"result": "worker", "error": ""}],
        tmp_path,
    )

    assert result[0]["result"] == "verdict"
    assert calls == [("verifier_1_10", "verifier_10_call.md")]
    assert ("verifier_1_10", "Verify 10", "Verifying Worker 10") in tui.worker_tabs


def test_verifier_setup_interrupt_submits_no_work_and_completes_tabs(tmp_path):
    prover, tui = make_prover(tmp_path)
    calls = []
    add_worker_tab = tui.add_worker_tab

    def interrupting_add_worker_tab(*args, **kwargs):
        prover._interrupt_count += 1
        return add_worker_tab(*args, **kwargs)

    tui.add_worker_tab = interrupting_add_worker_tab
    prover.__dict__["_run_verifier"] = lambda *_args: calls.append("called") or {
        "result": "verdict", "cost": 0.0,
        "duration_ms": 0, "raw": {}, "error": "",
    }
    tasks = [{"description": f"task {index}"} for index in range(3)]
    worker_resps = [{"result": f"worker {index}", "error": ""} for index in range(3)]

    result = prover._run_verifiers(tasks, worker_resps, tmp_path)

    assert calls == []
    assert all(tui.done.count(f"verifier_1_{index}") == 1 for index in range(3))
    assert result == {
        index: {
            "result": "(terminated by user)", "cost": 0.0,
            "duration_ms": 0, "raw": {}, "error": "interrupted",
        }
        for index in range(3)
    }
    assert tui.waiting[-1] == ""


def test_worker_admission_rechecks_interrupt_count_before_each_submit(monkeypatch, tmp_path):
    prover, _ = make_prover(tmp_path)
    submitted = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _call, task, *_args):
            submitted.append(task["description"])
            prover._interrupt_count += 1
            future = Future()
            future.set_result({"result": "worker", "cost": 0.0,
                               "duration_ms": 0, "raw": {}, "error": ""})
            return future

    monkeypatch.setattr(prover_module, "ThreadPoolExecutor", RecordingExecutor)

    prover._handle_spawn(
        {"tasks": [{"description": f"task {index}"} for index in range(3)]},
        tmp_path,
    )

    assert submitted == ["task 0"]
