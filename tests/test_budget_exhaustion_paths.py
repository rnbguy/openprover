import threading
from types import SimpleNamespace

from openprover import cli
from openprover.budget import Budget
from openprover.prover import Prover


class FakeTUI:
    autonomous = True

    def __init__(self):
        self.step_entries = [{}]

    def update_budget(self, _status):
        pass

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class FakeLLM:
    model = "fake"

    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.calls = 0

    def call(self, **_kwargs):
        self.calls += 1
        return next(self.responses)

    def chat(self, **_kwargs):
        self.calls += 1
        return next(self.responses)


def save_run_config(work_dir):
    work_dir.mkdir()
    cli._save_run_config(
        work_dir, planner_model="sonnet", worker_model="sonnet",
        budget_mode="tokens", budget_limit=2, conclude_after=0.99,
        max_workers=1, isolation=False, autonomous=True, mode="prove",
        lean_project_dir=None, lean_items=False, lean_worker_tools=False,
        provider_url="http://127.0.0.1:8000", answer_reserve=4096,
        history_budget=0, verifier=True,
    )


def make_prover(tmp_path, budget, llm):
    prover = Prover.__new__(Prover)
    prover.work_dir = tmp_path / "run"
    save_run_config(prover.work_dir)
    prover.budget = budget
    prover.tui = FakeTUI()
    prover._budget_persistence_lock = threading.Lock()
    prover._backoff_delay = prover._rate_limit_backoff = prover._transient_backoff = 4
    prover.planner_llm = llm
    prover.worker_llm = llm
    prover.repo = SimpleNamespace(list_summaries=lambda: "")
    prover.whiteboard = "before"
    prover.proof_text = ""
    prover.theorem_text = "theorem"
    prover.lean_theorem_text = ""
    prover.step_history = []
    prover.step_num = 1
    prover.max_workers = 1
    prover.history_budget = 0
    prover.isolation = False
    prover.mode = "prove"
    prover.lean_items = False
    prover._steps_since_productive = 0
    prover._respawn_plan = None
    prover._step_idx = 0
    prover._current_action_outputs = []
    prover._stream_cb = lambda *_args, **_kwargs: None
    return prover


def response(result, tokens, finish_reason):
    return {
        "result": result, "thinking": "", "cost": 1.0, "duration_ms": 1,
        "raw": {"model": "fake", "usage": {"output_tokens": tokens}},
        "finish_reason": finish_reason,
    }


def test_initial_planner_denial_saves_budget_exhausted_step_without_client(tmp_path):
    llm = FakeLLM()
    prover = make_prover(tmp_path, Budget("tokens", 2, initial_output_tokens=2), llm)

    assert prover._do_step() == "stop"

    meta = (prover.work_dir / "steps" / "step_001" / "meta.toml").read_text()
    assert llm.calls == 0
    assert 'status = "budget_exhausted"' in meta
    assert "budget_exhausted" in meta


def test_planner_phase_two_denial_preserves_first_response_metadata(monkeypatch, tmp_path):
    llm = FakeLLM([response("partial", 2, "length")])
    prover = make_prover(tmp_path, Budget("tokens", 2), llm)
    monkeypatch.setattr("openprover.prover.prompts.parse_planner_toml", lambda _text: None)

    assert prover._do_step() == "stop"

    meta = (prover.work_dir / "steps" / "step_001" / "meta.toml").read_text()
    assert llm.calls == 1
    assert 'status = "budget_exhausted"' in meta
    assert "output_tokens = 2" in meta


def test_exhausted_literature_search_skips_model_and_local_action_still_runs(tmp_path):
    llm = FakeLLM()
    prover = make_prover(tmp_path, Budget("tokens", 2, initial_output_tokens=2), llm)
    step_dir = prover.work_dir / "steps" / "step_001"
    step_dir.mkdir(parents=True)
    plans = [
        {"action": "literature_search", "search_query": "query", "search_context": ""},
        {"action": "write_whiteboard", "whiteboard": "after"},
    ]

    assert prover._execute_plans(plans, step_dir, {}) == "continue"

    result = (step_dir / "workers" / "result_0.md").read_text()
    assert llm.calls == 0
    assert result == "(skipped: budget exhausted)"
    assert prover.whiteboard == "after"
    assert 'status = "budget_exhausted"' in (step_dir / "meta.toml").read_text()


def test_exhausted_discussion_writes_existing_local_fallback_without_model(tmp_path):
    llm = FakeLLM()
    prover = make_prover(tmp_path, Budget("tokens", 2, initial_output_tokens=2), llm)

    prover._write_discussion()

    discussion = (prover.work_dir / "DISCUSSION.md").read_text()
    assert llm.calls == 0
    assert "Session ended after 1 steps." in discussion
    assert "## Final Whiteboard\n\nbefore" in discussion
