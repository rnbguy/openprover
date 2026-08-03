import sys
from types import SimpleNamespace

from openprover import cli


def test_inspect_initializes_real_prover_as_resumed(monkeypatch, tmp_path):
    (tmp_path / "WHITEBOARD.md").write_text("preserve me")
    (tmp_path / "THEOREM.md").write_text("theorem")
    cli._save_run_config(
        tmp_path,
        planner_model="sonnet", worker_model="sonnet", budget_mode="tokens",
        budget_limit=10, conclude_after=0.99, max_workers=1, isolation=False,
        autonomous=False, mode="prove", lean_project_dir=None, lean_items=False,
        lean_worker_tools=False, provider_url="http://127.0.0.1:8000",
        answer_reserve=4096, history_budget=0, verifier=True,
    )
    captured = {}

    class ProverSpy:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.planner_llm = SimpleNamespace(total_cost=0, call_count=0)
            self.worker_llm = SimpleNamespace(total_cost=0, call_count=0)
            self.work_dir = tmp_path
            self.budget = SimpleNamespace(total_output_tokens=0)
            self.mode = "prove"
            self._spending_limit_hit = False
            self._llm_error_exit = False

        def inspect(self):
            pass

    monkeypatch.setattr(cli, "Prover", ProverSpy)
    monkeypatch.setattr(sys, "argv", ["openprover", str(tmp_path), "--read-only", "--headless"])

    cli._cmd_prove()

    assert captured["resumed"] is True
    assert (tmp_path / "WHITEBOARD.md").read_text() == "preserve me"
