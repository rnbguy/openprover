import sys
from types import SimpleNamespace

from openprover import cli


def test_inspect_cleans_shared_client_before_lean_directory(monkeypatch, tmp_path):
    # Given
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
    cleanup_order = []
    client = SimpleNamespace(
        total_cost=0,
        call_count=0,
        cleanup=lambda: cleanup_order.append("client"),
    )
    lean_work_dir = SimpleNamespace(
        cleanup=lambda: cleanup_order.append("lean"),
    )

    class ProverSpy:
        def __init__(self, **kwargs):
            self.planner_llm = client
            self.worker_llm = client
            self.lean_work_dir = lean_work_dir
            self.work_dir = tmp_path
            self.budget = SimpleNamespace(total_output_tokens=0)
            self.mode = "prove"
            self._spending_limit_hit = False
            self._llm_error_exit = False

        def inspect(self):
            pass

    registered_cleanup = []
    monkeypatch.setattr(cli, "Prover", ProverSpy)
    monkeypatch.setattr(cli.atexit, "register", registered_cleanup.append)
    monkeypatch.setattr(sys, "argv", ["openprover", str(tmp_path), "--read-only", "--headless"])

    # When
    cli._cmd_prove()
    registered_cleanup[0]()

    # Then
    assert cleanup_order == ["client", "lean"]
