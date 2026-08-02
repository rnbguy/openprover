from pathlib import Path

from openprover.prover import Prover


class FakeTUI:
    def stream_start(self, _label: str, *, tab: str) -> None:
        pass

    def stream_end(self, *, tab: str) -> None:
        pass

    def tab_log(self, _tab: str, _text: str, *, color: str) -> None:
        pass


class RejectingCodex:
    def __init__(self) -> None:
        self.calls = 0
        self.soft_interrupt_cleared = 0

    def clear_soft_interrupt(self) -> None:
        self.soft_interrupt_cleared += 1

    def call(
        self,
        *,
        prompt: str,
        system_prompt: str,
        label: str,
        stream_callback,
        archive_path: Path | None,
        tool_callback=None,
        tool_start_callback=None,
        max_tokens: int | None = None,
    ) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "result": "partial",
                "thinking": "",
                "cost": 0.0,
                "duration_ms": 1,
                "raw": {},
                "finish_reason": "soft_interrupted",
            }
        return {
            "result": "final",
            "thinking": "",
            "cost": 0.0,
            "duration_ms": 1,
            "raw": {},
            "finish_reason": "stop",
        }


def test_worker_soft_interrupt_phase_two_omits_claude_only_keyword():
    prover = Prover.__new__(Prover)
    prover.worker_llm = RejectingCodex()
    prover.tui = FakeTUI()
    prover._stream_cb = lambda _tab, output_only=False: None
    prover._check_error_policy = lambda _error: "stop"

    response = prover._run_worker_single_turn("prompt", "system", "worker", None)

    assert response["result"] == "final"
    assert prover.worker_llm.calls == 2
    assert prover.worker_llm.soft_interrupt_cleared == 1
