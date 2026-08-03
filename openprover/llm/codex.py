"""Typed adapter for the local Codex SDK."""

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Final

from openai_codex import (
    ApprovalMode,
    Codex,
    CodexConfig,
    InvalidRequestError,
    Sandbox,
    TurnHandle,
)
from openai_codex.models import JsonObject
from openai_codex.types import ReasoningEffort, TurnStatus

from ._base import Interrupted, archive
from ._codex_events import (
    StreamCallback,
    StreamCallbacks,
    ToolCallback,
    ToolStartCallback,
    run_turn,
    thinking_from_items,
    usage_from_result,
)

MODEL: Final = "gpt-5.6-sol"
_CONFIG_OVERRIDES: Final = (
    "features.shell_tool=false",
    "features.multi_agent=false",
    "features.plugins=false",
    "features.apps=false",
)


class CodexTurnError(RuntimeError):
    """A failed Codex turn that retains its normalized response."""

    def __init__(self, message: str, response: dict):
        super().__init__(message)
        self.response = response


class CodexClient:
    """Run isolated, read-only high-reasoning Codex turns through the SDK."""

    context_length = 1_050_000

    def __init__(self, model: str, archive_dir: Path, max_output_tokens: int = 128_000):
        if model != MODEL:
            raise ValueError(f"CodexClient supports only model {MODEL!r}, got {model!r}")
        del max_output_tokens
        self.model = model
        self.archive_dir = archive_dir
        self.call_count = 0
        self.total_cost = 0.0
        self.mcp_config: JsonObject | None = None
        self._codex: Codex | None = None
        self._codex_home = None
        self._sdk_lock = threading.RLock()
        self._turn_lock = threading.Lock()
        self._interrupted = threading.Event()
        self._active_turns: dict[str, TurnHandle] = {}
        self._hard_turn_ids: set[str] = set()
        self._soft_turn_ids: set[str] = set()
        self._soft_generation = 0

    def interrupt(self) -> None:
        """Hard-interrupt every turn that is active at this instant."""
        self._interrupted.set()
        self._interrupt_active(hard=True)

    def soft_interrupt(self) -> None:
        """Interrupt active turns while preserving their completed partial output."""
        self._interrupt_active(hard=False)

    def cleanup(self) -> None:
        """Close the lazily created SDK runtime once."""
        with self._sdk_lock:
            codex = self._codex
            codex_home = self._codex_home
            self._codex = None
            self._codex_home = None
        try:
            if codex is not None:
                codex.close()
        finally:
            if codex_home is not None:
                codex_home.cleanup()

    def clear_interrupt(self) -> None:
        """Allow subsequent calls after a hard interrupt."""
        self._interrupted.clear()

    def clear_soft_interrupt(self) -> None:
        """Completed turns consume their own soft-interrupt state."""

    def call(
        self,
        prompt: str,
        system_prompt: str,
        json_schema: JsonObject | None = None,
        label: str = "",
        web_search: bool = False,
        stream_callback: StreamCallback | None = None,
        archive_path: Path | None = None,
        tool_callback: ToolCallback | None = None,
        tool_start_callback: ToolStartCallback | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Run one ephemeral Codex turn and normalize its result contract."""
        del max_tokens
        with self._turn_lock:
            soft_generation = self._soft_generation
        call_num = self._next_call_number()
        self._archive(
            call_num, label, prompt, system_prompt, json_schema, None, None, 0, archive_path
        )
        if self._interrupted.is_set():
            self._archive(
                call_num,
                label,
                prompt,
                system_prompt,
                json_schema,
                "interrupted",
                None,
                0,
                archive_path,
            )
            raise Interrupted()

        thread = self._ensure_codex().thread_start(
            approval_mode=ApprovalMode.deny_all,
            config=self._thread_config(web_search),
            developer_instructions=system_prompt,
            ephemeral=True,
            model=self.model,
            sandbox=Sandbox.read_only,
        )
        turn = thread.turn(
            prompt,
            approval_mode=ApprovalMode.deny_all,
            effort=ReasoningEffort.high,
            output_schema=json_schema,
            sandbox=Sandbox.read_only,
        )
        self._register_turn(turn, soft_generation)
        try:
            turn_result, streamed_result, streamed_thinking = run_turn(
                turn, StreamCallbacks(stream_callback, tool_callback, tool_start_callback)
            )
            duration_ms = turn_result.duration_ms or 0
            raw = {
                "turn_id": turn_result.id,
                "status": turn_result.status.value,
                "usage": usage_from_result(turn_result),
                "total_cost_usd": 0.0,
                "stop_reason": "interrupted" if turn_result.status is TurnStatus.interrupted else "stop",
            }
            error_response = {"raw": raw}
            hard, soft = self._interruption_for(turn.id)
            if hard or (turn_result.status is TurnStatus.interrupted and not soft):
                self._archive(
                    call_num, label, prompt, system_prompt, json_schema, "interrupted", raw,
                    duration_ms, archive_path,
                )
                raise Interrupted(error_response)
            if turn_result.status is not TurnStatus.interrupted and turn_result.error is not None:
                error = f"Codex turn error: {turn_result.error.message}"
                self._archive(
                    call_num, label, prompt, system_prompt, json_schema, error, raw,
                    duration_ms, archive_path,
                )
                raise CodexTurnError(error, error_response)
            if turn_result.status not in (TurnStatus.completed, TurnStatus.interrupted):
                error = f"Codex turn ended with unexpected status {turn_result.status.value}"
                self._archive(
                    call_num, label, prompt, system_prompt, json_schema, error, raw,
                    duration_ms, archive_path,
                )
                raise CodexTurnError(error, error_response)

            finish_reason = (
                "soft_interrupted"
                if soft and turn_result.status is TurnStatus.interrupted
                else "stop"
            )
            result = turn_result.final_response or (
                streamed_result if turn_result.status is TurnStatus.interrupted else ""
            )
            thinking = thinking_from_items(turn_result.items) or streamed_thinking
            raw["stop_reason"] = finish_reason
            self._archive(
                call_num, label, prompt, system_prompt, json_schema, None, raw, duration_ms,
                archive_path, thinking=thinking, result_text=result,
            )
            return {
                "result": result,
                "thinking": thinking,
                "cost": 0.0,
                "duration_ms": duration_ms,
                "raw": raw,
                "finish_reason": finish_reason,
            }
        finally:
            self._unregister_turn(turn.id)

    def _next_call_number(self) -> int:
        with self._turn_lock:
            self.call_count += 1
            return self.call_count

    def _ensure_codex(self) -> Codex:
        with self._sdk_lock:
            if self._codex is not None:
                return self._codex
            source_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
            source_auth = source_home / "auth.json"
            codex_home = tempfile.TemporaryDirectory()
            codex: Codex | None = None
            ready = False
            try:
                Path(codex_home.name).chmod(0o700)
                if source_auth.is_file():
                    isolated_auth = Path(codex_home.name) / "auth.json"
                    shutil.copyfile(source_auth, isolated_auth)
                    isolated_auth.chmod(0o600)
                codex = Codex(
                    CodexConfig(
                        config_overrides=_CONFIG_OVERRIDES,
                        cwd=codex_home.name,
                        env={"CODEX_HOME": codex_home.name},
                    )
                )
                self._validate_model(codex)
                self._codex = codex
                self._codex_home = codex_home
                ready = True
                return codex
            finally:
                if not ready:
                    if codex is not None:
                        codex.close()
                    codex_home.cleanup()

    def _validate_model(self, codex: Codex) -> None:
        model = next((entry for entry in codex.models().data if entry.model == self.model), None)
        if model is None:
            raise RuntimeError(f"Codex model catalog does not advertise {self.model!r}")
        efforts = {option.reasoning_effort for option in model.supported_reasoning_efforts}
        if ReasoningEffort.high not in efforts:
            raise RuntimeError(f"Codex model {self.model!r} does not advertise high reasoning")

    def _thread_config(self, web_search: bool) -> JsonObject:
        config: JsonObject = {"web_search": "live" if web_search else "disabled"}
        if self.mcp_config is not None and not web_search:
            config["mcp_servers"] = self.mcp_config["mcp_servers"]
        return config

    def _register_turn(self, turn: TurnHandle, soft_generation: int) -> None:
        with self._turn_lock:
            self._active_turns[turn.id] = turn
            hard = self._interrupted.is_set()
            soft = self._soft_generation != soft_generation
            if hard:
                self._hard_turn_ids.add(turn.id)
            if soft:
                self._soft_turn_ids.add(turn.id)
        if hard or soft:
            self._interrupt_turn(turn)

    def _unregister_turn(self, turn_id: str) -> None:
        with self._turn_lock:
            self._active_turns.pop(turn_id, None)
            self._hard_turn_ids.discard(turn_id)
            self._soft_turn_ids.discard(turn_id)

    def _interruption_for(self, turn_id: str) -> tuple[bool, bool]:
        with self._turn_lock:
            return turn_id in self._hard_turn_ids, turn_id in self._soft_turn_ids

    def _interrupt_active(self, *, hard: bool) -> None:
        with self._turn_lock:
            turn_ids = set(self._active_turns)
            if hard:
                self._hard_turn_ids.update(turn_ids)
            else:
                self._soft_generation += 1
                self._soft_turn_ids.update(turn_ids)
            turns = tuple(self._active_turns.values())
        for turn in turns:
            self._interrupt_turn(turn)

    @staticmethod
    def _interrupt_turn(turn: TurnHandle) -> None:
        try:
            turn.interrupt()
        except InvalidRequestError:
            return

    def _archive(
        self, call_num: int, label: str, prompt: str, system_prompt: str,
        json_schema: JsonObject | None, error: str | None, response: JsonObject | None, elapsed_ms: int,
        archive_path: Path | None, *, thinking: str = "", result_text: str = "",
    ) -> None:
        archive(
            self.model, self.archive_dir, call_num, label, prompt, system_prompt,
            json_schema, response, error, elapsed_ms, archive_path,
            thinking=thinking, result_text=result_text,
        )
