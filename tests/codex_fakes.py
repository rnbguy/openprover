from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock

import pytest
from openai_codex import (
    ApprovalMode,
    CodexConfig,
    InvalidRequestError,
    Sandbox,
    TurnResult,
)
from openai_codex.models import (
    AgentMessageDeltaNotification,
    ItemCompletedNotification,
    JsonObject,
    Notification,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)
from openai_codex.types import ReasoningEffort, ThreadItem, ThreadTokenUsage, TurnError, TurnStatus

import openprover.llm.codex as codex_module
from openprover.llm.codex import CodexClient


@dataclass(frozen=True, slots=True)
class FakeEffort:
    reasoning_effort: ReasoningEffort


@dataclass(frozen=True, slots=True)
class FakeModel:
    id: str
    model: str
    supported_reasoning_efforts: list[FakeEffort]


@dataclass(frozen=True, slots=True)
class FakeCatalog:
    data: list[FakeModel]


@dataclass(frozen=True, slots=True)
class FakeTurnRequest:
    approval_mode: ApprovalMode
    effort: ReasoningEffort
    sandbox: Sandbox
    output_schema: JsonObject | None


@dataclass(slots=True)
class FakeTurn:
    id: str
    result: TurnResult
    events: list[Notification] = field(default_factory=list)
    block: Event | None = None
    turn_started: Event | None = None
    turn_release: Event | None = None
    late_interrupt: bool = False
    started: Event = field(default_factory=Event)
    interrupts: int = 0

    def interrupt(self) -> None:
        self.interrupts += 1
        if self.late_interrupt:
            raise InvalidRequestError(-32600, "turn already completed")

    def run(self) -> TurnResult:
        self.started.set()
        if self.block is not None:
            self.block.wait(timeout=1)
        if self.result.status is TurnStatus.failed:
            if self.result.error is not None and self.result.error.message:
                raise RuntimeError(self.result.error.message)
            raise RuntimeError(f"turn failed with status {self.result.status.value}")
        return self.result

    def stream(self) -> Iterator[Notification]:
        self.started.set()
        if self.block is not None:
            self.block.wait(timeout=1)
        if self.events:
            yield from self.events
            return
        if self.result.final_response is not None:
            yield Notification(
                method="item/agentMessage/delta",
                payload=AgentMessageDeltaNotification(
                    delta=self.result.final_response,
                    item_id="message-1",
                    thread_id="thread-1",
                    turn_id=self.result.id,
                ),
            )
            yield Notification(
                method="item/completed",
                payload=ItemCompletedNotification(
                    item=ThreadItem.model_validate(
                        {
                            "type": "agentMessage",
                            "id": "message-1",
                            "phase": "final_answer",
                            "text": self.result.final_response,
                        }
                    ),
                    completed_at_ms=2,
                    thread_id="thread-1",
                    turn_id=self.result.id,
                ),
            )
        if self.result.usage is not None:
            yield Notification(
                method="thread/tokenUsage/updated",
                payload=ThreadTokenUsageUpdatedNotification(
                    thread_id="thread-1",
                    token_usage=self.result.usage,
                    turn_id=self.result.id,
                ),
            )
        yield completed_event(self.result.status, self.result.error, self.result.id)


@dataclass(slots=True)
class FakeThread:
    handle: FakeTurn
    turn_requests: list[FakeTurnRequest]

    def turn(
        self,
        _input: str,
        *,
        approval_mode: ApprovalMode,
        effort: ReasoningEffort,
        output_schema: JsonObject | None,
        sandbox: Sandbox,
    ) -> FakeTurn:
        if self.handle.turn_started is not None:
            self.handle.turn_started.set()
        if self.handle.turn_release is not None:
            self.handle.turn_release.wait(timeout=1)
        self.turn_requests.append(
            FakeTurnRequest(approval_mode, effort, sandbox, output_schema)
        )
        return self.handle


@dataclass(slots=True)
class FakeCodex:
    catalog: FakeCatalog
    turns: list[FakeTurn]
    configs: list[CodexConfig] = field(default_factory=list)
    thread_configs: list[JsonObject] = field(default_factory=list)
    turn_requests: list[FakeTurnRequest] = field(default_factory=list)
    close_calls: int = 0
    _lock: Lock = field(default_factory=Lock)

    def models(self) -> FakeCatalog:
        return self.catalog

    def thread_start(
        self,
        *,
        approval_mode: ApprovalMode,
        config: JsonObject,
        developer_instructions: str,
        ephemeral: bool,
        model: str,
        sandbox: Sandbox,
    ) -> FakeThread:
        assert approval_mode is ApprovalMode.deny_all
        assert developer_instructions == "system"
        assert ephemeral is True
        assert model == "gpt-5.6-sol"
        assert sandbox is Sandbox.read_only
        with self._lock:
            self.thread_configs.append(config)
            turn = self.turns.pop(0)
        return FakeThread(turn, self.turn_requests)

    def close(self) -> None:
        self.close_calls += 1


@dataclass(slots=True)
class FakeFactory:
    codex: FakeCodex

    def __call__(self, config: CodexConfig) -> FakeCodex:
        self.codex.configs.append(config)
        return self.codex


def catalog(*efforts: ReasoningEffort) -> FakeCatalog:
    return FakeCatalog(
        [
            FakeModel(
                id="catalog-entry-id",
                model="gpt-5.6-sol",
                supported_reasoning_efforts=[FakeEffort(effort) for effort in efforts],
            )
        ]
    )


def usage() -> ThreadTokenUsage:
    return ThreadTokenUsage.model_validate(
        {
            "last": {
                "inputTokens": 11,
                "cachedInputTokens": 2,
                "outputTokens": 7,
                "reasoningOutputTokens": 5,
                "totalTokens": 18,
            },
            "total": {
                "inputTokens": 11,
                "cachedInputTokens": 2,
                "outputTokens": 7,
                "reasoningOutputTokens": 5,
                "totalTokens": 18,
            },
        }
    )


def result(
    *,
    text: str = "final",
    status: TurnStatus = TurnStatus.completed,
    error: TurnError | None = None,
    turn_id: str = "turn-1",
) -> TurnResult:
    return TurnResult(
        id=turn_id,
        status=status,
        error=error,
        started_at=None,
        completed_at=None,
        duration_ms=123,
        final_response=text,
        items=[],
        usage=usage(),
    )


def completed_event(
    status: TurnStatus, error: TurnError | None = None, turn_id: str = "turn-1"
) -> Notification:
    turn: JsonObject = {"id": turn_id, "items": [], "status": status.value}
    if error is not None:
        turn["error"] = {"message": error.message}
    return Notification(
        method="turn/completed",
        payload=TurnCompletedNotification.model_validate({"threadId": "thread-1", "turn": turn}),
    )


def client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    turns: list[FakeTurn],
    catalog_response: FakeCatalog | None = None,
) -> tuple[CodexClient, FakeCodex]:
    fake = FakeCodex(catalog_response or catalog(ReasoningEffort.high), turns)
    monkeypatch.setattr(codex_module, "Codex", FakeFactory(fake))
    return CodexClient("gpt-5.6-sol", tmp_path), fake
