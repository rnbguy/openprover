from threading import Event, Thread

import pytest
from openai_codex.types import TurnStatus

from tests.codex_fakes import FakeTurn, client, result


def test_soft_interrupt_during_turn_start_is_registered(monkeypatch: pytest.MonkeyPatch, tmp_path):
    turn_started = Event()
    turn_release = Event()
    handle = FakeTurn(
        "turn-1",
        result(text="partial", status=TurnStatus.interrupted),
        turn_started=turn_started,
        turn_release=turn_release,
    )
    codex, _ = client(monkeypatch, tmp_path, [handle])
    outcomes: list[dict] = []
    thread = Thread(target=lambda: outcomes.append(codex.call("prompt", "system")))
    thread.start()
    assert turn_started.wait(timeout=1)

    codex.soft_interrupt()
    turn_release.set()
    thread.join(timeout=1)

    assert handle.interrupts == 1
    assert outcomes[0]["finish_reason"] == "soft_interrupted"


def test_late_interrupt_does_not_skip_other_active_turns(monkeypatch: pytest.MonkeyPatch, tmp_path):
    release = Event()
    first = FakeTurn(
        "turn-1", result(status=TurnStatus.interrupted), block=release, late_interrupt=True
    )
    second = FakeTurn("turn-2", result(turn_id="turn-2", status=TurnStatus.interrupted), block=release)
    codex, _ = client(monkeypatch, tmp_path, [first, second])
    outcomes: list[dict] = []
    threads = [
        Thread(target=lambda: outcomes.append(codex.call("prompt", "system"))),
        Thread(target=lambda: outcomes.append(codex.call("prompt", "system"))),
    ]
    for thread in threads:
        thread.start()
    assert first.started.wait(timeout=1)
    assert second.started.wait(timeout=1)

    codex.soft_interrupt()
    release.set()
    for thread in threads:
        thread.join(timeout=1)

    assert first.interrupts == 1
    assert second.interrupts == 1
    assert len(outcomes) == 2


def test_soft_interrupt_racing_completed_turn_keeps_stop_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    release = Event()
    handle = FakeTurn("turn-1", result(), block=release)
    codex, _ = client(monkeypatch, tmp_path, [handle])
    outcomes: list[dict] = []
    thread = Thread(target=lambda: outcomes.append(codex.call("prompt", "system")))
    thread.start()
    assert handle.started.wait(timeout=1)

    codex.soft_interrupt()
    release.set()
    thread.join(timeout=1)

    assert outcomes[0]["finish_reason"] == "stop"
