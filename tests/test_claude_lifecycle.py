import json
import signal

import pytest

import openprover.llm.claude as claude_module
from openprover.llm import Interrupted
from openprover.llm.claude import LLMClient


class FakePipe:
    def __init__(self, lines=(), *, write_error=None, on_read=None):
        self._lines = iter(lines)
        self._write_error = write_error
        self._on_read = on_read
        self.closed = False

    def write(self, text):
        if self._write_error is not None:
            raise self._write_error
        return len(text)

    def close(self):
        self.closed = True

    def readline(self):
        if self._on_read is not None:
            on_read = self._on_read
            self._on_read = None
            on_read()
        return next(self._lines, "")

    def read(self):
        return ""


class FakePopen:
    def __init__(
        self,
        *,
        communicate_error=None,
        communicate_output=('{"result": "ok"}', ""),
        stdin_error=None,
        stdout_lines=(),
        on_stdout_read=None,
    ):
        self.pid = 4321
        self.returncode = None
        self.stdin = FakePipe(write_error=stdin_error)
        self.stdout = FakePipe(stdout_lines, on_read=on_stdout_read)
        self.stderr = FakePipe()
        self._communicate_error = communicate_error
        self._communicate_output = communicate_output
        self.kill_calls = 0
        self.wait_calls = 0

    def communicate(self, *, input):
        if self._communicate_error is not None:
            raise self._communicate_error
        self.returncode = 0
        return self._communicate_output

    def poll(self):
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = -signal.SIGKILL

    def wait(self):
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def install_process(monkeypatch, process):
    popen_kwargs = []
    group_kills = []

    def popen(*_args, **kwargs):
        popen_kwargs.append(kwargs)
        return process

    def killpg(pid, sig):
        group_kills.append((pid, sig))
        process.returncode = -sig

    monkeypatch.setattr(claude_module.subprocess, "Popen", popen)
    monkeypatch.setattr(claude_module.os, "killpg", killpg)
    return popen_kwargs, group_kills


def assert_group_cleanup(client, process, group_kills):
    assert (
        client._active_procs,
        group_kills,
        process.wait_calls,
        process.kill_calls,
    ) == ([], [(process.pid, signal.SIGKILL)], 1, 0)


@pytest.mark.parametrize("streaming", [False, True])
def test_claude_starts_each_process_in_new_session(monkeypatch, tmp_path, streaming):
    lines = [json.dumps({"type": "result", "result": "ok"}) + "\n"] if streaming else []
    process = FakePopen(stdout_lines=lines)
    popen_kwargs, _ = install_process(monkeypatch, process)
    client = LLMClient("test-model", tmp_path)
    callback = (lambda _text, _kind: None) if streaming else None

    result = client.call("prompt", "system", stream_callback=callback)

    assert result["result"] == "ok"
    assert popen_kwargs[0]["start_new_session"] is True


def test_communicate_exception_cleans_process_group_and_preserves_error(monkeypatch, tmp_path):
    error = RuntimeError("communicate failed")
    process = FakePopen(communicate_error=error)
    _, group_kills = install_process(monkeypatch, process)
    client = LLMClient("test-model", tmp_path)

    with pytest.raises(RuntimeError) as raised:
        client.call("prompt", "system")

    assert raised.value is error
    assert_group_cleanup(client, process, group_kills)


def test_stdin_exception_cleans_streaming_process_group_and_preserves_error(monkeypatch, tmp_path):
    error = RuntimeError("stdin failed")
    process = FakePopen(stdin_error=error)
    _, group_kills = install_process(monkeypatch, process)
    client = LLMClient("test-model", tmp_path)

    with pytest.raises(RuntimeError) as raised:
        client.call("prompt", "system", stream_callback=lambda _text, _kind: None)

    assert raised.value is error
    assert_group_cleanup(client, process, group_kills)


def test_callback_exception_cleans_streaming_process_group_and_preserves_error(
    monkeypatch, tmp_path
):
    event = {
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}},
    }
    process = FakePopen(stdout_lines=[json.dumps(event) + "\n"])
    _, group_kills = install_process(monkeypatch, process)
    client = LLMClient("test-model", tmp_path)
    error = RuntimeError("callback failed")

    def fail_callback(_text, _kind):
        raise error

    with pytest.raises(RuntimeError) as raised:
        client.call("prompt", "system", stream_callback=fail_callback)

    assert raised.value is error
    assert_group_cleanup(client, process, group_kills)


def test_hard_interrupt_observed_by_stream_kills_process_group(monkeypatch, tmp_path):
    client = LLMClient("test-model", tmp_path)
    process = FakePopen(stdout_lines=["\n"], on_stdout_read=client._interrupted.set)
    _, group_kills = install_process(monkeypatch, process)

    with pytest.raises(Interrupted):
        client.call("prompt", "system", stream_callback=lambda _text, _kind: None)

    assert_group_cleanup(client, process, group_kills)


def test_soft_interrupt_observed_by_stream_kills_process_group(monkeypatch, tmp_path):
    client = LLMClient("test-model", tmp_path)
    process = FakePopen(stdout_lines=["\n"], on_stdout_read=client._soft_interrupted.set)
    _, group_kills = install_process(monkeypatch, process)

    result = client.call("prompt", "system", stream_callback=lambda _text, _kind: None)

    assert result["finish_reason"] == "soft_interrupted"
    assert_group_cleanup(client, process, group_kills)
