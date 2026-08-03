import urllib.error
from email.message import Message

import pytest

import openprover.llm.mistral as mistral_module
from openprover.llm import Interrupted
from openprover.llm.mistral import MistralClient


class FakeMonotonic:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeEvent:
    def __init__(self, clock, interrupt_on_wait=False):
        self.clock = clock
        self.interrupt_on_wait = interrupt_on_wait
        self.interrupted = False
        self.waits = []

    def is_set(self):
        return self.interrupted

    def wait(self, delay):
        self.waits.append(delay)
        self.clock.now += delay
        if self.interrupt_on_wait:
            self.interrupted = True
        return self.interrupted


def request_client(monkeypatch, tmp_path, *, interrupt_on_wait=False):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    client = MistralClient("leanstral", tmp_path)
    clock = FakeMonotonic()
    event = FakeEvent(clock, interrupt_on_wait)
    client.__dict__["_interrupted"] = event
    monkeypatch.setattr(mistral_module.time, "monotonic", clock)
    monkeypatch.setattr(mistral_module.time, "sleep", event.wait)
    return client, clock, event


def http_error(code):
    return urllib.error.HTTPError("https://example.test", code, "error", Message(), None)


def test_request_preserves_retry_delay_progression(monkeypatch, tmp_path):
    client, _, event = request_client(monkeypatch, tmp_path)
    response = object()
    timeouts = []

    def urlopen(_request, timeout):
        timeouts.append(timeout)
        if len(timeouts) <= 7:
            raise TimeoutError("temporary")
        return response

    monkeypatch.setattr(mistral_module.urllib.request, "urlopen", urlopen)

    assert client._request({}, timeout=1_000) is response
    assert event.waits == [2, 5, 15, 30, 60, 120, 120]


def test_request_stops_when_backoff_consumes_deadline(monkeypatch, tmp_path):
    client, _, event = request_client(monkeypatch, tmp_path)
    response = object()
    calls = []

    def urlopen(_request, timeout):
        calls.append(timeout)
        if len(calls) <= 2:
            raise TimeoutError("temporary")
        return response

    monkeypatch.setattr(mistral_module.urllib.request, "urlopen", urlopen)

    with pytest.raises(RuntimeError, match="request timed out"):
        client._request({}, timeout=3)

    assert calls == [3, 1]
    assert event.waits == [2, 1]


def test_request_clamps_urlopen_timeout_to_remaining_deadline(monkeypatch, tmp_path):
    client, clock, event = request_client(monkeypatch, tmp_path)
    response = object()
    timeouts = []

    def urlopen(_request, timeout):
        timeouts.append(timeout)
        clock.now += 1
        if len(timeouts) <= 3:
            raise TimeoutError("temporary")
        return response

    monkeypatch.setattr(mistral_module.urllib.request, "urlopen", urlopen)

    with pytest.raises(RuntimeError, match="request timed out"):
        client._request({}, timeout=10)

    assert timeouts == [10, 7, 1]
    assert event.waits == [2, 5]


@pytest.mark.parametrize(
    "error",
    [
        http_error(409),
        http_error(429),
        http_error(500),
        urllib.error.URLError("temporary"),
        TimeoutError("temporary"),
        ConnectionError("temporary"),
    ],
)
def test_request_retries_each_retryable_error(monkeypatch, tmp_path, error):
    client, _, event = request_client(monkeypatch, tmp_path)
    response = object()
    calls = []

    def urlopen(_request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise error
        return response

    monkeypatch.setattr(mistral_module.urllib.request, "urlopen", urlopen)

    assert client._request({}, timeout=10) is response
    assert calls == [10, 8]
    assert event.waits == [2]


def test_request_surfaces_nonretryable_400_without_retry(monkeypatch, tmp_path):
    client, _, event = request_client(monkeypatch, tmp_path)
    calls = []

    def urlopen(_request, timeout):
        calls.append(timeout)
        raise http_error(400)

    monkeypatch.setattr(mistral_module.urllib.request, "urlopen", urlopen)

    with pytest.raises(urllib.error.HTTPError, match="error"):
        client._request({}, timeout=10)

    assert calls == [10]
    assert event.waits == []


def test_request_interrupts_during_backoff_without_another_attempt(monkeypatch, tmp_path):
    client, _, event = request_client(monkeypatch, tmp_path, interrupt_on_wait=True)
    calls = []

    def urlopen(_request, timeout):
        calls.append(timeout)
        raise TimeoutError("temporary")

    monkeypatch.setattr(mistral_module.urllib.request, "urlopen", urlopen)

    with pytest.raises(Interrupted):
        client._request({}, timeout=10)

    assert calls == [10]
    assert event.waits == [2]
    assert event.is_set()


@pytest.mark.parametrize("timeout", [0, -1])
def test_request_with_nonpositive_timeout_makes_no_attempt(monkeypatch, tmp_path, timeout):
    client, _, _ = request_client(monkeypatch, tmp_path)
    calls = []

    def urlopen(_request, timeout):
        calls.append(timeout)
        pytest.fail("urlopen must not be called")

    monkeypatch.setattr(mistral_module.urllib.request, "urlopen", urlopen)

    with pytest.raises(RuntimeError, match="request timed out"):
        client._request({}, timeout=timeout)

    assert calls == []
