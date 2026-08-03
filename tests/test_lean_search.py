import runpy
import sys
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from openprover import cli
from openprover.lean import tools
from openprover.lean.search import SearchResponse, SearchResult, search


class StubResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return False

    def read(self) -> bytes:
        return self.payload


SEARCH_RESPONSE = b'''{
    "count": 1,
    "results": [{
        "id": 213536,
        "name": "Nat.Prime",
        "module": "Mathlib.Data.Nat.Prime.Defs",
        "source_text": "def Prime (p : Nat) := Irreducible p",
        "docstring": "A prime natural number.",
        "informalization": "Prime Number.",
        "undocumented_metadata": "ignored"
    }]
}'''


def test_search_uses_hosted_request_contract_and_parses_results(monkeypatch):
    # Given
    requests = []

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return StubResponse(SEARCH_RESPONSE)

    monkeypatch.setattr("openprover.lean.search.urlopen", urlopen)
    monkeypatch.setenv("LEAN_EXPLORE_API_URL", "https://search.example/api")

    # When
    response = search("Nat.Prime & divisibility", limit=2)

    # Then
    assert response.results == (
        SearchResult(
            name="Nat.Prime",
            module="Mathlib.Data.Nat.Prime.Defs",
            source_text="def Prime (p : Nat) := Irreducible p",
            docstring="A prime natural number.",
            informalization="Prime Number.",
        ),
    )
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.get_method() == "GET"
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("User-agent") == "openprover/1.0"
    assert request.data is None
    assert timeout == 30
    url = urlsplit(request.full_url)
    assert f"{url.scheme}://{url.netloc}{url.path}" == "https://search.example/api"
    assert parse_qs(url.query) == {
        "q": ["Nat.Prime & divisibility"],
        "limit": ["2"],
        "packages": ["Mathlib,Batteries,Init,Lean,Std"],
    }


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://search.example/api", 503, "unavailable", Message(), BytesIO()),
        URLError("offline"),
        TimeoutError("timed out"),
    ],
)
def test_search_propagates_transport_errors_after_one_attempt(monkeypatch, error):
    # Given
    attempts = []

    def urlopen(_request, timeout):
        attempts.append(timeout)
        raise error

    monkeypatch.setattr("openprover.lean.search.urlopen", urlopen)

    # When / Then
    with pytest.raises(type(error)) as raised:
        search("Nat.Prime")

    assert raised.value is error
    assert attempts == [30]


def test_search_propagates_validation_error_after_one_attempt(monkeypatch):
    # Given
    attempts = []

    def urlopen(_request, timeout):
        attempts.append(timeout)
        return StubResponse(b'{"results": [{"name": 1}]}')

    monkeypatch.setattr("openprover.lean.search.urlopen", urlopen)

    # When / Then
    with pytest.raises(ValidationError):
        search("Nat.Prime")

    assert attempts == [30]


def test_native_search_preserves_markdown_status_and_error_behavior(monkeypatch):
    # Given
    response = SearchResponse.model_validate_json(SEARCH_RESPONSE)
    monkeypatch.setattr(tools, "search", lambda query, limit=10: response)

    # When
    result, status = tools._tool_lean_search({"query": "Nat.Prime"}, "worker-1")

    # Then
    assert status == "ok"
    assert result == (
        "**Nat.Prime**  (Mathlib.Data.Nat.Prime.Defs)\n"
        "```lean\n"
        "def Prime (p : Nat) := Irreducible p\n"
        "```\n"
        "A prime natural number.\n"
        "Informalization: Prime Number."
    )

    def fail_search(_query, limit=10):
        raise URLError("offline")

    monkeypatch.setattr(tools, "search", fail_search)

    assert tools._tool_lean_search({"query": "Nat.Prime"}, "worker-1") == (
        "Search error: <urlopen error offline>",
        "error",
    )


def test_cli_removes_fetch_lean_data_subcommand():
    # Given / When / Then
    assert cli.SUBCOMMANDS == {"inspect"}
    assert not hasattr(cli, "_cmd_fetch_lean_data")


def test_script_routes_query_and_limit_to_shared_search(monkeypatch, capsys):
    # Given
    script = runpy.run_path(str(Path("scripts/lean_search.py")))
    calls = []
    response = SearchResponse.model_validate_json(SEARCH_RESPONSE)

    def search_stub(query: str, limit: int = 10):
        calls.append((query, limit))
        return response

    monkeypatch.setitem(script["main"].__globals__, "search", search_stub)
    monkeypatch.setattr(sys, "argv", ["lean_search.py", "Nat.Prime", "--limit", "2"])

    # When
    script["main"]()

    # Then
    assert calls == [("Nat.Prime", 2)]
    assert "1. Nat.Prime" in capsys.readouterr().out
