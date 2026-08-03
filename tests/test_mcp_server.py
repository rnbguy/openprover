from threading import Barrier
from types import SimpleNamespace

import anyio
import pytest
from mcp import Client
from mcp.server import MCPServer

from openprover import __version__
from openprover.lean import mcp_server

@pytest.fixture(autouse=True)
def reset_mcp_server_state(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(mcp_server, "_project_dir", tmp_path)
    monkeypatch.setattr(mcp_server, "_work_dir", mcp_server.LeanWorkDir(tmp_path))
    monkeypatch.setattr(mcp_server, "_store", "")


def test_mcp_server_registers_only_text_tools_with_openprover_version():
    async def check_server() -> None:
        async with Client(mcp_server.mcp, raise_exceptions=True) as client:
            assert isinstance(mcp_server.mcp, MCPServer)
            assert client.server_info.name == "lean_tools"
            assert client.server_info.version == __version__
            tools = (await client.list_tools()).tools
            assert [tool.name for tool in tools] == ["lean_verify", "lean_store", "lean_search"]
            assert all(tool.output_schema is None for tool in tools)

    anyio.run(check_server)


def test_lean_verify_returns_text_only_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mcp_server, "run_lean_check", lambda path, project_dir: (True, "", ""))

    async def call_verify() -> None:
        async with Client(mcp_server.mcp, raise_exceptions=True) as client:
            result = await client.call_tool("lean_verify", {"code": "example : True := by trivial"})
            assert [content.text for content in result.content] == ["OK - no errors"]
            assert result.is_error is False
            assert result.structured_content is None

    anyio.run(call_verify)


def test_lean_search_returns_text_only_success(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def search_stub(query: str, limit: int = 10):
        return SimpleNamespace(results=())

    async def run_sync(function, *args):
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(mcp_server, "search", search_stub)
    monkeypatch.setattr(mcp_server.anyio.to_thread, "run_sync", run_sync)

    async def call_search() -> None:
        async with Client(mcp_server.mcp, raise_exceptions=True) as client:
            result = await client.call_tool("lean_search", {"query": "Nat.Prime"})
            assert [content.text for content in result.content] == ["No results found"]
            assert result.is_error is False
            assert result.structured_content is None

    anyio.run(call_search)
    assert calls == [(search_stub, ("Nat.Prime", 10))]


def test_concurrent_lean_store_keeps_each_verified_update(monkeypatch: pytest.MonkeyPatch):
    simultaneous_checks = Barrier(2)

    def successful_check(path, project_dir):
        if getattr(mcp_server, "_store_lock", None) is None:
            simultaneous_checks.wait(timeout=5)
        return (True, "", "")

    monkeypatch.setattr(mcp_server, "run_lean_check", successful_check)
    results = []

    async def store(client: Client, code: str) -> None:
        results.append(await client.call_tool("lean_store", {"code": code}))

    async def store_concurrently() -> None:
        async with (
            Client(mcp_server.mcp, raise_exceptions=True) as client,
            anyio.create_task_group() as task_group,
        ):
            task_group.start_soon(store, client, "def first : Nat := 1")
            task_group.start_soon(store, client, "def second : Nat := 2")

    anyio.run(store_concurrently)

    assert all(result.is_error is False for result in results)
    assert "def first : Nat := 1" in mcp_server._store
    assert "def second : Nat := 2" in mcp_server._store
