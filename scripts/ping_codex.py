#!/usr/bin/env python3
"""Test Codex app-server client via a smoke/debug harness."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

try:
    from openprover.llm.codex import CodexClient, Interrupted
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from openprover.llm.codex import CodexClient, Interrupted


def _tool_summary(tool: str, args: dict) -> str:
    if "query" in args:
        return str(args["query"])[:60]
    if "code" in args and tool != "lean_verify":
        return str(args["code"]).strip().split("\n")[0][:60]
    return ""


def _print_tool_start(tool: str, args: dict):
    summary = _tool_summary(tool, args)
    line = f"[tool] > {tool}"
    if summary:
        line += f" - {summary}"
    print(f"\n{line}")


def _print_tool_done(tool: str, args: dict, result: str, status: str, duration_ms: int):
    icon = "x"
    if status == "ok":
        icon = "ok"
    elif status == "partial":
        icon = "partial"
    elif status == "running":
        icon = "..."
    summary = _tool_summary(tool, args)
    dur = f" ({duration_ms / 1000:.1f}s)" if duration_ms else ""
    line = f"[tool] > {tool} {icon}{dur}"
    if summary:
        line += f" - {summary}"
    print(f"\n{line}")
    if result:
        first = result.strip().split("\n", 1)[0][:120]
        if first:
            print(f"[tool]   result: {first}")


def _build_mcp_config(lean_project: str) -> dict:
    return {
        "mcp_servers": {
            "lean_tools": {
                "command": sys.executable,
                "args": ["-m", "openprover.lean.mcp_server"],
                "env": {
                    "LEAN_PROJECT_DIR": str(Path(lean_project).resolve()),
                    "LEAN_WORK_DIR": str(Path(lean_project).resolve()),
                },
                "enabled_tools": ["lean_verify", "lean_search"],
                "required": True,
                "startup_timeout_ms": 30000,
            }
        }
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test CodexClient app-server")
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Model name (default: gpt-5.4)",
    )
    parser.add_argument(
        "--prompt",
        default="What is the fundamental theorem of algebra? State it precisely.",
        help="Prompt to send",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--print-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print streamed thinking tokens when present (default: true)",
    )
    parser.add_argument(
        "--lean-project",
        default=None,
        help="Path to Lean project (enables lean MCP tools)",
    )
    parser.add_argument(
        "--interrupt-after",
        type=float,
        default=None,
        help="Soft-interrupt after N seconds and return partial output",
    )
    parser.add_argument(
        "--debug-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print debug callback/event info to stderr (default: false)",
    )
    return parser


def _check_codex_login() -> tuple[bool, str]:
    # Workaround: app-server startup can block before surfacing auth errors.
    try:
        proc = subprocess.run(
            ["codex", "login", "status"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return (
            False,
            "codex CLI not found on PATH; install Codex and run `codex login`.",
        )
    except subprocess.TimeoutExpired:
        return False, "codex login status timed out; run `codex login` and retry."

    if proc.returncode == 0:
        return True, ""

    detail = (proc.stderr or proc.stdout or "").strip().split("\n", 1)[0]
    if detail:
        return False, f"codex auth not ready ({detail}); run `codex login`."
    return False, "codex auth not ready; run `codex login`."


def main(argv=None):
    args = build_parser().parse_args(argv)

    ok, msg = _check_codex_login()
    if not ok:
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    archive_dir = Path(tempfile.mkdtemp(prefix="ping-codex-archive-"))
    client = CodexClient(args.model, archive_dir, max_output_tokens=args.max_tokens)
    if args.lean_project:
        client.mcp_config = _build_mcp_config(args.lean_project)

    dim = "\033[2m" if sys.stdout.isatty() else ""
    reset = "\033[0m" if sys.stdout.isatty() else ""
    saw_reasoning = False
    printed_any = False

    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    timer = None
    started = time.time()

    def on_stream(delta: str, kind: str):
        nonlocal saw_reasoning, printed_any
        if kind == "text":
            sys.stdout.write(delta)
            sys.stdout.flush()
            printed_any = True
            return
        if args.print_reasoning and kind == "thinking":
            if not saw_reasoning:
                sys.stdout.write(dim)
                saw_reasoning = True
            sys.stdout.write(delta)
            sys.stdout.flush()
            printed_any = True
            return
        if args.debug_events:
            print(f"[debug] ignored stream kind={kind}", file=sys.stderr)

    try:
        if args.interrupt_after is not None:
            timer = threading.Timer(args.interrupt_after, client.soft_interrupt)
            timer.daemon = True
            timer.start()
        response = client.call(
            prompt=args.prompt,
            system_prompt="You are a concise assistant.",
            label="ping_codex",
            stream_callback=on_stream,
            tool_start_callback=_print_tool_start,
            tool_callback=_print_tool_done,
            max_tokens=args.max_tokens,
        )
    except Interrupted:
        if saw_reasoning:
            sys.stdout.write(reset)
        print("\n(interrupted)")
        response = {
            "finish_reason": "interrupted",
            "duration_ms": int((time.time() - started) * 1000),
            "result": "",
            "thinking": "",
            "raw": {"usage": {}},
        }
    finally:
        if timer is not None:
            timer.cancel()
        if saw_reasoning:
            sys.stdout.write(reset)
            sys.stdout.flush()
        if not printed_any:
            print()
        client.cleanup()

    elapsed_ms = int((time.time() - started) * 1000)
    finish_reason = response.get("finish_reason", "?")
    duration_ms = response.get("duration_ms", elapsed_ms)
    usage = response.get("raw", {}).get("usage", {})

    print()
    print("-" * 60)
    print(f"Finish reason: {finish_reason}")
    print(f"Duration (ms): {duration_ms}")
    print(f"Prompt tokens:     {usage.get('prompt_tokens', '?')}")
    print(f"Completion tokens: {usage.get('completion_tokens', '?')}")
    print(f"Total tokens:      {usage.get('total_tokens', '?')}")
    if args.debug_events:
        print(
            "Raw keys:",
            json.dumps(sorted(response.get("raw", {}).keys())),
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
