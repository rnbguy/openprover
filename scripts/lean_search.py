#!/usr/bin/env python3
"""Standalone lean_search tool for testing and benchmarking."""

import argparse
import time

from openprover.lean.search import search


def main():
    parser = argparse.ArgumentParser(description="Standalone lean_search")
    parser.add_argument("query", nargs="?", help="Search query (omit for interactive mode)")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    args = parser.parse_args()

    def run_query(query: str):
        t0 = time.time()
        response = search(query, limit=args.limit)
        elapsed = time.time() - t0
        results = response.results
        print(f"--- {len(results)} results in {elapsed:.2f}s ---\n")
        if not results:
            print("No results found.\n")
            return
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.name}")
            if r.module:
                print(f"   module: {r.module}")
            if r.source_text:
                # Show first 3 lines of source
                lines = r.source_text.strip().splitlines()
                for line in lines[:3]:
                    print(f"   {line}")
                if len(lines) > 3:
                    print(f"   ... ({len(lines) - 3} more lines)")
            if r.docstring:
                print(f"   -- {r.docstring.strip().splitlines()[0]}")
            if r.informalization:
                print(f"   info: {r.informalization.strip().splitlines()[0]}")
            print()

    if args.query:
        run_query(args.query)
    else:
        print("Interactive mode — type queries, Ctrl+C to exit\n")
        while True:
            try:
                query = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            run_query(query)


if __name__ == "__main__":
    main()
