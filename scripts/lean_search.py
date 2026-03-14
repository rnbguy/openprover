#!/usr/bin/env python3
"""Standalone lean_search tool for testing and benchmarking."""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Standalone lean_search")
    parser.add_argument("query", nargs="?", help="Search query (omit for interactive mode)")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--local", action="store_true", default=True,
                        help="Use local data (default)")
    parser.add_argument("--no-local", action="store_false", dest="local",
                        help="Download data instead of using local")
    args = parser.parse_args()

    print("Initializing SearchEngine...", flush=True)
    t0 = time.time()
    from lean_explore.search import SearchEngine, Service
    engine = SearchEngine(use_local_data=args.local)
    service = Service(engine=engine)
    print(f"Engine ready in {time.time() - t0:.1f}s\n")

    import torch
    rerank = 0 if args.no_rerank else (25 if torch.cuda.is_available() else 0)
    if rerank:
        print(f"Reranking top {rerank} (CUDA available)")
    else:
        print("Reranking disabled" if args.no_rerank else "Reranking disabled (no CUDA)")
    print()

    import asyncio

    def run_query(query: str):
        t0 = time.time()
        results = asyncio.run(service.search(query, limit=args.limit, rerank_top=rerank))
        elapsed = time.time() - t0
        print(f"--- {len(results) if results else 0} results in {elapsed:.2f}s ---\n")
        if not results:
            print("No results found.\n")
            return
        for i, r in enumerate(results, 1):
            name = getattr(r, 'name', str(r))
            sig = getattr(r, 'signature', '') or ''
            doc = getattr(r, 'doc_string', '') or ''
            print(f"{i}. {name}")
            if sig:
                print(f"   {sig}")
            if doc:
                for line in doc.strip().splitlines()[:3]:
                    print(f"   {line}")
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
