from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .evidence_utils import (
    build_conversation_index,
    load_locomo10,
    resolve_evidence,
)


def validate(path: Path, limit: Optional[int] = 10) -> int:
    data = load_locomo10(path)
    if not data:
        print("No items found in locomo10.json")
        return 1
    item = data[0]

    index = build_conversation_index(item)
    qa_list = item.get("qa", [])
    n = len(qa_list)
    count = min(limit, n) if limit is not None else n

    print(f"Validating {count}/{n} QA entries against conversation index built from {path}…")
    errors = 0
    for i, qa in enumerate(qa_list[:count], start=1):
        q = qa.get("question")
        a = qa.get("answer")
        res = resolve_evidence(qa, index)
        print(f"\n[{i}] Q: {q}")
        if a is not None:
            print(f"    A: {a}")
        if not res:
            print("    (no evidence refs)")
            continue
        for r in res:
            if r["ok"]:
                sp = r.get("speaker") or "?"
                txt = (r.get("text") or "").replace("\n", " ")
                print(f"    - {r['ref']} :: {sp}: {txt}")
            else:
                errors += 1
                print(f"    - {r['ref']} :: ERROR: {r.get('error')}")

    if errors:
        print(f"\nCompleted with {errors} unresolved references.")
        return 2
    print("\nAll referenced evidence resolved successfully.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and preview evidence refs in locomo10.json")
    ap.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("locomo10.json")),
        help="Path to locomo10.json (default: alongside this script)",
    )
    ap.add_argument("--limit", type=int, default=10, help="Number of QA entries to check (default: 10; use 0 or negative for all)")
    args = ap.parse_args()

    path = Path(args.path)
    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    return validate(path, limit)


if __name__ == "__main__":
    raise SystemExit(main())

