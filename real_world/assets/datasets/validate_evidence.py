"""
validate_evidence: LoCoMo10 QA `evidence` 検証ツール

目的
- `locomo10.json` の先頭アイテムに含まれる QA の `evidence`（例: "D3:11"）を
  会話本文（`conversation.session_N` 配列）へマッピングし、対応する話者と発話を
  表示して検証します。

Usage
- uv + module 実行:
  uv run python -m real_world.assets.datasets.validate_evidence [PATH] [--limit N]
- 直接 Python 実行（環境による）:
  python3 -m real_world.assets.datasets.validate_evidence [PATH] [--limit N]

引数 / オプション
- PATH: `locomo10.json` のパス。省略時は本スクリプトと同フォルダの
  `real_world/assets/datasets/locomo10.json` を使用します。
- --limit N: 検証する QA 件数。
  - 省略時は 10 件
  - N <= 0 で全件検証

出力の見方
- 先頭に `Validating X/N QA entries ...` を出力します（X は検証件数、N は総件数）。
- 各 QA について以下を出力します:
  - `[i] Q: <question>`
  - `A: <answer>`（`answer` が存在する場合のみ）
  - `    (no evidence refs)`（`evidence` が空の場合）
  - `    - D#:line :: <speaker>: <text>`（参照が解決できた場合）
  - `    - D#:line :: ERROR: <reason>`（参照が解決できなかった場合）

`answer` / `adversarial_answer` について
- 一部の QA は `answer` を持たず、`adversarial_answer` のみを持つことがあります。
- 現状の出力は `answer` のみ表示します（`adversarial_answer` は非表示）。
- `adversarial_answer` の表示が必要であれば、本ツールにオプションを追加可能です。

終了コード
- 0: すべての参照が解決された（または `evidence` が空のみ）
- 1: `locomo10.json` にアイテムが無い
- 2: 未解決参照が 1 件以上あった

備考
- `evidence` は `;` 区切りで複数参照（例: "D8:6; D9:17"）を含む場合があります。
- 会話インデックスは `conversation.session_N` を `D{N}` に対応づけて構築します。
- メッセージには `dia_id` が含まれることがあります（表示はデフォルトでは省略）。

例
    $ uv run python -m real_world.assets.datasets.validate_evidence --limit 3
    Validating 3/199 QA entries against conversation index built from .../locomo10.json…
    
    [1] Q: When did Caroline go to the LGBTQ support group?
        A: 7 May 2023
        - D1:3 :: Caroline: I went to a LGBTQ support group yesterday ...
"""

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
    """`locomo10.json` の QA `evidence` を検証する。

    Parameters
    - path: `locomo10.json` のパス。
    - limit: 検証する QA 件数。None の場合は全件。0 以下は全件に展開して扱う。

    Returns
    - 0: 参照解決に失敗が無い
    - 1: ルート配列が空（検証対象なし）
    - 2: 1 件以上の未解決参照がある

    Notes
    - 現状は先頭アイテム（`data[0]`）のみ検証対象。
    - `answer` がない QA で `adversarial_answer` のみ存在する場合、`A:` 行は出力しません。
    """
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
