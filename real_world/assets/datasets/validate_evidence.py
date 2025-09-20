"""
validate_evidence: LoCoMo10 QA `evidence` 検証ツール

目的
- `locomo10.json` の各アイテムに含まれる QA の `evidence`（例: "D3:11"）を
  会話本文（`conversation.session_N` 配列）へマッピングし、対応する話者と発話を
  表示して検証します。

Usage
- uv + module 実行:
  uv run python -m real_world.assets.datasets.validate_evidence [PATH] [--limit N] [--strict [--strict-ts {required|warn|off}] [--rag RAG_PATH [--rag-id ID]]]
- 直接 Python 実行（環境による）:
  python3 -m real_world.assets.datasets.validate_evidence [PATH] [--limit N] [--strict [--strict-ts {required|warn|off}] [--rag RAG_PATH [--rag-id ID]]]

引数 / オプション
- PATH: `locomo10.json` のパス。省略時は本スクリプトと同フォルダの
  `real_world/assets/datasets/locomo10.json` を使用します。
- --limit N: 検証する QA 件数（各アイテムごとに先頭から N 件）。
  - 省略時は全件検証（標準動作）
  - N <= 0 でも全件検証（各アイテム全件）
- --strict: 厳格モード。`locomo10_rag.json` と突合し、全 `evidence` の発話が RAG 側にも存在すること、各 `session_N_date_time` が RAG 側の `timestamp` に含まれることを検証。
  - 注: `--strict` のチェックは `--limit` に関わらず、対象アイテム内の全 `evidence` 参照を走査します。
- --strict-ts {required|warn|off}: 厳格モードにおけるタイムスタンプ照合の扱い（既定: off）。
  - required: 1件でも不一致があれば失敗（終了コード 3）
  - warn: 不一致は警告のみ（終了コードは成功側）
  - off: タイムスタンプ照合を無効化（発話一致のみチェック）
- --rag RAG_PATH: RAG ファイルへのパス（省略時は `PATH` と同じフォルダの `locomo10_rag.json`）。
- --rag-id ID: RAG のトップレベルキー。`auto`（既定）で全アイテムを index=ID で自動突合。

出力の見方
- 先頭に `Validating X/N QA entries ...` もしくは `Validating all N QA entries ...` を出力します。
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
- 0: すべての参照が解決された（または `evidence` が空のみ）。`--strict` の場合は、発話一致に不一致がなく、`--strict-ts` の設定に応じた条件を満たす。
- 1: `locomo10.json` にアイテムが無い
- 2: 未解決参照が 1 件以上あった（`--strict` 未使用時）
- 3: `--strict` で RAG 突合に不一致があった（発話不一致、または `--strict-ts required` かつタイムスタンプ不一致）

備考
- `evidence` は `;` 区切りで複数参照（例: "D8:6; D9:17"）を含む場合があります。
- 会話インデックスは `conversation.session_N` を `D{N}` に対応づけて構築します。
- メッセージには `dia_id` が含まれることがあります（表示はデフォルトでは省略）。

例
    $ uv run python -m real_world.assets.datasets.validate_evidence
    Validating all QA entries against conversation index built from .../locomo10.json…
    
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
    load_locomo10_rag,
    strict_cross_validate,
    strict_cross_validate_dataset,
    resolve_evidence,
)


def validate(
    path: Path,
    limit: Optional[int] = None,
    *,
    strict: bool = False,
    strict_ts: str = "off",
    rag_path: Optional[Path] = None,
    rag_id: str = "auto",
) -> int:
    """`locomo10.json` の QA `evidence` を検証する。

    Parameters
    - path: `locomo10.json` のパス。
    - limit: 検証する QA 件数。None の場合は全件。0 以下は全件に展開して扱う。
    - strict: RAG とのクロス検証を有効化するか。
    - strict_ts: タイムスタンプ照合の扱い（"required" | "warn" | "off"）。
    - rag_path: RAG ファイルのパス（省略時は同フォルダの `locomo10_rag.json`）。
    - rag_id: RAG のトップレベルキー（デフォルト: "auto"）。

    Returns
    - 0: 参照解決に失敗が無い
    - 1: ルート配列が空（検証対象なし）
    - 2: 1 件以上の未解決参照がある

    Notes
    - 全アイテムが検証対象です（標準動作）。
    - `answer` がない QA で `adversarial_answer` のみ存在する場合、`A:` 行は出力しません。
    """
    try:
        data = load_locomo10(path)
    except Exception as e:
        print(f"Failed to load locomo10.json at {path}: {e}")
        return 1
    if not data:
        print("No items found in locomo10.json")
        return 1
    total_errors = 0
    # Collect unresolved refs for report file
    unresolved: dict[tuple[int, int], dict] = {}
    for item_idx, item in enumerate(data):
        index = build_conversation_index(item)
        qa_list = item.get("qa", [])
        n = len(qa_list)
        count = min(limit, n) if limit is not None else n
        if limit is None:
            header = f"Item {item_idx}: Validating all {n} QA entries (from {path})"
        else:
            header = f"Item {item_idx}: Validating {count}/{n} QA entries (from {path})"
        print("\n" + header)
        errors = 0
        for i, qa in enumerate(qa_list[:count], start=1):
            q = qa.get("question")
            a = qa.get("answer")
            res = resolve_evidence(qa, index)
            print(f"\n  [{i}] Q: {q}")
            if a is not None:
                print(f"      A: {a}")
            if not res:
                print("      (no evidence refs)")
                continue
            for r in res:
                if r["ok"]:
                    sp = r.get("speaker") or "?"
                    txt = (r.get("text") or "").replace("\n", " ")
                    print(f"      - {r['ref']} :: {sp}: {txt}")
                else:
                    errors += 1
                    print(f"      - {r['ref']} :: ERROR: {r.get('error')}")
                    # Accumulate unresolved details per QA
                    key = (item_idx, i)
                    entry = unresolved.get(key)
                    if not entry:
                        entry = {
                            "item_idx": item_idx,
                            "qa_idx": i,
                            "question": q,
                            "evidence": qa.get("evidence") or [],
                            "unresolved": [],
                        }
                        unresolved[key] = entry
                    entry["unresolved"].append({
                        "ref": r.get("ref"),
                        "reason": r.get("error") or "unknown",
                    })
        total_errors += errors
        if errors == 0:
            print("\n  All referenced evidence resolved successfully for this item.")
        else:
            print(f"\n  Completed with {errors} unresolved references for this item.")

    if total_errors:
        print(f"\nCompleted with {total_errors} unresolved references across all items.")
        # Write validate_result.txt alongside this script for manual inspection
        report_path = Path(__file__).with_name("validate_result.txt")
        try:
            lines = []
            lines.append("Evidence Validation Report (unresolved only)\n")
            lines.append(f"Dataset: {path}")
            lines.append(f"Total unresolved: {total_errors}\n")
            lines.append("Manual action required: 以下の参照は解決できませんでした。データセットを手動で確認・修正してください。\n")
            # Sort by item_idx, qa_idx
            for (it, qi) in sorted(unresolved.keys()):
                e = unresolved[(it, qi)]
                lines.append(f"- Item {e['item_idx']} / QA {e['qa_idx']}")
                if e.get("question") is not None:
                    lines.append(f"  Q: {e['question']}")
                if e.get("evidence"):
                    evs = ", ".join(map(str, e["evidence"]))
                    lines.append(f"  evidence: [{evs}]")
                lines.append("  unresolved:")
                for ur in e["unresolved"]:
                    lines.append(f"    - ref={ur.get('ref')} reason={ur.get('reason')}")
                lines.append("")
            report_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Report written to {report_path}. 手動でチェックし、当該データを修正してください。")
        except Exception as e:
            print(f"Failed to write validate_result.txt: {e}")
        return 2

    print("\nAll referenced evidence resolved successfully across all items.")

    if strict:
        # Load RAG and cross-validate all refs across items (not just limited QA)
        rp: Path
        if rag_path is not None:
            rp = Path(rag_path)
        else:
            rp = Path(__file__).with_name("locomo10_rag.json")
        try:
            rag = load_locomo10_rag(rp)
        except Exception as e:
            print(f"Failed to load locomo10_rag.json at {rp}: {e}")
            return 3
        print(f"\nStrict mode cross-check against {rp}")

        # Determine scope: auto -> all items by index; otherwise single id
        if rag_id and rag_id != "auto":
            # Validate single pair: map dataset index=int(rag_id)
            try:
                idx = int(rag_id)
            except ValueError:
                print(f"Strict mode: --rag-id must be an integer or 'auto', got '{rag_id}'")
                return 3
            if not (0 <= idx < len(data)):
                print(f"Strict mode: dataset index {idx} out of range (0..{len(data)-1})")
                return 3
            if rag_id not in rag:
                print(f"Strict mode: RAG id '{rag_id}' not found in {rp}")
                return 3
            item = data[idx]
            report = strict_cross_validate(item, rag[rag_id])
            missing = report["missing_refs"]
            ts_missing = report["session_timestamps_missing"]
            total_refs = report["total_refs"]
            print(f"- id={rag_id}: Total refs checked: {total_refs}")
            if ts_missing:
                if strict_ts == "required":
                    print("  Missing session timestamps in RAG:")
                    for ts in ts_missing:
                        print(f"    * {ts}")
                elif strict_ts == "warn":
                    print("  Warning: Missing session timestamps in RAG:")
                    for ts in ts_missing:
                        print(f"    * {ts}")
            if missing:
                print("  Missing refs in RAG (speaker/text not found):")
                for m in missing[:20]:
                    sp = m.get("speaker") or "?"
                    tx = (m.get("text") or "").replace("\n", " ")
                    print(f"    * {m['ref']} :: {sp}: {tx}  [{m.get('reason')}]")
                if len(missing) > 20:
                    print(f"    ... and {len(missing)-20} more")
            fail_ts = bool(ts_missing) and (strict_ts == "required")
            fail_text = bool(missing)
            if fail_text or fail_ts:
                print("\nStrict check FAILED: RAG cross-validation mismatches detected.")
                return 3
            print("\nStrict check PASSED: RAG cross-validation succeeded.")
        else:
            # Validate all items
            reports = strict_cross_validate_dataset(data, rag)
            total_refs = 0
            total_missing = 0
            total_ts_missing = 0
            for key in sorted(reports.keys(), key=int):
                r = reports[key]
                total_refs += r.get("total_refs", 0)
                missing = r.get("missing_refs", [])
                ts_missing = r.get("session_timestamps_missing", [])
                if r.get("missing_rag_entry"):
                    print(f"- id={key}: RAG entry missing")
                    total_missing += 1
                else:
                    if missing:
                        total_missing += len(missing)
                        print(f"- id={key}: {len(missing)} missing refs")
                    if ts_missing:
                        total_ts_missing += len(ts_missing)
                        if strict_ts in ("required", "warn"):
                            prefix = "Warning" if strict_ts == "warn" else "Missing"
                            print(f"- id={key}: {prefix} session timestamps: {len(ts_missing)}")
            print(f"- Total refs checked: {total_refs}")
            fail_ts = (strict_ts == "required") and (total_ts_missing > 0)
            fail_text = total_missing > 0
            if fail_text or fail_ts:
                print("\nStrict check FAILED: RAG cross-validation mismatches detected.")
                return 3
            print("\nStrict check PASSED: RAG cross-validation succeeded.")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and preview evidence refs in locomo10.json")
    ap.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("locomo10.json")),
        help="Path to locomo10.json (default: alongside this script)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Number of QA entries to check (default: all; use 0 or negative for all)")
    ap.add_argument("--strict", action="store_true", help="Enable strict mode: cross-check against locomo10_rag.json as well")
    ap.add_argument(
        "--strict-ts",
        choices=["required", "warn", "off"],
        default="off",
        help="Timestamp check policy in strict mode (default: off)",
    )
    ap.add_argument("--rag", dest="rag_path", help="Path to locomo10_rag.json (default: alongside the script)")
    ap.add_argument("--rag-id", default="auto", help="Top-level key in RAG JSON to use (default: 'auto')")
    args = ap.parse_args()

    path = Path(args.path)
    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    rag_path = Path(args.rag_path) if args.rag_path else None
    return validate(
        path,
        limit,
        strict=args.strict,
        strict_ts=args.strict_ts,
        rag_path=rag_path,
        rag_id=args.rag_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
