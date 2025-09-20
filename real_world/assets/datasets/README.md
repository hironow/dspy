# LoCoMo10 データセット概要（構造と最小例）

このディレクトリには、LoCoMo10 系のデータが格納されています。どちらの JSON も大きなファイルですが、ここに示す構造説明と小さな例を見れば、中身を直接開かなくても扱えるようにまとめています。

- `real_world/assets/datasets/locomo10.json`
- `real_world/assets/datasets/locomo10_rag.json`

## locomo10.json の構造

- 形式: 配列（リスト）。各要素が 1 つのシナリオ/会話セットを表すオブジェクト。
- 主なフィールド:
  - `qa`: 質問応答（Q/A）エントリの配列。
    - `question`: 文字列（例: "When did Caroline go to the LGBTQ support group?")
    - `answer`: 文字列または数値（例: `"7 May 2023"` や `2022`）
    - `evidence`: 文字列の配列。`"D<number>:<line>"` 形式の参照（例: `"D1:3"`）。
    - `category`: 整数（タスク/質問の種別を表すカテゴリ ID）

- `evidence` について:
  - `"D1:3"` のように、`D<文書番号>:<行番号>` を指す参照文字列です。
  - 参照先の文書本文は同一オブジェクト内の別フィールド（例: 文書配列/マップ）に置かれているケースが一般的です。`evidence` のフォーマットに従って `D#`（文書 ID 相当）と `line`（行/文のインデックス）を抽出し、照合してください。

- 最小例（構造を示すためのサンプル）:

```json
[
  {
    "qa": [
      {
        "question": "When did Caroline go to the LGBTQ support group?",
        "answer": "7 May 2023",
        "evidence": ["D1:3"],
        "category": 2
      },
      {
        "question": "When did Melanie paint a sunrise?",
        "answer": 2022,
        "evidence": ["D1:12"],
        "category": 2
      }
    ]
    /* …必要に応じて文書本文やメタ情報などが続く… */
  }
]
```

- Python 読み込み例:

```python
import json
from pathlib import Path

path = Path("real_world/assets/datasets/locomo10.json")
with path.open("r", encoding="utf-8") as f:
    data = json.load(f)  # data は list

for item in data:
    for qa in item.get("qa", []):
        q = qa["question"]
        a = qa["answer"]  # str または int などの場合あり
        ev = qa.get("evidence", [])  # ["D1:3", ...]
        cat = qa.get("category")
        # ここで処理
```

`answer` は文字列と数値が混在しうるため、型には注意してください。

## locomo10_rag.json の構造

- 形式: オブジェクト（マップ/辞書）。トップレベルのキーは文字列の数値 ID（例: `"0"`, `"1"`, …）。
- 各キーに対応する値は 1 つの会話セッションを表すオブジェクト。
- 主なフィールド:
  - `conversation`: メッセージの配列。
    - 各メッセージはオブジェクトで、少なくとも以下を持ちます:
      - `timestamp`: 文字列（例: "1:56 pm on 8 May, 2023"）
      - `speaker`: 文字列（例: "Caroline", "Melanie"）
      - `text`: 文字列（発話内容）

- 最小例（構造を示すためのサンプル）:

```json
{
  "0": {
    "conversation": [
      {
        "timestamp": "1:56 pm on 8 May, 2023",
        "speaker": "Caroline",
        "text": "Hey Mel! Good to see you! How have you been?"
      },
      {
        "timestamp": "1:56 pm on 8 May, 2023",
        "speaker": "Melanie",
        "text": "I'm swamped with the kids & work. What's up with you?"
      }
    ]
    /* …必要に応じて他のメタ情報が続く… */
  },
  "1": { "conversation": [/* … */] }
}
```

- Python 読み込み例:

```python
import json
from pathlib import Path

path = Path("real_world/assets/datasets/locomo10_rag.json")
with path.open("r", encoding="utf-8") as f:
    data = json.load(f)  # data は dict（キーは "0", "1", ...）

for conv_id, session in data.items():
    for msg in session.get("conversation", []):
        ts = msg.get("timestamp")
        sp = msg.get("speaker")
        tx = msg.get("text")
        # ここで処理
```

## 2 つのファイルの関係（利用イメージ）

- `locomo10_rag.json`: 時系列の会話本文を保持（RAG 用のコンテキストに相当）。
- `locomo10.json`: 会話から解答可能な質問集（Q/A）と、その根拠位置（`evidence`）を保持。
- RAG（検索・読み込み）で会話の一部を取り出し、`locomo10.json` の QA を評価に使う、という流れで組み合わせると用途が広がります。

## 実装時の注意

- ファイルサイズが大きいため、必要部分のみをストリーミング/チャンク読み込みするか、事前にインデックス化するのが推奨です。
- 文字コードは UTF-8。改行は LF（Unix 形式）。
- `answer` の型（文字列/数値混在）と `evidence` のパース（`D#:line` 形式）に注意してください。

### `evidence` 参照の簡易パーサ例（任意）

```python
def parse_evidence_tag(tag: str):
    # 例: "D3:13" -> ("D3", 13)
    doc_id, line_str = tag.split(":", 1)
    return doc_id, int(line_str)
```

この README だけで、両ファイルの一般的な構造と扱い方が分かるようにしてあります。詳細な内容確認が必要な場合のみ、実ファイルを開いてください。

## 付属ユーティリティ

- `real_world/assets/datasets/evidence_utils.py`
  - `build_conversation_index(item)`: `conversation.session_N` を走査して `{ "D{N}": [msg, ...] }` を作成。
  - `parse_all_evidence(evidence)`: `"D8:6; D9:17"` のような複合参照を展開。
  - `resolve_evidence(qa_entry, index)`: QA の `evidence` を実際の発話テキスト/話者に解決。

- `real_world/assets/datasets/validate_evidence.py`
  - 簡易検証 CLI。標準では全 QA を検証。`--limit` で先頭から指定件数のみプレビュー可能。
  - `--strict` を付けると `locomo10_rag.json` とのクロス検証も実施（`--rag`, `--rag-id` 指定可）。
  - `--strict-ts {required|warn|off}` でタイムスタンプ照合の厳密度を選択（既定: off）。

実行例（リポジトリルートで）:

```bash
# 標準: 全 QA を検証
uv run python -m real_world.assets.datasets.validate_evidence
# プレビュー（先頭から N 件のみ）
uv run python -m real_world.assets.datasets.validate_evidence --limit 10
# RAG とも突合（strict モード; 全件）
uv run python -m real_world.assets.datasets.validate_evidence --strict
# QA 用にタイムスタンプ照合を無効化して strict（全件）
uv run python -m real_world.assets.datasets.validate_evidence --strict --strict-ts off
```

プログラムからの利用例:

```python
from real_world.assets.datasets.evidence_utils import load_locomo10, build_conversation_index, resolve_evidence

data = load_locomo10("real_world/assets/datasets/locomo10.json")
item = data[0]
index = build_conversation_index(item)
qa = item["qa"][0]
resolved = resolve_evidence(qa, index)
for r in resolved:
    if r["ok"]:
        print(r["ref"], r["speaker"], r["text"])  # D#:line, 話者, 発話
```

## データ修正履歴（アノテーション）

最終更新: 2025-09-20（全件検証および strict 突合済み）

- 目的: `locomo10.json` の `evidence` 欄に存在した不正/不整合な参照を修正し、全件で参照解決可能にしました。
- 検証結果: `validate_evidence` で「All referenced evidence resolved successfully across all items.」を確認。
  また strict モードで `locomo10_rag.json` とも突合し、「Total refs checked: 2822」「Strict check PASSED」を確認。

修正内容（抜粋・Item/QA は 0 始まりの Item, 1 始まりの表示 QA に準拠）

- Item 3 / QA 59（What things has Nate reccomended to Joanna?）
  - 誤: `D10:19`（`session_10` に 19行目が存在せず範囲外）
  - 正: `D20:15`（バター代替の推奨発話に一致）

- Item 3 / QA 89（What is one of Joanna's favorite movies?）
  - 誤: `evidence: ["D1:18", "D", "D1:20"]`（無効トークン `D` を含む）
  - 正: `evidence: ["D1:18", "D1:20"]`

- Item 4 / QA 19（What authors has Tim read books from?）
  - 誤: `D:11:26`（コロン位置不正）
  - 正: `D11:26`

- Item 6 / QA 39（What happened to John's job situation in 2022?）
  - 誤: `D4:36`（`session_4` の長さ超過により範囲外）
  - 正: `D4:36` を削除（他の根拠 `D18:1`, `D18:7` により成立）

- Item 8 / QA 32（How might Evan and Sam's experiences ...?）
  - 誤: `"D9:1 D4:4 D4:6"`（スペース連結）
  - 正: `"D9:1", "D4:4", "D4:6"`（トークン分割）

- Item 8 / QA 39（What role does nature and the outdoors ...?）
  - 誤: `"D22:1 D22:2 D9:10 D9:11"`
  - 正: `"D22:1", "D22:2", "D9:10", "D9:11"`

- Item 8 / QA 47（How do Evan and Sam use creative outlets ...?）
  - 誤: `"D21:18 D21:22 D11:15 D11:19"`
  - 正: `"D21:18", "D21:22", "D11:15", "D11:19"`

補足

- 検証・修正には `validate_evidence` を使用。未解決がある場合は自動生成される `validate_result.txt` に未解決一覧（Item/QA/参照/理由）を出力するよう対応済みです。
- strict 突合ではテキスト正規化とタイムスタンプの簡易正規化（大文字小文字/空白の正規化）を行い、`rag-id=auto` デフォルトで全アイテムを `locomo10_rag.json` と照合します。
