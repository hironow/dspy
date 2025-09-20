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
  - 簡易検証 CLI。先頭の QA から指定件数だけ `evidence` を解決して表示。
  - `--strict` を付けると `locomo10_rag.json` とのクロス検証も実施（`--rag`, `--rag-id` 指定可）。
  - `--strict-ts {required|warn|off}` でタイムスタンプ照合の厳密度を選択（既定: off）。

実行例（リポジトリルートで）:

```bash
python -m real_world.assets.datasets.validate_evidence --limit 10
# すべて検証したい場合
python -m real_world.assets.datasets.validate_evidence --limit 0
# RAG とも突合（strict モード）
python -m real_world.assets.datasets.validate_evidence --strict --limit 0
# QA 用にタイムスタンプ照合を無効化して strict
python -m real_world.assets.datasets.validate_evidence --strict --strict-ts off --limit 0
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
