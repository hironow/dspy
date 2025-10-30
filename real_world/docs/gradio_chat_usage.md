# Gradio Chat Console Guide

`real_world/gradio_chat.py` 提供の軽量チャット UI を活用するためのメモです。OpenAI などの既存 LM を切り替えながら、応答に Good/Bad のフィードバックを付与し、結果を TSV 形式でダウンロードできます。

## 主な機能

- **バックエンド切替**  
  - 既定で「dummy (offline echo)」「openai/gpt-4o-mini」「openai/gpt-4o」を選択可能。  
  - `OPENAI_API_KEY` が未設定の場合は OpenAI オプション選択時にエラーを表示。
- **プロンプト調整**  
  - システムプロンプト欄で役割やスタイルを指定。  
  - 温度 (`temperature`) / 最大出力トークン (`max_tokens`) をスライダで変更。
- **フィードバック記録**  
  - 直近の assistant 応答に対し、👍 Good / 👎 Bad を付与。  
  - 履歴（ユーザ発話と応答）を前後コンテキスト付きで記録し、TSV にエクスポート可能。
- **ダウンロード**  
  - “Export feedback TSV” ボタンで `timestamp, user, assistant, label, message, history` の列を含む TSV を生成。  
  - 履歴はアプリの状態管理で保持され、TSV 出力のみファイルに書き出される。

## 実行方法

```bash
uv run python real_world/gradio_chat.py \
    --host 0.0.0.0 \
    --port 7860 \
    --share \
    --default-backend "openai/gpt-4o-mini"
```

### 主な CLI オプション

| オプション | 説明 |
| ---------- | ---- |
| `--host` / `--port` | Gradio のバインド先 |
| `--share` | 公開用の Gradio URL を生成 |
| `--queue` | `demo.queue()` を有効化し同時接続を制御 |
| `--default-backend` | 初期選択バックエンドを指定（リストに存在する必要あり） |
| `--theme` | Gradio のテーマ識別子 |

## バックエンド追加

`ChatBackend` は `OrderedDict` で backend 名 → LM ビルダを管理しています。  
OpenAI 以外のモデルを追加する場合、例えば:

```python
from real_world.gradio_chat import ChatBackend

backend = ChatBackend()
backend._builders["anthropic/claude-3-5-sonnet"] = lambda: anthropic_lm("claude-3-5-sonnet-20241022")
```

`real_world/helper.py` の LM ファクトリを併用すると API キー管理が統一できます。

## フィードバック TSV の構造

| 列 | 内容 |
| --- | --- |
| `timestamp` | ISO8601 (秒精度) |
| `user` | 対象のユーザ発話 |
| `assistant` | 直近のアシスタント応答 |
| `label` | `good` or `bad` |
| `message` | `assistant` と同一（将来拡張用） |
| `history` | 当該ターンまでの会話ログ（`User:/Assistant:` 形式、改行→ `\n` に変換） |

改行やタブは `_sanitize_cell` によりエスケープされるため、TSV を Excel や pandas で安全に読み込めます。

## テスト

非 UI 部分は `real_world/tests/test_gradio_chat.py` でカバーしています。

```bash
uv run pytest real_world/tests/test_gradio_chat.py
```

- `_build_messages` や `_format_history` のフォーマット検証  
- `ChatBackend` のダミー応答・任意バックエンド呼び出し  
- 環境変数チェック `_require_env`

Gradio UI 自体は統合テスト対象外ですが、フィードバックテーブルに関するロジックは unit test で検証可能です。

## カスタマイズ Tips

- **UI 変更**: `build_app` 内の Gradio Blocks を調整。テキストエリアの高さやボタン配置を変えるだけならこちらで完結。
- **永続化**: フィードバック TSV を即時アップロードしたい場合は、`export_feedback` 内で S3 や DB へアップロードする処理に置き換える。
- **多人数利用**: `--queue` を有効にし、必要であればユーザ識別子を入力欄に追加して TSV へ含める。
- **認証**: Gradio の `auth` パラメータを `demo.launch(auth=("user", "password"))` のように指定することで簡易ベーシック認証が可能。

---

このチャット UI は、DSPy プログラムの動作確認やプロンプト改善を素早く検証するための補助ツールとして位置づけています。フィードバック TSV を反復改善や評価データ作成に活用してください。
