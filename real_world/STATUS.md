# 実世界 TODO / FIXME 一覧

以下は `real_world/` 配下で確認された未対応の TODO/FIXME です。対応方針の検討や担当割り当ての際に参照してください。

- `real_world/simple_gepa_vector_rag_locomo.py:233` — LoCoMo データセット読み込み時の巨大ファイル処理を最適化するため、部分読み込み／ストリーミング対応が求められている。
- `real_world/simple_gepa_vector_rag_locomo.py:255` — RAG 検索時にエビデンス情報を用いて検索空間をフィルタリングする処理の実装が未完了。

今後新しい TODO/FIXME を追加・完了した場合は、本ファイルを併せて更新してください。

## 完了済み

- `real_world/gradio_chat.py` — Gradio ベースのチャット UI（バックエンド切替と good/bad フィードバック機能付き）。フィードバックは UI 上で TSV としてエクスポート可能です。CLI から `uv run python real_world/gradio_chat.py` で起動できます。
