# 実世界 GEPA 事例集 (v3+)

このフォルダは、DSPy v3+ における `dspy.GEPA` の最小・実用サンプル集です。スクリプトは「ダミーLMでのローカル検証」と「本物のLMでの実行」の両方に対応し、表示・コスト見積・保存などは共通ユーティリティに切り出しています。

## クイックスタート

- 依存:
  - `uv add dspy gepa`
  - OpenAI/Anthropic を使う場合は環境変数 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`

- ダミーLMで実行（外部通信なし）:
  - `uv run python real_world/simple_gepa_basic.py --dummy`
  - `--log-level DEBUG` で詳細ログ

- 本物のLMで実行（OpenAI例・helper経由）:
  - `uv run python real_world/simple_gepa_basic.py`
  - ほかの `simple_*.py` も同様（--dummy を付けなければ helper が OpenAI LM を初期化）

共通の「事前/事後表示」「コスト見積」「保存」は、`real_world/utils.py`・`real_world/cost.py`・`real_world/save.py` にまとめています。

---

## スクリプト（最小 GEPA パイプライン）

### simple_gepa_basic.py

- 概要: 2段のPredict（rewrite→predict）で日本語QAを解く最小例。
- Metric: 正解/不正解 + 短いテキストFB（GEPA時）
- データ: `factory.basic_qa_dummy(locale="ja")`
- 実LM: `helper` の `openai_gpt_4o_mini_lm`（タスク）/`openai_gpt_4o_lm`（反射）
- 表示/保存/見積: `utils.summarize_*` / `save.save_artifacts` / `cost.log_*`

### simple_gepa_task_metric.py

- 概要: 上記に「タスク特化メトリクス（同義語・近似一致・簡潔さ）」を導入。
- Metric: `score = 0.8*Correctness + 0.2*Brevity`、GEPA時は要素別FBを返す
- データ: `factory.task_metric_qa_dummy(locale="ja")`

### simple_gepa_structured_invoice.py

- 概要: 請求書テキスト→構造化抽出（extract→normalize）。
- Metric: スキーマ検証（vendor/date/amount/currency）。GEPA時は `pred_name/pred_trace` を使い、normalize 向けには from→to 提示（例: `31-12-2024 → 2024-12-31`, `¥ → JPY`）。
- データ: `factory.invoice_dummy(locale="ja")`

### simple_gepa_routed_sources.py

- 概要: Router × 3ソース（DB/RAG/Graph）× 条件付きリランク。
  - light policy（本番想定）: ルータで選んだ1ソースだけを呼ぶ
  - heavy policy（最適化/検証）: 全ソース取得＋リランク
- Metric: 最終テキストの包含一致 + pred_name別FB（route/rerank/source）
- データ: `factory.routed_sources_dummy(locale="ja")`

### simple_gepa_multimodal_caption.py

- 概要: 画像→説明（caption, keywords）。見ていない人に伝わる要点重視の説明を最適化。
- Metric: gold.keywords に対するカバレッジ（caption + pred.keywords）＋簡潔さの軽いペナルティ
- GEPA: MultiModalInstructionProposer を使い、画像を含む反射データから指示文（captionの方針）を最適化
- 注記: このスクリプトは reflection_lm と instruction_proposer の両方を同時に利用する最初の simple 例です
  - GEPA の要件（reflection_lm もしくは instruction_proposer のいずれか必須）に対し、本例は両方を指定
  - reflection_lm にはマルチモーダル対応モデル（例: gpt‑4o）、instruction_proposer には MultiModalInstructionProposer を使用
- データ: `factory.image_caption_dummy(locale="ja")`（dspy.Image をURLで供給）

### simple_gepa_multimodal_observe_compose.py

- 概要: 観測→作文の二段構成（analyze: objects/attributes/actions/scene/meta → compose: caption/keywords）。
- Metric: キーワード被覆＋簡潔さ。GEPA時は pred_name を活かし analyze/compose へ段ごとのFB。
- GEPA: MultiModalInstructionProposer を使用（画像を構造化のまま反射LMへ）。
- データ: `factory.image_caption_dummy(locale="ja")`

---

## ログの見方（よく出るメッセージ）

- Baseline:
  - `Running baseline evaluation on N validation examples...`
  - `PREDICTIVE-NOTE: 推定タスクLM呼び出し回数 (baseline) ≈ ...`（`cost.log_baseline_estimate` 由来）

- GEPA 事前見積:
  - `metric_calls ≈ ...` / `taskLM_calls ≈ ...`（`cost.log_gepa_estimate` 由来）
  - `adapter.evaluate(...)` / `propose_new_texts(...)` の呼び出し箇所メモ

- GEPA 実行/事後:
  - `GEPA compile finished.`
  - `Post-GEPA score: ...`
  - 候補の簡易表（Idx/Score/Best@Val/DiscoveryCalls/Best?）と BEFORE/AFTER 表（`utils.summarize_*`）
  - 事後実測ログ（`cost.log_recorded_gepa_cost` 由来）

---

## 保存（成果物）

- `save.save_artifacts()` により `real_world/exports/` に保存
  - baseline: `<prefix>-baseline-YYYYmmdd-HHMMSS.json`
  - optimized: `<prefix>-optimized-YYYYmmdd-HHMMSS.json`
  - detailed_results: `<prefix>-gepa-details-YYYYmmdd-HHMMSS.json`（可能な場合）

---

## 共有ヘルパー（共通モジュール）

- gradio_chat.py
  - Gradio 製の軽量チャット UI。OpenAI ヘルパーやダミー応答を切り替えて試しながら、最新応答を Good/Bad でラベル付けし、履歴付きで TSV ダウンロードが可能。
  - 詳細ガイド: [docs/gradio_chat_usage.md](docs/gradio_chat_usage.md)
  - 起動例: `uv run python real_world/gradio_chat.py --share`（CLI フラグでバックエンド初期値やテーマを指定可能）。

- helper.py
  - 本物のLMを簡単に取得する小さなヘルパ（OpenAI/Anthropic）。
  - 例: `from real_world.helper import openai_gpt_4o_mini_lm` / `configure_openai(...)`。

- dummy_lm.py
  - ダミーLM（DummyLM）の作成/設定を一本化。JSONAdapter を隠蔽し、安定パースを保証。
  - `make_dummy_lm_json(...)` / `configure_dummy_adapter(...)`。

- factory.py
  - ダミー/実データのファクトリ。全て `(trainset, valset)` を返す。
  - QA/invoice/routed の各種。`locale="ja"` のサポートあり。

- utils.py
  - `summarize_gepa_results(optimized, logger)`：GEPA候補の簡易表
  - `summarize_before_after(before_instructions, optimized, logger)`：指示文の BEFORE/AFTER 表

- cost.py
  - `log_baseline_estimate(...)`：ベースラインの推定呼数
  - `log_gepa_estimate(...)`：GEPAの予算見積（metric_calls / taskLM_calls）
  - `log_recorded_gepa_cost(...)`：最適化ログからの実測値

- save.py
  - `save_artifacts(program, optimized, save_dir, prefix, logger)`：baseline/optimized/detailed_results を保存

- [docs/simple_gepa_architecture.md](docs/simple_gepa_architecture.md)
  - 各サンプルの最適化後アーキテクチャ（アスキー図）と Evaluate/GEPA の役割説明

### Gradio チャット統合の構成

`gradio_chat.py` は単なる LM 実験 UI から発展し、GEPA サンプル群と疎結合で連携できるようになりました。構成イメージは以下の通りです。

```
+----------------------------+
| Gradio UI (gradio_chat.py) |
|  ├─ Backend dropdown       |
|  ├─ Program dropdown       |
|  └─ Chatbot + feedback log |
+-------------+--------------+
              | ProgramRequest
              v
+-----------------------------+
| GEPA Program Registry       |
| (gepa_chat_programs.py)     |
|  ├─ Simple GEPA Basic       |
|  ├─ Task Metric QA          |
|  ├─ Structured Invoice      |
|  ├─ Routed Sources          |
|  ├─ Vector RAG / LoCoMo     |
|  └─ LangExtract fallback    |
+-------------+---------------+
              | require_lm()
              v
+-----------------------------+
| DSPy LM backends (ChatBackend) |
|  ├─ openai/gpt-4o-mini         |
|  ├─ openai/gpt-4o              |
|  └─ dummy (offline echo)       |
+--------------------------------+
```

- Program dropdown で `Raw LM chat` を選択すると従来通りの LM 対話。具体的な `simple_gepa_*` を選ぶと、`ProgramRequest` 経由で最適化済みモジュールが呼び出されます。
- `gepa_chat_programs.py` は起動時に `real_world/exports/` から最新の `*-optimized-*.json`/`*.pkl` を自動ロードします。明示的に指定したい場合は環境変数で上書きできます。
  - `DSPY_GEPA_EXPORT_DIR`：検索するディレクトリを一括変更
  - `DSPY_GEPA_<SLUG>_OPTIMIZED`：特定プログラムの最適化済みファイルを直接指定（例: `DSPY_GEPA_SIMPLE_GEPA_BASIC_OPTIMIZED=/path/to/file.json`）
- 画像対応（`MultimodalTextbox`）により、`simple_gepa_multimodal_caption` や `simple_gepa_multimodal_observe_compose` などのマルチモーダル GEPA サンプルも、依存関係（`dspy.adapters.types.Image` が利用可能な環境）であればドロップダウンから直接呼び出せます。依存が不足している場合は自動的にリストから除外されます。

- 送信時の挙動：
  1. プログラム選択あり → `ChatBackend.require_lm()` で LM インスタンスを取得し、`ProgramRequest` を作成。
  2. プログラムなし → これまで通りメッセージ履歴を LM に渡して応答を生成。
  3. どちらの場合も履歴は `role/content` 形式で保持され、Good/Bad ラベルや TSV エクスポート機能はそのまま利用可能。

---

## 使い分け（Dummy と Real LMs）

- `--dummy` を付けると DummyLM を使い、外部コールは一切不要。
- 付けない場合は `helper` が OpenAI（既定）を初期化。
- `routed_sources` のみ policy を切替：
  - Baseline/Post は `rerank_policy="light"`
  - GEPA中は `rerank_policy="heavy"`

---

## GEPA のメトリクス（共通ルール）

- Evaluate: `metric(example, pred) -> float`
- GEPA: `metric(gold, pred, trace, pred_name, pred_trace) -> float | dspy.Prediction(score, feedback)`
  - Predictor単位のFB（pred_name/pred_trace）で「どこをどう直すか」を具体化

補足:

- 分類向けの厳格・再現重視（FN抑制）メトリクス例を `real_world/metrics_privacy_risk.py` に追加。
  - 出力の厳格バリデーション（"High Risk"/"Low Risk"）
  - 混同行列の簡易トレースログ
  - GEPA向けに常に `dspy.Prediction(score, feedback)` を返す実装
- メトリクス実装の共通ユーティリティを `real_world/metrics_utils.py` に追加。全サンプルで利用しています。
  - `confusion_outcomes(gold_pos, guess_pos, pred_claim)` で TP/FP/TN/FN の算出を統一
  - `safe_trace_log(trace, data)` でトレース出力を例外安全に記録
  - Evaluate ではスカラー、GEPA では `dspy.Prediction(score, feedback)` のルールを徹底
  - フィードバックはテキストのみ（絵文字・記号による意味づけは使用しない）

---

## 出力と保存

- ログにベースライン/最適化後のスコア、GEPA候補の簡易表、指示文の BEFORE/AFTER。
- `real_world/exports/` に baseline/optimized と GEPA 詳細（可能なら）を保存。

---

## よくある拡張

- metric の差し替え（構文検証、LLM-as-judge、ドメイン固有ルール）
- 実データの投入（`factory.*_from_records/csv/jsonl`）
- 反射モデルの強化（`openai_gpt_4o_lm`/Anthropicへ切替）
- Router/reranker の軽量化や policy 設計（本番は light、最適化は heavy）
