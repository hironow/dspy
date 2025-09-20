# 引き継ぎ書（next_person_doc）

この文書は、今回の作業内容・現状把握・重要な設計判断・今後の示唆をまとめた「遺言書」です。DSPy v3 系の GEPA まわり（optimize/metric/GEPA）と real_world の実例、とくにマルチモーダル例の整理が目的です。

---

## 1. 全体像（何を扱っているか）

- 本リポジトリは DSPy v3 系（`dspy`）に GEPA 最適化（`dspy.GEPA`）を統合した構成。
- GEPA（Genetic-Pareto）は「プログラム内の各 Predictor の instructions（自然言語仕様）」を最適化する進化的最適化器。
- 最低限の要点:
  - 「評価用」`Evaluate` と「最適化用」`GEPA` が同じ `metric` 関数を共有。
  - `Evaluate`: `metric(gold, pred, trace=None) -> float|int|bool`
  - `GEPA`: `metric(gold, pred, trace, pred_name, pred_trace) -> float | dspy.Prediction(score, feedback)`
  - GEPA が探索・更新するのは基本的に「instructions」。Signature の doc/desc は直接の最適化対象ではないが、Adapter が構造化プロンプトを組み立てる際に参照され、出力安定性・反射の質に間接的に効く。

参考コード（実装箇所）
- GEPA 本体（DSPy 側ラッパ）: `dspy/teleprompt/gepa/gepa.py`
- GEPA-Adapter（評価/反射データ生成）: `dspy/teleprompt/gepa/gepa_utils.py`
- 反射用プロポーザ（マルチモーダル対応）: `dspy/teleprompt/gepa/instruction_proposal.py`
- Evaluate 実装: `dspy/evaluate/evaluate.py`
- ブートストラップ＋トレース取得: `dspy/teleprompt/bootstrap_trace.py`

---

## 2. 今回の変更・追加（重要）

1) シンプルなマルチモーダル・キャプション例の微修正
- ファイル: `real_world/simple_gepa_multimodal_caption.py`
- 変更点: 簡潔性ペナルティの閾値の順序を修正（>300 文字に強いペナルティが正しく入るように）

2) 新しい二段構成（Observation→Composition）のマルチモーダル例を追加
- ファイル: `real_world/simple_gepa_multimodal_observe_compose.py`
- 特徴:
  - Signature を初導入（`Analyze`, `Compose`）
  - `analyze: image -> objects/attributes/actions/scene/meta`、`compose: observations -> caption/keywords`
  - GEPA 用メトリクスは pred_name を用いて段別フィードバック（analyze/compose）
  - trace 活用の段間整合性チェック（観測未反映・観測にない語の検出）
  - `add_format_failure_as_feedback=True` でスキーマ準拠を学習材料化
  - ダミー/実LM両対応、保存・要約・コスト見積（共通ユーティリティ使用）

3) real_world/README を更新
- 追加例の説明を追記し、実行意図・設計を明示。

---

## 3. 例の使い方（最低限）

- 依存（例）: `uv add dspy gepa`
- ダミーLM（外部通信なしでドライラン）:
  - `uv run python real_world/simple_gepa_multimodal_caption.py --dummy`
  - `uv run python real_world/simple_gepa_multimodal_observe_compose.py --dummy`
- 実LM（OpenAI 例／helper 経由）:
  - `uv run python real_world/simple_gepa_multimodal_caption.py`
  - `uv run python real_world/simple_gepa_multimodal_observe_compose.py`

出力: ベースライン/ポスト最適化スコア、GEPA 候補の簡易表、instructions の BEFORE/AFTER、保存されたアーティファクト（baseline/optimized/詳細）。

---

## 4. GEPA 運用メモ（要点）

- 予算設定は必須（3 択）: `auto` | `max_full_evals` | `max_metric_calls`。今回の例では dummy=手動、実運用=auto=light。
- 反射モデル: `reflection_lm` を推奨（強めのモデル）。今回は `MultiModalInstructionProposer` と組み合わせ、画像を構造化のまま反射LMへ渡す。
- `component_selector`:
  - 既定（`round_robin`）は挙動が観察しやすい。
  - 同時最適化を試すなら `"all"` も可（コメントで案内を記載済み）。
- `warn_on_score_mismatch`:
  - GEPA は現在、最終スコア（モジュールレベル）を前提。pred_name 付き呼び出しで異なるスコアを返すと警告され、モジュールスコアへ丸め込み。
  - 不要な混乱を避けるため、metric が返す score は基本「モジュールレベル」に揃えるのが無難。
- `track_stats`/`track_best_outputs`:
  - Pareto 前線やタスクごとの最良出力（推論時探索）を追跡可能。必要時のみ有効化。

---

## 5. メトリクス設計の指針（本例での実践）

- Evaluate 用: `metric(gold, pred, trace=None) -> float|int|bool`
- GEPA 用: `metric(gold, pred, trace, pred_name, pred_trace) -> float | dspy.Prediction(score, feedback)`
  - pred_name/pred_trace を使い、最適化対象の Predictor（段）ごとに具体的・短い改善ヒント（feedback）を返す。
  - 今回は「カバレッジ×簡潔性」を基本とし、GEPA 用には段別FB＋トレース整合性の指摘を追加。

Tips:
- 変えたいポリシー（例: 件数上限・順序・簡潔さ）は instructions にも明記（desc は最適化対象外）。
- 失敗（構文/パース）も学習材料にする場合は `add_format_failure_as_feedback=True` を活用（今回済み）。
- 過度な複雑化は避け、短く明確なフィードバック（不足/過剰/構造違反）を返すと効果的。

---

## 6. 重要ファイルと見どころ

- `real_world/simple_gepa_multimodal_caption.py`
  - 単段の Captioner。instructions 最適化の最小例。簡潔性ペナルティの分岐修正済み。

- `real_world/simple_gepa_multimodal_observe_compose.py`
  - 二段構成の Signature 版。trace を使って「観測→作文」の整合性を FB で促す。
  - コメントに補足（component_selector の選び方／軽量メトリクスの微調整案）。

- `dspy/teleprompt/gepa/gepa.py`
  - GEPA ラッパ実装（auto 予算、compile、引数の検証など）。

- `dspy/teleprompt/gepa/gepa_utils.py`
  - DSPyAdapter。トレース収集・反射データ整形・フィードバック統合・（必要に応じ）フォーマット失敗の構造指示付与。

- `dspy/teleprompt/gepa/instruction_proposal.py`
  - MultiModalInstructionProposer。画像など Type を構造化のまま反射LMに渡す。

- `real_world/cost.py`, `real_world/utils.py`, `real_world/save.py`
  - ログにコスト見積を出す／候補・BEFORE/AFTER の整形表示／成果物保存。

---

## 7. 既知の注意点・改善余地

- DspyGEPAResult.to_dict と保存の整合:
  - `dspy/teleprompt/gepa/gepa.py` の `to_dict()` が候補を dict 期待する一方、実際は Module を保持する経路があり得る。
  - これに対し `real_world/save.py` は try/fallback で Module→dict 相当を再構成して保存済み。将来は to_dict 側の一貫化を検討。

- Evaluate の一部引数（他箇所）:
  - 旧コード由来の未使用パラメータの痕跡があれば、不要に渡さないのが無難（挙動には影響なし）。

- 反射の多様性:
  - `reflection_minibatch_size=1` は速いが失敗の多様性が乏しい。2–3 に上げると段別の改善が安定化しやすい。

- 微調整の案（実装はしていない・コメントに記載済み）:
  - `meta` を caption に自然に反映していれば +ε の加点
  - `keywords` 過多（例: 6 以上）にごく小さな減衰

---

## 8. 次の担当者への提案（優先度順）

1) 二段構成例の挙動チェック
   - `--dummy` でのドライラン→実LM で少数例で実行。
   - `reflection_minibatch_size` を 2–3 に上げたときの改善量を比較。

2) component_selector の比較（オプション）
   - `round_robin` vs `"all"` での収束挙動・FB の質を比べる。

3) メトリクスの軽量拡張（必要に応じて）
   - `meta` 反映の小加点・`keywords` 過剰の軽微ペナルティなどを導入し、ルール遵守をもう一段バックアップ。

4) to_dict の一貫化（長期）
   - `DspyGEPAResult` 側の dict 化方針を詰め、保存ロジック側の fallback をシンプルにする。

---

## 9. ひとこと要約（要点復習）

- GEPA は instructions を最適化。metric は Evaluate/GEPA で引数が拡張され、GEPA 時は段別 FB を返すと効く。
- マルチモーダル反射は MultiModalInstructionProposer を使い、画像など Type を“構造化のまま”反射LMへ。
- 二段構成例は Signature を導入、trace で段間整合性をチェック。構文失敗も学習材料に。
- 予算・反射・component_selector・保存・コスト見積までひと通り整っている。

以上。よろしくお願いします。

