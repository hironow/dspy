# GEPA × LangExtract：利点と実践ガイド

このメモは、LangExtract を GEPA（dspy.GEPA）で最適化する意義と、実装・運用の勘所を短くまとめたものです。実装例は `real_world/simple_gepa_langextract.py` を参照してください。

## なぜ GEPA で LangExtract を最適化するか

- 反復の自動化と速度向上
  - 人手の「書く→試す→直す」を、メトリクス駆動で反射的に回す。短い予算でも改善が積み上がる。
- 具体フィードバックの活用（pred_name / pred_trace）
  - 不足クラス・JSON不備・厳密スパン/重複の違反などをピンポイントに指摘。暗黙知を metric に固定化できる。
- バッチ頑健性（Pareto 選択）
  - 検証集合全体の Pareto 前線を維持しつつ改善。単一テキストへの過適合を避ける。
- 予算管理と再現性
  - `auto` / `max_metric_calls` で探索量を制御。候補系列・スコア・使用コール数を保存し、Before/After を比較可能。
- 構文エラーも学習材料に
  - 出力フォーマット違反を構造ヒントとしてフィードバックし、外部ライブラリの期待仕様に収束させられる。

## GEPA が最適化するもの

- 対象は「Predictor の instructions（自然言語仕様）」
  - 実装例では `BuildLangExtractPrompt` が `prompt_description`（= LangExtract の prompt）と `examples_json`（= LangExtract の examples）を生成。
  - GEPA はこの Predictor の instructions を進化させ、より良い prompt/examples を出すように学習する。
- 入力は task / target_classes / style_hints（可変条件）
  - instructions は「一般則」。可変要件は入力として渡し、出力の品質で最適化。

## メトリクス設計（必須の勘所）

- Evaluate 用はモジュール単一スコア（例: 期待抽出のカバレッジ ∈ [0,1]）。
- GEPA 用は `dspy.Prediction(score, feedback)` を返す。
  - `pred_name == "build_prompt"` のとき `pred_trace` から `prompt_description` / `examples_json` を点検し、
    - 例のクラス網羅性
    - JSON の妥当性（配列であるか・構造が正しいか）
    - 厳密スパン・重複回避・属性の有無
    を具体テキストでフィードバック。
- まずは決定的な検査（ルール/スキーマ/文字列マッチ）を核に。LLM-as-judge はノイズが増えるため段階的に。

## 予算・反射の方針

- まずは `auto='light'` か少なめの `max_metric_calls`。改善が頭打ちになったら増やす。
- 反射の多様性が必要なら `reflection_minibatch_size=2..3` を検討。
- 反射 LM は強めを推奨（prompt 進化の質が上がる）。

## 実装の指針（Controller / Executor 分離）

- Controller（最適化対象）
  - `dspy.Predict(Signature)` が prompt と few-shot（JSON など）を出力。
  - GEPA はこの instructions を最適化。
- Executor（最適化対象外）
  - Controller の出力を LangExtract の API へ変換して呼ぶだけの薄い層。副作用を持たず、テスト容易に。
- 例外・移植性
  - 外部ライブラリは遅延 import。未インストールでも動く `--dummy` フォールバックを用意。
  - JSON パースや型変換は防御的に行う。

## 手動チューニングのみとの対比（課題）

- 勘と経験に依存し、差分・根拠が残りづらい（監査性・再現性の欠如）。
- 複数テキスト/目的のトレードオフ最適化が感覚頼みになりやすい。
- 例の網羅性や構文遵守の機械検査が抜け落ちがち（GEPA なら metric に落とし込める）。

## 参考

- 実装例: `real_world/simple_gepa_langextract.py`
- Before/After と候補の表示・保存: `real_world/utils.py`, `real_world/save.py`
- GEPA ラッパ/Adapter: `dspy/teleprompt/gepa/gepa.py`, `dspy/teleprompt/gepa/gepa_utils.py`

---

要するに、GEPA は「人手で都度作る prompt/examples」を、評価可能な原則に基づいて自動・系統的に洗練し、
再現性・説明責任・スケール性を備えた改善プロセスに変えてくれます。
