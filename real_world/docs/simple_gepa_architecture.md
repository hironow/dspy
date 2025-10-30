# シンプル GEPA アーキテクチャ (v3+)

以下は、このリポジトリのシンプル事例が「最適化（GEPA）後にどう動くか」のフローを、アスキーアートで表現したものです。各図の下に、Evaluate と GEPA で用いる metric が「何を最適化/評価しているか」を明記します。

---

## 7) GEPA 全体フロー（アスキーアート）

```
必要要素（Inputs）
  - Student Program (DSPy Module with Predictors)
  - Metric (GEPA用: gold, pred, trace, pred_name, pred_trace -> score | {score, feedback})
  - Reflection: reflection_lm または instruction_proposer（どちらか必須）
  - Budget: auto("light|medium|heavy") | max_metric_calls | max_full_evals（いずれか必須）
  - Optional: component_selector（"round_robin" | "all" | 関数）, use_merge, logging, track_stats

初期化
  [Student instructions]
        |
        v
  [Seed Candidate (component_name -> instruction)]
        |
        v
  (Option) Baseline Eval on valset  -->  baseline score

反復（Iterate until budget）
  ┌───────────────────────────────────────────────────────────────────────┐
  │ 1) Candidate Selection (Pareto Frontier)                              │
  │    - 現在の Pareto 前線から確率的に 1 候補を選ぶ                            │
  │                                                                       │
  │ 2) Component Selection                                                │
  │    - component_selector に従い、最適化対象 Predictor を選ぶ                │
  │                                                                       │
  │ 3) Minibatch Sampling (trainset から M 件)                             │
  │                                                                       │
  │ 4) Rollout + Trace Capture                                            │
  │    - 候補プログラムをミニバッチで実行し、各 Predictor のトレース取得            │
  │    - Metric を呼び、score + textual feedback を取得（pred_name付き）      │
  │    - 失敗（パース）は FailedPrediction として保持（構文指示に活用）           │
  │                                                                       │
  │ 5) Reflect & Propose (指示文の改良案)                                    │
  │    - reflective_dataset: {Inputs, Generated Outputs, Feedback, ...}   │
  │    - reflection_lm または instruction_proposer で new instruction       │
  │                                                                       │
  │ 6) Build New Candidate                                                │
  │    - 対象 Predictor の instructions を置き換えて新候補を生成                │
  │                                                                       │
  │ 7) Evaluate & Pareto Update                                           │
  │    - ミニバッチでベースライン比の改善を確認                                  │
  │    - 周期的/条件付きに valset で評価し、Pareto 前線を更新                    │
  │    - discovery_eval_counts などの統計を記録                              │
  │                                                                       │
  │ 8) (Optional) Merge/Crossover                                         │
  │    - 系統の異なる候補をマージし、新しい強い候補を探索.                         │
  └───────────────────────────────────────────────────────────────────────┘

終了（Budget 消費 or 収束）
  - valset 集計スコアが最大の候補（best candidate）を返す
  - track_stats=True なら詳細（候補群、Pareto、最良出力、メトリクス呼数 等）を格納

補助（Adapter の役割）
  - DspyAdapter が評価/トレース取得/反射用データ整形/提案器呼び出しを一手に担う
  - add_format_failure_as_feedback=True でパース失敗も学習材料へ（構造指示を自動付与）

Budget の目安（auto の例）
  - auto("light|medium|heavy") から候補数 n を決め、`auto_budget()` でおおよその metric_calls を見積
  - 代替: `max_metric_calls` を直接指定、または `max_full_evals * (|train|+|val|)`

出力（Outputs）
  - Optimized Program（最良 instructions を持つ DSPy Module）
  - Optional: Detailed Results（Pareto 前線・候補集・最良出力・ログディレクトリ 等）
```

## 1) simple_gepa_basic.py（SimpleQA: rewrite → predict）

```
[Input] question
    |
    v
[rewrite] dspy.Predict("question -> refined_question")  --(LM)--> refined_question
    |
    v
[predict] dspy.Predict("question -> answer")           --(LM)--> answer
    |
    v
[Output] answer

GEPA後: rewrite/predict の instructions（自然言語の指示文）が更新される可能性
```

- Evaluate の metric（ベーシック）
  - 目的: 正解/不正解（例: 日本語の一語回答の完全一致）を測定
  - 返り値: float（0/1）。複数例の平均を百分率で集計

- GEPA の metric（フィードバック対応）
  - 目的: 上記スコアに加え、誤答時の「どこが違うか」を短く文章化して返す
  - 返り値: dspy.Prediction(score=float, feedback=str)
  - pred_name/pred_trace: 指定された Predictor（rewrite か predict）に向けて、ピンポイントなFBを返す

---

## 2) simple_gepa_task_metric.py（SimpleQA: タスク特化メトリクス版）

```
[Input] question
    |
    v
[rewrite]  --(LM)--> refined_question
    |
    v
[predict]  --(LM)--> answer
    |
    v
[Output] answer

GEPA後: rewrite/predict の instructions が、タスク要件（簡潔/直接/一語）に沿う形へ進化
```

- Evaluate の metric（タスク特化）
  - 目的: タスク要件を反映したスコアリング
    - 同義語許容（青/ブルー, 黄色/イエロー）
    - 近似一致の部分点（edit distance<=1 を0.6など）
    - 簡潔さの部分点（1語・短いほど加点）
  - 返り値: 0.8*Correctness + 0.2*Brevity（[0,1]に丸め）

- GEPA の metric（フィードバック対応）
  - 目的: 上記の各要素（Correctness/Brevity）を短文で明確化し、改善すべき観点を提示
  - 返り値: dspy.Prediction(score, feedback)（要素別FB: “near-miss”, “単語が長い” など）
  - pred_name/pred_trace: rewrite/predict それぞれに適切な改善ヒントを与える

---

## 3) simple_gepa_structured_invoice.py（構造化抽出: extract → normalize）

```
[Input] invoice text
    |
    v
[extract]   dspy.Predict("text -> vendor, date, amount, currency")
    |
    v
[normalize] dspy.Predict("vendor, date, amount, currency -> vendor, date, amount, currency")
    |
    v
[Output] {vendor, date(YYYY-MM-DD), amount(float), currency(ISO)}

GEPA後: extract/normalize の instructions が、スキーマ適合・正規化ルール（ISO日付/ISO通貨/数値化）に沿うよう洗練
```

- Evaluate の metric（スキーマ検証）
  - 目的: vendor 非空 / date は ISO（YYYY-MM-DD）/ amount 数値 / currency ISO コード を満たすか
  - スコア: 1.0 を基準に、欠損・型違い・形式不正・Goldとの不一致を減点（[0,1]）

- GEPA の metric（フィードバック対応）
  - 目的: extract 向けには欠損/型違いの具体指摘、normalize 向けには from→to 提示（例: “31-12-2024 → 2024-12-31”, “¥ → JPY”, “7,890 → 7890.0”）
  - 返り値: dspy.Prediction(score, feedback)
  - pred_name/pred_trace: どの Predictor が何をどう直すべきかを明確に（対象フィールド・元値・正規化先）

---

## 4) simple_gepa_routed_sources.py（ルータ×3ソース×条件付きリランク）

```
Light policy（本番想定）

[Input] query
    |
    v
[route]  --(LM)--> source ∈ {db, rag, graph}
    |
    +--> if 'db'    : [from_db]    --(LM)--> text
    |    if 'rag'   : [from_rag]   --(LM)--> text
    |    if 'graph' : [from_graph] --(LM)--> text
    |
    v
[Output] text


Heavy policy（最適化/検証向け）

[Input] query
    |
    v
[route] --(LM)--> source（ログ用途）
    |
    +--[from_db]    --(LM)--> db_text
    +--[from_rag]   --(LM)--> rag_text
    +--[from_graph] --(LM)--> graph_text
    |
    v
[rerank] --(LM)--> best text
    |
    v
[Output] text

GEPA後: route / from_* / rerank の instructions が、適切なソース選択・候補選択に寄与するよう洗練
```

- Evaluate の metric（包含一致）
  - 目的: 最終 text が gold.answer を（部分一致で）含むか
  - 返り値: float（0/1）。Light policy で運用を想定

- GEPA の metric（フィードバック対応）
  - 目的: ルータ/リランカ/各ソースに対して、どの経路/候補が良かったかの理由を短文で提示
    - Router: gold.preferred_source（ヒント）と違う場合は切替を提案
    - Rerank: 候補群のうち gold により近いものを指摘（db/rag/graph のヒント）
    - Sources: 期待内容が含まれない場合は照会/取得の見直しを促す
  - 返り値: dspy.Prediction(score, feedback)
  - pred_name/pred_trace: 対象 Predictor（route/from_db/from_rag/from_graph/rerank）単位で具体的FB

---

## 5) simple_gepa_multimodal_caption.py（画像→ caption/keywords：単段）

```
[Input] image (dspy.Image)
    |
    v
[caption] dspy.Predict("image: dspy.Image -> caption: str, keywords: list[str]")  --(LM)--> {caption, keywords}
    |
    v
[Output] caption, keywords

GEPA後: caption の instructions（説明の方針）が更新される可能性
```

- Evaluate の metric（カバレッジ＋簡潔）
  - 目的: gold.keywords が caption+keywords に含まれる割合（coverage）、長文への軽いペナルティ
  - 返り値: float（[0,1]）。複数例の平均を百分率で集計

- GEPA の metric（フィードバック対応）
  - 目的: 足りないキーワードの列挙、簡潔化の促しなど、短いテキストFBを返す
  - 返り値: dspy.Prediction(score, feedback)
  - 反射: MultiModalInstructionProposer を使い、画像を構造化のまま反射LMへ

---

## 6) simple_gepa_multimodal_observe_compose.py（観測→作文：二段構成 + Signature）

```
[Input] image (dspy.Image)
    |
    v
[analyze]  dspy.Predict(Analyze: image -> objects, attributes, actions, scene, meta)  --(LM)--> observations
    |
    v
[compose]  dspy.Predict(Compose: observations -> caption, keywords)                    --(LM)--> {caption, keywords}
    |
    v
[Output] caption, keywords

GEPA後: analyze/compose の instructions が段別に洗練（観測語彙→説明の反映が進む）
```

- Evaluate の metric（カバレッジ＋簡潔）
  - 目的: gold.keywords の包含と簡潔性
  - 返り値: float（[0,1]）

- GEPA の metric（段別フィードバック + trace 整合性）
  - 目的: pred_name に応じた段別FBを返す（analyze: 観測の不足、compose: 説明の不足/簡潔化）
  - trace活用: analyze 出力（objects/attributes/actions/scene/meta）が caption/keywords に反映されているか検査
    - 観測未反映: 観測にあるが出力に現れない語を指摘
    - 観測にない語: 出力側にのみ現れる語を注意喚起
  - 構文失敗: add_format_failure_as_feedback=True により、フォーマット失敗も反射データに含み、
    「指定フィールド・型・構造の厳守」を FB として促す
  - 返り値: dspy.Prediction(score, feedback)
  - 反射: MultiModalInstructionProposer を使用（画像を構造化のまま反射LMへ）

備考:

- 本例は dspy.Signature（Analyze/Compose）を導入し、I/O の型・役割を明示。
- GEPA が最適化するのは instructions。Signature の doc/desc は最適化対象外だが、
  Adapter の構造化プロンプトや失敗時の構造指示に参照され、安定性に寄与。

---

## 共通メモ（Evaluate と GEPA の役割）

```
Evaluate:  
  - 「最終出力」を単純・決定的にスコアリング（float）し、全体スコア（%）を出す
  - コストが軽く、回帰検証や A/B 比較に向く

GEPA:  
  - 同じ metric をよりリッチに（score + feedback）呼び、失敗例のテキストFBを反射に活用  
  - pred_name/pred_trace により、どの Predictor をどう改善するかを具体化（instructions を進化）
  - 予算（auto / max_metric_calls）内で、反射→候補生成→評価→Pareto 追跡→最良候補を返す
```
## 8) GEPA フローチャート（ボックス図）

```
                +-----------------------+        +---------------------+
                |  Student Program      |        |   Metric            |
                |  (DSPy Module)        |        | (score & feedback)  |
                +-----------+-----------+        +----------+----------+
                            \                         /
                             \                       /
                              v                     v
                        +-----+-----------------------+-----+
                        |     Initialize Seed Candidate     |
                        |  (component -> instruction map)   |
                        +-------------------+---------------+
                                            |
                                            v
                                 +----------+-----------+
                                 |   Budget Controller  |
                                 |  (auto / max_* )     |
                                 +----------+-----------+
                                            |
                                  (metric calls remain?)
                                     yes / no
                                      |     \
                                      |      \---> no ---> +----------------------+
                                      |                    |  Return Best on Val  |
                                      |                    |  (Pareto aggregate)  |
                                      |                    +----------+-----------+
                                      |                               ^
                                      v                               |
                    +-----------------+-----------------+             |
                    |   Select Candidate from Pareto    |             |
                    +-----------------+-----------------+             |
                                      |
                                      v
                    +-----------------+-----------------+
                    |   Select Components to Mutate     |
                    |  (component_selector: rr / all)   |
                    +-----------------+-----------------+
                                      |
                                      v
                    +-----------------+-----------------+
                    |   Sample Minibatch from Train     |
                    +-----------------+-----------------+
                                      |
                                      v
                    +-----------------+-----------------+
                    |  Rollout Candidate on Minibatch   |
                    |  Capture Traces (per predictor)   |
                    +-----------------+-----------------+
                                      |
                                      v
                    +-----------------+-----------------+
                    | Evaluate with Metric               |
                    | - program score                    |
                    | - textual feedback (pred_name)     |
                    | - (format failures kept if enabled)|
                    +-----------------+-----------------+
                                      |
                                      v
                    +-----------------+-----------------+
                    | Reflect & Propose Instructions    |
                    | (reflection_lm / proposer)        |
                    +-----------------+-----------------+
                                      |
                                      v
                    +-----------------+-----------------+
                    |   Build New Candidate             |
                    +-----------------+-----------------+
                                      |
                                      v
                    +-----------------+-----------------+
                    |  Evaluate (mini/full) & Update    |
                    |  Pareto Frontier on Val           |
                    +-----------------+-----------------+
                                      |
                                      v
                    +-----------------+-----------------+
                    |  (Optional) Merge/Crossover       |
                    +-----------------+-----------------+
                                      |
                                      +---------------------------> loop
- 凡例 / 注意:
- component_selector: round_robin（既定）/ all / 関数
- add_format_failure_as_feedback: True でパース失敗も反射データへ
- track_stats / track_best_outputs: 最適化の詳細やバッチ最良出力を保持
- Pareto: 各 val 例で最良の候補集合（多様な戦略を維持）
```
