# Simple GEPA Architectures (v3+)

以下は、このリポジトリのシンプル事例が「最適化（GEPA）後にどう動くか」のフローを、アスキーアートで表現したものです。各図の下に、Evaluate と GEPA で用いる metric が「何を最適化/評価しているか」を明記します。

---

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

