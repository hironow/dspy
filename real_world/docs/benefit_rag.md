# GEPA × Vector RAG: Benefits and Practical Guide

このメモは、外部ベクタDBを用いた RAG を GEPA（dspy.GEPA）で最適化する意義と、実装・運用の勘所を短くまとめたものです。実装例は `real_world/simple_gepa_vector_rag.py`、アダプタIFは `real_world/vector_adapter.py` を参照してください。

## なぜ GEPA で RAG を最適化するか

- 反復の自動化と高速な学習
  - 「Rewrite → Retrieve → Answer」のサイクルをメトリクス駆動で反射的に回す。短い予算でも改善が積み上がる。
- 具体フィードバックの活用（pred_name / pred_trace）
  - rewrite にはクエリの不足/過剰・固有名詞/同義語の指摘、answer には根拠（passages）未反映や過剰説明の指摘など、段階別にピンポイントなFBを返せる。
- バッチ頑健性（Pareto 選択）
  - 検証集合全体の Pareto 前線を維持しつつ改善。特定質問への過適合を避け、RAGの汎化を促す。
- 予算管理と再現性
  - `auto` / `max_metric_calls` で探索量を制御。候補系列・スコア・使用コール数を保存し、Before/After を比較可能。
- 外部依存の吸収
  - ベクタDBは Adapter 経由の薄い呼び出しに統一（Pinecone/Weaviate/Qdrant/pgvector 等）。実/ダミーを差し替え可能。

## GEPA が最適化するもの

- 対象は「Predictor の instructions（自然言語仕様）」
  - 実装例では `rewrite: question -> rewritten_query` と `answer: question, passages -> answer` を最適化。
    - これは dspy.Signature の省略表記です（`question` はフィールド名で、型は `str` など）。
    - 例（dspy.Signature での定義）:

      ```python
      class Rewrite(dspy.Signature):
          """Rewrite the user question into a retrieval-friendly query."""
          question: str = dspy.InputField(desc="Original user question")
          rewritten_query: str = dspy.OutputField(desc="Simplified query preserving entities")

      class Answer(dspy.Signature):
          """Answer the question using only the provided passages as evidence."""
          question: str = dspy.InputField(desc="Original user question")
          passages: list[str] = dspy.InputField(desc="Retrieved context passages")
          answer: str = dspy.OutputField(desc="Concise, grounded answer")
      ```
  - 検索（retrieve）は Executor（Adapter 呼び出し）に分離し、最適化対象から外す。
- 入力は question と passages（retrieve 結果）
  - rewrite は一般則（固有名詞維持/同義語補完/曖昧語の明確化）、answer は根拠引用/簡潔性などの一般則を instructions に。

## Metric 設計（必須の勘所）

- モジュールスコアは決定的に：
  - 正答（EM/正規化一致）を主、根拠性（passages に解答表現が含まれるか）を副目的に。
  - 例: `score = 0.8 * correctness + 0.2 * groundedness`（groundedness は passages の文字列一致等のシンプル指標）。
- GEPA 用は `dspy.Prediction(score, feedback)` を返す。
  - `pred_name == "rewrite"` では「エンティティ抜け」「同義語展開不足」「冗長/過小」等をFB。
  - `pred_name == "answer"` では「根拠が本文に見当たらない」「引用を増やす」「簡潔に」など、`pred_trace` から `passages` を検分して具体FB。
- まずはルール/文字列マッチ中心の決定的評価で安定化。LLM-as-judge は段階的に採用（非決定性に留意）。

### 具体的なメトリクス例（擬似コード）

```python
def rag_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    def norm(s):
        return (s or "").strip().lower()

    g = norm(getattr(gold, "answer", ""))
    a = norm(getattr(pred, "answer", ""))
    correctness = 1.0 if g and a == g else 0.0

    # groundedness: answerが含まれるpassageがあるか
    grounded = 0.0
    passages = []
    if pred_name == "answer" and isinstance(pred_trace, list) and pred_trace:
        _, ans_inputs, _ = pred_trace[0]
        passages = ans_inputs.get("passages", []) if isinstance(ans_inputs, dict) else []
    if g and any(g in norm(p or "") for p in passages):
        grounded = 1.0

    score = round(0.8 * correctness + 0.2 * grounded, 3)
    if pred_name is None and pred_trace is None:
        return score

    fb = []
    if correctness == 1.0:
        fb.append("Correct answer.")
    else:
        fb.append(f"Expected '{g}' but got '{a}'.")

    if pred_name == "rewrite":
        fb.append("固有名詞/用語を保持し、同義語・必要語を補って検索再現性を上げてください。")
    elif pred_name == "answer":
        if grounded < 1.0:
            fb.append("根拠passagesに答えの表現が見当たりません。該当スニペットを引用してください。")
        fb.append("回答は簡潔に。引用は二重引用符などで明示。")
    else:
        fb.append("正答を維持しつつ、根拠に依拠してください。")

    return dspy.Prediction(score=score, feedback=" ".join(fb))
```

### groundedness 判定の代替案

- 厳格一致（現在の例）: 正規化文字列の包含。決定性と安定性を優先。
- ゆるやか一致: N-gram一致率、トークンF1、簡易類似度（未使用語除去後にしきい値）。ただし非決定性・コストに注意。
- 引用インデックス（拡張案）: answer側で citations を返し、参照passageの一致をチェック（CoT+引用の設計と併用）。

## pred_trace の把握（抜粋例）

GEPAは predictor単位で `pred_trace` を渡します。`[(predictor, inputs: dict, outputs: Prediction)]` 形式の先頭要素を参照すれば十分です。

```python
# answer predictor用 pred_trace の読み方
if pred_name == "answer" and isinstance(pred_trace, list) and pred_trace:
    _, ans_inputs, ans_outputs = pred_trace[0]
    # passagesはanswerの入力側に入っている
    passages = ans_inputs.get("passages", [])  # list[str]
    answer_text = getattr(ans_outputs, "answer", "")

    # 例: passagesのどの要素に答えが含まれたか（ログ/FB用）
    matched = [i for i, p in enumerate(passages) if norm(gold.answer) in norm(p)]
```

rewriteの場合も同様に、`pred_trace[0][1]` から `{"question": ...}`、`pred_trace[0][2]` から `{"rewritten_query": ...}` を取得できます。これにより「元質問で不足している語」「再書換えの過剰語」などを具体的にFBへ反映させられます。

## フィードバック文例（雛形）

- rewrite向け
  - 「質問内の固有名詞/数値/日付を保持し、一般語を具体化してください（例: 'それ'→対象名）。」
  - 「検索再現性のため、同義語・異表記（ローマ字/漢字）を括弧内に追加してください。」
  - 「不要な形容や主観語を削除し、N語以内に簡潔化してください。」
- answer向け
  - 「passages内の該当フレーズを二重引用符で引用し、1文で回答してください。」
  - 「根拠未一致です。該当するpassage断片を確認し、表現を一致させてください。」
  - 「冗長です。回答先頭で結論を述べ、根拠は最小限の引用に留めてください。」

## 評価の落とし穴と対処

- 非決定的な採点（LLM-as-judge）による揺れ
  - まずは決定的なヒューリスティックで安定化。必要時のみ段階的に導入。
- ドキュメントの更新/差し替えで指標が変化
  - Adapterにバージョン/メタ（インデックス日時など）を持たせ、run毎に記録する。
- 短いpassagesでの一致判定が難しい
  - 部分一致の前処理（記号/空白正規化）を徹底。長文は前後5〜10語のコンテキストも評価対象に入れる。

## 予算・反射の方針

- まずは `auto='light'` か少なめの `max_metric_calls`。改善が頭打ちになったら増やす。
- `reflection_minibatch_size=1..3` で失敗例を小分けに観察。反射 LM は強めが望ましい（rewrite/answer の質が上がる）。
- `track_stats=True` で候補・Pareto・最良出力を可視化し、探索経路を振り返る。

## 実装の指針（Controller / Executor / Adapter）

- Controller（最適化対象）
  - rewrite/answer の 2 Predictor。GEPA はこれらの instructions を進化させる。
- Executor（最適化対象外）
  - retrieve は Adapter 経由の呼び出し（引数：クエリ、返値：passages）。
  - 例外・エラーの扱いは薄く、決定的に（リトライやフォールバックは必要に応じて別層で）。
- Adapter（統一IF）
  - IF: `upsert(list[Document])`, `query(text, k=5, filter=None) -> list[QueryHit]`
  - `InMemoryTfIdfAdapter`（ダミー、依存なし）から開始し、Pinecone/Weaviate/Qdrant等の実装に置き換え可能。
  - メタデータ filter（完全一致）など最低限の運用フックを用意。

## 手動チューニングのみとの対比（課題）

- クエリ書き換え（rewrite）の良し悪しが経験則になりやすく、再現性が弱い。
- 回答（answer）の根拠性・引用整合は後回しになりがち（metric に入れると自然に改善対象になる）。
- ベクタDB入れ替え時の影響範囲が読みにくい（Adapter 経由だと影響が局所化）。

## 参考

- 実装例: `real_world/simple_gepa_vector_rag.py`
- Adapter IF/ダミー: `real_world/vector_adapter.py`
- JSONL準備/読込: `real_world/data_tools.py`
- Before/After と候補の表示・保存: `real_world/utils.py`, `real_world/save.py`
- GEPA ラッパ/Adapter: `dspy/teleprompt/gepa/gepa.py`, `dspy/teleprompt/gepa/gepa_utils.py`

---

要するに、GEPA は RAG の「クエリ最適化」と「根拠に基づく回答」の運用を、
評価可能な原則に基づいて自動・系統的に洗練し、再現性・説明責任・スケール性を備えた改善プロセスに変えてくれます。


## 目的次第だが、Adapter方式を基本にしやすい理由

実装/運用の観点では「どちらが絶対に良い」ではなく目的次第ですが、GEPA 前提の最適化と再現性・テスト容易性を重視する場合、simple_gepa_vector_rag の Adapter 方式を基本にする利点が多いです。

- 責務分離（Controller/Executor/Adapter）
  - rewrite/answer の instructions だけを最適化対象にし、retrieve は Adapter で外部依存を吸収。設計が明確で差し替えが容易。
- pred_trace/pred_name を活かしやすい
  - retrieve 結果（passages）を明示的に answer の入力へ渡すため、pred_trace から passages を簡単かつ確実に取得でき、groundedness 検査や段階別FBがシンプル。
- 再現性とテスト
  - InMemoryTfIdfAdapter のような決定的ダミーで端から端までpytest可能。外部サービス揺らぎの影響を受けにくい。
- 依存差し替え
  - VectorAdapter(upsert, query) という統一IFにより、Pinecone/Weaviate/Qdrant/pgvector 等の導入・切替が局所化。

一方で、既存プロダクションが dspy.settings.rm（Retrieval Module）で固まっている場合は、rm を活かすのも合理的です。その場合でも、answer 予測器に passages を必ず入力として渡し、pred_trace 経由でメトリクス/FBが扱えるようにするのがポイントです。

### rm を活かしつつ Adapter へ橋渡しする最小例

```python
from real_world.vector_adapter import VectorAdapter, QueryHit

class RMAdapter(VectorAdapter):
    def __init__(self, rm):
        self.rm = rm  # 例: dspy.settings.rm に設定済みの retriever

    def upsert(self, docs):
        raise NotImplementedError("Indexing is handled by the existing pipeline.")

    def query(self, text: str, *, k: int = 5, filter: dict | None = None):
        passages = self.rm(text, k=k)  # retriever の返り値を想定
        return [QueryHit(id=str(i), text=p, score=1.0) for i, p in enumerate(passages)]
```

このように、rm 既存資産を活かしつつ Adapter 方式の書き心地（責務分離・テスト性・pred_traceの取り回し）を得ることも可能です。
