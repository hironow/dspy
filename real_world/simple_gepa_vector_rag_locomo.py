"""
Vector RAG for LoCoMo10 dataset with GEPA (rewrite -> retrieve -> answer).

This script loads a single item from `real_world/assets/datasets/locomo10.json`,
flattens the conversation sessions into per-line documents (D#:line), upserts
them into a VectorAdapter, and optimizes a simple RAG pipeline with GEPA.

- Controller (optimized): rewrite, answer
- Executor (not optimized): adapter.query(rewritten_query) -> passages

Usage (dummy):
  uv run python real_world/simple_gepa_vector_rag_locomo.py \
      --locomo real_world/assets/datasets/locomo10.json \
      [--locomo-rag real_world/assets/datasets/locomo10_rag.json] \
      --item-index 0 \
      --top-k 5 \
      --dummy

Notes
-----
- The LoCoMo JSON may contain numeric answers; this script normalizes via str() inside the metric.
- Groundedness uses a simple substring match in retrieved passages.

Cautions / Considerations (using real conversational data)
---------------------------------------------------------
- Privacy/PII: Treat all conversation text as potentially sensitive.
  - Do not upload raw text or prompts to third‑party services without proper approval.
  - When enabling external logging (e.g., W&B), review contents and disable or scrub if required.
- Licensing: Confirm dataset license/terms of use before distribution and external calls.
- Determinism & Reproducibility:
  - The demo defaults to a deterministic in‑memory retriever (TF‑IDF) and a dummy LM when `--dummy` is set.
  - Real LMs and remote retrievers can introduce non‑determinism; pin seeds, versions, and snapshots where possible.
- Scale & Performance:
  - LoCoMo files can be large. Upserting entire corpora may require chunking/streaming, memory limits, and background indexing.
  - Consider sharding, `--top-k` tuning, and simple metadata filters on the adapter to bound costs.
- Evaluation Nuances:
  - Answers in LoCoMo can be strings, dates, or numbers; we cast to `str()` for comparison. If you need stricter checks,
    add task‑specific normalization (e.g., date formats, casing, punctuation).
  - Groundedness here is a substring heuristic over `passages`. For higher precision, align answers to evidence indices
    (D#:line) and/or implement citation indices from the answer predictor.
- Internationalization:
  - Timestamps and locale‑specific tokens may affect retrieval. Consider normalizing time expressions and numerals for recall.
- Safety:
  - Keep Controller/Executor separation: only the rewrite/answer instructions are optimized; retrieval is a thin, auditable
    adapter call. This limits the blast radius of changes and helps with incident triage.
"""

from __future__ import annotations

import argparse

from loguru import logger

import dspy
from real_world.assets.datasets.evidence_utils import (
    build_conversation_index,
    load_locomo10,
    load_locomo10_rag,
)
from real_world.cli import add_standard_args, setup_logging
from real_world.cost import log_baseline_estimate, log_gepa_estimate, log_recorded_gepa_cost
from real_world.dummy_lm import configure_dummy_adapter, make_dummy_lm_json
from real_world.save import save_artifacts
from real_world.utils import summarize_before_after, summarize_gepa_results
from real_world.vector_adapter import Document, InMemoryTfIdfAdapter, VectorAdapter
from real_world.wandb import get_wandb_args, make_run_name
from real_world.metrics_utils import confusion_outcomes, safe_trace_log


class VectorRAG(dspy.Module):
    def __init__(self, adapter: VectorAdapter, *, top_k: int = 5):
        super().__init__()
        self.rewrite = dspy.Predict("question -> rewritten_query")
        self.answer = dspy.Predict("question, passages: list[str] -> answer")
        self.adapter = adapter
        self.top_k = top_k

    def forward(self, question: str):
        # Use per-predictor LMs if provided (dummy mode)
        rw_lm = getattr(self, "_rewrite_lm", None)
        if rw_lm is not None:
            with dspy.context(lm=rw_lm):
                rw = self.rewrite(question=question)
        else:
            rw = self.rewrite(question=question)
        rq = getattr(rw, "rewritten_query", None) or question
        hits = self.adapter.query(rq, k=self.top_k)
        passages = [h.text for h in hits] or [""]
        ans_lm = getattr(self, "_answer_lm", None)
        if ans_lm is not None:
            with dspy.context(lm=ans_lm):
                return self.answer(question=question, passages=passages)
        return self.answer(question=question, passages=passages)


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def rag_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name: str | None = None, pred_trace=None):
    g = _normalize(str(getattr(gold, "answer", "")))
    a = _normalize(str(getattr(pred, "answer", "")))
    correctness = 1.0 if g and a == g else 0.0

    grounded = 0.0
    try:
        if pred_name == "answer" and isinstance(pred_trace, list) and pred_trace:
            _, ans_inputs, _ = pred_trace[0]
            passages = ans_inputs.get("passages", []) if isinstance(ans_inputs, dict) else []
        else:
            passages = []
        if g and any(g in _normalize(p or "") for p in passages):
            grounded = 1.0
    except Exception:
        grounded = 0.0

    score = round(0.8 * correctness + 0.2 * grounded, 3)

    # Binary framing reused for trace and feedback
    gold_pos = bool(g)
    guess_pos = bool(g) and (a == g)
    pred_claim = bool(a)
    conf = confusion_outcomes(gold_pos, guess_pos, pred_claim)
    num_passages = len(passages) if "passages" in locals() and isinstance(passages, list) else 0

    # Trace essentials for reflection/debug
    safe_trace_log(
        trace,
        {
            "gold": g,
            "pred": a,
            "correctness": correctness,
            "grounded": grounded,
            "num_passages": num_passages,
            "confusion": conf,
        },
    )

    if pred_name is None and pred_trace is None:
        return score

    # Feedback using TP/FN/FP/TN branches
    fb: list[str] = []
    if conf["TP"]:
        fb.append("正しい回答です。")
        if pred_name == "answer" and grounded < 1.0:
            fb.append("根拠スニペットを引用し、証拠を明示してください。")
    elif conf["FN"]:
        fb.append(f"不一致です。正解は '{g}' です。")
    elif conf["FP"]:
        fb.append("不要な回答です（正解がないケース）。幻覚を避けてください。")
    else:
        fb.append("正しい無回答です。")

    if pred_name == "rewrite":
        fb.append("固有名詞/数字/日付を保持し、曖昧語を具体化して検索再現性を上げてください。")
    elif pred_name == "answer":
        if grounded < 1.0:
            fb.append("根拠passagesに答えの表現が見当たりません。該当スニペットを引用してください。")
        fb.append("回答は簡潔に。引用は二重引用符などで明示。")
    else:
        fb.append("正答を維持しつつ、根拠に依拠してください。")

    return dspy.Prediction(score=score, feedback=" ".join(fb))


def _flatten_locomo_item_to_docs(item: dict) -> list[Document]:
    index = build_conversation_index(item)  # {"D1": [msg, ...], ...}
    docs: list[Document] = []
    for doc_id, msgs in index.items():
        for i, msg in enumerate(msgs, start=1):
            text = str(msg.get("text") or "").strip()
            if not text:
                continue
            meta = {
                "doc": doc_id,
                "line": i,
                "speaker": msg.get("speaker"),
                "dia_id": msg.get("dia_id"),
            }
            docs.append(Document(id=f"{doc_id}:{i}", text=text, meta=meta))
    return docs


def _locomo_item_to_qa_examples(item: dict) -> list[dspy.Example]:
    exs: list[dspy.Example] = []
    for qa in item.get("qa", []):
        q = str(qa.get("question") or "").strip()
        a = qa.get("answer")
        if not q:
            continue
        exs.append(dspy.Example(question=q, answer=str(a) if a is not None else "").with_inputs("question"))
    return exs


def _flatten_locomo_rag_to_docs(rag_data: dict[str, dict]) -> list[Document]:
    """Flatten locomo10_rag.json into per-line documents.

    Each top-level key (e.g., "0", "1", ...) is treated as a conversation bucket.
    We assign doc ids like R{conv_id}:{line} and store meta with conv_id/line/speaker.
    """
    docs: list[Document] = []
    for conv_id, session in rag_data.items():
        msgs = session.get("conversation", []) or []
        for i, msg in enumerate(msgs, start=1):
            text = str(msg.get("text") or "").strip()
            if not text:
                continue
            meta = {
                "conv_id": conv_id,
                "line": i,
                "speaker": msg.get("speaker"),
                "timestamp": msg.get("timestamp"),
            }
            docs.append(Document(id=f"R{conv_id}:{i}", text=text, meta=meta))
    return docs


def main():
    parser = argparse.ArgumentParser()
    add_standard_args(parser, default_save_prefix="simple_gepa_vector_rag_locomo")
    parser.add_argument("--locomo", default="real_world/assets/datasets/locomo10.json", help="Path to locomo10.json")
    parser.add_argument("--locomo-rag", default=None, help="Optional path to locomo10_rag.json (RAG corpus)")
    parser.add_argument(
        "--item-index", type=int, default=0, help="Which item index in locomo10.json to use (default: 0)"
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Starting LoCoMo Vector RAG + GEPA demo")

    # TODO(Perf/LargeFiles): 巨大ファイル部分読み込み
    # - 現在は load_locomo10 で JSON 全体を読み込む。
    # - 代替案: ijson 等のストリーミングで `--item-index` の該当要素のみを抽出する
    #   `load_locomo10_item(path, item_index)` のようなヘルパーを evidence_utils.py 側に実装して差し替える。
    data = load_locomo10(args.locomo)
    assert data and 0 <= args.item_index < len(data), (
        f"Invalid item-index {args.item_index} for {args.locomo} (len={len(data)})"
    )
    item = data[args.item_index]

    # Build adapter index (prefer locomo_rag as corpus if provided)
    adapter: VectorAdapter = InMemoryTfIdfAdapter()
    if args.locomo_rag:
        rag_data = load_locomo10_rag(args.locomo_rag)
        docs = _flatten_locomo_rag_to_docs(rag_data)
        logger.info("Indexed {} conversation lines from locomo10_rag.json.", len(docs))
    else:
        docs = _flatten_locomo_item_to_docs(item)
        logger.info("Indexed {} conversation lines from locomo10.json item {}.", len(docs), args.item_index)
    adapter.upsert(docs)

    # Build QA examples
    # TODO(RetrievalFilter/Evidence): エビデンスで検索空間を絞る
    # - InMemoryTfIdfAdapter.query は meta に対する単純な等価フィルターをサポートしている。
    # - QA の evidence から doc_id（例: "D1"）を抽出し、adapter.query(..., filter={'doc': 'D1'})
    #   のようにセッション単位で候補を限定する実装を追加する。
    # - 実装指針:
    #   1) _locomo_item_to_qa_examples で Example に `evidence_docs` 等の補助フィールドを持たせる
    #   2) VectorRAG.forward で pred_trace/inputs から当該フィールドを参照し filter を渡す
    #   3) 複数 doc の場合は filter を拡張するか doc ごとに query して結合する
    all_examples = _locomo_item_to_qa_examples(item)
    if not all_examples:
        logger.warning("No QA examples found in the selected item.")
    # For simplicity, use the same for train/val (or split with data_tools if desired)
    trainset = all_examples
    valset = all_examples

    # Build program
    program = VectorRAG(adapter=adapter, top_k=int(args.top_k))
    program.rewrite.signature = program.rewrite.signature.with_instructions(
        "質問の要点（固有名詞/数字/日付）を保持し、曖昧語を具体化して再表現してください。"
    )
    program.answer.signature = program.answer.signature.with_instructions(
        "与えられたpassagesのみを根拠に、短く正確に回答してください。根拠の表現は可能な限り引用してください。"
    )

    # LMs
    if args.dummy:

        def rw_responses():
            while True:
                yield {"rewritten_query": "質問 要点"}

        def ans_responses():
            while True:
                yield {"answer": ""}

        rw_lm = make_dummy_lm_json(rw_responses())
        ans_lm = make_dummy_lm_json(ans_responses())
        program._rewrite_lm = rw_lm
        program._answer_lm = ans_lm
        configure_dummy_adapter(lm=ans_lm)
        # Reflection LM must be an iterator of dict outputs, not a function
        reflection_lm = make_dummy_lm_json(iter([{"improved_instruction": "固有名詞/日付を保持しつつ簡潔に。"}] * 1000))
    else:
        from real_world.helper import openai_gpt_4o_lm, openai_gpt_4o_mini_lm

        dspy.settings.configure(lm=openai_gpt_4o_mini_lm)
        reflection_lm = openai_gpt_4o_lm

    # Baseline
    from dspy.evaluate import Evaluate

    evaluator = Evaluate(devset=valset, metric=rag_metric, display_progress=False, num_threads=1)
    logger.info("Baseline evaluation on {} validation examples...", len(valset))
    log_baseline_estimate(valset_size=len(valset), num_predictors=len(program.predictors()), logger=logger)
    baseline = evaluator(program)
    logger.success("Baseline score: {}", baseline.score)

    # GEPA
    gepa = dspy.GEPA(
        metric=rag_metric,
        reflection_lm=reflection_lm,
        auto="light" if not args.dummy else None,
        max_metric_calls=40 if args.dummy else None,
        reflection_minibatch_size=1,
        track_stats=True,
        **get_wandb_args(project="real_world", run_name=make_run_name(args.save_prefix), enabled=not args.dummy),
    )

    logger.info("Running GEPA compile (max_metric_calls={} auto={})...", gepa.max_metric_calls, gepa.auto)
    log_gepa_estimate(
        gepa=gepa,
        num_predictors=len(program.predictors()),
        valset_size=len(valset),
        trainset_size=len(trainset := trainset),
        logger=logger,
    )
    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    logger.success("GEPA compile finished.")

    logger.info("Evaluating optimized program on validation set...")
    improved = evaluator(optimized)
    logger.success("Post-GEPA score: {}", improved.score)

    summarize_gepa_results(optimized, logger, top_k=10)
    before = {n: p.signature.instructions for n, p in program.named_predictors()}
    summarize_before_after(before, optimized, logger)
    if hasattr(optimized, "detailed_results") and optimized.detailed_results is not None:
        log_recorded_gepa_cost(optimized.detailed_results, num_predictors=len(program.predictors()), logger=logger)

    # Save artifacts
    save_artifacts(
        program,
        optimized,
        save_dir=args.save_dir,
        prefix=args.save_prefix,
        logger=logger,
        save_details=True,
    )


if __name__ == "__main__":
    main()
