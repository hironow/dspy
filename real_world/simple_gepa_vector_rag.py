"""
Vector RAG with GEPA (rewrite -> retrieve -> answer), JSONL-only.

- Module:
  - rewrite = dspy.Predict("question -> rewritten_query")
  - answer = dspy.Predict("question, passages: list[str] -> answer")
  - retrieve is an Executor calling a VectorAdapter (pluggable).

- Adapter:
  - Default: InMemoryTfIdfAdapter (deterministic, no deps)
  - Real: implement VectorAdapter (e.g., PineconeAdapter) and pass via code/flag

- Data (JSONL):
  - docs.jsonl: {"id": str, "text": str, "meta"?: object}
  - qa_train.jsonl / qa_val.jsonl: {"question": str, "answer": str}

Usage:
  uv run python real_world/simple_gepa_vector_rag.py \
      --docs-jsonl path/to/docs.jsonl \
      --train-qa path/to/qa_train.jsonl \
      --val-qa path/to/qa_val.jsonl \
      --dummy

GEPA compile requirements:
- metric(gold, pred, trace, pred_name, pred_trace)
- exactly one of: auto | max_full_evals | max_metric_calls
- reflection_lm or instruction_proposer
- trainset (and recommended valset)
"""

from __future__ import annotations

import argparse
from typing import Any

from loguru import logger

import dspy
from real_world.cli import add_standard_args, setup_logging
from real_world.cost import log_baseline_estimate, log_gepa_estimate, log_recorded_gepa_cost
from real_world.dummy_lm import configure_dummy_adapter, make_dummy_lm_json
from real_world.utils import summarize_before_after, summarize_gepa_results
from real_world.wandb import get_wandb_args, make_run_name
from real_world.vector_adapter import Document, InMemoryTfIdfAdapter, VectorAdapter
from real_world import data_tools as DT


class VectorRAG(dspy.Module):
    def __init__(self, adapter: VectorAdapter, *, top_k: int = 5):
        super().__init__()
        self.rewrite = dspy.Predict("question -> rewritten_query")
        self.answer = dspy.Predict("question, passages: list[str] -> answer")
        self.adapter = adapter
        self.top_k = top_k

    def forward(self, question: str):
        rw = self.rewrite(question=question)
        rq = getattr(rw, "rewritten_query", None) or question
        # Executor: query vector DB
        hits = self.adapter.query(rq, k=self.top_k)
        passages = [h.text for h in hits] or [""]
        return self.answer(question=question, passages=passages)


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def rag_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
):
    """Simple RAG metric combining correctness and groundedness.

    - correctness: exact match on normalized answers
    - groundedness: whether any passage contains the (normalized) answer substring
    score = 0.8*correctness + 0.2*groundedness

    In GEPA mode, returns feedback tailored to rewrite/answer using pred_name/pred_trace.
    """
    g = _normalize(str(getattr(gold, "answer", "")))
    a = _normalize(str(getattr(pred, "answer", "")))
    correctness = 1.0 if g and a == g else 0.0

    # Extract passages from answer pred inputs via pred_trace if available
    grounded = 0.0
    try:
        # pred_trace holds [(predictor, inputs, outputs)] for the target predictor
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

    if pred_name is None and pred_trace is None:
        return score

    # Feedback
    fb: list[str] = []
    if correctness == 1.0:
        fb.append("Correct answer.")
    else:
        fb.append(f"Expected '{g}' but got '{a}'.")

    if pred_name == "rewrite":
        fb.append("Ensure the rewritten query preserves key entities and intent; expand synonyms if recall is low.")
    elif pred_name == "answer":
        if grounded < 1.0:
            fb.append("Ground passages don't clearly contain the target phrase; cite matching snippet(s).")
        fb.append("Answer concisely; quote exact phrases when relevant.")
    else:
        fb.append("Maintain correctness; prefer answers grounded in retrieved passages.")

    return dspy.Prediction(score=score, feedback=" ".join(fb))


def _load_docs_jsonl(path: str) -> list[Document]:
    rows = DT.load_jsonl(path)
    docs: list[Document] = []
    for r in rows:
        doc_id = str(r.get("id") or r.get("_id") or len(docs))
        text = str(r.get("text") or "")
        meta = r.get("meta") if isinstance(r.get("meta"), dict) else None
        if text.strip():
            docs.append(Document(id=doc_id, text=text, meta=meta))
    return docs


def main():
    parser = argparse.ArgumentParser()
    add_standard_args(parser, default_save_prefix="simple_gepa_vector_rag")
    parser.add_argument("--docs-jsonl", required=False, help="Path to JSONL with documents: {id,text,meta?}")
    parser.add_argument("--train-qa", required=False, help="Path to JSONL with train QA: {question,answer}")
    parser.add_argument("--val-qa", required=False, help="Path to JSONL with val QA: {question,answer}")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Starting Vector RAG + GEPA demo")

    # Adapter selection (use in-memory unless you swap in a real adapter)
    adapter: VectorAdapter = InMemoryTfIdfAdapter()

    # Build module
    program = VectorRAG(adapter=adapter, top_k=int(args.top_k))
    program.rewrite.signature = program.rewrite.signature.with_instructions(
        "質問の要点（固有名詞/同義語）を補いながら簡潔に再表現してください。"
    )
    program.answer.signature = program.answer.signature.with_instructions(
        "与えられたパッセージ(passages)のみを根拠に、短く正確に回答してください。根拠の表現は可能な限り引用してください。"
    )

    # Data loading (JSONL). If not provided, use a tiny dummy fallback.
    if args.docs_jsonl:
        docs = _load_docs_jsonl(args.docs_jsonl)
    else:
        docs = [
            Document(id="d1", text="The capital of Japan is Tokyo."),
            Document(id="d2", text="Mount Fuji is near Tokyo in Japan."),
        ]
    adapter.upsert(docs)

    if args.train_qa and args.val_qa:
        trainset = DT.load_qa_examples_from_jsonl(args.train_qa)
        valset = DT.load_qa_examples_from_jsonl(args.val_qa)
    else:
        # Fallback tiny QA
        trainset = [
            dspy.Example(question="日本の首都は？", answer="東京").with_inputs("question"),
            dspy.Example(question="富士山はどこにありますか？", answer="東京近郊").with_inputs("question"),
        ]
        valset = trainset

    if args.dummy:
        # Dummy LMs
        def rw_responses():
            while True:
                yield {"rewritten_query": "日本 首都"}

        def ans_responses():
            while True:
                yield {"answer": "東京"}

        rw_lm = make_dummy_lm_json(rw_responses())
        ans_lm = make_dummy_lm_json(ans_responses())
        program._rewrite_lm = rw_lm
        program._answer_lm = ans_lm
        configure_dummy_adapter(lm=ans_lm)
        reflection_lm = make_dummy_lm_json(lambda: iter([{"improved_instruction": "固有名詞を保持しつつ簡潔に。"}]*1000))
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


if __name__ == "__main__":
    main()

