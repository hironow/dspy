"""
Routed multi-source retrieval with optional reranking (v3+): DB / RAG / Graph

- Router picks an information source from {db, rag, graph}
- Heavy policy (for GEPA/training): fetch from all sources and run a reranker
- Light policy (for prod): fetch only from routed source (fast)
- Metric uses (gold, pred, trace, pred_name, pred_trace) to give tailored feedback

Run (no external calls):
  uv run python real_world/simple_gepa_routed_sources.py --dummy

Switch policy at runtime:
  - Heavy (default here when optimizing): with dspy.context(rerank_policy="heavy")
  - Light (default otherwise): dspy.settings.configure(rerank_policy="light")
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import json
from typing import Any

from loguru import logger

import dspy
from dspy.adapters.json_adapter import JSONAdapter
from real_world.helper import openai_gpt_4o_mini_lm, openai_gpt_4o_lm
from real_world.factory import routed_sources_dummy


class RoutedSources(dspy.Module):
    def __init__(self):
        super().__init__()
        self.route = dspy.Predict("query -> source")
        self.from_db = dspy.Predict("query -> text")
        self.from_rag = dspy.Predict("query -> text")
        self.from_graph = dspy.Predict("query -> text")
        # Reranker chooses the best final text among candidates
        self.rerank = dspy.Predict("query, db_text, rag_text, graph_text -> text")

    def forward(self, query: str):
        # Routing first (use per-predictor LM if attached)
        rt_lm = getattr(self, "_route_lm", None)
        if rt_lm is not None:
            with dspy.context(lm=rt_lm):
                r = self.route(query=query)
        else:
            r = self.route(query=query)
        chosen = getattr(r, "source", "rag")

        policy = dspy.settings.get("rerank_policy", "light")

        if policy == "heavy":
            # Fetch from all sources (training/GEPA) with per-predictor LMs if provided
            db_lm = getattr(self, "_from_db_lm", None)
            rag_lm = getattr(self, "_from_rag_lm", None)
            gr_lm = getattr(self, "_from_graph_lm", None)

            if db_lm is not None:
                with dspy.context(lm=db_lm):
                    db = self.from_db(query=query)
            else:
                db = self.from_db(query=query)

            if rag_lm is not None:
                with dspy.context(lm=rag_lm):
                    rag = self.from_rag(query=query)
            else:
                rag = self.from_rag(query=query)

            if gr_lm is not None:
                with dspy.context(lm=gr_lm):
                    g = self.from_graph(query=query)
            else:
                g = self.from_graph(query=query)

            # Rerank/select best
            rr_lm = getattr(self, "_rerank_lm", None)
            if rr_lm is not None:
                with dspy.context(lm=rr_lm):
                    final = self.rerank(
                        query=query,
                        db_text=db.text,
                        rag_text=rag.text,
                        graph_text=g.text,
                    )
            else:
                final = self.rerank(
                    query=query,
                    db_text=db.text,
                    rag_text=rag.text,
                    graph_text=g.text,
                )
            return final
        else:
            # Light policy: call only the selected source
            if chosen == "db":
                db_lm = getattr(self, "_from_db_lm", None)
                if db_lm is not None:
                    with dspy.context(lm=db_lm):
                        return self.from_db(query=query)
                return self.from_db(query=query)
            elif chosen == "graph":
                gr_lm = getattr(self, "_from_graph_lm", None)
                if gr_lm is not None:
                    with dspy.context(lm=gr_lm):
                        return self.from_graph(query=query)
                return self.from_graph(query=query)
            else:
                rag_lm = getattr(self, "_from_rag_lm", None)
                if rag_lm is not None:
                    with dspy.context(lm=rag_lm):
                        return self.from_rag(query=query)
                return self.from_rag(query=query)


def routed_metric_with_feedback(
    gold: Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
):
    """
    Task-specific metric for routed multi-source retrieval.

    - Correctness: final text should match gold.answer (substring OK for this demo)
    - Preferred source: gold.preferred_source is a hint. Router/reranker feedback uses it.
    - Uses trace/pred_name/pred_trace to tailor actionable feedback for each predictor.

    Returns float (Evaluate) or dspy.Prediction(score, feedback) for GEPA.
    """
    gold_answer = str(getattr(gold, "answer", "")).strip()
    preferred = str(getattr(gold, "preferred_source", "")).strip()  # db|rag|graph (hint)
    final_text = str(getattr(pred, "text", "")).strip()

    # Simple correctness: substring match
    if not gold_answer:
        base = 0.0
    else:
        base = 1.0 if gold_answer.lower() in final_text.lower() else 0.0

    # Evaluate mode (no pred_name/pred_trace): return scalar only
    if pred_name is None and pred_trace is None:
        return base

    # Build feedback using predictor-level view
    fb: list[str] = []

    # Helper to extract routed choice from full trace
    def find_routed_source(full_trace) -> str | None:
        try:
            for (p, inputs, outputs) in full_trace or []:
                if hasattr(outputs, "source"):
                    return getattr(outputs, "source", None)
        except Exception:
            pass
        return None

    routed_source = find_routed_source(trace)

    # Program-level feedback
    if pred_name is None:
        if base == 1.0:
            fb.append("Output matches expected content.")
        else:
            fb.append(f"Mismatch: expected contains '{gold_answer}', got '{final_text[:50]}'.")
        if preferred and routed_source and routed_source != preferred:
            fb.append(f"Consider routing to '{preferred}' for this query.")

    # Router-specific feedback
    elif pred_name == "route":
        # pred_trace carries the route predictor's outputs
        try:
            _, p_inputs, p_outputs = pred_trace[0]
            chosen = getattr(p_outputs, "source", None)
        except Exception:
            chosen = None

        if preferred and chosen and chosen != preferred:
            fb.append(
                f"Router: Prefer '{preferred}' over '{chosen}'. Use lightweight heuristics (e.g., keywords) to route."
            )
        else:
            fb.append("Router: Source selection acceptable. Keep heuristics simple and fast.")

    # Source-specific feedback
    elif pred_name in {"from_db", "from_rag", "from_graph"}:
        # For simplicity, only comment on correctness vs. gold
        if base < 1.0:
            fb.append(
                f"{pred_name}: Returned text didn't include expected content '{gold_answer}'. Ensure source query/lookup is appropriate."
            )
        else:
            fb.append(f"{pred_name}: Good. Keep the result concise and relevant.")

    # Reranker-specific feedback
    elif pred_name == "rerank":
        try:
            _, p_inputs, _ = pred_trace[0]
            db_t = str(p_inputs.get("db_text", ""))
            rag_t = str(p_inputs.get("rag_text", ""))
            gr_t = str(p_inputs.get("graph_text", ""))
        except Exception:
            db_t = rag_t = gr_t = ""

        # If incorrect, guide toward candidate that matches gold better
        if base < 1.0:
            hints = []
            if gold_answer.lower() in db_t.lower():
                hints.append("db")
            if gold_answer.lower() in rag_t.lower():
                hints.append("rag")
            if gold_answer.lower() in gr_t.lower():
                hints.append("graph")
            if hints:
                fb.append(f"Rerank: Prefer candidate(s) from {hints} matching expected content.")
            else:
                fb.append("Rerank: None of the candidates matched expected content; refine upstream or selection rules.")
        else:
            fb.append("Rerank: Good selection; keep cheap features decisive (schema/keywords/length).")

    else:
        fb.append("Program: Maintain correctness with minimal latency.")

    feedback = " ".join(fb).strip()
    return dspy.Prediction(score=base, feedback=feedback)


## dataset is provided by real_world.factory for consistency across demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Use DummyLM for a local dry run")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--save-dir", default="real_world/exports")
    parser.add_argument("--save-prefix", default="simple_gepa_routed")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    logger.info("Starting routed multi-source example (DB/RAG/Graph)")

    program = RoutedSources()
    program.route.signature = program.route.signature.with_instructions(
        "与えられた質問の意図に基づき、'db' / 'rag' / 'graph' から最も適切な情報源を一つ選んで 'source' に出力してください。"
    )
    program.from_db.signature = program.from_db.signature.with_instructions(
        "DBからクエリに対応する短いテキストを返してください（例: メールアドレスなど）。"
    )
    program.from_rag.signature = program.from_rag.signature.with_instructions(
        "RAG（外部知識）からクエリに対応する短い要約テキストを返してください。"
    )
    program.from_graph.signature = program.from_graph.signature.with_instructions(
        "GraphDBからクエリに対応する関係情報を短く記述してください。"
    )
    program.rerank.signature = program.rerank.signature.with_instructions(
        "与えられた候補（db_text, rag_text, graph_text）の中から、最も質問に適合する一つを 'text' に出力してください。"
    )

    before = {n: p.signature.instructions for n, p in program.named_predictors()}

    trainset, valset = routed_sources_dummy(locale="ja")
    logger.info("Dataset — train: {}, val: {}", len(trainset), len(valset))

    if args.dummy:
        logger.info("Configuring DummyLMs (JSONAdapter) for router/sources/reranker")
        from dspy.utils.dummies import DummyLM
        import itertools

        # Router LM cycles: db -> rag -> graph
        def route_responses():
            while True:
                yield {"source": "db"}
                yield {"source": "rag"}
                yield {"source": "graph"}

        # DB source outputs (good for first, irrelevant for others)
        def db_responses():
            while True:
                yield {"text": "user42@example.com"}
                yield {"text": "N/A"}
                yield {"text": "N/A"}

        # RAG source outputs (good for second)
        def rag_responses():
            while True:
                yield {"text": "N/A"}
                yield {"text": "ポリシー更新: Policy updated in 2023"}
                yield {"text": "N/A"}

        # Graph source outputs (good for third)
        def graph_responses():
            while True:
                yield {"text": "N/A"}
                yield {"text": "N/A"}
                yield {"text": "NodeA と NodeB はエッジXで接続: NodeA connected to NodeB via edge X"}

        # Reranker picks the correct candidate for each example
        def rerank_responses():
            while True:
                yield {"text": "user42@example.com"}  # choose DB
                yield {"text": "Policy updated in 2023"}  # choose RAG
                yield {"text": "NodeA connected to NodeB via edge X"}  # choose Graph

        route_lm = DummyLM(route_responses(), adapter=JSONAdapter())
        db_lm = DummyLM(db_responses(), adapter=JSONAdapter())
        rag_lm = DummyLM(rag_responses(), adapter=JSONAdapter())
        graph_lm = DummyLM(graph_responses(), adapter=JSONAdapter())
        rerank_lm = DummyLM(rerank_responses(), adapter=JSONAdapter())

        # Attach per-predictor LMs
        program._route_lm = route_lm
        program._from_db_lm = db_lm
        program._from_rag_lm = rag_lm
        program._from_graph_lm = graph_lm
        program._rerank_lm = rerank_lm
        # Use reranker LM also as reflection LM in dummy mode
        reflection_lm = rerank_lm

        # Global defaults (light by default, can override with context)
        dspy.settings.configure(lm=rag_lm, adapter=JSONAdapter(), rerank_policy="light")
    else:
        logger.info("Configuring real LMs via helper (OpenAI).")
        task_lm = openai_gpt_4o_mini_lm
        dspy.settings.configure(lm=task_lm, rerank_policy="light")
        reflection_lm = openai_gpt_4o_lm

    # Baseline
    from dspy.evaluate import Evaluate
    evaluator = Evaluate(devset=valset, metric=routed_metric_with_feedback, display_progress=False, num_threads=1)

    logger.info("Baseline (light policy)")
    with dspy.context(rerank_policy="light"):
        base = evaluator(program)
    logger.success("Baseline score: {}", base.score)

    # GEPA (heavy policy)
    gepa = dspy.GEPA(
        metric=routed_metric_with_feedback,
        max_metric_calls=60,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=1,
        track_stats=True,
    )

    logger.info("Running GEPA compile (heavy policy)...")
    with dspy.context(rerank_policy="heavy"):
        optimized = gepa.compile(program, trainset=trainset, valset=valset)
    logger.success("GEPA compile finished.")

    logger.info("Post-GEPA eval (light policy)")
    with dspy.context(rerank_policy="light"):
        improved = evaluator(optimized)
    logger.success("Post-GEPA score: {}", improved.score)

    # Save artifacts
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bpath = os.path.join(args.save_dir, f"{args.save_prefix}-baseline-{stamp}.json")
        opath = os.path.join(args.save_dir, f"{args.save_prefix}-optimized-{stamp}.json")
        program.save(bpath)
        optimized.save(opath)
        logger.success("Saved baseline: {}", bpath)
        logger.success("Saved optimized: {}", opath)
        if hasattr(optimized, "detailed_results") and optimized.detailed_results is not None:
            dr_path = os.path.join(args.save_dir, f"{args.save_prefix}-gepa-details-{stamp}.json")
            with open(dr_path, "w", encoding="utf-8") as f:
                json.dump(optimized.detailed_results.to_dict(), f, ensure_ascii=False, indent=2)
            logger.success("Saved GEPA details: {}", dr_path)
    except Exception as e:
        logger.warning("Failed to save artifacts: {}", e)


if __name__ == "__main__":
    main()
