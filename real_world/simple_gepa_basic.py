"""
Minimal GEPA example (v3+):

- Defines a tiny DSPy program (Predict-based QA)
- Implements a GEPA-friendly metric (score + feedback)
- Runs GEPA with small train/val sets
- Prints pre/post optimization scores

Usage:
  - Dry run with DummyLM (no external calls):
      python real_world/simple_gepa_basic.py --dummy

  - Real LM:
      1) Replace LM configuration below with your provider/model.
      2) Run: python real_world/simple_gepa_basic.py

Required GEPA compile args shown here:
  - metric(gold, pred, trace, pred_name, pred_trace)
  - exactly one of: auto | max_metric_calls | max_full_evals
  - reflection_lm or instruction_proposer
  - trainset (and recommended valset)
"""

from __future__ import annotations

import argparse
import sys
from loguru import logger
import dspy
from dspy import Example


class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question: str):
        return self.predict(question=question)


def qa_metric_with_feedback(gold: Example, pred: dspy.Prediction, trace=None, pred_name: str | None = None, pred_trace=None):
    """
    GEPA-friendly metric: returns either a float or a dict-like Prediction with `score` and `feedback`.

    - gold.answer vs pred.answer exact match (case-insensitive)
    - Provide simple, useful textual feedback
    - pred_name/pred_trace are provided by GEPA for predictor-level reflection (not required to use)
    """
    gold_ans = str(getattr(gold, "answer", "")).strip().lower()
    pred_ans = str(getattr(pred, "answer", "")).strip().lower()

    correct = 1.0 if gold_ans == pred_ans and gold_ans != "" else 0.0

    if correct:
        fb = "Correct answer."
    else:
        fb = f"Expected '{gold_ans}' but got '{pred_ans}'."
        if pred_name is not None:
            fb += f" Target predictor: {pred_name}."

    # Return a plain float for standard evaluation (Evaluate),
    # and (score, feedback) when GEPA asks for predictor/program feedback.
    if pred_name is None and pred_trace is None:
        return correct
    return dspy.Prediction(score=correct, feedback=fb)


def build_tiny_dataset():
    """日本語のごく簡単なQAデータセット（数学を避ける）。"""
    # 小規模でシンプル。必要に応じて拡張してください。
    trainset = [
        Example(question="空の色は何色ですか？", answer="青").with_inputs("question"),
        Example(question="バナナの色は何色ですか？", answer="黄色").with_inputs("question"),
    ]
    valset = [
        Example(question="晴れた日の海の色は何色ですか？", answer="青").with_inputs("question"),
        Example(question="熟したバナナの色は何色ですか？", answer="黄色").with_inputs("question"),
    ]
    return trainset, valset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Use DummyLM for a local dry run")
    parser.add_argument("--log-level", default="INFO", help="Log level for loguru (e.g., DEBUG, INFO, SUCCESS, WARNING)")
    args = parser.parse_args()

    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    logger.info("Starting minimal GEPA example (v3+)")
    logger.debug("DSPy version: {}", getattr(dspy, "__version__", "unknown"))

    # 1) Build a tiny program and dataset
    program = SimpleQA()
    # 日本語のタスク説明（プロンプト説明）を付与
    program.predict.signature = program.predict.signature.with_instructions(
        "次の日本語の質問に、短く正確に回答してください。回答は名詞一語を目指してください。"
    )
    trainset, valset = build_tiny_dataset()
    logger.info("Built tiny dataset — train: {}, val: {}", len(trainset), len(valset))
    logger.debug("Train sample: {}", trainset[0] if trainset else None)
    logger.debug("Val sample: {}", valset[0] if valset else None)

    # 2) LM configuration
    #    - For a real LM, replace the next block with your provider/model
    #    - Example:
    #        task_lm = dspy.LM(model="gpt-4o-mini", temperature=0.0)
    #        reflection_lm = dspy.LM(model="gpt-4o", temperature=0.7)
    #        dspy.settings.configure(lm=task_lm)
    #
    if args.dummy:
        logger.info("Configuring DummyLM for both task and reflection (no external calls).")
        from dspy.utils.dummies import DummyLM

        # A short sequence of answers for our tiny set (日本語)
        # The main LM is used to answer questions
        task_lm = DummyLM([
            {"answer": "青"},
            {"answer": "黄色"},
            {"answer": "青"},
            {"answer": "黄色"},
            {"answer": "青"},
        ])
        dspy.settings.configure(lm=task_lm)
        lm_mode = "dict" if isinstance(task_lm.answers, dict) else "sequence"
        logger.debug("Dummy task LM configured (mode={}, follow_examples={}).", lm_mode, task_lm.follow_examples)

        # Reflection LM proposes improved instructions（日本語）
        reflection_lm = DummyLM([
            {"improved_instruction": "簡潔かつ事実に基づいて回答してください。"},
            {"improved_instruction": "できるだけ直接的に回答してください。"},
            {"improved_instruction": "冗長表現を避け、簡潔な表現を用いてください。"},
        ])
    else:
        logger.warning("Real LM mode selected, but configuration is a placeholder in this example.")
        # Placeholder: set your real models here
        # task_lm = dspy.LM(model="<your-task-model>", temperature=0.0)
        # reflection_lm = dspy.LM(model="<your-strong-reflection-model>", temperature=0.7)
        # dspy.settings.configure(lm=task_lm)
        raise RuntimeError(
            "Please run with --dummy for a local dry run, or edit LM configuration for a real provider."
        )

    # 3) Evaluate baseline
    from dspy.evaluate import Evaluate

    evaluator = Evaluate(devset=valset, metric=qa_metric_with_feedback, display_progress=False)
    logger.info("Running baseline evaluation on {} validation examples...", len(valset))
    baseline = evaluator(program)
    logger.success("Baseline score: {}", baseline.score)

    # 4) Run GEPA (choose one of: auto | max_metric_calls | max_full_evals)
    logger.info(
        "Configuring GEPA (auto={}, track_stats={}, reflection_lm={})",
        "light",
        True,
        "yes" if 'reflection_lm' in locals() and reflection_lm is not None else "no",
    )

    gepa = dspy.GEPA(
        metric=qa_metric_with_feedback,
        auto="light",  # or set: max_metric_calls=..., or max_full_evals=...
        reflection_lm=reflection_lm,
        track_stats=True,
    )

    logger.info("Starting GEPA compile with train={} and val={}...", len(trainset), len(valset))
    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    logger.success("GEPA compile finished.")

    # 5) Evaluate post-optimization
    logger.info("Evaluating optimized program on validation set...")
    improved = evaluator(optimized)
    logger.success("Post-GEPA score: {}", improved.score)

    delta = round((improved.score - baseline.score), 2)
    if delta > 0:
        logger.success("Improvement vs baseline: +{}", delta)
    else:
        logger.warning("No improvement vs baseline (delta = {}).", delta)

    # Optional: inspect run details (Pareto candidates, scores, etc.)
    if hasattr(optimized, "detailed_results"):
        dr = optimized.detailed_results
        logger.info("GEPA proposed {} candidates.", len(dr.candidates))
        if dr.val_aggregate_scores:
            best_score = dr.val_aggregate_scores[dr.best_idx]
            logger.info("Best validation score (Pareto selection): {}", best_score)

            # Show a short summary of top-3 scores
            top_scores = sorted(dr.val_aggregate_scores, reverse=True)[:3]
            logger.debug("Top scores (top-3): {}", top_scores)

            # Pretty print a compact Pareto summary table (no extra deps)
            table = _format_gepa_results_table(dr, top_k=10)
            if table:
                logger.info("\n{}", table)
                # Brief legend to help interpretation
                logger.info(
                    "\nNotes:\n"
                    "- Score: aggregate validation score (higher is better).\n"
                    "- Best@Val: number of validation items where this candidate is best (Pareto coverage).\n"
                    "- DiscoveryCalls: cumulative metric calls when this candidate was discovered (earlier is smaller).\n"
                    "- Best?: '*' marks the globally best candidate by aggregate score (returned program)."
                )


def _format_gepa_results_table(dr, top_k: int = 10) -> str:
    """Create a compact ASCII table summarizing GEPA candidate results.

    Columns:
      - Idx: candidate index
      - Score: aggregate validation score
      - Best@Val: how many val tasks where this candidate is on the Pareto front (best for that task)
      - DiscoveryCalls: cumulative metric calls when this candidate was discovered (if available)
      - Best?: '*' marks global best by aggregate score
    """
    try:
        n = len(dr.val_aggregate_scores)
        if n == 0:
            return ""

        # Compute coverage per candidate using per_val_instance_best_candidates
        coverage = [0] * n
        for s in getattr(dr, "per_val_instance_best_candidates", []) or []:
            for i in s:
                if 0 <= i < n:
                    coverage[i] += 1

        rows = []
        for i, score in enumerate(dr.val_aggregate_scores):
            cov = coverage[i] if i < len(coverage) else 0
            disc = "-"
            try:
                disc = getattr(dr, "discovery_eval_counts", [None] * n)[i]
                if disc is None:
                    disc = "-"
            except Exception:
                disc = "-"
            is_best = "*" if i == dr.best_idx else ""
            rows.append((i, score, cov, disc, is_best))

        # Sort by Score desc, then coverage desc
        rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
        rows = rows[: min(top_k, len(rows))]

        headers = ("Idx", "Score", "Best@Val", "DiscoveryCalls", "Best?")

        # Compute column widths
        cols = list(zip(*([headers] + [(str(i), f"{s:.3f}", str(c), str(d), b) for i, s, c, d, b in rows])))
        widths = [max(len(x) for x in col) for col in cols]

        def fmt_row(cells):
            return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=False))

        line = "-+-".join("-" * w for w in widths)

        out = [fmt_row(headers), line]
        for i, s, c, d, b in rows:
            out.append(fmt_row((str(i), f"{s:.3f}", str(c), str(d), b)))

        return "\n".join(out)
    except Exception:
        # Be conservative: if anything goes wrong, skip the table
        return ""


if __name__ == "__main__":
    main()
