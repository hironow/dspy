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
import math
import os
import time
import json
from loguru import logger
import dspy
from real_world.helper import openai_gpt_4o_mini_lm, openai_gpt_4o_lm
from real_world.factory import basic_qa_dummy
from real_world.dummy_lm import make_dummy_lm_json, configure_dummy_adapter
from real_world.utils import summarize_gepa_results, summarize_before_after
from real_world.cost import log_baseline_estimate, log_gepa_estimate, log_recorded_gepa_cost


class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # 2つのPredictorを持つ例（モジュールの個数分だけ指示文が最適化対象）
        self.rewrite = dspy.Predict("question -> refined_question")
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question: str):
        # 1) 質問の簡略化/言い換え（rewrite専用LMがあればそれを使う）
        rw_lm = getattr(self, "_rewrite_lm", None)
        if rw_lm is not None:
            with dspy.context(lm=rw_lm):
                rew = self.rewrite(question=question)
        else:
            rew = self.rewrite(question=question)
        rq = getattr(rew, "refined_question", None) or question
        # 2) 言い換え済みの質問で最終回答（predict専用LMがあればそれを使う）
        pr_lm = getattr(self, "_predict_lm", None)
        if pr_lm is not None:
            with dspy.context(lm=pr_lm):
                return self.predict(question=rq)
        else:
            return self.predict(question=rq)


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


## dataset is provided by real_world.factory for consistency across demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Use DummyLM for a local dry run")
    parser.add_argument("--log-level", default="INFO", help="Log level for loguru (e.g., DEBUG, INFO, SUCCESS, WARNING)")
    parser.add_argument("--save-dir", default="real_world/exports", help="Directory to save DSPy-standard program artifacts (.json)")
    parser.add_argument("--save-prefix", default="simple_gepa", help="Filename prefix for saved artifacts")
    args = parser.parse_args()

    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    logger.info("Starting minimal GEPA example (v3+)")
    logger.debug("DSPy version: {}", getattr(dspy, "__version__", "unknown"))

    # 1) Build a tiny program and dataset
    program = SimpleQA()
    # 日本語のタスク説明（プロンプト説明）を付与（複数Predictor分）
    program.rewrite.signature = program.rewrite.signature.with_instructions(
        "与えられた日本語の質問を、意味を保ったまま簡潔に言い換えてください。曖昧さを避け、不要な語を省いてください。"
    )
    program.predict.signature = program.predict.signature.with_instructions(
        "次の日本語の質問に、短く正確に回答してください。回答は名詞一語を目指してください。"
    )
    # Capture baseline instructions before optimization (for BEFORE/AFTER output)
    before_instructions = {
        name: pred.signature.instructions for name, pred in program.named_predictors()
    }
    trainset, valset = basic_qa_dummy(locale="ja")
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
        import itertools

        # Separate generators per predictor to ensure outputs always match
        # the expected fields for that predictor's signature.
        def infinite_rewrite_responses():
            while True:
                yield {"refined_question": "簡潔な質問"}

        def infinite_predict_responses():
            i = 0
            while True:
                yield {"answer": ("青" if (i % 2 == 0) else "黄色")}
                i += 1

        def infinite_reflection_responses():
            phrases = [
                "簡潔かつ事実に基づいて回答してください。",
                "できるだけ直接的に回答してください。",
                "冗長表現を避け、簡潔な表現を用いてください。",
            ]
            for p in itertools.cycle(phrases):
                yield {"improved_instruction": p}

        # Dedicated LMs per-predictor so outputs always match expected fields
        rewrite_lm = make_dummy_lm_json(infinite_rewrite_responses())
        predict_lm = make_dummy_lm_json(infinite_predict_responses())
        # Attach to program so forward() can use them with dspy.context
        program._rewrite_lm = rewrite_lm
        program._predict_lm = predict_lm
        # Configure default LM and JSON adapter to match DummyLM outputs
        # Note: predictors use per-predictor contexts above, but adapter must match parsing format.
        configure_dummy_adapter(lm=predict_lm)
        logger.debug("Dummy per-predictor LMs configured with JSON adapter (infinite seq). Global adapter set.")
        reflection_lm = make_dummy_lm_json(infinite_reflection_responses())
    else:
        logger.info("Configuring real LMs via helper (OpenAI).")
        # Task LM (default for program)
        task_lm = openai_gpt_4o_mini_lm
        dspy.settings.configure(lm=task_lm)
        # Reflection LM (stronger model recommended)
        reflection_lm = openai_gpt_4o_lm

    # 3) Evaluate baseline
    from dspy.evaluate import Evaluate

    evaluator = Evaluate(devset=valset, metric=qa_metric_with_feedback, display_progress=False, num_threads=1)
    logger.info("Running baseline evaluation on {} validation examples...", len(valset))
    preds = len(program.predictors())
    log_baseline_estimate(valset_size=len(valset), num_predictors=preds, logger=logger)
    baseline = evaluator(program)
    logger.success("Baseline score: {}", baseline.score)

    # 4) Run GEPA (choose one of: auto | max_metric_calls | max_full_evals)
    # Configure GEPA (smaller budget in dummy mode to avoid long runs)
    auto_mode_for_log = "manual" if args.dummy else "light"
    logger.info(
        "Configuring GEPA (auto={}, track_stats={}, reflection_lm={})",
        auto_mode_for_log,
        True,
        "yes" if 'reflection_lm' in locals() and reflection_lm is not None else "no",
    )

    if args.dummy:
        gepa = dspy.GEPA(
            metric=qa_metric_with_feedback,
            max_metric_calls=60,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=1,
            track_stats=True,
        )
    else:
        gepa = dspy.GEPA(
            metric=qa_metric_with_feedback,
            auto="light",  # or set: max_metric_calls=..., or max_full_evals=...
            reflection_lm=reflection_lm,
            track_stats=True,
        )

    log_gepa_estimate(
        gepa=gepa,
        num_predictors=preds,
        valset_size=len(valset),
        trainset_size=len(trainset),
        logger=logger,
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

    # Optional: compact GEPA summary table (common util)
    summarize_gepa_results(optimized, logger, top_k=10)

    # BEFORE / AFTER as one table (common util)
    summarize_before_after(before_instructions, optimized, logger)

    logger.info("FYI: この例ではGEPAは各Predictorの命令文（instructions）を主に最適化します。重み学習は行いません。")
    if hasattr(optimized, "detailed_results") and optimized.detailed_results is not None:
        log_recorded_gepa_cost(optimized.detailed_results, num_predictors=preds, logger=logger)

    # Save artifacts
    from real_world.save import save_artifacts
    save_artifacts(program, optimized, save_dir=args.save_dir, prefix=args.save_prefix, logger=logger, save_details=True)


# (Common display helpers moved to real_world.utils)


if __name__ == "__main__":
    main()
