"""
Task-specific metric GEPA example (v3+), still simple but a bit more practical.

- Reuses a tiny DSPy program (Predict-based QA)
- Implements a task-specific metric with feedback:
  - synonym-aware correctness
  - brevity preference (single short word)
  - near-miss partial credit via edit distance (<=1)
  - multi-objective weighted score: score = 0.8*correctness + 0.2*brevity
- Runs GEPA with a small train/val split
- Works locally with --dummy (no external calls) using JSONAdapter for stable parsing

Usage:
  uv run python real_world/simple_gepa_task_metric.py --dummy

Notes:
- The metric accepts (gold, pred, trace, pred_name, pred_trace) and returns
  either a float or dspy.Prediction(score=..., feedback=...).
"""

from __future__ import annotations

import argparse
import sys
import time

from loguru import logger

import dspy
from real_world.cost import log_baseline_estimate, log_gepa_estimate, log_recorded_gepa_cost
from real_world.dummy_lm import configure_dummy_adapter, make_dummy_lm_json
from real_world.factory import task_metric_qa_dummy
from real_world.helper import openai_gpt_4o_lm, openai_gpt_4o_mini_lm
from real_world.utils import summarize_before_after, summarize_gepa_results
from real_world.wandb import get_wandb_args


class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.Predict("question -> refined_question")
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question: str):
        rw_lm = getattr(self, "_rewrite_lm", None)
        if rw_lm is not None:
            with dspy.context(lm=rw_lm):
                rew = self.rewrite(question=question)
        else:
            rew = self.rewrite(question=question)
        rq = getattr(rew, "refined_question", None) or question

        pr_lm = getattr(self, "_predict_lm", None)
        if pr_lm is not None:
            with dspy.context(lm=pr_lm):
                return self.predict(question=rq)
        else:
            return self.predict(question=rq)


def _normalize_color(s: str) -> str:
    s = (s or "").strip().lower()
    # Very small synonym map for demo
    blue = {"青", "あお", "ブルー", "blue"}
    yellow = {"黄色", "きいろ", "イエロー", "yellow", "黄"}
    if s in blue:
        return "青"
    if s in yellow:
        return "黄色"
    return s


def qa_metric_task_specific(
    gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name: str | None = None, pred_trace=None
):
    """
    Task-specific metric:
    - Correctness: allow tiny synonym set (青/ブルー, 黄色/イエロー...). Near-miss (edit distance<=1) gets partial credit.
    - Brevity: prefer a single short word (<=5 chars). Single but long gets partial credit.
    - Final score: 0.8*correctness + 0.2*brevity. GEPA receives textual feedback explaining both components.

    Modes:
    - Evaluate mode (called without `pred_name/pred_trace`): return a scalar in [0,1].
    - GEPA mode (called with `pred_name/pred_trace`): return `dspy.Prediction(score, feedback)`; you may use
      `pred_name`/`pred_trace` to tailor guidance for `rewrite` or `predict` if desired.
    """
    gold_ans_raw = str(getattr(gold, "answer", "")).strip()
    pred_ans_raw = str(getattr(pred, "answer", "")).strip()

    gold_norm = _normalize_color(gold_ans_raw)
    pred_norm = _normalize_color(pred_ans_raw)

    # --- Correctness (with near-miss partial credit) ---
    def _edit_distance(a: str, b: str) -> int:
        # Simple DP Levenshtein for short strings
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # delete
                    dp[i][j - 1] + 1,  # insert
                    dp[i - 1][j - 1] + cost,  # replace
                )
        return dp[n][m]

    if gold_norm and pred_norm == gold_norm:
        s_correct = 1.0
    elif gold_norm and pred_norm:
        s_correct = 0.6 if _edit_distance(gold_norm, pred_norm) <= 1 else 0.0
    else:
        s_correct = 0.0

    # --- Brevity (format preference) ---
    single_word = (" " not in pred_ans_raw) and len(pred_ans_raw) > 0
    short_enough = len(pred_ans_raw) <= 5
    if single_word and short_enough:
        s_brevity = 1.0
    elif single_word:
        s_brevity = 0.6
    else:
        s_brevity = 0.0

    score = round(0.8 * s_correct + 0.2 * s_brevity, 3)

    if pred_name is None and pred_trace is None:
        return score

    # --- Feedback ---
    fb = []
    if s_correct == 1.0:
        fb.append("Correctness: exact/synonym match.")
    elif s_correct >= 0.6:
        fb.append(f"Correctness: near-miss (minor typo). Expected '{gold_norm}', got '{pred_ans_raw}'.")
    else:
        fb.append(f"Correctness: mismatch. Expected '{gold_norm}', got '{pred_ans_raw}'.")

    if s_brevity == 1.0:
        fb.append("Brevity: good (single short word).")
    elif s_brevity >= 0.6:
        fb.append("Brevity: single word but long; prefer a shorter noun (<=5 chars).")
    else:
        fb.append("Brevity: too verbose; answer should be one noun.")

    if pred_name:
        fb.append(f"Target predictor: {pred_name}.")

    feedback = " ".join(fb)
    return dspy.Prediction(score=score, feedback=feedback)


## dataset is provided by real_world.factory for consistency across demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Use DummyLM for a local dry run")
    parser.add_argument("--log-level", default="INFO", help="Log level (DEBUG/INFO/...)")
    parser.add_argument("--save-dir", default="real_world/exports", help="Directory to save DSPy artifacts (.json)")
    parser.add_argument("--save-prefix", default="simple_gepa_task", help="Filename prefix for saved artifacts")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    logger.info("Starting GEPA example with task-specific metric")

    program = SimpleQA()
    program.rewrite.signature = program.rewrite.signature.with_instructions(
        "与えられた日本語の質問を、意味を保ったまま簡潔に言い換えてください。曖昧さを避け、不要な語を省いてください。"
    )
    program.predict.signature = program.predict.signature.with_instructions(
        "次の日本語の質問に、短く正確に回答してください。回答は名詞一語を目指してください。"
    )

    before_instructions = {name: p.signature.instructions for name, p in program.named_predictors()}

    trainset, valset = task_metric_qa_dummy(locale="ja")
    logger.info("Dataset — train: {}, val: {}", len(trainset), len(valset))

    if args.dummy:
        logger.info("Configuring DummyLM (JSON) for task + reflection (no external calls)")
        import itertools

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
                "回答は名詞一語を目指してください。",
            ]
            for p in itertools.cycle(phrases):
                yield {"improved_instruction": p}

        rewrite_lm = make_dummy_lm_json(infinite_rewrite_responses())
        predict_lm = make_dummy_lm_json(infinite_predict_responses())
        program._rewrite_lm = rewrite_lm
        program._predict_lm = predict_lm
        configure_dummy_adapter(lm=predict_lm)
        reflection_lm = make_dummy_lm_json(infinite_reflection_responses())
    else:
        logger.info("Configuring real LMs via helper (OpenAI).")
        task_lm = openai_gpt_4o_mini_lm
        dspy.settings.configure(lm=task_lm)
        reflection_lm = openai_gpt_4o_lm

    # Baseline
    from dspy.evaluate import Evaluate

    evaluator = Evaluate(devset=valset, metric=qa_metric_task_specific, display_progress=False, num_threads=1)
    logger.info("Evaluating baseline on {} validation examples...", len(valset))
    log_baseline_estimate(valset_size=len(valset), num_predictors=len(program.predictors()), logger=logger)
    baseline = evaluator(program)
    logger.success("Baseline score: {}", baseline.score)

    # GEPA
    gepa = dspy.GEPA(
        metric=qa_metric_task_specific,
        max_metric_calls=60,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=1,
        track_stats=True,
        **get_wandb_args(
            project="real_world",
            run_name=f"{args.save_prefix}-{time.strftime('%Y%m%d-%H%M%S')}",
            enabled=not args.dummy,
        ),
    )

    logger.info("Running GEPA compile (max_metric_calls={})...", gepa.max_metric_calls)
    log_gepa_estimate(
        gepa=gepa,
        num_predictors=len(program.predictors()),
        valset_size=len(valset),
        trainset_size=len(trainset),
        logger=logger,
    )
    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    logger.success("GEPA compile finished.")

    # Post
    logger.info("Evaluating optimized program on validation set...")
    improved = evaluator(optimized)
    logger.success("Post-GEPA score: {}", improved.score)

    summarize_gepa_results(optimized, logger, top_k=10)
    summarize_before_after(before_instructions, optimized, logger)
    if hasattr(optimized, "detailed_results") and optimized.detailed_results is not None:
        log_recorded_gepa_cost(optimized.detailed_results, num_predictors=len(program.predictors()), logger=logger)

    # Save artifacts
    from real_world.save import save_artifacts

    save_artifacts(
        program, optimized, save_dir=args.save_dir, prefix=args.save_prefix, logger=logger, save_details=True
    )


if __name__ == "__main__":
    main()
