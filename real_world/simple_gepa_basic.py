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
from dspy import Example


class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # 2つのPredictorを持つ例（モジュールの個数分だけ指示文が最適化対象）
        self.rewrite = dspy.Predict("question -> refined_question")
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question: str):
        # 1) 質問の簡略化/言い換え
        rew = self.rewrite(question=question)
        rq = getattr(rew, "refined_question", None) or question
        # 2) 言い換え済みの質問で最終回答
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

        # 2つのPredictor呼び出しに対応するため、出力フィールドを交互に用意
        # 例: rewrite -> refined_question, predict -> answer
        task_responses = []
        for i in range(60):  # 少し多めに用意
            task_responses.append({"refined_question": "簡潔な質問"})
            task_responses.append({"answer": ("青" if i % 2 == 0 else "黄色")})

        task_lm = DummyLM(task_responses)
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
    # Predictive notes for real API mode
    preds = len(program.predictors())
    baseline_calls = len(valset) * preds
    logger.info(
        "PREDICTIVE-NOTE [CALLSITE]: 実APIではここで 'Evaluate(program)' -> 'program(**example.inputs())' -> タスクLM への推論リクエストが発生します。"
    )
    logger.info(
        "PREDICTIVE-NOTE: 想定呼び出し回数 ≈ バリデーション件数 × Predictor数 = {} × {} = {}",
        len(valset),
        preds,
        baseline_calls,
    )
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

    # Predictive notes for GEPA in real API mode
    # Determine num_candidates from GEPA auto mode when possible
    auto_mode = getattr(gepa, "auto", None)
    auto_map = {"light": 6, "medium": 12, "heavy": 18}
    n_candidates = auto_map.get(auto_mode, None)

    # Approximate metric-call budget
    approx_budget = None
    approx_budget_min = None
    approx_budget_max = None
    if auto_mode is not None and n_candidates is not None:
        try:
            # Mid (library default assumptions)
            approx_budget = gepa.auto_budget(
                num_preds=preds,
                num_candidates=n_candidates,
                valset_size=len(valset),
            )

            # Compute a plausible range by varying minibatch size (M) and full_eval_steps (m)
            # Lower bound: smaller M + larger m (fewer periodic full evals)
            min_M = getattr(gepa, "reflection_minibatch_size", 3) or 3
            min_M = max(1, int(min_M))
            m_hi = 10  # fewer periodic full evals vs default (5)

            budget_min_via_M = gepa.auto_budget(
                num_preds=preds,
                num_candidates=n_candidates,
                valset_size=len(valset),
                minibatch_size=min_M,
            )
            budget_min_via_Mm = gepa.auto_budget(
                num_preds=preds,
                num_candidates=n_candidates,
                valset_size=len(valset),
                minibatch_size=min_M,
                full_eval_steps=m_hi,
            )

            # Upper bound: larger M (close to valset) + smaller m (more frequent full evals)
            max_M = max(35, len(valset))
            m_lo = 1
            budget_max_via_M = gepa.auto_budget(
                num_preds=preds,
                num_candidates=n_candidates,
                valset_size=len(valset),
                minibatch_size=max_M,
            )
            budget_max_via_Mm = gepa.auto_budget(
                num_preds=preds,
                num_candidates=n_candidates,
                valset_size=len(valset),
                minibatch_size=max_M,
                full_eval_steps=m_lo,
            )

            approx_budget_min = min(budget_min_via_M, budget_min_via_Mm)
            approx_budget_max = max(budget_max_via_M, budget_max_via_Mm)
        except Exception:
            approx_budget = None
    elif getattr(gepa, "max_full_evals", None) is not None:
        approx_budget = gepa.max_full_evals * (len(trainset) + len(valset))
    elif getattr(gepa, "max_metric_calls", None) is not None:
        approx_budget = gepa.max_metric_calls

    # Rough estimate of trials N (mirrors internal logic approximately)
    N = None
    if n_candidates is not None:
        try:
            N = int(max(2 * (preds * 2) * math.log2(n_candidates), 1.5 * n_candidates))
        except Exception:
            N = None

    if approx_budget is not None:
        if approx_budget_min is not None and approx_budget_max is not None:
            logger.info(
                "PREDICTIVE-NOTE RANGE: metric_calls ≈ {} .. {}（mid ≈ {}、auto='{}'）",
                approx_budget_min,
                approx_budget_max,
                approx_budget,
                auto_mode if auto_mode is not None else "manual",
            )
        else:
            logger.info(
                "PREDICTIVE-NOTE: 実APIではGEPA最適化中にタスクLMへの評価/メトリック呼び出しが概ね ~{} 回（auto='{}'想定）発生します。",
                approx_budget,
                auto_mode if auto_mode is not None else "manual",
            )

    # Approximate LM call counts (task vs reflection)
    if approx_budget is not None:
        approx_task_lm_calls = approx_budget * preds
        if approx_budget_min is not None and approx_budget_max is not None:
            logger.info(
                "PREDICTIVE-NOTE RANGE: taskLM_calls ≈ ({}..{}) × {} ≈ {}..{}",
                approx_budget_min,
                approx_budget_max,
                preds,
                approx_budget_min * preds,
                approx_budget_max * preds,
            )
        logger.info(
            "PREDICTIVE-NOTE: タスクLMの推定呼び出し回数（mid） ≈ metric_calls × Predictor数 = {} × {} ≈ {}",
            approx_budget,
            preds,
            approx_task_lm_calls,
        )

    logger.info(
        "PREDICTIVE-NOTE [CALLSITE]: 実APIではGEPA内部で 'adapter.evaluate(...)' がタスクLMを呼び、'propose_new_texts(...)' が反射LM(または提案器)を呼び出します。"
    )
    if N is not None:
        logger.info(
            "PREDICTIVE-NOTE: 反射LMの呼び出しは試行回数Nに概ね比例（N≈{}）。各試行で対象Predictor向けの改良命令を1回程度生成する想定です。",
            N,
        )
    logger.info(
        "PREDICTIVE-NOTE: reflection_minibatch_size={}, candidate_n={}, predictor_count={}",
        getattr(gepa, "reflection_minibatch_size", None),
        n_candidates,
        preds,
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

        # Predictive post-hoc notes if real APIs were used
        if getattr(dr, "total_metric_calls", None) is not None:
            logger.info(
                "PREDICTIVE-NOTE: 実APIならタスクLMへのメトリック呼び出し総数はおよそ {} 回（記録ベース）となります。",
                dr.total_metric_calls,
            )
        if getattr(dr, "num_full_val_evals", None) is not None:
            logger.info(
                "PREDICTIVE-NOTE: フル評価（全Valでの集計）回数はおよそ {} 回（記録ベース）。",
                dr.num_full_val_evals,
            )
        if getattr(dr, "log_dir", None):
            logger.info(
                "PREDICTIVE-NOTE: 実API運用ではログ/成果物は '{}' に蓄積（トレースや候補スナップショットの保存先）。",
                dr.log_dir,
            )

    # BEFORE / AFTER as one table
    after_instructions = {
        name: pred.signature.instructions for name, pred in optimized.named_predictors()
    }

    ba_table = _format_before_after_instructions_table(before_instructions, after_instructions)
    if ba_table:
        logger.info("\n{}", ba_table)
        logger.info(
            "\nNotes:\n"
            "- Predictor: 予測器名。\n"
            "- Before/After: 最適化前/後の命令文（長文は一部省略）。\n"
            "- Changed?: '*' は命令文が更新されたことを示します。"
        )

    # Summarize change
    keys_union = sorted(set(before_instructions) | set(after_instructions))
    changed = sum(1 for k in keys_union if before_instructions.get(k) != after_instructions.get(k))
    if changed == 0:
        logger.warning("Instructions unchanged. With Dummy reflection, proposals may be identical or conservative.")
    else:
        logger.success("Instructions updated by GEPA ({} updated).", changed)

    logger.info("FYI: この例ではGEPAは各Predictorの命令文（instructions）を主に最適化します。重み学習は行いません。")

    # Save programs in DSPy standard format (.json state)
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        baseline_path = os.path.join(args.save_dir, f"{args.save_prefix}-baseline-{stamp}.json")
        optimized_path = os.path.join(args.save_dir, f"{args.save_prefix}-optimized-{stamp}.json")

        program.save(baseline_path)
        optimized.save(optimized_path)
        logger.success("Saved baseline program: {}", baseline_path)
        logger.success("Saved optimized program: {}", optimized_path)

        # Save detailed_results (if any) as JSON for later inspection
        if hasattr(optimized, "detailed_results") and optimized.detailed_results is not None:
            try:
                dr_dict = optimized.detailed_results.to_dict()
                dr_path = os.path.join(args.save_dir, f"{args.save_prefix}-gepa-details-{stamp}.json")
                with open(dr_path, "w", encoding="utf-8") as f:
                    json.dump(dr_dict, f, ensure_ascii=False, indent=2)
                logger.success("Saved GEPA detailed results: {}", dr_path)
            except Exception as e:
                logger.warning("Failed to save GEPA detailed results: {}", e)
    except Exception as e:
        logger.warning("Failed to save DSPy programs: {}", e)


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


def _format_before_after_instructions_table(before: dict[str, str], after: dict[str, str], max_col_width: int = 90) -> str:
    """Create an ASCII table showing BEFORE/AFTER instructions per predictor.

    Columns:
      - Predictor
      - Before
      - After
      - Changed?

    Long text is truncated to keep the table compact.
    """
    try:
        def compact(s: str) -> str:
            # Replace newlines with spaces and collapse whitespace
            s = " ".join(str(s).split())
            if len(s) > max_col_width:
                return s[: max_col_width - 3] + "..."
            return s

        keys = sorted(set(before) | set(after))
        if not keys:
            return ""

        rows = []
        for k in keys:
            b = compact(before.get(k, ""))
            a = compact(after.get(k, ""))
            ch = "*" if before.get(k, "") != after.get(k, "") else ""
            rows.append((k, b, a, ch))

        headers = ("Predictor", "Before", "After", "Changed?")

        # Compute column widths
        cols = list(zip(*([headers] + rows)))
        widths = [max(len(str(x)) for x in col) for col in cols]

        def fmt_row(cells):
            return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=False))

        line = "-+-".join("-" * w for w in widths)

        out = [fmt_row(headers), line]
        for r in rows:
            out.append(fmt_row(r))
        return "\n".join(out)
    except Exception:
        return ""


if __name__ == "__main__":
    main()
