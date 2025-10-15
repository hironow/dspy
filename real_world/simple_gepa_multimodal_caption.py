"""
Multimodal captioning with GEPA (v3+): image -> caption, keywords

- Module: caption = dspy.Predict("image: dspy.Image -> caption: str, keywords: list[str]")
- Metric: coverage of gold.keywords in (pred.caption + pred.keywords) with brevity hint
- GEPA: optimize caption instructions; use MultiModalInstructionProposer for reflection
- Supports dummy (no external calls) and real LMs (OpenAI via helper)

Usage:
  uv run python real_world/simple_gepa_multimodal_caption.py --dummy

GEPA compile requirements:
- metric(gold, pred, trace, pred_name, pred_trace)
- exactly one of: auto | max_full_evals | max_metric_calls
- reflection_lm or instruction_proposer (e.g., MultiModalInstructionProposer)
- trainset (and recommended valset)
"""

from __future__ import annotations

import argparse

from loguru import logger

import dspy
from dspy.adapters.types import Image as DspyImage
from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer
from real_world.cli import add_standard_args, setup_logging
from real_world.cost import log_baseline_estimate, log_gepa_estimate, log_recorded_gepa_cost
from real_world.dummy_lm import configure_dummy_adapter, make_dummy_lm_json
from real_world.factory import image_caption_dummy
from real_world.helper import openai_gpt_4o_lm, openai_gpt_4o_mini_lm
from real_world.save import save_artifacts
from real_world.utils import summarize_before_after, summarize_gepa_results
from real_world.wandb import get_wandb_args, make_run_name
from real_world.metrics_utils import confusion_outcomes, safe_trace_log


class Captioner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.caption = dspy.Predict("image: dspy.Image -> caption: str, keywords: list[str]")

    def forward(self, image: DspyImage):
        lm = getattr(self, "_caption_lm", None)
        if lm is not None:
            with dspy.context(lm=lm):
                return self.caption(image=image)
        return self.caption(image=image)


def _normalize_words(words: list[str]) -> set[str]:
    return {str(w or "").strip().lower() for w in words if str(w or "").strip()}


def caption_metric(
    gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name: str | None = None, pred_trace=None
):
    """Coverage-based metric with brevity hint.

    gold.keywords: list[str] expected key ideas
    pred: caption: str, keywords: list[str]
    score = coverage ∈ [0,1], penalize overly long caption slightly

    Modes:
    - Evaluate mode (called without `pred_name/pred_trace`): return a scalar in [0,1].
    - GEPA mode (called with `pred_name/pred_trace`): return `dspy.Prediction(score, feedback)` and optionally
      use `pred_name`/`pred_trace` to provide short, actionable guidance for the predictor.
    """
    gold_k = _normalize_words(list(getattr(gold, "keywords", []) or []))
    pred_caption = str(getattr(pred, "caption", ""))
    pred_k = _normalize_words(list(getattr(pred, "keywords", []) or []))

    # compute coverage: a keyword is found if appears in caption or in pred keywords
    caption_lower = pred_caption.lower()
    hits = 0
    for k in gold_k:
        if not k:
            continue
        if (k in caption_lower) or (k in pred_k):
            hits += 1
    coverage = (hits / max(1, len(gold_k))) if gold_k else 0.0

    # brevity penalty for very long captions (soft)
    penalty = 1.0
    if len(pred_caption) > 300:
        penalty = 0.8
    elif len(pred_caption) > 180:
        penalty = 0.9

    score = round(max(0.0, min(1.0, coverage * penalty)), 3)

    # Binary framing reused for trace and feedback
    gold_pos = bool(gold_k)
    guess_pos = gold_pos and (coverage == 1.0)
    pred_claim = bool(pred_caption.strip()) or bool(pred_k)
    conf = confusion_outcomes(gold_pos, guess_pos, pred_claim)

    # Trace essentials for reflection/debug
    safe_trace_log(
        trace,
        {
            "coverage": coverage,
            "missing": len([k for k in gold_k if (k not in pred_k and k not in caption_lower)]),
            "caption_len": len(pred_caption),
            "score": score,
            "confusion": conf,
        },
    )

    if pred_name is None and pred_trace is None:
        return score

    # feedback via TP/FN/FP/TN
    missing = [k for k in gold_k if (k not in pred_k and k not in caption_lower)]
    fb_parts = []
    if conf["TP"]:
        fb_parts.append("主要キーワードを十分に含んでいます。")
    elif conf["FN"]:
        fb_parts.append(f"不足キーワード: {missing}")
    elif conf["FP"]:
        fb_parts.append("不要な説明やキーワードがあります。必要がない場合は簡潔に。")
    else:  # TN
        fb_parts.append("無回答が正しいケースです。不要な記述を避けてください。")

    if len(pred_caption) > 180:
        fb_parts.append("説明は簡潔に（1〜2文、要点先述）。")
    fb_parts.append("観点: 主体/属性（色・形）/行為・状態/背景の順に短く。")
    if pred_name:
        fb_parts.append(f"Target predictor: {pred_name}.")
    return dspy.Prediction(score=score, feedback=" ".join(fb_parts))


def main():
    parser = argparse.ArgumentParser()
    add_standard_args(parser, default_save_prefix="simple_gepa_multimodal")
    args = parser.parse_args()

    setup_logging(args.log_level)

    logger.info("Starting multimodal captioning GEPA example")

    program = Captioner()
    program.caption.signature = program.caption.signature.with_instructions(
        "画像を見ていない人に伝えるため、主体→属性（色・形）→行為・状態→背景の順で、1〜2文の簡潔な説明とキーワードを出力してください。"
    )

    before = {n: p.signature.instructions for n, p in program.named_predictors()}

    trainset, valset = image_caption_dummy(locale="ja")
    logger.info("Dataset — train: {}, val: {}", len(trainset), len(valset))

    if args.dummy:
        # Dummy caption outputs alternate between partial and full coverage
        def caption_responses():
            i = 0
            while True:
                if i % 2 == 0:
                    yield {"caption": "茶色の犬が屋外でこちらを見る。", "keywords": ["犬", "茶色"]}
                else:
                    yield {"caption": "赤い車が道路に停まっている。背景に建物。", "keywords": ["車", "道路", "赤"]}
                i += 1

        def reflection_responses():
            # Short instruction improvements
            while True:
                yield {"improved_instruction": "主体/属性/行為/背景の順で1文に要点をまとめ、キーワードも併記。"}

        caption_lm = make_dummy_lm_json(caption_responses())
        program._caption_lm = caption_lm
        configure_dummy_adapter(lm=caption_lm)
        reflection_lm = make_dummy_lm_json(reflection_responses())
    else:
        # Real LMs: task uses a lighter model, reflection uses a stronger multimodal model
        task_lm = openai_gpt_4o_mini_lm
        dspy.settings.configure(lm=task_lm)
        reflection_lm = openai_gpt_4o_lm

    from dspy.evaluate import Evaluate

    evaluator = Evaluate(devset=valset, metric=caption_metric, display_progress=False, num_threads=1)

    logger.info("Baseline evaluation on {} validation examples...", len(valset))
    log_baseline_estimate(valset_size=len(valset), num_predictors=len(program.predictors()), logger=logger)
    baseline = evaluator(program)
    logger.success("Baseline score: {}", baseline.score)

    # GEPA with multimodal instruction proposer
    proposer = MultiModalInstructionProposer()
    gepa = dspy.GEPA(
        metric=caption_metric,
        max_metric_calls=60 if args.dummy else None,
        auto=None if args.dummy else "light",
        reflection_lm=reflection_lm,
        instruction_proposer=proposer,
        reflection_minibatch_size=1,
        track_stats=True,
        **get_wandb_args(project="real_world", run_name=make_run_name(args.save_prefix), enabled=not args.dummy),
    )

    log_gepa_estimate(
        gepa=gepa,
        num_predictors=len(program.predictors()),
        valset_size=len(valset),
        trainset_size=len(trainset),
        logger=logger,
    )

    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    logger.success("GEPA compile finished.")

    logger.info("Evaluating optimized program on validation set...")
    improved = evaluator(optimized)
    logger.success("Post-GEPA score: {}", improved.score)

    summarize_gepa_results(optimized, logger, top_k=10)
    summarize_before_after(before, optimized, logger)
    if hasattr(optimized, "detailed_results") and optimized.detailed_results is not None:
        log_recorded_gepa_cost(optimized.detailed_results, num_predictors=len(program.predictors()), logger=logger)

    save_artifacts(
        program, optimized, save_dir=args.save_dir, prefix=args.save_prefix, logger=logger, save_details=True
    )


if __name__ == "__main__":
    main()
