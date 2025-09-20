"""
Multimodal captioning (Observation -> Composition) with GEPA (v3+)

- Modules:
  - analyze = dspy.Predict("image: dspy.Image -> objects: list[str], attributes: list[str], actions: list[str], scene: str, meta: list[str]")
  - compose = dspy.Predict("objects: list[str], attributes: list[str], actions: list[str], scene: str, meta: list[str] -> caption: str, keywords: list[str]")
- Metric: coverage of gold.keywords in (pred.caption + pred.keywords) with brevity hint
  - GEPAでは pred_name を用いて analyze / compose に別々の改善FBを返す
- GEPA: MultiModalInstructionProposer により、画像を含む反射データから各段の指示文を最適化
- ダミー/実LM両対応
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from loguru import logger

import dspy
from dspy.adapters.types import Image as DspyImage
from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer

from real_world.helper import openai_gpt_4o_mini_lm, openai_gpt_4o_lm
from real_world.factory import image_caption_dummy
from real_world.dummy_lm import make_dummy_lm_json, configure_dummy_adapter
from real_world.utils import summarize_gepa_results, summarize_before_after
from real_world.cost import log_baseline_estimate, log_gepa_estimate, log_recorded_gepa_cost
from real_world.wandb import get_wandb_args
from real_world.save import save_artifacts


class ObsComposeCaptioner(dspy.Module):
    def __init__(self):
        super().__init__()
        # 1) 観測ステージ（objects/attributes/actions/scene/meta）
        self.analyze = dspy.Predict(
            "image: dspy.Image -> objects: list[str], attributes: list[str], actions: list[str], scene: str, meta: list[str]"
        )
        # 2) 作文ステージ（caption/keywords）
        self.compose = dspy.Predict(
            "objects: list[str], attributes: list[str], actions: list[str], scene: str, meta: list[str] -> caption: str, keywords: list[str]"
        )

    def forward(self, image: DspyImage):
        # Optional per-stage LMs
        a_lm = getattr(self, "_analyze_lm", None)
        c_lm = getattr(self, "_compose_lm", None)

        if a_lm is not None:
            with dspy.context(lm=a_lm):
                obs = self.analyze(image=image)
        else:
            obs = self.analyze(image=image)

        kwargs = dict(
            objects=getattr(obs, "objects", []) or [],
            attributes=getattr(obs, "attributes", []) or [],
            actions=getattr(obs, "actions", []) or [],
            scene=getattr(obs, "scene", "") or "",
            meta=getattr(obs, "meta", []) or [],
        )

        if c_lm is not None:
            with dspy.context(lm=c_lm):
                return self.compose(**kwargs)
        return self.compose(**kwargs)


def _normalize_words(words: list[str]) -> set[str]:
    return {str(w or "").strip().lower() for w in words if str(w or "").strip()}


def caption_metric(
    gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name: str | None = None, pred_trace=None
):
    """Coverage-based metric with brevity hint and stage-aware feedback."""
    gold_k = _normalize_words(list(getattr(gold, "keywords", []) or []))
    pred_caption = str(getattr(pred, "caption", ""))
    pred_k = _normalize_words(list(getattr(pred, "keywords", []) or []))

    caption_lower = pred_caption.lower()
    hits = 0
    for k in gold_k:
        if not k:
            continue
        if (k in caption_lower) or (k in pred_k):
            hits += 1
    coverage = (hits / max(1, len(gold_k))) if gold_k else 0.0

    penalty = 1.0
    if len(pred_caption) > 300:
        penalty = 0.8
    elif len(pred_caption) > 180:
        penalty = 0.9

    score = round(max(0.0, min(1.0, coverage * penalty)), 3)

    if pred_name is None and pred_trace is None:
        return score

    missing = [k for k in gold_k if (k not in pred_k and k not in caption_lower)]
    fb_parts: list[str] = []

    if pred_name == "analyze":
        if missing:
            fb_parts.append(
                f"観測に不足: {missing}（対象物・属性・行為・背景の候補として抽出してください）"
            )
        else:
            fb_parts.append("観測は主要要素を網羅しています。")
        fb_parts.append(
            "画像から主体→属性（色・形）→行為→背景の順で、簡潔に語彙を列挙。メタ情報も加味。"
        )
    elif pred_name == "compose":
        if missing:
            fb_parts.append(f"説明に不足: {missing}（1〜2文で必ず触れる）")
        else:
            fb_parts.append("説明は主要キーワードを十分に含んでいます。")
        if len(pred_caption) > 180:
            fb_parts.append("説明は簡潔に（1〜2文、要点先述）。")
        fb_parts.append(
            "順序: 主体/属性/行為/背景。観測語彙（objects/attributes/actions/scene/meta）を自然に統合。"
        )
    else:
        if missing:
            fb_parts.append(f"不足キーワード: {missing}")
        else:
            fb_parts.append("主要キーワードを十分に含んでいます。")
        if len(pred_caption) > 180:
            fb_parts.append("説明は簡潔に（1〜2文、要点先述）。")
        fb_parts.append("観点: 主体/属性（色・形）/行為・状態/背景の順に短く。")

    if pred_name:
        fb_parts.append(f"Target predictor: {pred_name}.")
    return dspy.Prediction(score=score, feedback=" ".join(fb_parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Use DummyLM (no external calls)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--save-dir", default="real_world/exports")
    parser.add_argument("--save-prefix", default="simple_gepa_multimodal_oc")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    logger.info("Starting multimodal Obs→Compose GEPA example")

    program = ObsComposeCaptioner()
    program.analyze.signature = program.analyze.signature.with_instructions(
        "画像から主要な観測語彙を抽出してください。順序は 主体→属性（色・形）→行為→背景。meta にはEXIF等の補足があれば短語で。"
    )
    program.compose.signature = program.compose.signature.with_instructions(
        "与えられた観測（objects/attributes/actions/scene/meta）を自然に統合し、1〜2文の簡潔な説明と keywords を出力してください。"
    )

    before = {n: p.signature.instructions for n, p in program.named_predictors()}

    trainset, valset = image_caption_dummy(locale="ja")
    logger.info("Dataset — train: {}, val: {}", len(trainset), len(valset))

    if args.dummy:
        # Dummy LMs for two-stage pipeline
        def analyze_responses():
            i = 0
            while True:
                if i % 2 == 0:
                    yield {
                        "objects": ["犬"],
                        "attributes": ["茶色"],
                        "actions": ["見る"],
                        "scene": "屋外",
                        "meta": [],
                    }
                else:
                    yield {
                        "objects": ["車"],
                        "attributes": ["赤"],
                        "actions": ["停車"],
                        "scene": "道路",
                        "meta": ["建物"],
                    }
                i += 1

        def compose_responses():
            i = 0
            while True:
                if i % 2 == 0:
                    yield {
                        "caption": "茶色の犬が屋外でこちらを見る。",
                        "keywords": ["犬", "茶色"],
                    }
                else:
                    yield {
                        "caption": "赤い車が道路に停まっている。背景に建物。",
                        "keywords": ["車", "道路", "赤"],
                    }
                i += 1

        def reflection_responses():
            while True:
                yield {
                    "improved_instruction": "観測は主体/属性/行為/背景を箇条書き、作文は1文に要点を統合し keywords 併記。"
                }

        analyze_lm = make_dummy_lm_json(analyze_responses())
        compose_lm = make_dummy_lm_json(compose_responses())
        program._analyze_lm = analyze_lm
        program._compose_lm = compose_lm
        configure_dummy_adapter(lm=compose_lm)
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

    proposer = MultiModalInstructionProposer()
    gepa = dspy.GEPA(
        metric=caption_metric,
        max_metric_calls=60 if args.dummy else None,
        auto=None if args.dummy else "light",
        reflection_lm=reflection_lm,
        instruction_proposer=proposer,
        reflection_minibatch_size=1,
        track_stats=True,
        **get_wandb_args(
            project="real_world",
            run_name=f"{args.save_prefix}-{time.strftime('%Y%m%d-%H%M%S')}",
            enabled=not args.dummy,
        ),
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

