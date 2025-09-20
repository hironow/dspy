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

from loguru import logger

import dspy
from dspy.adapters.types import Image as DspyImage
from dspy.teleprompt.bootstrap_trace import FailedPrediction
from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer
from real_world.cost import log_baseline_estimate, log_gepa_estimate, log_recorded_gepa_cost
from real_world.dummy_lm import configure_dummy_adapter, make_dummy_lm_json
from real_world.factory import image_caption_dummy
from real_world.helper import openai_gpt_4o_lm, openai_gpt_4o_mini_lm
from real_world.save import save_artifacts
from real_world.utils import summarize_before_after, summarize_gepa_results
from real_world.wandb import get_wandb_args


# Signatures: using typed fields improves clarity and parsing stability
#
# NOTE: このファイルは、当リポジトリのシンプルなマルチモーダル例で
#       初めて dspy.Signature を導入しているサンプルです。
#   - GEPA が直接最適化するのは signature.instructions（指示文）です。
#   - ここでのクラス docstring / 各フィールドの desc は最適化の「対象」ではありませんが、
#     Adapter が構造化プロンプトを組み立てる際に参照され、
#     推論の出力安定性や、構文失敗時の「この構造に従ってください」の指示に間接的に効きます。
#   - そのため「変化させたい方針」は instructions にも明記し、
#     desc には構造・型・件数目安などを簡潔に記述する方針にしています。
#
# 以上を踏まえて、analyze/compose の I/O を Signature として定義します。
class Analyze(dspy.Signature):
    """Extract concise observations from the image.

    Output short, discriminative tokens:
    - objects: main nouns (1–4)
    - attributes: colors/shapes (1–4)
    - actions: key actions/states (0–3)
    - scene: a short phrase (e.g., "屋外")
    - meta: optional hints (e.g., "建物")
    """

    image: dspy.Image = dspy.InputField(desc="Input image (structured)")
    objects: list[str] = dspy.OutputField(desc="Main objects (1–4 short nouns)")
    attributes: list[str] = dspy.OutputField(desc="Colors/shapes (1–4)")
    actions: list[str] = dspy.OutputField(desc="Key actions/states (0–3)")
    scene: str = dspy.OutputField(desc="One short phrase, e.g., '屋外'")
    meta: list[str] = dspy.OutputField(desc="Optional hints, short tokens")


class Compose(dspy.Signature):
    """Compose a concise 1–2 sentence caption using observations."""

    objects: list[str] = dspy.InputField(desc="From Analyze.objects")
    attributes: list[str] = dspy.InputField(desc="From Analyze.attributes")
    actions: list[str] = dspy.InputField(desc="From Analyze.actions")
    scene: str = dspy.InputField(desc="From Analyze.scene")
    meta: list[str] = dspy.InputField(desc="From Analyze.meta")
    caption: str = dspy.OutputField(desc="1–2 sentences. Subject→attributes→action→background.")
    keywords: list[str] = dspy.OutputField(desc="Short nouns (<=5)")


class ObsComposeCaptioner(dspy.Module):
    def __init__(self):
        super().__init__()
        # 1) 観測ステージ（objects/attributes/actions/scene/meta）
        self.analyze = dspy.Predict(Analyze)
        # 2) 作文ステージ（caption/keywords）
        self.compose = dspy.Predict(Compose)

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
    """
    Coverage-based metric with brevity hint and stage-aware feedback.

    Modes:
    - Evaluate mode (called without `pred_name/pred_trace`): return a scalar in [0,1].
    - GEPA mode (called with `pred_name/pred_trace`): return `dspy.Prediction(score, feedback)` and use
      `pred_name`/`pred_trace` to provide analyzer/compose-specific guidance (e.g., reflect missing
      observations in the final caption, JSON/format hints on failures).
    """
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

    # Trace-based consistency check between analyze -> compose
    # - Extract last analyze outputs from the full trace
    analyze_obs_tokens: set[str] = set()
    try:
        if isinstance(trace, list):
            for _pred, _inputs, _outputs in reversed(trace):
                # Skip failed parses from analyze
                if isinstance(_outputs, FailedPrediction):
                    continue
                # Identify analyze by its output fields (objects/attributes/actions/scene/meta)
                try:
                    out_items = dict(_outputs.items()) if hasattr(_outputs, "items") else {}
                except Exception:
                    out_items = {}
                if {
                    "objects",
                    "attributes",
                    "actions",
                    "scene",
                    "meta",
                }.issubset(set(out_items.keys())):
                    obs_objects = _normalize_words(list(out_items.get("objects", []) or []))
                    obs_attributes = _normalize_words(list(out_items.get("attributes", []) or []))
                    obs_actions = _normalize_words(list(out_items.get("actions", []) or []))
                    obs_scene = (
                        {str(out_items.get("scene", "")).strip().lower()} if out_items.get("scene", "") else set()
                    )
                    obs_meta = _normalize_words(list(out_items.get("meta", []) or []))
                    analyze_obs_tokens = obs_objects | obs_attributes | obs_actions | obs_scene | obs_meta
                    break
    except Exception:
        pass

    # Reflect missing-from-composition observations and possibly spurious keywords
    if analyze_obs_tokens:
        not_reflected = [tok for tok in sorted(analyze_obs_tokens) if (tok not in caption_lower and tok not in pred_k)]
        if not_reflected:
            fb_parts.append(f"観測未反映: {not_reflected}")

        spurious = [tok for tok in sorted(pred_k) if tok not in analyze_obs_tokens]
        if spurious:
            fb_parts.append(f"観測にない語（要注意）: {spurious}")

    # If the target predictor's trace indicates a formatting failure, surface a schema hint
    try:
        if isinstance(pred_trace, list) and any(isinstance(t[2], FailedPrediction) for t in pred_trace):
            fb_parts.append("出力の構文/形式に従ってください（指定フィールドのみ、型・構造を厳守）。")
    except Exception:
        pass

    if pred_name == "analyze":
        if missing:
            fb_parts.append(f"観測に不足: {missing}（対象物・属性・行為・背景の候補として抽出してください）")
        else:
            fb_parts.append("観測は主要要素を網羅しています。")
        fb_parts.append("画像から主体→属性（色・形）→行為→背景の順で、簡潔に語彙を列挙。メタ情報も加味。")
    elif pred_name == "compose":
        if missing:
            fb_parts.append(f"説明に不足: {missing}（1〜2文で必ず触れる）")
        else:
            fb_parts.append("説明は主要キーワードを十分に含んでいます。")
        if len(pred_caption) > 180:
            fb_parts.append("説明は簡潔に（1〜2文、要点先述）。")
        fb_parts.append("順序: 主体/属性/行為/背景。観測語彙（objects/attributes/actions/scene/meta）を自然に統合。")
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
        # 解析失敗（パースエラー）を反射データに含め、スキーマ遵守を学習材料にする
        add_format_failure_as_feedback=True,
        track_stats=True,
        **get_wandb_args(
            project="real_world",
            run_name=f"{args.save_prefix}-{time.strftime('%Y%m%d-%H%M%S')}",
            enabled=not args.dummy,
        ),
    )

    # NOTE: component_selector の選択について
    # - 既定（round_robin）は段ごとの改善が観察しやすい
    # - 同時最適化を試すなら component_selector="all" を指定し、両段の協調改善を促す選択肢もある
    #   例: dspy.GEPA(..., component_selector="all", ...)

    # NOTE: メトリクスの微調整（軽量のまま）について
    # - meta 反映の軽い加点：meta に有用語があり caption に自然に反映されていれば +ε
    # - keywords 過剰抑制：keywords が過多（例: 6 以上）のときにごく小さな減衰を掛ける
    # - 本デモではスコア式は維持し、GEPA 時フィードバックに観測未反映/観測にない語を追記して改善を誘導

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
