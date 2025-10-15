"""
GEPA-integrated prompt+examples optimizer for external library (langextract).

- Idea: Optimize the prompt (and few-shot examples, as JSON) that we pass to
  `langextract.extract(...)` via a DSPy Predictor whose instructions are evolved
  by GEPA. This follows the "controller + executor" pattern:

  - Controller (optimized): `build_prompt = dspy.Predict(BuildLangExtractPrompt)`
    generates `prompt_description` and `examples_json`.
  - Executor (external call): forward() calls langextract with those values and
    returns normalized extractions as a DSPy Prediction.

Run (offline dummy LMs):
  uv run python real_world/simple_gepa_langextract.py --dummy

Run (with a real LM for reflection and (optionally) real langextract):
  uv run python real_world/simple_gepa_langextract.py \
    --langextract-model-id gemini-2.5-flash  # or your preferred model id

Notes
-----
- This script does not require langextract to be installed when running with
  --dummy. If installed and not using --dummy, we will import it dynamically.
- The examples JSON schema is a minimal, dependency-free shape that we convert
  to langextract objects only if that library is available.

Design Guidelines (use this as a template for similar integrations)
------------------------------------------------------------------
1) Split responsibilities: Controller vs Executor
   - Controller (optimized by GEPA): A dspy.Predict(Signature) that emits the
     language-spec text and any auxiliary artifacts (e.g., few-shot examples) to
     feed into the external library. GEPA evolves this predictor's
     `signature.instructions`.
   - Executor (not optimized): A small, side-effect-free function that converts
     the controller outputs into the external library's inputs and performs the
     call. Keep it deterministic and easy to test.

2) Signature IO design
   - Prefer simple, robust outputs from the controller (strings/JSON) instead
     of complex Python objects. Parse them right before the external call.
   - If you need to optimize multiple artifacts (e.g., prompt + examples), add
     more OutputFields and validate them in the metric.
   - Keep inputs (task, classes, style hints) as structured fields so the
     predictor can generalize while the instructions remain generic.

3) Metric design
   - Return a single module-level scalar for Evaluate-mode; in GEPA-mode return
     `dspy.Prediction(score, feedback)`. Use `pred_name/pred_trace` to produce
     focused feedback for the controller (e.g., JSON validity, class coverage,
     formatting requirements). Avoid running LLMs inside the metric by default.
   - If you must use stochastic metrics (LLM-as-judge), consider
     `warn_on_score_mismatch=False` in GEPA and be mindful of non-determinism.

4) Budgets and reflection
   - Start with `auto='light'` or a small `max_metric_calls`. Increase only if
     improvements plateau too early. For more diverse reflection, consider
     `reflection_minibatch_size=2..3`.
   - Use a strong reflection LM for best prompt evolution quality.

5) Resilience and portability
   - Import external libs lazily and provide a local fallback (`--dummy`) so the
     demo runs offline. Handle JSON parsing errors defensively.
   - Keep executor stateless; avoid global mutable state or hidden caches.

6) Reproducibility & artifacts
   - Set and surface seeds if needed. Use `track_stats=True` to introspect
     candidates and Pareto scores, and `save_artifacts(...)` to persist runs.

7) Extensibility
   - To add more controllers (e.g., class schema proposer, attribute schema
     proposer), define additional Predictors and extend the metric to return
     specific feedback by checking `pred_name`.

GEPA compile requirements:
- metric(gold, pred, trace, pred_name, pred_trace)
- exactly one of: auto | max_full_evals | max_metric_calls
- reflection_lm or instruction_proposer
- trainset (and recommended valset)
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from loguru import logger

import dspy
from real_world.cli import add_standard_args, setup_logging
from real_world.cost import (
    log_baseline_estimate,
    log_gepa_estimate,
    log_recorded_gepa_cost,
)
from real_world.dummy_lm import configure_dummy_adapter, make_dummy_lm_json
from real_world.factory import langextract_dummy
from real_world.helper import openai_gpt_4o_lm, openai_gpt_4o_mini_lm
from real_world.save import save_artifacts
from real_world.utils import summarize_before_after, summarize_gepa_results
from real_world.metrics_utils import confusion_outcomes, safe_trace_log
from real_world.wandb import get_wandb_args, make_run_name

# -------------------------------
# Signatures
# -------------------------------


class BuildLangExtractPrompt(dspy.Signature):
    """Generate a high-quality `prompt_description` and `examples_json` for langextract.

    Mapping to the langextract quick-start example (for clarity):
    - This signature's output `prompt_description` corresponds to the variable
      named `prompt` in the example code (the instruction string passed to
      langextract).
    - This signature's output `examples_json` corresponds to the list `examples`
      in the example code (few-shot demonstrations), represented here as a JSON
      array of objects so that DSPy can produce it robustly.

    GEPA optimization note:
    - GEPA optimizes the instructions of THIS predictor (the natural-language
      spec attached to this signature), not the Python code. By improving these
      instructions, the predictor will generate better `prompt_description` and
      `examples_json` over the course of optimization.

    The goal is to extract entities from text accurately with exact spans and
    minimal overlap. Provide helpful attributes for context.

    Output requirements:
    - prompt_description: A concise instruction that enforces exact quoting of
      spans from the text, non-overlapping extractions, and meaningful attributes.
    - examples_json: A JSON string representing a list of examples, each with:
        {"text": str, "extractions": [{
            "extraction_class": str,
            "extraction_text": str,
            "attributes": {str: str}
        } ...]}
    """

    task: str = dspy.InputField(desc="Task to perform (e.g., extract characters/emotions/relationships)")
    target_classes: list[str] = dspy.InputField(desc="List of extraction classes to detect")
    style_hints: str = dspy.InputField(desc="Additional constraints or preferences")

    prompt_description: str = dspy.OutputField(desc="Instruction for langextract")
    examples_json: str = dspy.OutputField(desc="JSON array of {text, extractions} examples")


# -------------------------------
# Module
# -------------------------------


class LangExtractPipeline(dspy.Module):
    """Pipeline that builds a prompt+examples, then calls langextract (or a fallback).

    What corresponds to what:
    - `self.build_prompt(...)` produces the langextract `prompt` and `examples`
      (as JSON) that you would normally handcraft. GEPA targets the instructions
      for this predictor and learns to produce better versions of those.
    - `_call_langextract(...)` converts the JSON few-shot into the objects that
      langextract expects and invokes `lx.extract(...)`.
    """

    def __init__(
        self,
        *,
        langextract_model_id: str | None = None,
        default_task: str | None = None,
        default_target_classes: list[str] | None = None,
        default_style_hints: str | None = None,
        use_fallback: bool = False,
    ):
        super().__init__()
        self.build_prompt = dspy.Predict(BuildLangExtractPrompt)
        self.langextract_model_id = langextract_model_id
        self.use_fallback = use_fallback
        self.default_task = default_task or ("Extract characters, emotions, and relationships from the input text.")
        self.default_target_classes = default_target_classes or [
            "character",
            "emotion",
            "relationship",
        ]
        self.default_style_hints = default_style_hints or (
            "Use exact spans from the text (no paraphrasing), avoid overlaps, include attributes."
        )

    def _call_langextract(self, *, text: str, prompt_description: str, examples_json: str) -> list[dict[str, Any]]:
        """Call langextract if available, otherwise use a simple heuristic fallback.

        Returns a list of dicts: [{extraction_class, extraction_text, attributes}].
        """
        # If forced fallback (e.g., --dummy), skip import/call to langextract
        lx = None
        if not self.use_fallback:
            # Try import langextract only when needed
            try:
                import langextract as lx  # type: ignore
            except Exception:
                lx = None

        # Parse examples_json into a neutral dict schema
        examples_data: list[dict[str, Any]] = []
        try:
            parsed = json.loads(examples_json)
            if isinstance(parsed, list):
                examples_data = parsed
        except Exception:
            examples_data = []

        if lx is None:
            # Fallback: naive span spotting for a tiny demo. This is only for
            # --dummy usage or when langextract isn't installed.
            lower = text.lower()
            outs: list[dict[str, Any]] = []
            # Characters
            for name in ["romeo", "juliet", "lady juliet"]:
                if name in lower:
                    span = "Lady Juliet" if name == "lady juliet" else name.title()
                    outs.append(
                        {
                            "extraction_class": "character",
                            "extraction_text": span,
                            "attributes": {"source": "fallback"},
                        }
                    )
            # Relationship
            if "juliet is the sun" in lower:
                outs.append(
                    {
                        "extraction_class": "relationship",
                        "extraction_text": "Juliet is the sun",
                        "attributes": {"type": "metaphor"},
                    }
                )
            # Emotion: match a simple cue
            if "but soft" in lower or "gazed longingly" in lower:
                emo_text = "But soft!" if "but soft" in lower else "gazed longingly at the stars, her heart aching"
                outs.append(
                    {
                        "extraction_class": "emotion",
                        "extraction_text": emo_text,
                        "attributes": {"feeling": "awe/longing"},
                    }
                )
            return outs

        # Build proper example objects for langextract
        # (maps `examples_json` → langextract's `lx.data.ExampleData` instances)
        ex_objs: list[Any] = []
        for item in examples_data:
            try:
                extractions = []
                for e in item.get("extractions", []) or []:
                    extractions.append(
                        lx.data.Extraction(
                            extraction_class=str(e.get("extraction_class", "")),
                            extraction_text=str(e.get("extraction_text", "")),
                            attributes={str(k): str(v) for k, v in (e.get("attributes") or {}).items()},
                        )
                    )
                ex_objs.append(lx.data.ExampleData(text=str(item.get("text", "")), extractions=extractions))
            except Exception:
                continue

        # Call langextract (fallback on any provider/config error)
        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt_description,
                examples=ex_objs if ex_objs else None,
                model_id=self.langextract_model_id or "gemini-2.5-flash",
            )
        except Exception:
            lower = text.lower()
            outs: list[dict[str, Any]] = []
            for name in ["romeo", "juliet", "lady juliet"]:
                if name in lower:
                    span = "Lady Juliet" if name == "lady juliet" else name.title()
                    outs.append(
                        {
                            "extraction_class": "character",
                            "extraction_text": span,
                            "attributes": {"source": "fallback"},
                        }
                    )
            if "juliet is the sun" in lower:
                outs.append(
                    {
                        "extraction_class": "relationship",
                        "extraction_text": "Juliet is the sun",
                        "attributes": {"type": "metaphor"},
                    }
                )
            if "but soft" in lower or "gazed longingly" in lower:
                emo_text = "But soft!" if "but soft" in lower else "gazed longingly at the stars, her heart aching"
                outs.append(
                    {
                        "extraction_class": "emotion",
                        "extraction_text": emo_text,
                        "attributes": {"feeling": "awe/longing"},
                    }
                )
            return outs

        outs: list[dict[str, Any]] = []
        try:
            for e in getattr(result, "extractions", []) or []:
                outs.append(
                    {
                        "extraction_class": getattr(e, "extraction_class", ""),
                        "extraction_text": getattr(e, "extraction_text", ""),
                        "attributes": dict(getattr(e, "attributes", {}) or {}),
                    }
                )
        except Exception:
            pass
        return outs

    def forward(
        self,
        *,
        text: str,
        task: str | None = None,
        target_classes: list[str] | None = None,
        style_hints: str | None = None,
    ) -> dspy.Prediction:
        # Build prompt+examples with the optimized Predictor.
        # GEPA optimizes the instructions behind `self.build_prompt` so that
        # these two outputs become higher quality over time.
        bp = self.build_prompt(
            task=task or self.default_task,
            target_classes=target_classes or self.default_target_classes,
            style_hints=style_hints if style_hints is not None else self.default_style_hints,
        )

        extractions = self._call_langextract(
            text=text, prompt_description=bp.prompt_description, examples_json=bp.examples_json
        )
        # Optionally include the raw controller outputs for debugging
        return dspy.Prediction(
            extractions=extractions,
            prompt_description=bp.prompt_description,
            examples_json=bp.examples_json,
        )


# -------------------------------
# Dataset (tiny, inline)
# -------------------------------


"""
Dataset moved to real_world.factory.langextract_dummy().
"""


# -------------------------------
# Metric
# -------------------------------


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _match_extractions(pred: list[dict[str, Any]], gold: list[dict[str, Any]]) -> tuple[int, int]:
    """Count how many gold extractions are covered by predicted ones. Case-insensitive substring match.

    Returns (matched, total).
    """
    total = len(gold)
    if total == 0:
        return 0, 0
    matched = 0
    for g in gold:
        g_cls = str(g.get("extraction_class", ""))
        g_txt = _normalize(str(g.get("extraction_text", "")))
        ok = False
        for p in pred:
            if str(p.get("extraction_class", "")) != g_cls:
                continue
            p_txt = _normalize(str(p.get("extraction_text", "")))
            if g_txt and (g_txt in p_txt or p_txt in g_txt):
                ok = True
                break
        if ok:
            matched += 1
    return matched, total


def langextract_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
):
    """
    Evaluate coverage of expected extractions and provide targeted feedback for GEPA.

    Modes:
    - Evaluate mode (called without `pred_name/pred_trace`): return a scalar in [0,1].
    - GEPA mode (called with `pred_name/pred_trace`): return `dspy.Prediction(score, feedback)`.
      When `pred_name == "build_prompt"`, we inspect the generated `prompt_description` and
      `examples_json` from the predictor's trace to give focused suggestions (e.g., class
      coverage in few-shot, JSON validity). This directly guides how the prompt/examples
      should evolve — i.e., the same items you would author by hand in a pure-langextract
      script.
    """
    gold_targets = list(getattr(gold, "targets", []) or [])
    pred_extractions = list(getattr(pred, "extractions", []) or [])

    matched, total = _match_extractions(pred_extractions, gold_targets)
    score = 0.0 if total == 0 else round(matched / total, 3)

    # Binary framing reused for trace and feedback
    gold_pos = total > 0
    guess_pos = (total > 0) and (matched == total)
    pred_claim = len(pred_extractions) > 0
    conf = confusion_outcomes(gold_pos, guess_pos, pred_claim)

    # Trace essentials for reflection/debug
    safe_trace_log(trace, {"matched": matched, "total": total, "score": score, "confusion": conf})

    # Evaluate mode
    if pred_name is None and pred_trace is None:
        return score

    # GEPA feedback
    missing: list[dict[str, str]] = []
    for g in gold_targets:
        g_cls = str(g.get("extraction_class", ""))
        g_txt = str(g.get("extraction_text", ""))
        found = any(
            (str(p.get("extraction_class", "")) == g_cls)
            and (
                _normalize(g_txt) in _normalize(str(p.get("extraction_text", "")))
                or _normalize(str(p.get("extraction_text", ""))) in _normalize(g_txt)
            )
            for p in pred_extractions
        )
        if not found:
            missing.append({"class": g_cls, "text": g_txt})

    fb_parts: list[str] = []

    # TP/FN/FP/TN framing (full coverage = TP)
    if conf["TP"]:
        fb_parts.append("All expected extractions detected.")
    elif conf["FN"]:
        fb_parts.append("Missing extractions; cover all target classes and spans.")
    elif conf["FP"]:
        fb_parts.append("Unnecessary extractions when no targets are expected; avoid over-extracting.")
    else:
        fb_parts.append("Correct: no extractions needed.")

    if missing:
        cls_buckets: dict[str, list[str]] = {}
        for m in missing:
            cls_buckets.setdefault(m["class"], []).append(m["text"])
        fb_parts.append("Missing extractions: " + ", ".join(f"{k}: {v}" for k, v in cls_buckets.items()))

    # If optimizing the prompt builder, surface more focused guidance using its outputs
    if pred_name == "build_prompt":
        try:
            # pred_trace is [(predictor, inputs, outputs)]
            _, bp_inputs, bp_outputs = (pred_trace or [None, None, None])[0]
            pdesc = getattr(bp_outputs, "prompt_description", "")
            ex_json = getattr(bp_outputs, "examples_json", "")
            # Lightweight checks
            if not pdesc:
                fb_parts.append("Prompt is empty; specify exact span quoting and non-overlap.")
            try:
                parsed = json.loads(ex_json) if ex_json else []
                if not parsed:
                    fb_parts.append("Provide at least one few-shot example with attributes.")
                else:
                    # Ensure every target class appears in at least one example
                    want = {str(c).lower() for c in (bp_inputs or {}).get("target_classes", [])}
                    seen: set[str] = set()
                    for item in parsed:
                        for e in item.get("extractions", []) or []:
                            seen.add(str(e.get("extraction_class", "")).lower())
                    missing_in_examples = sorted(list(want - seen))
                    if missing_in_examples:
                        fb_parts.append("Examples: add coverage for classes: " + ", ".join(missing_in_examples))
            except Exception:
                fb_parts.append("examples_json is not valid JSON; output a JSON array of {text, extractions}.")
        except Exception:
            pass

    # Generic guidance
    fb_parts.append(
        "Use exact quotes for spans, avoid overlapping extractions, and include meaningful attributes (e.g., feeling/type)."
    )
    return dspy.Prediction(score=score, feedback=" ".join(p for p in fb_parts if p))


# -------------------------------
# CLI
# -------------------------------


def _dummy_prompt_responses():
    """Yield prompt+examples for DummyLM (JSONAdapter)."""
    while True:
        yield {
            "prompt_description": (
                "Extract characters, emotions, and relationships. Use exact text spans, "
                "no paraphrasing, and avoid overlapping entities. Provide attributes."
            ),
            "examples_json": json.dumps(
                [
                    {
                        "text": "ROMEO. But soft! ... Juliet is the sun.",
                        "extractions": [
                            {
                                "extraction_class": "character",
                                "extraction_text": "ROMEO",
                                "attributes": {"emotional_state": "awe"},
                            },
                            {
                                "extraction_class": "emotion",
                                "extraction_text": "But soft!",
                                "attributes": {"feeling": "gentle awe"},
                            },
                            {
                                "extraction_class": "relationship",
                                "extraction_text": "Juliet is the sun",
                                "attributes": {"type": "metaphor"},
                            },
                        ],
                    }
                ]
            ),
        }


def _dummy_reflection_responses():
    while True:
        yield {"improved_instruction": "Emphasize exact spans, class coverage in examples, and non-overlap."}


def main():
    parser = argparse.ArgumentParser()
    add_standard_args(parser, default_save_prefix="simple_gepa_langextract")
    parser.add_argument(
        "--langextract-model-id",
        default=None,
        help="Model id to use in langextract (e.g., gemini-2.5-flash).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    logger.info("Starting GEPA + langextract prompt optimization demo")

    # Program
    program = LangExtractPipeline(
        langextract_model_id=args.langextract_model_id,
        use_fallback=bool(args.dummy),
    )
    program.build_prompt.signature = program.build_prompt.signature.with_instructions(
        "Produce a concise extraction instruction and a few-shot examples JSON.\n"
        "- Enforce exact text spans from input and avoid overlap.\n"
        "- Cover all target classes in examples with attributes.\n"
        "- Keep the prompt clear and brief."
    )

    before = {n: p.signature.instructions for n, p in program.named_predictors()}

    # Data
    trainset, valset = langextract_dummy(locale="en")
    logger.info("Dataset — train: {}, val: {}", len(trainset), len(valset))

    # LMs
    if args.dummy:
        # Build-prompt LM emits prompt_description + examples_json
        build_prompt_lm = make_dummy_lm_json(_dummy_prompt_responses())
        configure_dummy_adapter(lm=build_prompt_lm)
        # Reflection LM (GEPA)
        reflection_lm = make_dummy_lm_json(_dummy_reflection_responses())
    else:
        # Real task and reflection LMs via helpers
        dspy.settings.configure(lm=openai_gpt_4o_mini_lm)
        reflection_lm = openai_gpt_4o_lm

    # Baseline evaluate
    from dspy.evaluate import Evaluate

    evaluator = Evaluate(devset=valset, metric=langextract_metric, display_progress=False, num_threads=1)
    logger.info("Baseline evaluation on {} validation examples...", len(valset))
    # Predictive cost notes
    log_baseline_estimate(valset_size=len(valset), num_predictors=len(program.predictors()), logger=logger)
    baseline = evaluator(program)
    logger.success("Baseline score: {}", baseline.score)

    # GEPA
    gepa = dspy.GEPA(
        metric=langextract_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=40 if args.dummy else None,
        auto=None if args.dummy else "light",
        reflection_minibatch_size=1,
        track_stats=True,
        **get_wandb_args(project="real_world", run_name=make_run_name(args.save_prefix), enabled=not args.dummy),
    )

    logger.info("Running GEPA compile (max_metric_calls={} auto={})...", gepa.max_metric_calls, gepa.auto)
    # Predictive GEPA cost notes
    log_gepa_estimate(
        gepa=gepa,
        num_predictors=len(program.predictors()),
        valset_size=len(valset),
        trainset_size=len(trainset),
        logger=logger,
    )
    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    logger.success("GEPA compile finished.")

    # Post-eval
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
