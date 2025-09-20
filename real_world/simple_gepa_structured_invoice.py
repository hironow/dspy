"""
Structured invoice IE with task-specific metric using trace/pred_name/pred_trace (v3+).

- Program: extract -> normalize over invoice text
- Metric (invoice_metric_with_feedback):
  - Uses trace/pred_name/pred_trace to tailor feedback per predictor
  - Validates schema (vendor/date/amount/currency), types, and formats
  - Encourages ISO date (YYYY-MM-DD) and ISO currency code (e.g., JPY)

Run (no external calls):
  uv run python real_world/simple_gepa_structured_invoice.py --dummy
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Any

from loguru import logger

import dspy
from real_world.cost import log_baseline_estimate, log_gepa_estimate, log_recorded_gepa_cost
from real_world.dummy_lm import configure_dummy_adapter, make_dummy_lm_json
from real_world.factory import invoice_dummy
from real_world.helper import openai_gpt_4o_lm, openai_gpt_4o_mini_lm
from real_world.utils import summarize_before_after, summarize_gepa_results
from real_world.wandb import get_wandb_args


class InvoiceIE(dspy.Module):
    def __init__(self):
        super().__init__()
        # Structured extraction from free text
        self.extract = dspy.Predict("text -> vendor, date, amount, currency")
        # Normalization to canonical forms
        self.normalize = dspy.Predict("vendor, date, amount, currency -> vendor, date, amount, currency")

    def forward(self, text: str):
        # 1) Extract
        ex_lm = getattr(self, "_extract_lm", None)
        if ex_lm is not None:
            with dspy.context(lm=ex_lm):
                s1 = self.extract(text=text)
        else:
            s1 = self.extract(text=text)

        # 2) Normalize
        nm_lm = getattr(self, "_normalize_lm", None)
        kwargs = dict(vendor=s1.vendor, date=s1.date, amount=s1.amount, currency=s1.currency)
        if nm_lm is not None:
            with dspy.context(lm=nm_lm):
                s2 = self.normalize(**kwargs)
        else:
            s2 = self.normalize(**kwargs)

        return s2


ISO_CURRENCIES = {"USD", "EUR", "JPY", "GBP", "CNY"}
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_iso_date(s: str) -> bool:
    return bool(ISO_DATE_PATTERN.match(s or ""))


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def invoice_metric_with_feedback(
    gold: Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
):
    """
    Task-specific metric for structured invoice extraction/normalization.

    - Validates vendor (non-empty), date (ISO YYYY-MM-DD), amount (float), currency (ISO code)
    - Uses pred_name/pred_trace to tailor feedback:
      * extract: missing/incorrect fields, parsing guidance
      * normalize: canonicalization advice (date/currency/amount formats)
    - Uses trace to mention upstream context when helpful

    Returns:
      - float when pred_name/pred_trace are None (Evaluate)
      - dspy.Prediction(score, feedback) for GEPA (per-predictor feedback)
    """
    # Gold reference
    gold_vendor = _safe_str(getattr(gold, "vendor", "")).strip()
    gold_date = _safe_str(getattr(gold, "date", "")).strip()
    gold_currency = _safe_str(getattr(gold, "currency", "")).strip().upper()
    gold_amount = _to_float(getattr(gold, "amount", None))
    gold_text = _safe_str(getattr(gold, "text", ""))

    # Predicted
    pred_vendor = _safe_str(getattr(pred, "vendor", "")).strip()
    pred_date = _safe_str(getattr(pred, "date", "")).strip()
    pred_currency = _safe_str(getattr(pred, "currency", "")).strip().upper()
    pred_amount = _to_float(getattr(pred, "amount", None))

    errors: list[str] = []
    score = 1.0

    # Vendor
    if not pred_vendor:
        errors.append("Missing vendor")
        score -= 0.25
    elif gold_vendor and pred_vendor.lower() != gold_vendor.lower():
        # simple strict comparison in this toy example
        errors.append(f"Vendor mismatch (got '{pred_vendor}', expected '{gold_vendor}')")
        score -= 0.15

    # Date
    if not pred_date:
        errors.append("Missing date")
        score -= 0.25
    elif not _is_iso_date(pred_date):
        errors.append(f"Date not ISO (YYYY-MM-DD): '{pred_date}'")
        score -= 0.15
    elif gold_date and pred_date != gold_date:
        errors.append(f"Date mismatch (got '{pred_date}', expected '{gold_date}')")
        score -= 0.10

    # Amount
    if pred_amount is None:
        errors.append(f"Amount not numeric: '{getattr(pred, 'amount', None)}'")
        score -= 0.20
    elif gold_amount is not None:
        if abs(pred_amount - gold_amount) > 0.01:
            errors.append(f"Amount mismatch (got {pred_amount}, expected {gold_amount})")
            score -= 0.10

    # Currency
    if not pred_currency:
        errors.append("Missing currency")
        score -= 0.25
    elif pred_currency not in ISO_CURRENCIES:
        errors.append(f"Currency not ISO code: '{pred_currency}'")
        score -= 0.15
    elif gold_currency and pred_currency != gold_currency:
        errors.append(f"Currency mismatch (got '{pred_currency}', expected '{gold_currency}')")
        score -= 0.10

    score = max(0.0, min(1.0, round(score, 3)))

    # Evaluate mode: return scalar only
    if pred_name is None and pred_trace is None:
        return score

    # GEPA mode: build tailored feedback using pred_name/pred_trace
    fb: list[str] = []

    # Summarize failures
    if errors:
        fb.append("; ".join(errors))
    else:
        fb.append("All fields valid against schema.")

    # Add predictor-specific suggestions
    try:
        if isinstance(pred_trace, list) and len(pred_trace) > 0:
            _pred, p_inputs, p_outputs = pred_trace[0]
        else:
            p_inputs, p_outputs = {}, pred
    except Exception:
        p_inputs, p_outputs = {}, pred

    if pred_name == "extract":
        fb.append(
            "Extractor: Ensure all fields are captured with correct types. "
            "Use patterns like 'YYYY-MM-DD' for dates and parse numeric amounts."
        )
        # Point to any missing keys from predictor output
        out_keys = (
            set(getattr(p_outputs, "keys", lambda: p_outputs.keys())())
            if hasattr(p_outputs, "keys")
            else set(p_outputs.__dict__.keys() if hasattr(p_outputs, "__dict__") else [])
        )
        needed = {"vendor", "date", "amount", "currency"}
        missing = sorted(needed - out_keys)
        if missing:
            fb.append(f"Missing keys in extract output: {missing}")

        # Source hint using module input text
        if gold_text:
            fb.append("When parsing, look for 'Vendor:', 'Date:', 'Amount:', 'Currency:' labels in the text.")

    elif pred_name == "normalize":
        fb.append(
            "Normalizer: Convert date to ISO 'YYYY-MM-DD', currency symbol to ISO code (e.g., '¥' -> 'JPY'), "
            "and ensure amount is numeric."
        )
        # Highlight specific normalizations with from -> to suggestions (bounded, concrete)
        src_date = None
        src_curr = None
        src_amt = None
        if isinstance(p_inputs, dict):
            src_date = _safe_str(p_inputs.get("date", ""))
            src_curr = _safe_str(p_inputs.get("currency", ""))
            src_amt = p_inputs.get("amount", None)

        # Date normalization
        if not _is_iso_date(pred_date):
            if gold_date:
                fb.append(f"Date normalization: '{src_date or pred_date}' -> '{gold_date}' (ISO).")
            elif src_date or pred_date:
                fb.append(f"Date normalization: '{src_date or pred_date}' -> 'YYYY-MM-DD' (ISO).")

        # Currency normalization
        if pred_currency not in ISO_CURRENCIES:
            target_cur = gold_currency or "ISO code like 'JPY'"
            fb.append(f"Currency normalization: '{(src_curr or pred_currency) or ''}' -> '{target_cur}'.")

        # Amount normalization
        if pred_amount is None:
            normalized_hint = None
            if isinstance(src_amt, str) and src_amt:
                # Strip non-numeric hints (keep digits, dot, minus)
                stripped = re.sub(r"[^0-9.\-]", "", src_amt)
                parsed = _to_float(stripped)
                if parsed is not None:
                    normalized_hint = parsed
            if normalized_hint is None and gold_amount is not None:
                normalized_hint = gold_amount
            if normalized_hint is not None:
                fb.append(f"Amount normalization: '{_safe_str(src_amt)}' -> {normalized_hint} (numeric).")
            else:
                fb.append("Amount normalization: ensure a numeric float (e.g., '7,890' -> 7890.0).")

    else:
        fb.append("Program: Maintain schema validity across steps.")

    feedback = " ".join(s for s in fb if s).strip()
    return dspy.Prediction(score=score, feedback=feedback)


## dataset is provided by real_world.factory for consistency across demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Use DummyLM for a local dry run")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--save-dir", default="real_world/exports")
    parser.add_argument("--save-prefix", default="simple_gepa_invoice")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    logger.info("Starting structured invoice GEPA example")

    program = InvoiceIE()
    program.extract.signature = program.extract.signature.with_instructions(
        "与えられた請求テキストから vendor（会社名）, date（YYYY-MM-DD）, amount（数値）, currency（ISOコード）を抽出してください。"
    )
    program.normalize.signature = program.normalize.signature.with_instructions(
        "抽出結果を正規化してください。dateはYYYY-MM-DD、currencyはISOコード（例: JPY）に統一し、amountは数値で返してください。"
    )

    before = {n: p.signature.instructions for n, p in program.named_predictors()}

    trainset, valset = invoice_dummy(locale="ja")
    logger.info("Dataset — train: {}, val: {}", len(trainset), len(valset))

    if args.dummy:
        logger.info("Configuring DummyLMs (JSONAdapter) for extract/normalize/reflection")
        import itertools

        # Extract LM: produces structured outputs; second example intentionally non-ISO date and symbol currency
        def extract_responses():
            while True:
                yield {"vendor": "Acme Corp", "date": "2024-12-31", "amount": 1234.56, "currency": "USD"}
                yield {"vendor": "Tokyo Shop", "date": "31-12-2024", "amount": 7890, "currency": "¥"}

        # Normalize LM: passthrough to trigger normalization feedback (kept simple)
        def normalize_responses():
            while True:
                yield {"vendor": "Acme Corp", "date": "2024-12-31", "amount": 1234.56, "currency": "USD"}
                yield {"vendor": "Tokyo Shop", "date": "31-12-2024", "amount": 7890, "currency": "¥"}

        def reflection_responses():
            phrases = [
                "日付はISO形式(YYYY-MM-DD)、通貨はISOコード（例: JPY）に統一してください。",
                "欠損フィールドがないようにし、スキーマのキー名を厳密に守ってください。",
                "通貨記号はISOコードへ、ロケール依存の日付はISO日付へ正規化してください。",
            ]
            for p in itertools.cycle(phrases):
                yield {"improved_instruction": p}

        extract_lm = make_dummy_lm_json(extract_responses())
        normalize_lm = make_dummy_lm_json(normalize_responses())
        reflection_lm = make_dummy_lm_json(reflection_responses())

        program._extract_lm = extract_lm
        program._normalize_lm = normalize_lm
        configure_dummy_adapter(lm=normalize_lm)
    else:
        logger.info("Configuring real LMs via helper (OpenAI).")
        task_lm = openai_gpt_4o_mini_lm
        dspy.settings.configure(lm=task_lm)
        reflection_lm = openai_gpt_4o_lm

    from dspy.evaluate import Evaluate

    evaluator = Evaluate(
        devset=valset,
        metric=invoice_metric_with_feedback,
        display_progress=False,
        num_threads=1,
    )
    logger.info("Baseline evaluation on {} validation examples...", len(valset))
    log_baseline_estimate(valset_size=len(valset), num_predictors=len(program.predictors()), logger=logger)
    baseline = evaluator(program)
    logger.success("Baseline score: {}", baseline.score)

    gepa = dspy.GEPA(
        metric=invoice_metric_with_feedback,
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

    logger.info("Evaluating optimized program on validation set...")
    improved = evaluator(optimized)
    logger.success("Post-GEPA score: {}", improved.score)

    summarize_gepa_results(optimized, logger, top_k=10)
    before_instructions = before
    summarize_before_after(before_instructions, optimized, logger)
    if hasattr(optimized, "detailed_results") and optimized.detailed_results is not None:
        log_recorded_gepa_cost(optimized.detailed_results, num_predictors=len(program.predictors()), logger=logger)

    after = {n: p.signature.instructions for n, p in optimized.named_predictors()}
    changed = sum(1 for k in set(before) | set(after) if before.get(k) != after.get(k))
    logger.info("Instructions changed: {}", changed)

    from real_world.save import save_artifacts

    save_artifacts(
        program, optimized, save_dir=args.save_dir, prefix=args.save_prefix, logger=logger, save_details=True
    )


if __name__ == "__main__":
    main()
