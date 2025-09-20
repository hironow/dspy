"""
Lightweight cost/latency estimation helpers for the demo scripts.

These functions only log simple estimates — they do not mutate objects.
Pass a logger compatible with .info(...). If omitted, results are printed.
"""

from __future__ import annotations

import math
from typing import Any


def _log(logger, msg: str, *args):
    if logger is not None:
        logger.info(msg, *args)
    else:
        try:
            print(msg.format(*args))
        except Exception:
            print(msg, *args)


def log_baseline_estimate(*, valset_size: int, num_predictors: int, logger=None) -> int:
    """Estimate baseline task-LM calls: roughly valset_size × num_predictors.

    Returns the estimated number of calls.
    """
    calls = valset_size * num_predictors
    _log(
        logger,
        "PREDICTIVE-NOTE [CALLSITE]: Evaluate(program) が各例を評価し、その過程で program(**inputs) により各 Predictor ぶんタスクLM への推論が発生します。",
    )
    _log(
        logger,
        "PREDICTIVE-NOTE: 推定タスクLM呼び出し回数 (baseline) ≈ バリデーション件数 × Predictor数 = {} × {} = {}",
        valset_size,
        num_predictors,
        calls,
    )
    return calls


def log_gepa_estimate(
    *,
    gepa: Any,
    num_predictors: int,
    valset_size: int,
    trainset_size: int,
    logger=None,
) -> dict[str, int | None]:
    """Log a rough estimate of GEPA metric calls and task-LM calls.

    Returns a dict with keys: metric_calls_mid/min/max, task_calls_mid/min/max.
    Some values may be None if not computable.
    """
    # Map for auto candidates (mirrors GEPA defaults)
    auto_map = {"light": 6, "medium": 12, "heavy": 18}

    metric_mid: int | None = None
    metric_min: int | None = None
    metric_max: int | None = None

    auto_mode = getattr(gepa, "auto", None)
    n_candidates = auto_map.get(auto_mode, None)

    if auto_mode is not None and n_candidates is not None:
        try:
            metric_mid = gepa.auto_budget(
                num_preds=num_predictors,
                num_candidates=n_candidates,
                valset_size=valset_size,
            )
            # Build a plausible range by varying minibatch size and full eval steps
            min_M = max(1, int(getattr(gepa, "reflection_minibatch_size", 3) or 3))
            m_hi = 10
            max_M = max(35, valset_size)
            m_lo = 1

            b_min_1 = gepa.auto_budget(
                num_preds=num_predictors,
                num_candidates=n_candidates,
                valset_size=valset_size,
                minibatch_size=min_M,
            )
            b_min_2 = gepa.auto_budget(
                num_preds=num_predictors,
                num_candidates=n_candidates,
                valset_size=valset_size,
                minibatch_size=min_M,
                full_eval_steps=m_hi,
            )
            b_max_1 = gepa.auto_budget(
                num_preds=num_predictors,
                num_candidates=n_candidates,
                valset_size=valset_size,
                minibatch_size=max_M,
            )
            b_max_2 = gepa.auto_budget(
                num_preds=num_predictors,
                num_candidates=n_candidates,
                valset_size=valset_size,
                minibatch_size=max_M,
                full_eval_steps=m_lo,
            )
            metric_min = min(b_min_1, b_min_2)
            metric_max = max(b_max_1, b_max_2)
        except Exception:
            metric_mid = None
    elif getattr(gepa, "max_full_evals", None) is not None:
        metric_mid = gepa.max_full_evals * (trainset_size + valset_size)
        metric_min = metric_max = metric_mid
    elif getattr(gepa, "max_metric_calls", None) is not None:
        metric_mid = gepa.max_metric_calls
        metric_min = metric_max = metric_mid

    # Trials (rough)
    N = None
    if n_candidates is not None:
        try:
            N = int(max(2 * (num_predictors * 2) * math.log2(n_candidates), 1.5 * n_candidates))
        except Exception:
            N = None

    # Logging
    if metric_mid is not None:
        if metric_min is not None and metric_max is not None and metric_min != metric_max:
            _log(
                logger,
                "PREDICTIVE-NOTE RANGE: metric_calls ≈ {} .. {}（mid ≈ {}、auto='{}'）",
                metric_min,
                metric_max,
                metric_mid,
                auto_mode if auto_mode is not None else "manual",
            )
        else:
            _log(
                logger,
                "PREDICTIVE-NOTE: 実APIではGEPA最適化中に評価関数 (metric) の呼び出しが概ね ~{} 回（auto='{}'想定）発生します。",
                metric_mid,
                auto_mode if auto_mode is not None else "manual",
            )

        # Task-LM call estimates
        task_mid = metric_mid * num_predictors
        if metric_min is not None and metric_max is not None and metric_min != metric_max:
            _log(
                logger,
                "PREDICTIVE-NOTE RANGE: taskLM_calls ≈ ({}..{}) × {} ≈ {}..{}",
                metric_min,
                metric_max,
                num_predictors,
                metric_min * num_predictors,
                metric_max * num_predictors,
            )
        _log(
            logger,
            "PREDICTIVE-NOTE: タスクLMの推定呼び出し回数（mid） ≈ metric_calls × Predictor数 = {} × {} ≈ {}",
            metric_mid,
            num_predictors,
            task_mid,
        )

    _log(
        logger,
        "PREDICTIVE-NOTE [CALLSITE]: GEPA内部で adapter.evaluate(...) がタスクLMを、propose_new_texts(...) が反射LM(または提案器)を呼び出します。",
    )
    if N is not None:
        _log(
            logger,
            "PREDICTIVE-NOTE: 反射LM呼び出しは試行回数Nに概ね比例（N≈{}）。",
            N,
        )

    return dict(
        metric_calls_mid=metric_mid,
        metric_calls_min=metric_min,
        metric_calls_max=metric_max,
        task_calls_mid=(metric_mid * num_predictors) if metric_mid is not None else None,
        task_calls_min=(metric_min * num_predictors) if metric_min is not None else None,
        task_calls_max=(metric_max * num_predictors) if metric_max is not None else None,
    )


def log_recorded_gepa_cost(dr: Any, *, num_predictors: int, logger=None) -> None:
    """Log recorded costs from detailed_results, if available."""
    if not dr:
        return
    total = getattr(dr, "total_metric_calls", None)
    if total is not None:
        _log(logger, "PREDICTIVE-NOTE: metric_calls（評価関数の呼び出し回数, 記録値） ≈ {}", total)
        try:
            _log(
                logger,
                "PREDICTIVE-NOTE: 推定タスクLM呼び出し回数（記録値に基づく） ≈ metric_calls × Predictor数 = {} × {} ≈ {}",
                total,
                num_predictors,
                total * num_predictors,
            )
        except Exception:
            pass
    nfull = getattr(dr, "num_full_val_evals", None)
    if nfull is not None:
        _log(logger, "PREDICTIVE-NOTE: フル評価（全Valでの集計）回数はおよそ {} 回（記録ベース）。", nfull)
    run_dir = getattr(dr, "log_dir", None)
    if run_dir:
        _log(logger, "PREDICTIVE-NOTE: 実API運用ではログ/成果物は '{}' に蓄積。", run_dir)


__all__ = [
    "log_baseline_estimate",
    "log_gepa_estimate",
    "log_recorded_gepa_cost",
]

