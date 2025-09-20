"""
Simple save helpers for demo scripts.

save_artifacts(program, optimized, save_dir, prefix, logger=None, save_details=True)
  - Saves baseline and optimized programs (DSPy standard .json state)
  - Optionally saves GEPA detailed results as JSON (with fallback)
  - Returns dict with file paths
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


def _log(logger, level: str, msg: str, *args):
    if logger is None:
        print(msg.format(*args))
        return
    logfn = getattr(logger, level, logger.info)
    logfn(msg, *args)


def _dr_to_dict(dr: Any) -> dict[str, Any] | None:
    try:
        return dr.to_dict()
    except Exception:
        try:
            cand_texts = []
            for cand in getattr(dr, "candidates", []) or []:
                if hasattr(cand, "named_predictors"):
                    cand_texts.append({name: p.signature.instructions for name, p in cand.named_predictors()})
                elif isinstance(cand, dict):
                    cand_texts.append(cand)
                else:
                    cand_texts.append(str(cand))
            return {
                "candidates": cand_texts,
                "val_aggregate_scores": getattr(dr, "val_aggregate_scores", []),
                "per_val_instance_best_candidates": [list(s) for s in getattr(dr, "per_val_instance_best_candidates", []) or []],
                "discovery_eval_counts": getattr(dr, "discovery_eval_counts", []),
                "seed": getattr(dr, "seed", None),
            }
        except Exception:
            return None


def save_artifacts(
    program,
    optimized,
    *,
    save_dir: str,
    prefix: str,
    logger=None,
    save_details: bool = True,
) -> dict[str, str | None]:
    os.makedirs(save_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    baseline_path = os.path.join(save_dir, f"{prefix}-baseline-{stamp}.json")
    optimized_path = os.path.join(save_dir, f"{prefix}-optimized-{stamp}.json")

    try:
        program.save(baseline_path)
        optimized.save(optimized_path)
        _log(logger, "success", "Saved baseline program: {}", baseline_path)
        _log(logger, "success", "Saved optimized program: {}", optimized_path)
    except Exception as e:
        _log(logger, "warning", "Failed to save DSPy programs: {}", e)

    details_path = None
    if save_details and hasattr(optimized, "detailed_results") and optimized.detailed_results is not None:
        details_path = os.path.join(save_dir, f"{prefix}-gepa-details-{stamp}.json")
        dr_dict = _dr_to_dict(optimized.detailed_results)
        if dr_dict is not None:
            try:
                with open(details_path, "w", encoding="utf-8") as f:
                    json.dump(dr_dict, f, ensure_ascii=False, indent=2)
                _log(logger, "success", "Saved GEPA detailed results: {}", details_path)
            except Exception as e:
                _log(logger, "warning", "Failed to save GEPA details: {}", e)
                details_path = None

    return dict(baseline=baseline_path, optimized=optimized_path, details=details_path)


__all__ = ["save_artifacts"]

