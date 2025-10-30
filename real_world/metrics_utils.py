from __future__ import annotations

from typing import Any, Dict


def confusion_outcomes(gold_pos: bool, guess_pos: bool, pred_claim: bool) -> Dict[str, int]:
    """Return a dict with TP, FP, TN, FN given binary events.

    - gold_pos: whether the gold label/target is present/positive
    - guess_pos: whether the prediction correctly affirms the positive case
    - pred_claim: whether the prediction claims a non-empty/affirmative output
    """
    TP = int(gold_pos and guess_pos)
    FN = int(gold_pos and not guess_pos)
    FP = int((not gold_pos) and pred_claim)
    TN = int((not gold_pos) and (not pred_claim))
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}


def safe_trace_log(trace: Any, data: Dict[str, Any]) -> None:
    """Safely log a small payload to trace if available."""
    if trace is None or not hasattr(trace, "log"):
        return
    try:
        trace.log(data)
    except Exception:
        # Swallow logging errors to avoid interfering with metric evaluation
        pass
