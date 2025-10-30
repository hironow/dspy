"""
GEPA-friendly metrics for classification tasks.

This module provides a recall-biased privacy risk metric suitable for
GEPA optimization. It follows DSPy tutorial patterns:

- Always return dspy.Prediction(score, feedback) when used by GEPA.
- Feedback echoes the correct label and gives actionable guidance.
- Strict output validation for label space.
- Optional trace logging for lightweight introspection.

Usage:
    from real_world.metrics_privacy_risk import risk_metric_gepa, gepa_metric

    metric = gepa_metric  # or customize via risk_metric_gepa(...)

Expected example/pred fields:
- example.ideal: gold label (e.g., "High Risk" or "Low Risk")
- example.answer_pos_label / example.answer_neg_label (optional overrides)
- example.explanation / example.comment (optional context for feedback)
- pred.risk: predicted label string
"""

from __future__ import annotations

import dspy


def _canon(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s.lower()


def risk_metric_gepa(
    tp_reward: float = 1.0,
    fn_penalty: float = 1.0,
    fp_penalty: float = 0.25,
    tn_reward: float = 0.10,
):
    """
    GEPA-friendly metric for privacy risk classification.

    Design:
    - Strict label space: {"High Risk", "Low Risk"} (case-insensitive),
      with optional per-example overrides via `answer_pos_label` / `answer_neg_label`.
    - Recall-biased shaping: FN receives the largest penalty; FP a smaller one;
      TN small reward; TP full reward (customizable).
    - Always returns dspy.Prediction(score, feedback). Evaluate can still accept
      the `score` field from this object.
    - Appends optional teaching snippet if `example.comment` is present.
    - Logs a compact confusion outcome into trace when available.
    """

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        # Bind labels (allow overrides in example)
        POS = _canon(getattr(example, "answer_pos_label", "High Risk"))
        NEG = _canon(getattr(example, "answer_neg_label", "Low Risk"))

        gold_raw = getattr(example, "ideal", "")
        gold = _canon(gold_raw)

        # Parse prediction strictly (validate and teach)
        guess_raw = getattr(pred, "risk", "")
        guess = _canon(guess_raw)
        if guess not in {POS, NEG}:
            feedback = (
                "Output format error. The 'risk' field must be exactly 'High Risk' or 'Low Risk'—"
                f"you responded with '{guess_raw}'.\n"
                f"Correct label for this example is '{gold_raw}'.\n"
                "Instruction: output ONLY one of the two exact strings with no extra text."
            )
            if hasattr(example, "explanation") and example.explanation:
                feedback += f"\nContext: {example.explanation}"
            return dspy.Prediction(score=0.0, feedback=feedback)

        # Confusion outcomes
        TP = int(gold == POS and guess == POS)
        FN = int(gold == POS and guess == NEG)
        FP = int(gold == NEG and guess == POS)
        TN = int(gold == NEG and guess == NEG)

        # Trace: essentials for reflection
        if trace is not None and hasattr(trace, "log"):
            try:
                trace.log(
                    {
                        "gold": gold,
                        "guess": guess,
                        "pos_label": POS,
                        "neg_label": NEG,
                        "confusion": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
                    }
                )
            except Exception:
                pass

        # Score shaping (recall-biased)
        if TP:
            score = tp_reward
            feedback = (
                f"Correct: '{gold_raw}'. Keep recall strong when you see any of these cues: "
                "use of user content for training, improvement, or research; "
                "perpetual, irrevocable, sublicensable, or 'for any purpose' licenses; "
                "sale or broad sharing of personal data; storing entire conversations or logs."
            )
        elif FN:
            score = max(0.0, 1.0 - fn_penalty)
            feedback = (
                f"Incorrect (missed high risk). Correct label is '{gold_raw}'. "
                "When any high-risk cue appears, even softly worded, prefer 'High Risk'. "
                "Common cues include training on user content, broad or perpetual licenses, sale or broad sharing, and "
                "retaining full conversations or logs."
            )
        elif FP:
            score = max(0.0, 1.0 - fp_penalty)
            feedback = (
                f"Incorrect (false high risk). Correct label is '{gold_raw}'. "
                "Avoid flagging routine or narrowly scoped uses without broad grants: basic service delivery, "
                "standard security statements, or limited or opt-in analytics without training or broad license language."
            )
        else:  # TN
            score = tn_reward
            feedback = (
                f"Correct: '{gold_raw}'. "
                "Maintain specificity and require explicit high-risk cues before choosing 'High Risk'."
            )

        # Optional teaching snippet
        extra = getattr(example, "comment", None)
        if extra:
            feedback += (
                "\nHere's the reason for the ground truth label. Think about what you can learn from this to improve your future answers: "
                f"{extra}"
            )

        score = float(max(0.0, min(1.0, score)))
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


# Preconfigured GEPA-style metric instance (slightly higher fp penalty to not over-bias recall).
gepa_metric = risk_metric_gepa(
    tp_reward=1.0,
    fn_penalty=1.0,
    fp_penalty=0.33,
    tn_reward=0.10,
)
