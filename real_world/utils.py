"""
Minimal utilities to display GEPA results and BEFORE/AFTER instruction diffs
in a consistent, lightweight way across demos.
"""

from __future__ import annotations

from typing import Any

import dspy


def _format_gepa_results_table(dr: Any, top_k: int = 10) -> str:
    try:
        n = len(dr.val_aggregate_scores)
        if n == 0:
            return ""

        # Coverage per candidate (how many val tasks picked it on the Pareto front)
        coverage = [0] * n
        for s in getattr(dr, "per_val_instance_best_candidates", []) or []:
            for i in s:
                if 0 <= i < n:
                    coverage[i] += 1

        rows = []
        for i, score in enumerate(dr.val_aggregate_scores):
            cov = coverage[i] if i < len(coverage) else 0
            disc = getattr(dr, "discovery_eval_counts", [None] * n)
            disc_i = disc[i] if i < len(disc) and disc[i] is not None else "-"
            is_best = "*" if i == dr.best_idx else ""
            rows.append((i, score, cov, disc_i, is_best))

        rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
        rows = rows[: min(top_k, len(rows))]

        headers = ("Idx", "Score", "Best@Val", "DiscoveryCalls", "Best?")
        cols = list(
            zip(*([headers] + [(str(i), f"{s:.3f}", str(c), str(d), b) for i, s, c, d, b in rows]), strict=False)
        )
        widths = [max(len(x) for x in col) for col in cols]

        def fmt_row(cells):
            return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=False))

        line = "-+-".join("-" * w for w in widths)
        out = [fmt_row(headers), line]
        for i, s, c, d, b in rows:
            out.append(fmt_row((str(i), f"{s:.3f}", str(c), str(d), b)))
        return "\n".join(out)
    except Exception:
        return ""


def _format_before_after_instructions_table(
    before: dict[str, str], after: dict[str, str], max_col_width: int = 90
) -> str:
    try:

        def compact(s: str) -> str:
            s = " ".join(str(s or "").split())
            return s if len(s) <= max_col_width else s[: max_col_width - 3] + "..."

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
        cols = list(zip(*([headers] + rows), strict=False))
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


def summarize_gepa_results(optimized: dspy.Module, logger=None, *, top_k: int = 10) -> None:
    dr = getattr(optimized, "detailed_results", None)
    if not dr:
        return
    if logger:
        logger.info("GEPA proposed {} candidates.", len(dr.candidates))
        if dr.val_aggregate_scores:
            logger.info("Best validation score (Pareto selection): {}", dr.val_aggregate_scores[dr.best_idx])
        table = _format_gepa_results_table(dr, top_k=top_k)
        if table:
            logger.info("\n{}", table)
    else:
        print(f"GEPA proposed {len(dr.candidates)} candidates.")
        if dr.val_aggregate_scores:
            print(f"Best validation score (Pareto selection): {dr.val_aggregate_scores[dr.best_idx]}")
        table = _format_gepa_results_table(dr, top_k=top_k)
        if table:
            print("\n" + table)


def summarize_before_after(
    before_instructions: dict[str, str], optimized: dspy.Module, logger=None, *, max_col_width: int = 90
) -> int:
    after_instructions = {name: pred.signature.instructions for name, pred in optimized.named_predictors()}
    table = _format_before_after_instructions_table(
        before_instructions, after_instructions, max_col_width=max_col_width
    )
    changed = sum(
        1
        for k in set(before_instructions) | set(after_instructions)
        if before_instructions.get(k) != after_instructions.get(k)
    )
    if table:
        if logger:
            logger.info("\n{}", table)
        else:
            print(table)
    if logger:
        if changed == 0:
            logger.info("Instructions unchanged.")
        else:
            logger.info("Instructions updated ({} changed).", changed)
    else:
        print(f"Instructions updated ({changed} changed)." if changed else "Instructions unchanged.")
    return changed


__all__ = [
    "summarize_gepa_results",
    "summarize_before_after",
]
