"""
Minimal helpers to pass Weights & Biases config to dspy.GEPA.

Usage:
  from real_world.wandb import get_wandb_args
  gepa = dspy.GEPA(..., **get_wandb_args(project="real_world", run_name="my-run"))

Behavior:
  - If enabled=False or WANDB_API_KEY is not set, returns an empty dict.
  - Otherwise returns {use_wandb, wandb_api_key, wandb_init_kwargs} suitable for dspy.GEPA.
"""

from __future__ import annotations

import os
from typing import Any
import time


def get_wandb_args(
    *,
    project: str = "dspy-gepa",
    run_name: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Return kwargs to enable W&B in dspy.GEPA, or {} when disabled/invalid.

    - enabled=False forces no-op.
    - Requires WANDB_API_KEY in environment.
    """
    if not enabled:
        return {}
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        return {}
    init: dict[str, Any] = {"project": project}
    if run_name:
        init["name"] = run_name
    if tags:
        init["tags"] = tags
    if config:
        init["config"] = config
    return dict(use_wandb=True, wandb_api_key=api_key, wandb_init_kwargs=init)


__all__ = ["get_wandb_args"]


def make_run_name(prefix: str, *, ts_format: str = "%Y%m%d-%H%M%S", suffix: str | None = None) -> str:
    """Construct a standardized W&B run name.

    Example: make_run_name("simple_gepa_basic") -> "simple_gepa_basic-20250101-123456"

    Args:
        prefix: A short identifier for the script or experiment.
        ts_format: time.strftime format for timestamp portion.
        suffix: Optional extra text appended after the timestamp.

    Returns:
        A string suitable for wandb.init(name=...).
    """
    stamp = time.strftime(ts_format)
    base = f"{prefix}-{stamp}"
    return f"{base}-{suffix}" if suffix else base

__all__.append("make_run_name")
