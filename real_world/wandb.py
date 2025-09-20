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
