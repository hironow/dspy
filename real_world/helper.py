"""
Helper utilities to construct real LMs for DSPy demos.

Usage patterns
--------------

1) 直接エイリアスを使う（推奨の簡便法）

   from real_world.helper import openai_gpt_4o_mini_lm
   import dspy
   dspy.settings.configure(lm=openai_gpt_4o_mini_lm)

   # 反射用などに別モデル
   from real_world.helper import openai_gpt_4o_lm
   reflection_lm = openai_gpt_4o_lm

   # Anthropic の例
   from real_world.helper import anthropic_claude_3_opus_20240229_lm

2) 明示ファクトリを使う（任意のモデル名を指定）

   from real_world.helper import openai_lm, anthropic_lm
   task_lm = openai_lm("gpt-4o-mini", temperature=0.0)
   reflection_lm = anthropic_lm("claude-3-5-sonnet-20240620", temperature=0.7)

3) 既定を一括設定

   from real_world.helper import configure_openai
   configure_openai("gpt-4o-mini", temperature=0.0)

備考
----
環境変数 OPENAI_API_KEY / ANTHROPIC_API_KEY を利用します。
エイリアス名は `openai_<model>_lm` / `anthropic_<model>_lm` 形式で、
model 部分はハイフンをアンダースコアで置き換えた表記です（例: gpt-4o-mini → gpt_4o_mini）。
"""

from __future__ import annotations

import os
from typing import Any

import dspy

_LM_CACHE: dict[str, dspy.LM] = {}


def _mk_key(provider: str, model: str, kwargs: dict[str, Any]) -> str:
    # Create a stable cache key for (provider, model, kwargs)
    # Only include a small set of common kwargs to avoid cache explosion.
    keys = ("temperature", "max_tokens", "top_p", "model_type")
    kitems = {k: kwargs.get(k) for k in keys if k in kwargs}
    return f"{provider}/{model}|{tuple(sorted(kitems.items()))}"


def openai_lm(model: str, /, **kwargs) -> dspy.LM:
    """Factory for OpenAI models. Example: openai_lm("gpt-4o-mini", temperature=0.0)."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    full = f"openai/{model}"
    key = _mk_key("openai", model, kwargs)
    if key not in _LM_CACHE:
        _LM_CACHE[key] = dspy.LM(full, api_key=api_key, **kwargs)
    return _LM_CACHE[key]


def anthropic_lm(model: str, /, **kwargs) -> dspy.LM:
    """Factory for Anthropic models. Example: anthropic_lm("claude-3-opus-20240229", temperature=0.7)."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    full = f"anthropic/{model}"
    key = _mk_key("anthropic", model, kwargs)
    if key not in _LM_CACHE:
        _LM_CACHE[key] = dspy.LM(full, api_key=api_key, **kwargs)
    return _LM_CACHE[key]


def configure_openai(model: str = "gpt-4o-mini", /, **kwargs) -> dspy.LM:
    """Convenience: configure global settings with an OpenAI LM and return it."""
    lm = openai_lm(model, **kwargs)
    dspy.settings.configure(lm=lm)
    return lm


def configure_anthropic(model: str = "claude-3-opus-20240229", /, **kwargs) -> dspy.LM:
    """Convenience: configure global settings with an Anthropic LM and return it."""
    lm = anthropic_lm(model, **kwargs)
    dspy.settings.configure(lm=lm)
    return lm


def __getattr__(name: str) -> dspy.LM:
    """
    Lazy alias accessor.

    - openai_gpt_4o_mini_lm -> dspy.LM("openai/gpt-4o-mini")
    - anthropic_claude_3_opus_20240229_lm -> dspy.LM("anthropic/claude-3-opus-20240229")

    変換規則: アンダースコア(_) をハイフン(-)へ置換し、末尾の _lm を除去します。
    """
    suffix = "_lm"
    if not name.endswith(suffix):
        raise AttributeError(name)

    def to_model(seg: str) -> str:
        # underscores to hyphens
        return seg.replace("_", "-")

    if name.startswith("openai_"):
        seg = name[len("openai_") : -len(suffix)]  # gpt_4o_mini
        model = to_model(seg)
        return openai_lm(model)

    if name.startswith("anthropic_"):
        seg = name[len("anthropic_") : -len(suffix)]
        model = to_model(seg)
        return anthropic_lm(model)

    raise AttributeError(name)


__all__ = [
    # factories
    "openai_lm",
    "anthropic_lm",
    "configure_openai",
    "configure_anthropic",
    # note: alias names are provided dynamically via __getattr__
]
