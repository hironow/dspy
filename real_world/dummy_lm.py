"""
Utilities for building DummyLMs with a JSON adapter and configuring DSPy
settings accordingly. Centralizes JSONAdapter import so demo scripts do not
need to reference it directly.
"""

from __future__ import annotations

from typing import Any, Iterable

import dspy
from dspy.adapters.json_adapter import JSONAdapter
from dspy.utils.dummies import DummyLM

_JSON_ADAPTER: JSONAdapter | None = None


def json_adapter() -> JSONAdapter:
    """Return a shared JSONAdapter instance for DummyLM usage."""
    global _JSON_ADAPTER
    if _JSON_ADAPTER is None:
        _JSON_ADAPTER = JSONAdapter()
    return _JSON_ADAPTER


def make_dummy_lm_json(
    responses: Iterable[dict[str, Any]] | dict[str, dict[str, Any]] | list[dict[str, Any]],
    *,
    follow_examples: bool = False,
) -> DummyLM:
    """Create a DummyLM that emits JSON-formatted outputs matching DSPy signatures.

    - `responses` can be a generator/iterator of dicts, a list of dicts, or a mapping
      from input string → dict output (DummyLM supports these modes).
    - Outputs are formatted via JSONAdapter so that DSPy parsers reliably match
      signature output fields.
    """
    return DummyLM(responses, follow_examples=follow_examples, adapter=json_adapter())


def configure_dummy_adapter(*, lm: Any, **overrides) -> None:
    """Configure DSPy settings to use the given LM plus JSONAdapter.

    Any extra `overrides` (e.g., rerank_policy="light") are passed to dspy.settings.configure.
    """
    dspy.settings.configure(lm=lm, adapter=json_adapter(), **overrides)


__all__ = [
    "json_adapter",
    "make_dummy_lm_json",
    "configure_dummy_adapter",
]
