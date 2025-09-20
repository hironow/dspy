"""
Lightweight helpers to build real DSPy datasets from CSV/JSONL/records.

This module complements `real_world.factory` (which focuses on tiny dummy datasets)
by providing simple utilities for:

- Loading rows from CSV/JSONL
- Cleaning and normalizing text fields
- Renaming/selecting keys
- Removing duplicates or records with missing required fields
- Train/validation splits
- Converting dict records into `dspy.Example` objects with declared input fields

Minimal usage example
---------------------
from real_world import data_tools as DT

rows = DT.load_csv("my_data.csv")
rows = DT.rename_keys(rows, {"question_text": "question", "label": "answer"})
rows = DT.drop_missing(rows, required_keys=["question", "answer"])
rows = DT.normalize_text(rows, fields=["question", "answer"])
train_rows, val_rows = DT.split_train_val(rows, val_ratio=0.25, seed=13)

import dspy
train = DT.to_examples(train_rows, input_keys=["question"])   # answer remains gold field
val = DT.to_examples(val_rows, input_keys=["question"])       # use in Evaluate/GEPA

Notes
-----
- These are intentionally simple utilities (stdlib only). Extend as needed
  (e.g., stratified splits, schema enforcement, more sophisticated cleaning).
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import dspy

# -------- I/O --------


def load_csv(path: str | Path, *, encoding: str = "utf-8", delimiter: str = ",") -> list[dict[str, Any]]:
    """Load rows from a CSV file into a list of dicts (header required)."""
    p = Path(path)
    with p.open("r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(row) for row in reader]


def load_jsonl(path: str | Path, *, encoding: str = "utf-8") -> list[dict[str, Any]]:
    """Load rows from a JSONL file into a list of dicts."""
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], *, encoding: str = "utf-8") -> None:
    """Save rows (iterable of dict-like) to JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding=encoding) as f:
        for r in rows:
            f.write(json.dumps(dict(r), ensure_ascii=False))
            f.write("\n")


# -------- Cleaning / transforms --------


def normalize_whitespace(s: Any) -> str:
    """Convert value to string and collapse whitespace to single spaces."""
    s = "" if s is None else str(s)
    return re.sub(r"\s+", " ", s.strip())


def normalize_text(rows: list[dict[str, Any]], *, fields: Sequence[str]) -> list[dict[str, Any]]:
    """Normalize whitespace for specified text fields in-place (returns same list)."""
    for r in rows:
        for k in fields:
            if k in r:
                r[k] = normalize_whitespace(r[k])
    return rows


def rename_keys(rows: list[dict[str, Any]], key_map: Mapping[str, str]) -> list[dict[str, Any]]:
    """Rename keys in each row according to key_map: {old: new}."""
    out: list[dict[str, Any]] = []
    for r in rows:
        nr = dict(r)
        for old, new in key_map.items():
            if old in nr:
                nr[new] = nr.pop(old)
        out.append(nr)
    return out


def select_keys(rows: list[dict[str, Any]], keep: Sequence[str]) -> list[dict[str, Any]]:
    """Project each row to a subset of keys."""
    keep_set = set(keep)
    return [{k: v for k, v in r.items() if k in keep_set} for r in rows]


def drop_missing(rows: list[dict[str, Any]], *, required_keys: Sequence[str]) -> list[dict[str, Any]]:
    """Drop rows that miss any of the required keys or have empty-string values."""
    req = list(required_keys)
    out: list[dict[str, Any]] = []
    for r in rows:
        if all((k in r) and (str(r[k]).strip() != "") for k in req):
            out.append(r)
    return out


def dedupe(rows: list[dict[str, Any]], *, by_keys: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Remove duplicate rows by key tuple (order preserved)."""
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    if not by_keys:
        by_keys = list(rows[0].keys()) if rows else []
    for r in rows:
        key = tuple(r.get(k) for k in by_keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# -------- Splits / stats --------

T = TypeVar("T")


def split_train_val(rows: Sequence[T], *, val_ratio: float = 0.2, seed: int = 0) -> tuple[list[T], list[T]]:
    """Random train/val split with fixed seed for reproducibility."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    k_val = max(1, int(round(len(rows) * val_ratio))) if len(rows) > 1 else len(rows)
    val_idx = set(idx[:k_val])
    train: list[T] = []
    val: list[T] = []
    for i, r in enumerate(rows):
        (val if i in val_idx else train).append(r)
    return train, val


def dataset_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return lightweight stats: size and per-key missing counts."""
    size = len(rows)
    key_missing: dict[str, int] = {}
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    for k in keys:
        key_missing[k] = sum(1 for r in rows if (k not in r) or (str(r[k]).strip() == ""))
    return {"size": size, "missing_by_key": key_missing}


# -------- Example conversion --------


def to_examples(rows: Sequence[Mapping[str, Any]], *, input_keys: Sequence[str]) -> list[dspy.Example]:
    """Convert dict rows to dspy.Example with declared inputs.

    Non-input keys remain as gold/reference fields (e.g., `answer`, `keywords`).
    Types are left as-is; convert upstream if needed.
    """
    examples: list[dspy.Example] = []
    input_set = set(input_keys)
    for r in rows:
        ex = dspy.Example(**dict(r)).with_inputs(*[k for k in r.keys() if k in input_set])
        examples.append(ex)
    return examples


__all__ = [
    # I/O
    "load_csv",
    "load_jsonl",
    "save_jsonl",
    # cleaning/transforms
    "normalize_whitespace",
    "normalize_text",
    "rename_keys",
    "select_keys",
    "drop_missing",
    "dedupe",
    # splits/stats
    "split_train_val",
    "dataset_stats",
    # example conversion
    "to_examples",
]
