"""
Lightweight helpers to build real DSPy datasets from JSONL/records.

This module complements `real_world.factory` (which focuses on tiny dummy datasets)
by providing simple utilities for JSONL-based pipelines:

- Loading rows from JSONL
- Cleaning and normalizing text fields
- Renaming/selecting keys
- Removing duplicates or records with missing required fields
- Train/validation splits
- Converting dict records into `dspy.Example` objects with declared input fields

Minimal usage example (JSONL)
-----------------------------
from real_world import data_tools as DT

rows = DT.load_jsonl("my_data.jsonl")
rows = DT.rename_keys(rows, {"question_text": "question", "label": "answer"})
rows = DT.drop_missing(rows, required_keys=["question", "answer"])
rows = DT.normalize_text(rows, fields=["question", "answer"])
train_rows, val_rows = DT.split_train_val(rows, val_ratio=0.25, seed=13)
DT.save_jsonl("train.jsonl", train_rows)
DT.save_jsonl("val.jsonl", val_rows)

import dspy
train = DT.to_examples(train_rows, input_keys=["question"])   # answer remains gold field
val = DT.to_examples(val_rows, input_keys=["question"])       # use in Evaluate/GEPA

Notes
-----
- These are intentionally simple utilities (stdlib only). Extend as needed
  (e.g., stratified splits, schema enforcement, more sophisticated cleaning).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import dspy

# -------- I/O (JSONL) --------


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
    # I/O (JSONL)
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
    # high-level dataset builders (JSONL)
    "prepare_qa_from_jsonl",
    "prepare_invoice_from_jsonl",
    "prepare_routed_from_jsonl",
    "prepare_image_caption_from_jsonl",
    "prepare_langextract_from_jsonl",
    # domain-specific loaders (JSONL -> Examples)
    "load_qa_examples_from_jsonl",
    "load_invoice_examples_from_jsonl",
    "load_routed_examples_from_jsonl",
    "load_image_caption_examples_from_jsonl",
    "load_langextract_examples_from_jsonl",
]


# -------- Domain-specific pipelines (JSONL -> JSONL + Examples) --------


def _ensure_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        try:
            # strip currency symbols etc.
            s = re.sub(r"[^0-9.\-]", "", str(v))
            return float(s) if s else None
        except Exception:
            return None


def _parse_keywords(value: Any) -> list[str]:
    """Parse keywords from JSONL cell (JSON string/list) into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_whitespace(x) for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    # try JSON array first
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return [normalize_whitespace(x) for x in arr if str(x).strip()]
    except Exception:
        pass
    # fallback: split by comma
    return [normalize_whitespace(x) for x in s.split(",") if x.strip()]


def prepare_qa_from_jsonl(
    jsonl_path: str | Path,
    out_train_jsonl: str | Path,
    out_val_jsonl: str | Path,
    *,
    question_key: str = "question",
    answer_key: str = "answer",
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    rows = load_jsonl(jsonl_path)
    if question_key != "question" or answer_key != "answer":
        rows = rename_keys(rows, {question_key: "question", answer_key: "answer"})
    rows = select_keys(rows, ["question", "answer"])
    rows = drop_missing(rows, required_keys=["question", "answer"])
    rows = normalize_text(rows, fields=["question", "answer"])
    train_rows, val_rows = split_train_val(rows, val_ratio=val_ratio, seed=seed)
    save_jsonl(out_train_jsonl, train_rows)
    save_jsonl(out_val_jsonl, val_rows)
    return to_examples(train_rows, input_keys=["question"]), to_examples(val_rows, input_keys=["question"])


def prepare_invoice_from_jsonl(
    jsonl_path: str | Path,
    out_train_jsonl: str | Path,
    out_val_jsonl: str | Path,
    *,
    text_key: str = "text",
    vendor_key: str = "vendor",
    date_key: str = "date",
    amount_key: str = "amount",
    currency_key: str = "currency",
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    rows = load_jsonl(jsonl_path)
    key_map = {}
    for src, dst in [
        (text_key, "text"),
        (vendor_key, "vendor"),
        (date_key, "date"),
        (amount_key, "amount"),
        (currency_key, "currency"),
    ]:
        if src != dst:
            key_map[src] = dst
    if key_map:
        rows = rename_keys(rows, key_map)
    # convert amount to float if possible
    for r in rows:
        r["amount"] = _ensure_float(r.get("amount"))
    rows = select_keys(rows, ["text", "vendor", "date", "amount", "currency"])
    rows = drop_missing(rows, required_keys=["text", "vendor", "date", "currency"])  # amount can be None
    rows = normalize_text(rows, fields=["text", "vendor", "date", "currency"])
    train_rows, val_rows = split_train_val(rows, val_ratio=val_ratio, seed=seed)
    save_jsonl(out_train_jsonl, train_rows)
    save_jsonl(out_val_jsonl, val_rows)
    return to_examples(train_rows, input_keys=["text"]), to_examples(val_rows, input_keys=["text"])


def prepare_routed_from_jsonl(
    jsonl_path: str | Path,
    out_train_jsonl: str | Path,
    out_val_jsonl: str | Path,
    *,
    query_key: str = "query",
    answer_key: str = "answer",
    preferred_source_key: str = "preferred_source",
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    rows = load_jsonl(jsonl_path)
    key_map = {}
    for src, dst in [
        (query_key, "query"),
        (answer_key, "answer"),
        (preferred_source_key, "preferred_source"),
    ]:
        if src != dst:
            key_map[src] = dst
    if key_map:
        rows = rename_keys(rows, key_map)
    rows = select_keys(rows, ["query", "answer", "preferred_source"])
    rows = drop_missing(rows, required_keys=["query", "answer"])  # preferred_source optional but recommended
    rows = normalize_text(rows, fields=["query", "answer", "preferred_source"])
    # normalize preferred_source to {db, rag, graph}
    for r in rows:
        ps = str(r.get("preferred_source", "")).strip().lower()
        if ps not in {"db", "rag", "graph"}:
            r["preferred_source"] = ps or "rag"
    train_rows, val_rows = split_train_val(rows, val_ratio=val_ratio, seed=seed)
    save_jsonl(out_train_jsonl, train_rows)
    save_jsonl(out_val_jsonl, val_rows)
    return to_examples(train_rows, input_keys=["query"]), to_examples(val_rows, input_keys=["query"])


def prepare_image_caption_from_jsonl(
    jsonl_path: str | Path,
    out_train_jsonl: str | Path,
    out_val_jsonl: str | Path,
    *,
    image_url_key: str = "image_url",
    keywords_key: str = "keywords",
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build image-caption dataset from JSONL with fields [image_url, keywords] (or compatible).

    keywords can be a JSON array string or a comma-separated string.
    JSONL stores {"image_url": str, "keywords": list[str]} per row.
    """
    rows_raw = load_jsonl(jsonl_path)
    rows: list[dict[str, Any]] = []
    for r in rows_raw:
        url = r.get(image_url_key)
        if not url and isinstance(r.get("image"), dict):
            url = r["image"].get("url")
        kw = r.get(keywords_key)
        rows.append({"image_url": url, "keywords": _parse_keywords(kw)})
    rows = select_keys(rows, ["image_url", "keywords"])
    rows = drop_missing(rows, required_keys=["image_url"])  # keywords can be empty
    train_rows, val_rows = split_train_val(rows, val_ratio=val_ratio, seed=seed)
    save_jsonl(out_train_jsonl, train_rows)
    save_jsonl(out_val_jsonl, val_rows)

    # Convert to Examples: image is dspy.Image(url=...)
    def _rows_to_examples(rr: Sequence[Mapping[str, Any]]):
        exs: list[dspy.Example] = []
        for r in rr:
            exs.append(
                dspy.Example(image=dspy.Image(url=r["image_url"]), keywords=list(r.get("keywords", []))).with_inputs(
                    "image"
                )
            )
        return exs

    return _rows_to_examples(train_rows), _rows_to_examples(val_rows)


def prepare_langextract_from_jsonl(
    jsonl_path: str | Path,
    out_train_jsonl: str | Path,
    out_val_jsonl: str | Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build langextract dataset from JSONL with rows: {text: str, targets: list[dict]}"""
    rows = load_jsonl(jsonl_path)
    rows = select_keys(rows, ["text", "targets"])  # preserve expected fields
    rows = drop_missing(rows, required_keys=["text"])  # targets may be empty
    rows = normalize_text(rows, fields=["text"])  # don't touch targets
    train_rows, val_rows = split_train_val(rows, val_ratio=val_ratio, seed=seed)
    save_jsonl(out_train_jsonl, train_rows)
    save_jsonl(out_val_jsonl, val_rows)
    return to_examples(train_rows, input_keys=["text"]), to_examples(val_rows, input_keys=["text"])


# -------- Domain-specific loaders (JSONL -> Examples) --------


def load_qa_examples_from_jsonl(path: str | Path) -> list[dspy.Example]:
    rows = load_jsonl(path)
    return to_examples(rows, input_keys=["question"])


def load_invoice_examples_from_jsonl(path: str | Path) -> list[dspy.Example]:
    rows = load_jsonl(path)
    # ensure amount is float if stored as string
    for r in rows:
        r["amount"] = _ensure_float(r.get("amount"))
    return to_examples(rows, input_keys=["text"])


def load_routed_examples_from_jsonl(path: str | Path) -> list[dspy.Example]:
    rows = load_jsonl(path)
    return to_examples(rows, input_keys=["query"])


def load_image_caption_examples_from_jsonl(path: str | Path) -> list[dspy.Example]:
    rows = load_jsonl(path)
    exs: list[dspy.Example] = []
    for r in rows:
        img_url = r.get("image_url")
        if not img_url:
            image_field = r.get("image")
            if isinstance(image_field, dict):
                img_url = image_field.get("url")
            elif isinstance(image_field, str):
                img_url = image_field
        if not img_url or not str(img_url).strip():
            # Skip malformed rows that don't have a usable URL.
            continue
        kws = r.get("keywords") or []
        exs.append(dspy.Example(image=dspy.Image(url=img_url), keywords=kws).with_inputs("image"))
    return exs


def load_langextract_examples_from_jsonl(path: str | Path) -> list[dspy.Example]:
    rows = load_jsonl(path)
    return to_examples(rows, input_keys=["text"])
