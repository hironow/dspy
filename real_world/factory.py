"""
Dataset factories for real_world demos.

Goal
----
- Provide small, ready-to-use dataset builders for the demo scripts.
- Offer both "dummy" (synthetic) and "real" (file/records-based) factories.

Return type
-----------
Every factory returns a tuple: (trainset, valset), where each element is a
list[dspy.Example]. Each example should call `.with_inputs(<field_names>)` so
the program knows which fields are inputs.

How to use
----------
from real_world import factory as F

# Simple QA (basic + task-metric variants reuse the same shape)
train, val = F.basic_qa_dummy(locale="ja")

# Structured invoice
train, val = F.invoice_dummy()

# Routed sources (DB/RAG/Graph)
train, val = F.routed_sources_dummy()

# Real datasets from in-memory records
train, val = F.basic_qa_from_pairs([(q, a), ...], val_ratio=0.4)
train, val = F.invoice_from_records([{...}, ...], val_ratio=0.5)
train, val = F.routed_sources_from_records([{...}, ...])

# Real datasets from files (CSV/JSONL)
train, val = F.invoice_from_csv("invoices.csv")
train, val = F.routed_sources_from_csv("routed.csv")

Notes
-----
These helpers are minimal and dependency-free (std lib only). For more
complex pipelines, extend as needed (schema validation, dedup, stratified
split, etc.).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Sequence

import dspy

# -------------------------------
# Internal utilities
# -------------------------------


def _split_train_val[T](items: Sequence[T], val_ratio: float = 0.5, seed: int = 0) -> tuple[list[T], list[T]]:
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    k_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else len(items)
    val_idx = set(idx[:k_val])
    train: list[T] = []
    val: list[T] = []
    for i, it in enumerate(items):
        (val if i in val_idx else train).append(it)
    return train, val


def _ensure_str(x: Any) -> str:
    return "" if x is None else str(x)


# -------------------------------
# 1) Simple QA (basic + task-metric variants)
# -------------------------------


def basic_qa_dummy(locale: str = "ja") -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Very small Japanese QA dataset (dummy)."""
    if locale == "ja":
        train = [
            dspy.Example(question="空の色は何色ですか？", answer="青").with_inputs("question"),
            dspy.Example(question="バナナの色は何色ですか？", answer="黄色").with_inputs("question"),
        ]
        val = [
            dspy.Example(question="晴れた日の海の色は何色ですか？", answer="青").with_inputs("question"),
            dspy.Example(question="熟したバナナの色は何色ですか？", answer="黄色").with_inputs("question"),
        ]
        return train, val
    else:
        train = [
            dspy.Example(question="What color is the sky?", answer="blue").with_inputs("question"),
            dspy.Example(question="What color is a ripe banana?", answer="yellow").with_inputs("question"),
        ]
        val = [
            dspy.Example(question="What color is the ocean on a sunny day?", answer="blue").with_inputs("question"),
            dspy.Example(question="What color is a banana?", answer="yellow").with_inputs("question"),
        ]
        return train, val


def basic_qa_from_pairs(
    pairs: Sequence[tuple[str, str]], *, val_ratio: float = 0.5, seed: int = 0
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build QA dataset from (question, answer) pairs."""
    exs = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in pairs]
    return _split_train_val(exs, val_ratio=val_ratio, seed=seed)


# Alias for task-metric demo (same shape)
task_metric_qa_dummy = basic_qa_dummy
task_metric_qa_from_pairs = basic_qa_from_pairs


# -------------------------------
# 2) Structured invoice extraction
# -------------------------------


def invoice_dummy(locale: str = "en") -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Two tiny records; the 2nd intentionally non-ISO for normalization demo.

    locale:
        - "en" (default): English fields and hints
        - "ja": Japanese text labels in the free-text `text` field
    """
    if locale == "ja":
        train = [
            dspy.Example(
                text="請求書: ベンダー=Acme Corp; 日付=2024-12-31; 金額=1234.56; 通貨=USD",
                vendor="Acme Corp",
                date="2024-12-31",
                amount=1234.56,
                currency="USD",
            ).with_inputs("text"),
            dspy.Example(
                text="請求書: ベンダー=Tokyo Shop; 日付=31-12-2024; 金額=7890; 通貨=¥",
                vendor="Tokyo Shop",
                date="2024-12-31",
                amount=7890.0,
                currency="JPY",
            ).with_inputs("text"),
        ]
        return train, train
    else:
        train = [
            dspy.Example(
                text="Invoice: Vendor=Acme Corp; Date=2024-12-31; Amount=1234.56; Currency=USD",
                vendor="Acme Corp",
                date="2024-12-31",
                amount=1234.56,
                currency="USD",
            ).with_inputs("text"),
            dspy.Example(
                text="Invoice: Vendor=Tokyo Shop; Date=31-12-2024; Amount=7890; Currency=¥",
                vendor="Tokyo Shop",
                date="2024-12-31",
                amount=7890.0,
                currency="JPY",
            ).with_inputs("text"),
        ]
        return train, train


def invoice_from_records(
    records: Sequence[dict[str, Any]], *, val_ratio: float = 0.5, seed: int = 0
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build invoice dataset from dict records.

    Expected keys per record: text, vendor, date, amount, currency
    """
    exs: list[dspy.Example] = []
    for r in records:
        ex = dspy.Example(
            text=_ensure_str(r.get("text")),
            vendor=_ensure_str(r.get("vendor")),
            date=_ensure_str(r.get("date")),
            amount=r.get("amount"),
            currency=_ensure_str(r.get("currency")),
        ).with_inputs("text")
        exs.append(ex)
    return _split_train_val(exs, val_ratio=val_ratio, seed=seed)


def invoice_from_csv(
    path: str | Path, *, val_ratio: float = 0.5, seed: int = 0
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return invoice_from_records(rows, val_ratio=val_ratio, seed=seed)


def invoice_from_jsonl(
    path: str | Path, *, val_ratio: float = 0.5, seed: int = 0
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return invoice_from_records(rows, val_ratio=val_ratio, seed=seed)


# -------------------------------
# 3) Routed multi-source (DB / RAG / Graph)
# -------------------------------


def routed_sources_dummy(locale: str = "en") -> tuple[list[dspy.Example], list[dspy.Example]]:
    if locale == "ja":
        train = [
            dspy.Example(
                query="ID 42 のユーザーのメールアドレスは？",
                answer="user42@example.com",
                preferred_source="db",
            ).with_inputs("query"),
            dspy.Example(
                query="最新のポリシー更新を要約してください。",
                answer="Policy updated in 2023",
                preferred_source="rag",
            ).with_inputs("query"),
            dspy.Example(
                query="NodeA と NodeB の関係を説明してください。",
                answer="NodeA connected to NodeB via edge X",
                preferred_source="graph",
            ).with_inputs("query"),
        ]
        return train, train
    else:
        train = [
            dspy.Example(
                query="What is the email for user id 42?",
                answer="user42@example.com",
                preferred_source="db",
            ).with_inputs("query"),
            dspy.Example(
                query="Summarize the latest policy update.",
                answer="Policy updated in 2023",
                preferred_source="rag",
            ).with_inputs("query"),
            dspy.Example(
                query="Describe the relation between NodeA and NodeB.",
                answer="NodeA connected to NodeB via edge X",
                preferred_source="graph",
            ).with_inputs("query"),
        ]
        return train, train


def routed_sources_from_records(
    records: Sequence[dict[str, Any]], *, val_ratio: float = 0.5, seed: int = 0
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build routed multi-source dataset from dict records.

    Expected keys per record: query, answer, preferred_source (one of db/rag/graph)
    """
    exs: list[dspy.Example] = []
    for r in records:
        ex = dspy.Example(
            query=_ensure_str(r.get("query")),
            answer=_ensure_str(r.get("answer")),
            preferred_source=_ensure_str(r.get("preferred_source")),
        ).with_inputs("query")
        exs.append(ex)
    return _split_train_val(exs, val_ratio=val_ratio, seed=seed)


def routed_sources_from_csv(
    path: str | Path, *, val_ratio: float = 0.5, seed: int = 0
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return routed_sources_from_records(rows, val_ratio=val_ratio, seed=seed)


def routed_sources_from_jsonl(
    path: str | Path, *, val_ratio: float = 0.5, seed: int = 0
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return routed_sources_from_records(rows, val_ratio=val_ratio, seed=seed)


__all__ = [
    # basic QA
    "basic_qa_dummy",
    "basic_qa_from_pairs",
    # task metric variant (aliases)
    "task_metric_qa_dummy",
    "task_metric_qa_from_pairs",
    # invoices
    "invoice_dummy",
    "invoice_from_records",
    "invoice_from_csv",
    "invoice_from_jsonl",
    # routed sources
    "routed_sources_dummy",
    "routed_sources_from_records",
    "routed_sources_from_csv",
    "routed_sources_from_jsonl",
    # image captioning
    "image_caption_dummy",
]


# -------------------------------
# 4) Multimodal image captioning
# -------------------------------


def image_caption_dummy(locale: str = "ja") -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Tiny multimodal caption dataset (image -> caption/keywords).

    We provide images as dspy.Image with URLs (no download required here). Gold has
    keywords to evaluate simple coverage.
    """
    # Simple placeholder images (public picsum endpoints)
    img1 = dspy.Image(url="https://picsum.photos/id/237/300/200")  # dog
    img2 = dspy.Image(url="https://picsum.photos/id/1025/300/200")  # nature
    img3 = dspy.Image(url="https://picsum.photos/id/1062/300/200")  # car/street-ish

    if locale == "ja":
        train = [
            dspy.Example(image=img1, keywords=["犬", "屋外", "茶色"]).with_inputs("image"),
            dspy.Example(image=img2, keywords=["山", "空", "緑"]).with_inputs("image"),
            dspy.Example(image=img3, keywords=["車", "道路", "赤"]).with_inputs("image"),
        ]
    else:
        train = [
            dspy.Example(image=img1, keywords=["dog", "outdoor", "brown"]).with_inputs("image"),
            dspy.Example(image=img2, keywords=["mountain", "sky", "green"]).with_inputs("image"),
            dspy.Example(image=img3, keywords=["car", "road", "red"]).with_inputs("image"),
        ]

    # keep val = train for the demo simplicity
    return train, train
