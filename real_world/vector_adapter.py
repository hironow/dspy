"""
Vector DB adapter interfaces and a simple in-memory TF-IDF implementation.

This module makes adapter usage explicit and convenient in the simple_* demos.

Usage (in-memory):
    from real_world.vector_adapter import Document, InMemoryTfIdfAdapter

    adapter = InMemoryTfIdfAdapter()
    adapter.upsert([
        Document(id="d1", text="Tokyo is the capital of Japan"),
        Document(id="d2", text="Mount Fuji is near Tokyo"),
    ])
    hits = adapter.query("capital of japan", k=2)
    # hits -> list[QueryHit(id, text, score, meta)]

To plug in a real vector DB, implement VectorAdapter and pass it to the demo.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Document:
    id: str
    text: str
    meta: dict[str, Any] | None = None


@dataclass
class QueryHit:
    id: str
    text: str
    score: float
    meta: dict[str, Any] | None = None


class VectorAdapter(Protocol):
    def upsert(self, docs: list[Document]) -> None: ...
    def query(self, text: str, *, k: int = 5, filter: dict[str, Any] | None = None) -> list[QueryHit]: ...


class InMemoryTfIdfAdapter:
    """Deterministic in-memory TF-IDF adapter (no external deps).

    - Tokenizes on word characters, lowercases
    - Builds IDF over inserted docs; supports repeated upserts (updates inverted index)
    - Returns top-k by cosine similarity between TF-IDF vectors
    - Optional naive filter by exact meta key equality (if provided)
    """

    def __init__(self):
        self._docs: dict[str, Document] = {}
        self._bow: dict[str, dict[str, int]] = {}  # id -> term -> tf
        self._df: dict[str, int] = {}  # term -> doc freq
        self._N: int = 0

    @staticmethod
    def _tokens(s: str) -> list[str]:
        return re.findall(r"\w+", (s or "").lower())

    def _rebuild_df(self):
        self._df.clear()
        for tf in self._bow.values():
            for term in tf.keys():
                self._df[term] = self._df.get(term, 0) + 1
        self._N = max(1, len(self._bow))

    def upsert(self, docs: list[Document]) -> None:
        for d in docs:
            self._docs[d.id] = d
            tf: dict[str, int] = {}
            for t in self._tokens(d.text):
                tf[t] = tf.get(t, 0) + 1
            self._bow[d.id] = tf
        self._rebuild_df()

    def _tfidf_vec(self, tf: dict[str, int]) -> dict[str, float]:
        vec: dict[str, float] = {}
        for term, f in tf.items():
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self._N + 1) / (df + 1)) + 1.0  # smoothed idf
            vec[term] = f * idf
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a.keys()) & set(b.keys())
        num = sum(a[t] * b[t] for t in common)
        da = math.sqrt(sum(v * v for v in a.values()))
        db = math.sqrt(sum(v * v for v in b.values()))
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)

    def query(self, text: str, *, k: int = 5, filter: dict[str, Any] | None = None) -> list[QueryHit]:
        q_tf: dict[str, int] = {}
        for t in self._tokens(text):
            q_tf[t] = q_tf.get(t, 0) + 1
        q_vec = self._tfidf_vec(q_tf)

        hits: list[tuple[str, float]] = []
        for doc_id, tf in self._bow.items():
            d = self._docs[doc_id]
            if filter:
                ok = True
                for fk, fv in filter.items():
                    if (d.meta or {}).get(fk) != fv:
                        ok = False
                        break
                if not ok:
                    continue
            score = self._cosine(q_vec, self._tfidf_vec(tf))
            if score > 0:
                hits.append((doc_id, score))

        hits.sort(key=lambda x: x[1], reverse=True)
        top = hits[: max(1, k)]
        return [QueryHit(id=i, text=self._docs[i].text, score=s, meta=self._docs[i].meta) for i, s in top]


class PineconeAdapter:  # Example stub for a real adapter
    """Placeholder for a real Pinecone (or other) adapter.

    Implement the same interface as VectorAdapter:
        - upsert(self, docs: list[Document])
        - query(self, text: str, *, k: int = 5, filter: dict | None = None)

    This stub is provided so the demo can show how to swap adapters via CLI/env
    without pulling heavy dependencies in this repo.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError("PineconeAdapter is a stub. Implement using your environment's SDK and drop it in.")
