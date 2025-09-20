# Handover: Vector RAG + GEPA (Adapter-first design)

This note is a practical handoff for the two new demos and their support code:

- real_world/simple_gepa_vector_rag.py
- real_world/simple_gepa_vector_rag_locomo.py
- real_world/vector_adapter.py (adapter interface + in‑memory TF‑IDF)
- real_world/data_tools.py (JSONL utilities and dataset builders)
- real_world/benefit_rag.md (why GEPA for RAG, metric/trace examples)

The goal: keep RAG retrieval a thin, auditable “Executor” and optimize only natural‑language instructions (rewrite/answer) with GEPA. The adapter pattern makes swapping real vector DBs straightforward while enabling deterministic local runs.

## TL;DR (Run These)

- Synthetic sample (deterministic):
  - `uv run python real_world/simple_gepa_vector_rag.py \
      --docs-jsonl real_world/data/docs.jsonl \
      --train-qa real_world/data/qa_train.jsonl \
      --val-qa real_world/data/qa_val.jsonl \
      --top-k 3 --dummy`
- LoCoMo (real dataset, item‑level):
  - `uv run python real_world/simple_gepa_vector_rag_locomo.py \
      --locomo real_world/assets/datasets/locomo10.json \
      --item-index 0 --top-k 5 --dummy`
  - Optional corpus override (use whole conversation file as corpus):
  - `... --locomo-rag real_world/assets/datasets/locomo10_rag.json`

Artifacts are saved to `--save-dir` (default `real_world/exports`) with `--save-prefix`.

## File Roles

- simple_gepa_vector_rag.py
  - Flat JSONL corpora (docs.jsonl {id,text,meta?}) and QA JSONL ({question,answer}).
  - Module: rewrite(question→rewritten_query), answer(question,passages→answer).
  - Retrieval is explicit: `adapter.query(rewritten_query)` → passages (Executor, not optimized).
- simple_gepa_vector_rag_locomo.py
  - For LoCoMo10: flattens conversations to per‑line docs (D#:line) or consumes `locomo10_rag.json` as a corpus; QA comes from `locomo10.json`.
  - Cautions about real data (privacy/licensing/reproducibility) are documented at the top docstring.
- vector_adapter.py
  - `VectorAdapter(Protocol)`: `upsert(docs)`, `query(text, k, filter)`; `Document`, `QueryHit` dataclasses.
  - `InMemoryTfIdfAdapter`: deterministic local retriever (no extra deps), with simple meta filtering.
  - `PineconeAdapter` is a stub (implement to wire a real backend).
- data_tools.py
  - JSONL‑only helpers: `load_jsonl/save_jsonl`, text normalization, splits, conversion to `dspy.Example`.
  - Domain builders: `prepare_*_from_jsonl` and `load_*_examples_from_jsonl`.
- benefit_rag.md
  - Why GEPA + RAG, metric patterns, pred_trace usage, adapter vs rm comparison.

## Architecture (Controller / Executor / Adapter)

- Controller (optimized by GEPA): rewrite, answer. Only their `signature.instructions` evolve.
- Executor (not optimized): the explicit call to the adapter (`adapter.query`), feeds `passages` into `answer`.
- Adapter (swap point): unify vector DB calls behind `VectorAdapter` IF.

## Metric & pred_trace

- Default metric in both demos: `score = 0.8*correctness + 0.2*groundedness`.
  - correctness: normalized EM on `pred.answer` vs gold.
  - groundedness: gold answer substring appears in `passages` (from pred_trace of `answer`).
- pred_name‑aware feedback:
  - rewrite: “keep entities, expand synonyms if recall is low, remove fluff”.
  - answer: “quote evidence, keep concise; if groundedness=0, cite matching snippet(s)”.

## Dummy vs Real LMs

- `--dummy`: uses `make_dummy_lm_json(...)` + JSONAdapter for robust parsing.
  - Per‑predictor LM contexts are applied in `forward` to avoid field mismatch.
  - Reflection LM must receive an ITERATOR of dicts, not a function. Example used:
    - `make_dummy_lm_json(iter([{"improved_instruction": "..."}] * 1000))`
- Real LMs: set via helpers (OpenAI), recommend stronger reflection LM for better proposals.

## Common Pitfalls (and fixes)

- Reflection LM TypeError: "function object is not an iterator"
  - Root cause: passing a function instead of an iterator to `DummyLM`.
  - Fix: wrap with `iter([...])` and pass the iterator directly.
- Deep copy warnings in GEPA: `_rewrite_lm/_answer_lm`
  - Harmless: GEPA deep‑copies programs; DummyLM falls back to shallow copy. Safe to ignore.
- Groundedness too lax/strict
  - It is a substring heuristic by default. Improve by adding token F1 / citation indices or evidence alignment.

## Datasets

- Synthetic: `real_world/data/docs.jsonl`, `qa_train.jsonl`, `qa_val.jsonl` (added for a quick deterministic run).
- LoCoMo10: see `real_world/assets/datasets/README.md`.
  - `locomo10.json` = QA + evidence refs (`D#:line`), may also embed sessions per item.
  - `locomo10_rag.json` = corpus (conversation per conv_id). Optional override as RAG corpus.
  - Indexing in `simple_gepa_vector_rag_locomo`:
    - From locomo10.json: session_N → `D#` → `Document(id=f"D#:{line}")`.
    - From locomo10_rag.json: `Document(id=f"R{conv_id}:{line}")`.

## Saving & Tracking

- Both demos call `save_artifacts(program, optimized, save_details=True)`.
- W&B support via `get_wandb_args(..., run_name=make_run_name(prefix))`.
- `benefit_rag.md` documents metric/trace and adapter vs rm trade‑offs.

## Adapter vs rm (Retriever Module)

- If you must keep `dspy.settings.rm`, create a thin bridge that implements `VectorAdapter` by delegating to `rm(query,k)` and wrapping return into `QueryHit`.
- Prefer the adapter path for: clearer pred_trace handling, testability, deterministic dummy, and easy backend swaps.

## Tests & Utilities

- `real_world/tests/test_data_tools.py`: schema‑compatibility between `data_tools.to_examples` and factory dummies.
- `real_world/tests/test_data_tools_jsonl_roundtrip.py`: end‑to‑end JSONL prepare→save→load as Examples.

## Extension Ideas (backlog)

- Scale:
  - Chunked indexing for large corpora, metadata filters, sharding, pagination.
- Groundedness & citations:
  - Add citation indices from `answer` and align with passages / evidence.
- Real adapter(s):
  - Implement Pinecone/Qdrant/Weaviate/pgvector adapters using `VectorAdapter`.
- rm bridge:
  - Provide `RMAdapter` and a `--use-rm` switch for environments invested in `settings.rm`.
- Data splitting in LoCoMo:
  - Option to split QA into train/val within `*_locomo` using `data_tools.split_train_val`.

## Quick Reference (Key APIs)

- VectorAdapter: `upsert(list[Document])`, `query(text, k=5, filter=None) -> list[QueryHit]`
- GEPA constructor basics:
  - `GEPA(metric, auto='light' | max_metric_calls=..., reflection_lm=..., track_stats=True, ...)`
- Per‑predictor LM contexts in Module.forward:
  - `with dspy.context(lm=self._rewrite_lm): self.rewrite(...)`
  - `with dspy.context(lm=self._answer_lm): self.answer(...)`

If you need to switch to a real backend, start by implementing the adapter in `vector_adapter.py`, test it with the synthetic JSONL demo, then point the LoCoMo demo at your live corpus. Keep retrieval auditable and keep metrics deterministic whenever possible.
