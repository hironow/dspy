# Handover (LoCoMo dataset) – Maintenance & Usage Guide

This is a practical handoff for maintaining and using the LoCoMo10 dataset within this repo. It captures structure, validators, known pitfalls, recent fixes, and how to wire the dataset into the Vector RAG + GEPA demo safely and deterministically.

## Files of Interest

- real_world/assets/datasets/locomo10.json
  - QA per item + per‑session conversation (session_N → D#)
- real_world/assets/datasets/locomo10_rag.json
  - Alternative RAG corpus (conversation per conv_id)
- real_world/assets/datasets/evidence_utils.py
  - Evidence parsing/resolution and RAG strict cross‑validation helpers
- real_world/assets/datasets/validate_evidence.py
  - CLI validator for QA evidence and strict cross‑checks
- real_world/simple_gepa_vector_rag_locomo.py
  - LoCoMo Vector RAG demo + GEPA prompt optimization
- real_world/vector_adapter.py
  - In‑memory TF‑IDF adapter (deterministic), adapter interface for swapping real vector DBs

## Dataset Structure (short)

- Evidence format: "D#:line" (1‑based lines). D1 corresponds to session_1, etc.
- In locomo10.json, a typical item has:
  - conversation.session_N: list of message objects
  - qa: [{question, answer, evidence: ["D1:3", ...], category}]
- In locomo10_rag.json, top‑level keys are string IDs ("0", "1", ...), each with conversation: [...].

## Validators and How To Use Them

- Quick run (all items, all QA):
  - uv run python -m real_world.assets.datasets.validate_evidence
- Strict cross‑check against locomo10_rag.json:
  - uv run python -m real_world.assets.datasets.validate_evidence --strict --strict-ts off
  - Notes:
    - --rag-id auto (default) cross‑checks every dataset item against rag[str(i)].
    - --strict-ts off/warn/required controls timestamp checks (text normalization is applied; off is recommended for QA workflows).
- Preview few QA only:
  - uv run python -m real_world.assets.datasets.validate_evidence --limit 10
- Unresolved evidence handling:
  - When any unresolved refs exist, a validate_result.txt is written next to the scripts with a concise list (Item/QA/evidence/unresolved refs & reasons). This is intended to guide manual edits without opening whole JSON files.

## Known Pitfalls (and what the validator does)

- 1‑based vs 0‑based: lines are 1‑based in evidence; validator converts to 0‑based.
- Invalid tokens: e.g., "D", "D:11:26", space‑concatenated refs like "D9:1 D4:4".
  - The validator now flags invalid formats instead of silently skipping them.
- Range errors: e.g., D10:19 when session_10 has 16 lines.
- Text mismatches in strict mode:
  - Text is normalized (quotes/dashes/whitespace/NBSP); timestamps are compared case‑insensitively with simple normalization.

## Recent Fixes (already committed)

We repaired seven issues in locomo10.json. After these changes:

- All referenced evidence resolved successfully across all items.
- Strict mode cross‑validation: Total refs checked: 2822. Passed.

Edits (Item index is 0‑based; QA numbering is the printed 1‑based index):

- Item 3 / QA 59 – D10:19 → D20:15 (session_10 has only 16 lines; butter alternative advice is at D20:15)
- Item 3 / QA 89 – removed invalid token D; evidence is now ["D1:18", "D1:20"]
- Item 4 / QA 19 – D:11:26 → D11:26
- Item 6 / QA 39 – removed out‑of‑range D4:36 (kept D18:1, D18:7)
- Item 8 / QA 32 – split "D9:1 D4:4 D4:6" into ["D9:1","D4:4","D4:6"]
- Item 8 / QA 39 – split "D22:1 D22:2 D9:10 D9:11" into separate tokens
- Item 8 / QA 47 – split "D21:18 D21:22 D11:15 D11:19" into separate tokens

These changes are documented in real_world/assets/datasets/README.md under 「データ修正履歴（アノテーション）」。

## How To Manually Patch Incorrect Evidence (without loading whole files)

- Use ripgrep to find the target QA:
  - rg -n "<question text>" real_world/assets/datasets/locomo10.json
- Inspect a narrow range:
  - sed -n 'LINE_START,LINE_ENDp' real_world/assets/datasets/locomo10.json
- Identify the bad token and correct it
  a) Out‑of‑range lines → find intended line by searching the phrase in the item’s sessions or in locomo10_rag.json via strict run.
  b) Invalid token formats → fix to "D#:line" or remove clearly spurious entries.
- Re‑run the validator; if unresolved remain, check validate_result.txt and repeat.

Tip: D# maps to conversation.session_N; lines are 1‑based. "dia_id" fields (e.g., "D10:11") can help you confirm the ground truth.

## evidence_utils.py – What’s Inside

- EvidenceRef (doc, line) and parsers (semicolon‑separated expansion supported)
- build_conversation_index(item) → { "D#": [msg, ...] }
- resolve_evidence(qa, index) → list of {ref, ok, speaker, text, dia_id, error}
  - Detects invalid evidence (format/type) and emits errors.
- strict_cross_validate(item, rag_session)
  - Check whether each resolved (speaker, normalized_text) exists in RAG
  - Also checks session_*_date_time against RAG timestamps (normalized)
- strict_cross_validate_dataset(dataset, rag) – maps every item i to rag[str(i)]

## Vector RAG + GEPA (LoCoMo)

- Script: real_world/simple_gepa_vector_rag_locomo.py
- What it does:
  - Flattens item conversations into per‑line documents id=f"D#:{line}" or consumes locomo10_rag.json as corpus id=f"R{conv_id}:{line}".
  - Builds QA examples from the same item.
  - Retrieval is explicit via VectorAdapter; GEPA optimizes only rewrite/answer instructions, not retrieval.
- Deterministic dummy run:
  - uv run python real_world/simple_gepa_vector_rag_locomo.py --locomo real_world/assets/datasets/locomo10.json --item-index 0 --top-k 5 --dummy
- TODOs left in code (for the next person):
  1) Large-file streaming load
     - Replace full load with ijson-based streaming to extract only --item-index.
  2) Evidence-driven retrieval filtering
     - Parse doc_ids from QA evidence and pass adapter.query(..., filter={'doc': 'D#'}) to restrict search space.

## Do / Don’t

- Do
  - Validate early with validate_evidence; use strict mode before publishing changes.
  - Use ripgrep + narrow sed windows; avoid opening huge JSONs when patching.
  - Keep Controller/Executor separation in demos; retrieval remains auditable.
  - Document fixes in README’s 修正履歴.
- Don’t
  - Change evidence semantics to 0‑based; it is 1‑based by design.
  - Silently drop bad tokens; fix or remove them explicitly.
  - Upload raw conversations to third‑party services without approval.

## Privacy, Licensing, Reproducibility

- Treat all conversation text as potentially sensitive.
- Confirm dataset license/terms for any external sharing.
- Prefer deterministic runs (dummy LM + in‑memory adapter) for CI/regressions; pin seeds/versions if you switch to real backends.

## Quick Commands

- Validate all QA:
  - uv run python -m real_world.assets.datasets.validate_evidence
- Validate + strict cross‑check:
  - uv run python -m real_world.assets.datasets.validate_evidence --strict --strict-ts off
- LoCoMo Vector RAG + GEPA (dummy):
  - uv run python real_world/simple_gepa_vector_rag_locomo.py --locomo real_world/assets/datasets/locomo10.json --item-index 0 --top-k 5 --dummy

If you hit a new unresolved, start with validate_result.txt, patch with ripgrep + sed windows, and keep the 修正履歴 current. Good luck!
