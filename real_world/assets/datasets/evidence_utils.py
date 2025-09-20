from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


EVIDENCE_RE = re.compile(r"^(D\d+):(\d+)$")


@dataclass(frozen=True)
class EvidenceRef:
    doc: str  # e.g., "D1"
    line: int  # 1-based

    @classmethod
    def parse(cls, tag: str) -> Optional["EvidenceRef"]:
        m = EVIDENCE_RE.match(tag.strip())
        if not m:
            return None
        return cls(doc=m.group(1), line=int(m.group(2)))


def parse_evidence_item(item: str) -> List[EvidenceRef]:
    """
    Parse a single evidence item which may contain multiple refs separated by ';'.
    Example: "D8:6; D9:17" -> [EvidenceRef("D8", 6), EvidenceRef("D9", 17)]
    """
    parts = [p.strip() for p in item.split(";") if p.strip()]
    refs: List[EvidenceRef] = []
    for p in parts:
        ref = EvidenceRef.parse(p)
        if ref:
            refs.append(ref)
    return refs


def parse_all_evidence(evidence: Iterable[str]) -> List[EvidenceRef]:
    refs: List[EvidenceRef] = []
    for item in evidence:
        refs.extend(parse_evidence_item(item))
    return refs


def load_locomo10(path: Path | str) -> List[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):  # safety
        raise TypeError("locomo10.json root must be a list")
    return data


def load_locomo10_rag(path: Path | str) -> Dict[str, dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):  # safety
        raise TypeError("locomo10_rag.json root must be a dict")
    return data


def build_conversation_index(item: dict) -> Dict[str, List[dict]]:
    """
    From a single locomo10 item, build {"D1": [msg, ...], "D2": [...], ...}.
    It scans conversation.session_{n} arrays (list values) and maps them to D{n}.
    """
    conv = item.get("conversation") or {}
    index: Dict[str, List[dict]] = {}
    for k, v in conv.items():
        if isinstance(v, list) and k.startswith("session_"):
            # k looks like "session_1", "session_2", ...
            try:
                num = int(k.split("_", 1)[1])
            except Exception:
                continue
            index[f"D{num}"] = v
    return index


def build_all_conversation_indices(dataset: List[dict]) -> List[Dict[str, List[dict]]]:
    """Build conversation indices for all items in the dataset."""
    return [build_conversation_index(item) for item in dataset]


def resolve_evidence(
    qa_entry: dict,
    index: Dict[str, List[dict]],
) -> List[dict]:
    """
    Return list of {"ref": "D#:line", "ok": bool, "speaker": str|None, "text": str|None, "dia_id": str|None, "error": str|None}
    in the same order as refs appear.

    - Detects invalid evidence tokens (e.g., wrong format) and reports them as errors.
    - Supports semicolon-separated tokens inside each evidence string.
    """
    evidence = qa_entry.get("evidence") or []
    out: List[dict] = []
    for item in evidence:
        if not isinstance(item, str):
            # Non-string evidence entry
            out.append({
                "ref": str(item), "ok": False, "speaker": None, "text": None, "dia_id": None,
                "error": "invalid ref type (must be str)",
            })
            continue
        parts = [p.strip() for p in item.split(";") if p.strip()]
        if not parts:
            # Empty string or only delimiters -> ignore silently
            continue
        for p in parts:
            ref = EvidenceRef.parse(p)
            if not ref:
                out.append({
                    "ref": p, "ok": False, "speaker": None, "text": None, "dia_id": None,
                    "error": "invalid ref format (expected D#:line)",
                })
                continue
            entry: dict = {"ref": f"{ref.doc}:{ref.line}", "ok": False, "speaker": None, "text": None, "dia_id": None, "error": None}
            msgs = index.get(ref.doc)
            if msgs is None:
                entry["error"] = f"doc {ref.doc} not found"
                out.append(entry)
                continue
            idx = ref.line - 1  # 1-based -> 0-based
            if not (0 <= idx < len(msgs)):
                entry["error"] = f"line {ref.line} out of range for {ref.doc} (len={len(msgs)})"
                out.append(entry)
                continue
            msg = msgs[idx]
            entry["ok"] = True
            entry["speaker"] = msg.get("speaker")
            entry["text"] = msg.get("text")
            entry["dia_id"] = msg.get("dia_id")
            out.append(entry)
    return out


def iter_qa_with_resolution(item: dict) -> Iterator[Tuple[dict, List[dict]]]:
    index = build_conversation_index(item)
    for qa in item.get("qa", []):
        yield qa, resolve_evidence(qa, index)


# --- Strict cross-validation against RAG conversation ---

_QUOTE_MAP = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "—": "-",
    "–": "-",
    "…": "...",
    "\u00a0": " ",  # non-breaking space
}


def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    out = s
    for k, v in _QUOTE_MAP.items():
        out = out.replace(k, v)
    # Collapse whitespace
    out = " ".join(out.split())
    return out


def build_rag_conversation_map(rag_session: dict) -> Tuple[set, List[dict]]:
    """
    Build a set of (speaker, normalized_text) pairs from RAG conversation.
    Return (pair_set, messages_list) where messages_list is the original list.
    """
    msgs = rag_session.get("conversation") or []
    pair_set = set()
    for m in msgs:
        sp = (m.get("speaker") or "").strip()
        tx = normalize_text(m.get("text"))
        pair_set.add((sp, tx))
    return pair_set, msgs


def strict_cross_validate(
    item: dict,
    rag_session: dict,
) -> dict:
    """
    Cross-validate that every resolved evidence line exists in locomo10_rag conversation.

    Returns a report dict with keys:
      - missing_refs: list of {ref, speaker, text} pairs not found in RAG
      - session_timestamps_missing: list of session_*_date_time strings not present in RAG timestamps
      - total_refs: int total refs checked
    """
    index = build_conversation_index(item)
    pair_set, rag_msgs = build_rag_conversation_map(rag_session)

    missing_refs: List[dict] = []
    total_refs = 0
    for qa in item.get("qa", []):
        res = resolve_evidence(qa, index)
        for r in res:
            total_refs += 1
            if not r.get("ok"):
                # Already invalid vs locomo10; count as missing in strict mode as well.
                missing_refs.append({
                    "ref": r.get("ref"),
                    "speaker": r.get("speaker"),
                    "text": r.get("text"),
                    "reason": r.get("error") or "unresolved in base",
                })
                continue
            sp = (r.get("speaker") or "").strip()
            tx_norm = normalize_text(r.get("text"))
            if (sp, tx_norm) not in pair_set:
                missing_refs.append({
                    "ref": r.get("ref"),
                    "speaker": sp,
                    "text": r.get("text"),
                    "reason": "not found in RAG conversation",
                })

    # Check that each session_*_date_time appears somewhere in RAG timestamps
    conv = item.get("conversation") or {}
    session_ts_values: List[str] = [
        v for k, v in conv.items()
        if isinstance(v, str) and k.startswith("session_") and k.endswith("_date_time")
    ]

    def _normalize_timestamp(s: Optional[str]) -> str:
        # Keep it simple: reuse text normalization and compare case-insensitively
        return normalize_text((s or "").strip()).lower()

    rag_timestamps = {
        _normalize_timestamp(m.get("timestamp"))
        for m in rag_msgs if m.get("timestamp") is not None
    }
    session_timestamps_missing = [
        ts for ts in session_ts_values
        if _normalize_timestamp(ts) not in rag_timestamps
    ]

    return {
        "missing_refs": missing_refs,
        "session_timestamps_missing": session_timestamps_missing,
        "total_refs": total_refs,
    }


def strict_cross_validate_dataset(
    dataset: List[dict],
    rag: Dict[str, dict],
) -> Dict[str, dict]:
    """
    Cross-validate all dataset items against RAG by index (item i -> rag[str(i)]).
    Returns mapping of item index (str) to per-item report as produced by strict_cross_validate.
    If a RAG entry is missing, the report contains keys:
      - missing_rag_entry: True
      - total_refs: computed refs for that item
    """
    reports: Dict[str, dict] = {}
    for i, item in enumerate(dataset):
        rag_key = str(i)
        if rag_key not in rag:
            # Still compute how many refs exist to report context
            index = build_conversation_index(item)
            total_refs = 0
            for qa in item.get("qa", []):
                # Count all tokens (valid or invalid) via resolve_evidence for consistency
                total_refs += len(resolve_evidence(qa, index))
            reports[rag_key] = {
                "missing_rag_entry": True,
                "total_refs": total_refs,
                "missing_refs": [
                    {"ref": None, "reason": "rag session missing"}
                ],
                "session_timestamps_missing": [],
            }
            continue
        reports[rag_key] = strict_cross_validate(item, rag[rag_key])
    return reports
