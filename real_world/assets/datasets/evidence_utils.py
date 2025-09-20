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


def resolve_evidence(
    qa_entry: dict,
    index: Dict[str, List[dict]],
) -> List[dict]:
    """
    Return list of {"ref": "D#:line", "ok": bool, "speaker": str|None, "text": str|None, "dia_id": str|None, "error": str|None}
    in the same order as refs appear.
    """
    evidence = qa_entry.get("evidence") or []
    refs = parse_all_evidence(evidence)
    out: List[dict] = []
    for ref in refs:
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

