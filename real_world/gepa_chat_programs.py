"""
Bridging utilities that expose the GEPA sample programs through a simple
chat-friendly interface.

The goal is to keep the original `real_world/simple_gepa_*.py` scripts
unchanged while providing lightweight adapters that can be invoked from the
Gradio chat UI (or other front-ends). Each adapter:

- Wraps the underlying DSPy program in a small object with a uniform `run(...)`
  method that accepts a text prompt plus optional configuration.
- Executes the program inside a `dspy.context(lm=...)` scope so the caller can
  decide which language model to use (OpenAI helper, dummy JSON LM, etc.).
- Returns a `ProgramResponse` carrying both the primary text to display in a
  chat bubble and structured metadata that UIs can render separately.

Only text-centric programs are fully supported today. Multimodal demos still
require richer UI widgets (image upload, etc.), so they are registered as
unavailable with an explanatory note.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, Mapping, MutableMapping, Protocol, Sequence

from loguru import logger

import dspy

# ----------------------------
# Type aliases
# ----------------------------

HistoryMessage = dict[str, str]
History = Sequence[HistoryMessage]


@dataclass
class ProgramRequest:
    """Canonical request passed to each chat adapter."""

    prompt: str
    history: History
    lm: dspy.LM | None = None
    options: Mapping[str, Any] | None = None


@dataclass
class ProgramResponse:
    """Result returned by each adapter."""

    text: str
    details: Mapping[str, Any] | None = None
    raw: Any | None = None


class ChatProgram(Protocol):
    """Protocol for chat-ready program adapters."""

    slug: str
    display_name: str
    description: str
    input_hint: str
    modalities: frozenset[str]

    def run(self, request: ProgramRequest) -> ProgramResponse: ...


def _lm_scope(lm: dspy.LM | None) -> Iterator[None]:
    """Create a context manager that sets the DSPy LM only when provided."""
    if lm is None:
        return contextlib.nullcontext()
    return dspy.context(lm=lm)


class _BaseProgram(ChatProgram):
    """Common helper that wraps error handling for adapters."""

    slug = "base"
    display_name = "Base Program"
    description = ""
    input_hint = "テキストを入力してください。"
    modalities = frozenset({"text"})
    artifact_prefix: str | None = None
    optimized_path: str | None = None

    def run(self, request: ProgramRequest) -> ProgramResponse:
        try:
            return self._predict(request)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Program %s failed with %s", self.slug, exc)
            return ProgramResponse(text=f"[ERROR] {exc}")

    def _predict(self, request: ProgramRequest) -> ProgramResponse:
        raise NotImplementedError

    def _load_optimized_state(self) -> None:
        if not hasattr(self, "program"):
            return
        if not self.artifact_prefix:
            return
        path = _resolve_optimized_path(self.slug, self.artifact_prefix)
        if not path:
            return
        try:
            self.program.load(path)
            self.optimized_path = str(path)
            logger.info("Loaded optimized state for {} from {}", self.slug, path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load optimized state for {} from {}: {}", self.slug, path, exc)


def _latest_user_prompt(history: History, fallback: str) -> str:
    """Use the latest user message from the chat history as the working prompt."""
    for message in reversed(history):
        if message.get("role") == "user":
            return message.get("content", "").strip()
    return fallback.strip()


_EXPORT_DIR_ENV = "DSPY_GEPA_EXPORT_DIR"
_OPTIMIZED_ENV_TEMPLATE = "DSPY_GEPA_{slug}_OPTIMIZED"


def _collect_optimized_candidates(directory: Path, prefix: str) -> list[Path]:
    patterns = [f"{prefix}-optimized-*.json", f"{prefix}-optimized-*.pkl"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(directory.glob(pattern))
    return sorted([p for p in candidates if p.is_file()])


def _resolve_optimized_path(slug: str, prefix: str | None) -> Path | None:
    slug_env = _OPTIMIZED_ENV_TEMPLATE.format(slug=slug.upper())
    explicit = os.getenv(slug_env)
    if explicit:
        target = Path(explicit).expanduser()
        if target.is_file():
            return target
        if target.is_dir() and prefix:
            candidates = _collect_optimized_candidates(target, prefix)
            if candidates:
                return candidates[-1]
        logger.warning("Environment variable {} did not point to a usable file: {}", slug_env, explicit)

    if not prefix:
        return None

    export_dir = Path(os.getenv(_EXPORT_DIR_ENV, "real_world/exports")).expanduser()
    if not export_dir.exists():
        return None
    candidates = _collect_optimized_candidates(export_dir, prefix)
    if candidates:
        return candidates[-1]
    return None


# ----------------------------
# Simple GEPA adapters
# ----------------------------


class _SimpleQAProgram(_BaseProgram):
    """Shared logic for the SimpleQA-based demos."""

    _rewrite_instruction = (
        "与えられた日本語の質問を、意味を保ったまま簡潔に言い換えてください。曖昧さを避け、不要な語を省いてください。"
    )
    _predict_instruction = "次の日本語の質問に、短く正確に回答してください。回答は名詞一語を目指してください。"

    def __init__(self, module_path: str, class_name: str = "SimpleQA"):
        module = __import__(module_path, fromlist=[class_name])
        simple_qa_cls = getattr(module, class_name)
        self.program: dspy.Module = simple_qa_cls()
        self.program.rewrite.signature = self.program.rewrite.signature.with_instructions(self._rewrite_instruction)
        self.program.predict.signature = self.program.predict.signature.with_instructions(self._predict_instruction)
        self._load_optimized_state()

    def _predict(self, request: ProgramRequest) -> ProgramResponse:
        prompt = request.prompt.strip() or _latest_user_prompt(request.history, "")
        if not prompt:
            return ProgramResponse(text="（入力が空です）")
        with _lm_scope(request.lm):
            pred = self.program(question=prompt)
        answer = getattr(pred, "answer", "")
        text = answer if answer else "（回答が生成されませんでした）"
        return ProgramResponse(text=text, details={"program": self.slug}, raw=pred)


class SimpleGEPABasicProgram(_SimpleQAProgram):
    slug = "simple_gepa_basic"
    display_name = "Simple GEPA Basic QA"
    description = "Rewrite→Answer の2段構成で短い回答を生成する最小のGEPAサンプル（最適化前）。"
    artifact_prefix = "simple_gepa"

    def __init__(self):
        super().__init__("real_world.simple_gepa_basic")


class SimpleGEPATaskMetricProgram(_SimpleQAProgram):
    slug = "simple_gepa_task_metric"
    display_name = "Simple GEPA Task Metric QA"
    description = "タスク特化メトリックを持つ SimpleQA バリエーション。基本構造は Basic と同じ。"
    artifact_prefix = "simple_gepa_task"

    def __init__(self):
        super().__init__("real_world.simple_gepa_task_metric")


class SimpleGEPAInvoiceProgram(_BaseProgram):
    slug = "simple_gepa_structured_invoice"
    display_name = "Simple GEPA Structured Invoice"
    description = "請求書テキストから (vendor, date, amount, currency) を抽出・正規化する2段モジュール。"
    input_hint = "請求書テキストを入力してください（例: ベンダー、日付、金額、通貨を含む列挙形式）。"
    artifact_prefix = "simple_gepa_invoice"

    _extract_instruction = "与えられた請求テキストから vendor（会社名）, date（YYYY-MM-DD）, amount（数値）, currency（ISOコード）を抽出してください。"
    _normalize_instruction = "抽出結果を正規化してください。dateはYYYY-MM-DD、currencyはISOコード（例: JPY）に統一し、amountは数値で返してください。"

    def __init__(self):
        module = __import__("real_world.simple_gepa_structured_invoice", fromlist=["InvoiceIE"])
        invoice_cls = module.InvoiceIE
        self.program: dspy.Module = invoice_cls()
        self.program.extract.signature = self.program.extract.signature.with_instructions(self._extract_instruction)
        self.program.normalize.signature = self.program.normalize.signature.with_instructions(
            self._normalize_instruction
        )
        self._load_optimized_state()

    def _predict(self, request: ProgramRequest) -> ProgramResponse:
        prompt = request.prompt.strip() or _latest_user_prompt(request.history, "")
        if not prompt:
            return ProgramResponse(text="（入力が空です）")
        with _lm_scope(request.lm):
            pred = self.program(text=prompt)
        fields = {
            "vendor": getattr(pred, "vendor", ""),
            "date": getattr(pred, "date", ""),
            "amount": getattr(pred, "amount", ""),
            "currency": getattr(pred, "currency", ""),
        }
        lines = [f"{key}: {value}" for key, value in fields.items()]
        return ProgramResponse(text="\n".join(lines), details={"program": self.slug, **fields}, raw=pred)


class SimpleGEPARoutedSourcesProgram(_BaseProgram):
    slug = "simple_gepa_routed_sources"
    display_name = "Simple GEPA Routed Sources"
    description = "質問に応じて DB / RAG / Graph から情報源をルーティングし、最終回答を選ぶサンプル。"
    input_hint = "問い合わせ内容を入力してください。モデルが適切な情報源を選択し短いテキストを返します。"
    artifact_prefix = "simple_gepa_routed"

    _instructions: ClassVar[dict[str, str]] = {
        "route": "与えられた質問の意図に基づき、'db' / 'rag' / 'graph' から最も適切な情報源を一つ選んで 'source' に出力してください。",
        "from_db": "DBからクエリに対応する短いテキストを返してください（例: メールアドレスなど）。",
        "from_rag": "RAG（外部知識）からクエリに対応する短い要約テキストを返してください。",
        "from_graph": "GraphDBからクエリに対応する関係情報を短く記述してください。",
        "rerank": "与えられた候補（db_text, rag_text, graph_text）の中から、最も質問に適合する一つを 'text' に出力してください。",
    }

    def __init__(self):
        module = __import__("real_world.simple_gepa_routed_sources", fromlist=["RoutedSources"])
        routed_cls = module.RoutedSources
        self.program: dspy.Module = routed_cls()
        for name, instruction in self._instructions.items():
            predictor = getattr(self.program, name)
            predictor.signature = predictor.signature.with_instructions(instruction)
        self._load_optimized_state()

    def _predict(self, request: ProgramRequest) -> ProgramResponse:
        prompt = request.prompt.strip() or _latest_user_prompt(request.history, "")
        if not prompt:
            return ProgramResponse(text="（入力が空です）")
        with _lm_scope(request.lm):
            final = self.program(query=prompt)
        text = getattr(final, "text", "")
        if not text:
            text = "（回答が生成されませんでした）"
        return ProgramResponse(text=text, details={"program": self.slug}, raw=final)


class SimpleGEPAVectorRAGProgram(_BaseProgram):
    slug = "simple_gepa_vector_rag"
    display_name = "Simple GEPA Vector RAG"
    description = "Rewrite→Vector検索→Answer の最小RAGパイプライン（TF-IDF擬似ベクトルDB付き）。"
    input_hint = "ナレッジベースから答えを探したい質問を入力してください。"
    artifact_prefix = "simple_gepa_vector_rag"

    _rewrite_instruction = "質問の要点（固有名詞/同義語）を補いながら簡潔に再表現してください。"
    _answer_instruction = "与えられたパッセージ(passages)のみを根拠に、短く正確に回答してください。根拠の表現は可能な限り引用してください。"

    def __init__(self):
        module = __import__("real_world.simple_gepa_vector_rag", fromlist=["VectorRAG", "Document"])
        vector_cls = module.VectorRAG
        document_cls = module.Document
        adapter_cls = module.InMemoryTfIdfAdapter
        self.adapter = adapter_cls()
        docs = [
            document_cls(id="d1", text="The capital of Japan is Tokyo."),
            document_cls(id="d2", text="Mount Fuji is near Tokyo in Japan."),
        ]
        self.adapter.upsert(docs)
        self.program: dspy.Module = vector_cls(adapter=self.adapter, top_k=3)
        self.program.rewrite.signature = self.program.rewrite.signature.with_instructions(self._rewrite_instruction)
        self.program.answer.signature = self.program.answer.signature.with_instructions(self._answer_instruction)
        self._load_optimized_state()

    def _predict(self, request: ProgramRequest) -> ProgramResponse:
        prompt = request.prompt.strip() or _latest_user_prompt(request.history, "")
        if not prompt:
            return ProgramResponse(text="（入力が空です）")
        with _lm_scope(request.lm):
            pred = self.program(question=prompt)
        answer = getattr(pred, "answer", "")
        text = answer if answer else "（回答が生成されませんでした）"
        return ProgramResponse(text=text, details={"program": self.slug}, raw=pred)


class SimpleGEPAVectorRAGLoCoMoProgram(_BaseProgram):
    slug = "simple_gepa_vector_rag_locomo"
    display_name = "Simple GEPA Vector RAG LoCoMo"
    description = "LoCoMo10 データセットから対話ログをTF-IDFにインデックスし、質問応答するRAGサンプル。"
    input_hint = "LoCoMo対話に関する質問を入力してください。"
    artifact_prefix = "simple_gepa_vector_rag_locomo"

    _rewrite_instruction = "固有名詞・話題語を保持しつつ、検索に適した形に簡潔に言い換えてください。"
    _answer_instruction = (
        "passages に含まれる情報のみを根拠に、短く正確に回答してください。必要に応じて引用を明示してください。"
    )

    def __init__(self):
        module = __import__(
            "real_world.simple_gepa_vector_rag_locomo",
            fromlist=["VectorRAG", "InMemoryTfIdfAdapter", "load_locomo10", "_flatten_locomo_item_to_docs"],
        )
        vector_cls = module.VectorRAG
        adapter_cls = module.InMemoryTfIdfAdapter
        load_locomo10 = module.load_locomo10
        flatten_docs = module._flatten_locomo_item_to_docs
        dataset_path = Path("real_world/assets/datasets/locomo10.json")
        if not dataset_path.exists():
            raise FileNotFoundError(
                "LoCoMo10 dataset not found at real_world/assets/datasets/locomo10.json. "
                "Provide the dataset to enable this adapter."
            )
        data = load_locomo10(str(dataset_path))
        if not data:
            raise ValueError("LoCoMo10 dataset is empty.")
        item = data[0]
        docs = flatten_docs(item)
        adapter = adapter_cls()
        adapter.upsert(docs)
        self.program: dspy.Module = vector_cls(adapter=adapter, top_k=5)
        self.program.rewrite.signature = self.program.rewrite.signature.with_instructions(self._rewrite_instruction)
        self.program.answer.signature = self.program.answer.signature.with_instructions(self._answer_instruction)
        self._load_optimized_state()

    def _predict(self, request: ProgramRequest) -> ProgramResponse:
        prompt = request.prompt.strip() or _latest_user_prompt(request.history, "")
        if not prompt:
            return ProgramResponse(text="（入力が空です）")
        with _lm_scope(request.lm):
            pred = self.program(question=prompt)
        answer = getattr(pred, "answer", "")
        text = answer if answer else "（回答が生成されませんでした）"
        return ProgramResponse(text=text, details={"program": self.slug}, raw=pred)


class SimpleGEPALangExtractProgram(_BaseProgram):
    slug = "simple_gepa_langextract"
    display_name = "Simple GEPA LangExtract Prompt Builder"
    description = "langextract 用プロンプト/例を生成し、簡易フォールバック推論で抽出結果を返すサンプル。"
    input_hint = "対象テキストを貼り付けてください。抽出されたエンティティを箇条書きで返します。"
    artifact_prefix = "simple_gepa_langextract"

    def __init__(self):
        module = __import__("real_world.simple_gepa_langextract", fromlist=["LangExtractPipeline"])
        pipeline_cls = module.LangExtractPipeline
        # use_fallback=True で langextract 非依存のナイーブ抽出を行う
        self.program: dspy.Module = pipeline_cls(use_fallback=True)
        self._load_optimized_state()

    def _predict(self, request: ProgramRequest) -> ProgramResponse:
        prompt = request.prompt.strip() or _latest_user_prompt(request.history, "")
        if not prompt:
            return ProgramResponse(text="（入力が空です）")
        opts = dict(request.options or {})
        call_kwargs = {
            "text": opts.get("text") or prompt,
            "task": opts.get("task"),
            "target_classes": opts.get("target_classes"),
            "style_hints": opts.get("style_hints"),
        }
        with _lm_scope(request.lm):
            pred = self.program(**call_kwargs)
        extractions = list(getattr(pred, "extractions", []) or [])
        if not extractions:
            text = "（抽出結果なし）"
        else:
            lines = []
            for item in extractions:
                cls = item.get("extraction_class", "")
                span = item.get("extraction_text", "")
                attrs = item.get("attributes") or {}
                suffix = ""
                if attrs:
                    attr_text = ", ".join(f"{k}={v}" for k, v in attrs.items())
                    suffix = f" ({attr_text})"
                lines.append(f"- [{cls}] {span}{suffix}")
            text = "\n".join(lines)
        details: MutableMapping[str, Any] = {"program": self.slug}
        details["prompt_description"] = getattr(pred, "prompt_description", "")
        details["examples_json"] = getattr(pred, "examples_json", "")
        return ProgramResponse(text=text, details=details, raw=pred)


class UnsupportedProgram(_BaseProgram):
    """Placeholder for demos that require additional UI modalities (e.g., images)."""

    def __init__(self, slug: str, display_name: str, reason: str):
        self.slug = slug
        self.display_name = display_name
        self.description = reason
        self.input_hint = reason

    def _predict(self, request: ProgramRequest) -> ProgramResponse:
        reason = f"{self.display_name} は現在のテキストチャットUIでは利用できません。{self.description}"
        return ProgramResponse(text=reason)


# ----------------------------
# Registry
# ----------------------------


@dataclass(frozen=True)
class ProgramDescriptor:
    slug: str
    display_name: str
    description: str
    builder: Callable[[], ChatProgram]
    modalities: frozenset[str] = frozenset({"text"})
    available: bool = True
    note: str | None = None


_DESCRIPTORS: dict[str, ProgramDescriptor] = {}
_INSTANCES: dict[str, ChatProgram] = {}


def _register(descriptor: ProgramDescriptor) -> None:
    if descriptor.slug in _DESCRIPTORS:
        raise ValueError(f"Duplicate program slug: {descriptor.slug}")
    _DESCRIPTORS[descriptor.slug] = descriptor


def list_programs() -> Sequence[ProgramDescriptor]:
    return list(_DESCRIPTORS.values())


def get_program(slug: str) -> ChatProgram:
    descriptor = _DESCRIPTORS.get(slug)
    if descriptor is None:
        raise KeyError(f"Unknown program slug: {slug}")
    if not descriptor.available:
        note = descriptor.note or "Program is marked unavailable."
        raise RuntimeError(f"Program '{slug}' is not available: {note}")
    if slug not in _INSTANCES:
        _INSTANCES[slug] = descriptor.builder()
    return _INSTANCES[slug]


# Register text-ready adapters
_register(
    ProgramDescriptor(
        slug=SimpleGEPABasicProgram.slug,
        display_name=SimpleGEPABasicProgram.display_name,
        description=SimpleGEPABasicProgram.description,
        builder=SimpleGEPABasicProgram,
    )
)
_register(
    ProgramDescriptor(
        slug=SimpleGEPATaskMetricProgram.slug,
        display_name=SimpleGEPATaskMetricProgram.display_name,
        description=SimpleGEPATaskMetricProgram.description,
        builder=SimpleGEPATaskMetricProgram,
    )
)
_register(
    ProgramDescriptor(
        slug=SimpleGEPAInvoiceProgram.slug,
        display_name=SimpleGEPAInvoiceProgram.display_name,
        description=SimpleGEPAInvoiceProgram.description,
        builder=SimpleGEPAInvoiceProgram,
    )
)
_register(
    ProgramDescriptor(
        slug=SimpleGEPARoutedSourcesProgram.slug,
        display_name=SimpleGEPARoutedSourcesProgram.display_name,
        description=SimpleGEPARoutedSourcesProgram.description,
        builder=SimpleGEPARoutedSourcesProgram,
    )
)
_register(
    ProgramDescriptor(
        slug=SimpleGEPAVectorRAGProgram.slug,
        display_name=SimpleGEPAVectorRAGProgram.display_name,
        description=SimpleGEPAVectorRAGProgram.description,
        builder=SimpleGEPAVectorRAGProgram,
    )
)
_register(
    ProgramDescriptor(
        slug=SimpleGEPAVectorRAGLoCoMoProgram.slug,
        display_name=SimpleGEPAVectorRAGLoCoMoProgram.display_name,
        description=SimpleGEPAVectorRAGLoCoMoProgram.description,
        builder=SimpleGEPAVectorRAGLoCoMoProgram,
    )
)
_register(
    ProgramDescriptor(
        slug=SimpleGEPALangExtractProgram.slug,
        display_name=SimpleGEPALangExtractProgram.display_name,
        description=SimpleGEPALangExtractProgram.description,
        builder=SimpleGEPALangExtractProgram,
    )
)

# Register placeholders for multimodal demos (currently unavailable in text chat)
_register(
    ProgramDescriptor(
        slug="simple_gepa_multimodal_caption",
        display_name="Simple GEPA Multimodal Caption",
        description="画像アップロードが必要なため、テキストのみのチャットUIでは未対応です。",
        builder=lambda: UnsupportedProgram(
            slug="simple_gepa_multimodal_caption",
            display_name="Simple GEPA Multimodal Caption",
            reason="画像を入力できるUIを追加してください。",
        ),
        modalities=frozenset({"image"}),
        available=False,
        note="Requires image upload component.",
    )
)
_register(
    ProgramDescriptor(
        slug="simple_gepa_multimodal_observe_compose",
        display_name="Simple GEPA Multimodal Observe & Compose",
        description="画像＋テキストの複合入力が必要なため、現行UIでは未対応です。",
        builder=lambda: UnsupportedProgram(
            slug="simple_gepa_multimodal_observe_compose",
            display_name="Simple GEPA Multimodal Observe & Compose",
            reason="画像入力と複数フィールドが必要です。",
        ),
        modalities=frozenset({"image", "text"}),
        available=False,
        note="Requires image upload + structured inputs.",
    )
)
