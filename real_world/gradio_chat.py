"""
Lightweight Gradio chat UI that can switch between existing LM backends and capture feedback.

Features
--------
- Dropdown to choose between supported DSPy LMs (OpenAI helpers or a local dummy echo).
- Optional system prompt plus temperature / max_tokens controls for quick experiments.
- Use inline 👍 / 👎 controls to label the latest assistant response and maintain a feedback table.
- Runs entirely as a Gradio Blocks app so it can be launched via CLI.

Usage
-----
    uv run python real_world/gradio_chat.py

Command line flags allow changing host/port/share options and the default backend.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import tempfile
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Literal

import gradio as gr
from loguru import logger

import dspy
from real_world.gepa_chat_programs import (
    ProgramRequest,
    get_program,
    list_programs,
)
from real_world.helper import openai_lm

# Chatbot message representation (OpenAI style).
Message = dict[str, str]
History = list[Message]

# ----------------------------
# Backend management
# ----------------------------


def _require_env(var: str) -> None:
    if not os.getenv(var):
        raise RuntimeError(f"{var} is not set. Please export {var} (or select another backend) to use this option.")


class ChatBackend:
    """Factory/cache wrapper for supported LMs plus a deterministic dummy reply."""

    def __init__(self):
        # Keep order stable for dropdown display.
        self._builders: OrderedDict[str, Any] = OrderedDict(
            [
                ("dummy (offline echo)", None),
                ("openai/gpt-4o-mini", partial(self._build_openai_lm, "gpt-4o-mini")),
                ("openai/gpt-4o", partial(self._build_openai_lm, "gpt-4o")),
            ]
        )
        self._cache: dict[str, dspy.LM] = {}

    @staticmethod
    def _build_openai_lm(model: str) -> dspy.LM:
        _require_env("OPENAI_API_KEY")
        # openai_lm caches instances internally; we keep a narrow cache here as well.
        return openai_lm(model)

    @property
    def options(self) -> list[str]:
        return list(self._builders.keys())

    def require_lm(self, key: str) -> dspy.LM:
        """Return a DSPy LM instance for program integration."""
        if key == "dummy (offline echo)":
            raise ValueError("Dummy backend does not provide a DSPy LM instance.")
        return self._get_lm(key)

    def _get_lm(self, key: str) -> dspy.LM:
        if key not in self._builders:
            raise ValueError(f"Unsupported backend: {key}")
        if key == "dummy (offline echo)":
            raise ValueError("Dummy backend does not expose a DSPy LM.")
        if key not in self._cache:
            builder = self._builders[key]
            if builder is None:
                raise ValueError(f"Backend {key} has no builder.")
            self._cache[key] = builder()
            logger.info("Instantiated backend {}", key)
        return self._cache[key]

    def generate(
        self,
        backend_key: str,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if backend_key == "dummy (offline echo)":
            user_utterance = ""
            for msg in reversed(list(messages)):
                if msg.get("role") == "user":
                    user_utterance = msg.get("content", "")
                    break
            reply = f"（ダミー応答）{user_utterance.strip() or '入力が空です。'}"
            logger.debug("Dummy backend reply: {}", reply)
            return reply

        lm = self._get_lm(backend_key)
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)

        outputs = lm(messages=list(messages), **kwargs)
        if not outputs:
            return ""
        first = outputs[0]
        if isinstance(first, dict):
            text = first.get("text") or first.get("content")
        else:
            text = first
        text = text or ""
        logger.debug("Backend {} returned {} chars", backend_key, len(text))
        return str(text)


# ----------------------------
# Gradio UI helpers
# ----------------------------


def _build_messages(history: History, user_message: str, system_prompt: str | None) -> History:
    msgs: History = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_message})
    return msgs


LOG_HEADERS = ["timestamp", "user", "assistant", "label", "message", "history"]


def _log_rows(log: list[dict[str, str]]) -> list[list[str]]:
    return [[entry[h] for h in LOG_HEADERS] for entry in log]


def _format_history(history: History) -> str:
    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role:
            lines.append(f"{role.capitalize()}: {content}")
    return "\\n".join(lines)


def _sanitize_cell(value: str) -> str:
    return value.replace("\t", "    ").replace("\r", " ").replace("\n", "\\n")


def _resolve_theme(theme: str | None) -> Any | None:
    if not theme:
        return None
    if "/" in theme:
        return theme
    themes_module = getattr(gr, "themes", None)
    if themes_module is None:
        logger.warning("Gradio themes module unavailable; ignoring theme '{}'.", theme)
        return None
    candidate = getattr(themes_module, theme, None)
    if candidate is None:
        logger.warning("Unknown theme '{}'; using default theme.", theme)
        return None
    try:
        return candidate()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to initialize theme '{}': {}", theme, exc)
        return None


def build_app(default_backend: str | None = None, *, theme: str | None = None) -> gr.Blocks:
    manager = ChatBackend()
    default_choice = default_backend if default_backend in manager.options else manager.options[0]
    descriptors = [d for d in list_programs() if d.available]
    descriptor_map = {d.slug: d for d in descriptors}
    PROGRAM_NONE = "__lm_only__"

    resolved_theme = _resolve_theme(theme)
    blocks_kwargs: dict[str, Any] = {"title": "DSPy Chat Console"}
    if resolved_theme is not None:
        blocks_kwargs["theme"] = resolved_theme

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            """
            # DSPy Chat Console

            - Select an LM backend (OpenAI helpers or a deterministic offline dummy).
            - Optional system prompt + generation controls.
            - Use the 👍 / 👎 icons on each assistant reply to quickly label and create a dataset.
            """
        )

        backend_choice = gr.Dropdown(
            manager.options,
            value=default_choice,
            label="Backend",
            info="Choose which DSPy LM to invoke. Dummy mode requires no API keys.",
        )
        program_choice = gr.Dropdown(
            [PROGRAM_NONE] + [d.slug for d in descriptors],
            value=PROGRAM_NONE,
            label="DSPy Program",
            info="Select a pre-compiled GEPA program or fall back to raw LM chat.",
        )
        program_info = gr.Markdown("Raw LM chat (no DSPy program).")
        system_prompt = gr.Textbox(
            label="System prompt (optional)",
            placeholder="例: あなたは丁寧な日本語で回答するアシスタントです。",
            lines=2,
        )

        with gr.Row():
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.2,
                step=0.05,
                label="Temperature",
            )
            max_tokens = gr.Slider(
                minimum=16,
                maximum=2048,
                value=512,
                step=16,
                label="Max tokens",
            )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=420,
            type="messages",
        )
        default_placeholder = "メッセージを入力して送信してください。"
        user_input = gr.Textbox(
            label="User message",
            placeholder=default_placeholder,
            lines=2,
        )
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear conversation")

        def describe_program(slug: str):
            if slug == PROGRAM_NONE:
                return (
                    gr.update(value="Raw LM chat (no DSPy program)."),
                    gr.update(placeholder=default_placeholder),
                )
            descriptor = descriptor_map.get(slug)
            try:
                program = get_program(slug)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to load program {}: {}", slug, exc)
                return (
                    gr.update(value=f"[ERROR] プログラム {slug} のロードに失敗しました。{exc}"),
                    gr.update(placeholder=default_placeholder),
                )
            display = descriptor.display_name if descriptor else slug
            info_lines = [f"**{display}**"]
            if descriptor and descriptor.description:
                info_lines.append("")
                info_lines.append(descriptor.description)
            optimized_path = getattr(program, "optimized_path", None)
            if optimized_path:
                info_lines.append("")
                info_lines.append(f"使用中の最適化済みモデル: `{optimized_path}`")
            hint = getattr(program, "input_hint", default_placeholder)
            info_lines.append("")
            info_lines.append(f"入力ヒント: {hint}")
            return (
                gr.update(value="\n".join(info_lines)),
                gr.update(placeholder=hint or default_placeholder),
            )

        program_choice.change(
            describe_program,
            inputs=[program_choice],
            outputs=[program_info, user_input],
        )

        feedback_status = gr.Markdown("")
        feedback_table = gr.Dataframe(
            headers=LOG_HEADERS,
            value=[],
            datatype=["str"] * len(LOG_HEADERS),
            interactive=False,
            label="Feedback log",
        )
        feedback_state = gr.State([])  # list[dict[str, str]]
        export_btn = gr.Button("Export feedback TSV")
        export_file = gr.File(label="Download TSV", interactive=False)

        def respond(
            message: str,
            history: History,
            backend: str,
            sys_prompt: str,
            temp: float,
            max_tok: int,
            program_slug: str,
        ):
            if not message.strip():
                return history, ""
            try:
                history = history or []
                if program_slug and program_slug != PROGRAM_NONE:
                    lm = manager.require_lm(backend)
                    program = get_program(program_slug)
                    request = ProgramRequest(
                        prompt=message,
                        history=history,
                        lm=lm,
                        options={
                            "system_prompt": sys_prompt,
                            "temperature": temp,
                            "max_tokens": max_tok,
                            "backend": backend,
                        },
                    )
                    result = program.run(request)
                    reply = result.text
                else:
                    messages = _build_messages(history, message, sys_prompt)
                    reply = manager.generate(
                        backend,
                        messages,
                        temperature=temp,
                        max_tokens=max_tok,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Generation failed: {}", exc)
                reply = f"[ERROR] {exc}"
            updated = list(history) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            return updated, ""

        send_btn.click(
            respond,
            inputs=[
                user_input,
                chatbot,
                backend_choice,
                system_prompt,
                temperature,
                max_tokens,
                program_choice,
            ],
            outputs=[chatbot, user_input],
        )

        def clear():
            return [], [], gr.update(value=[]), gr.update(value=""), gr.update(value=None)

        clear_btn.click(
            clear,
            inputs=None,
            outputs=[chatbot, feedback_state, feedback_table, feedback_status, export_file],
        )

        def record_feedback(
            label: Literal["good", "bad"],
            history: History,
            log_state: list[dict[str, str]],
            index: int | None = None,
        ):
            if not history:
                return (
                    log_state,
                    gr.update(value=_log_rows(log_state)),
                    gr.update(value="ラベル付け可能な応答がありません。"),
                )
            assistant_positions = [i for i, msg in enumerate(history) if msg.get("role") == "assistant"]
            if not assistant_positions:
                return (
                    log_state,
                    gr.update(value=_log_rows(log_state)),
                    gr.update(value="ラベル付け可能な応答がありません。"),
                )
            message_idx: int
            if index is None:
                message_idx = assistant_positions[-1]
            else:
                idx = int(index)
                if 0 <= idx < len(history) and history[idx].get("role") == "assistant":
                    message_idx = idx
                else:
                    if idx < 0:
                        idx = 0
                    if idx >= len(assistant_positions):
                        idx = len(assistant_positions) - 1
                    message_idx = assistant_positions[idx]
            assistant_reply = history[message_idx].get("content", "")
            if not assistant_reply:
                return (
                    log_state,
                    gr.update(value=_log_rows(log_state)),
                    gr.update(value="ラベル付け可能な応答がありません。"),
                )
            user_msg = ""
            for j in range(message_idx - 1, -1, -1):
                if history[j].get("role") == "user":
                    user_msg = history[j].get("content", "")
                    break
            history_text = _format_history(history)
            new_entry = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "user": user_msg,
                "assistant": assistant_reply,
                "label": label,
                "message": assistant_reply,
                "history": history_text,
            }
            new_log = list(log_state) + [new_entry]
            pair_idx = assistant_positions.index(message_idx) + 1
            msg = (
                f"応答 #{pair_idx} を 👍 Good として記録しました。"
                if label == "good"
                else f"応答 #{pair_idx} を 👎 Bad として記録しました。"
            )
            return new_log, gr.update(value=_log_rows(new_log)), gr.update(value=msg)

        def handle_like(data: gr.LikeData, history: History, log_state: list[dict[str, str]]):
            if data.liked is None:
                return (
                    log_state,
                    gr.update(value=_log_rows(log_state)),
                    gr.update(value="👍 / 👎 を選択してください。"),
                )
            label = "good" if data.liked else "bad"
            return record_feedback(label, history, log_state, getattr(data, "index", None))

        chatbot.like(
            handle_like,
            inputs=[chatbot, feedback_state],
            outputs=[feedback_state, feedback_table, feedback_status],
        )

        def export_feedback(log_state: list[dict[str, str]]):
            if not log_state:
                return None, gr.update(value="フィードバックがありません。")
            with tempfile.NamedTemporaryFile(
                "w",
                delete=False,
                encoding="utf-8",
                newline="",
                suffix=".tsv",
            ) as tmp:
                tmp.write("\t".join(LOG_HEADERS) + "\n")
                for entry in log_state:
                    row = "\t".join(_sanitize_cell(entry[h]) for h in LOG_HEADERS)
                    tmp.write(row + "\n")
                path = tmp.name
            filename = Path(path).name
            logger.info("Exported feedback TSV to {}", path)
            return path, gr.update(value=f"TSVを書き出しました: {filename}")

        export_btn.click(export_feedback, inputs=[feedback_state], outputs=[export_file, feedback_status])

    return demo


# ----------------------------
# CLI entrypoint
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the DSPy Gradio chat console.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL")
    parser.add_argument(
        "--default-backend",
        default=None,
        help="Initial backend selection (must match dropdown options).",
    )
    parser.add_argument("--queue", action="store_true", help="Enable Gradio queue() for concurrency.")
    parser.add_argument("--theme", default=None, help="Optional Gradio theme identifier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_app(default_backend=args.default_backend, theme=args.theme)
    if args.queue:
        demo = demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
