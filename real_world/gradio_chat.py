"""
Lightweight Gradio chat UI that can switch between existing LM backends and capture feedback.

Features
--------
- Dropdown to choose between supported DSPy LMs (OpenAI helpers or a local dummy echo).
- Optional system prompt plus temperature / max_tokens controls for quick experiments.
- Good / Bad buttons to label the latest assistant response and maintain a feedback table.
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
from typing import Any, Iterable, List, Literal

import gradio as gr
from loguru import logger

import dspy
from real_world.helper import openai_lm

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
    def options(self) -> List[str]:
        return list(self._builders.keys())

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


def _build_messages(
    history: list[tuple[str, str | None]],
    user_message: str,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for user, assistant in history:
        msgs.append({"role": "user", "content": user})
        if assistant:
            msgs.append({"role": "assistant", "content": assistant})
    msgs.append({"role": "user", "content": user_message})
    return msgs


LOG_HEADERS = ["timestamp", "user", "assistant", "label", "message", "history"]


def _log_rows(log: list[dict[str, str]]) -> list[list[str]]:
    return [[entry[h] for h in LOG_HEADERS] for entry in log]


def _format_history(history: list[tuple[str, str | None]]) -> str:
    lines: list[str] = []
    for user_msg, assistant_msg in history:
        lines.append(f"User: {user_msg}")
        if assistant_msg:
            lines.append(f"Assistant: {assistant_msg}")
    return "\\n".join(lines)


def _sanitize_cell(value: str) -> str:
    return value.replace("\t", "    ").replace("\r", " ").replace("\n", "\\n")


def build_app(default_backend: str | None = None) -> gr.Blocks:
    manager = ChatBackend()
    default_choice = default_backend if default_backend in manager.options else manager.options[0]

    with gr.Blocks(title="DSPy Chat Console") as demo:
        gr.Markdown(
            """
            # DSPy Chat Console

            - Select an LM backend (OpenAI helpers or a deterministic offline dummy).
            - Optional system prompt + generation controls.
            - Label the most recent assistant reply as 👍 Good / 👎 Bad to build a quick dataset.
            """
        )

        backend_choice = gr.Dropdown(
            manager.options,
            value=default_choice,
            label="Backend",
            info="Choose which DSPy LM to invoke. Dummy mode requires no API keys.",
        )
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
        )
        user_input = gr.Textbox(
            label="User message",
            placeholder="メッセージを入力して送信してください。",
            lines=2,
        )
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear conversation")

        with gr.Row():
            mark_good = gr.Button("👍 Good", variant="secondary")
            mark_bad = gr.Button("👎 Bad", variant="secondary")
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
            history: list[tuple[str, str | None]],
            backend: str,
            sys_prompt: str,
            temp: float,
            max_tok: int,
        ):
            if not message.strip():
                return history, ""
            try:
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
            updated = history + [(message, reply)]
            return updated, ""

        send_btn.click(
            respond,
            inputs=[user_input, chatbot, backend_choice, system_prompt, temperature, max_tokens],
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
            history: list[tuple[str, str | None]],
            log_state: list[dict[str, str]],
        ):
            if not history or not history[-1][1]:
                return (
                    log_state,
                    gr.update(value=_log_rows(log_state)),
                    gr.update(value="ラベル付け可能な応答がありません。"),
                )
            assistant_reply = history[-1][1] or ""
            history_text = _format_history(history)
            new_entry = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "user": history[-1][0],
                "assistant": assistant_reply,
                "label": label,
                "message": assistant_reply,
                "history": history_text,
            }
            new_log = list(log_state) + [new_entry]
            msg = (
                "最新の応答を 👍 Good として記録しました。"
                if label == "good"
                else "最新の応答を 👎 Bad として記録しました。"
            )
            return new_log, gr.update(value=_log_rows(new_log)), gr.update(value=msg)

        mark_good.click(
            lambda h, s: record_feedback("good", h, s),
            inputs=[chatbot, feedback_state],
            outputs=[feedback_state, feedback_table, feedback_status],
        )
        mark_bad.click(
            lambda h, s: record_feedback("bad", h, s),
            inputs=[chatbot, feedback_state],
            outputs=[feedback_state, feedback_table, feedback_status],
        )

        def export_feedback(log_state: list[dict[str, str]]):
            if not log_state:
                return None, gr.update(value="フィードバックがありません。")
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="") as tmp:
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
    demo = build_app(default_backend=args.default_backend)
    if args.queue:
        demo = demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=False,
        theme=args.theme,
    )


if __name__ == "__main__":
    main()
