"""
Common CLI and logging helpers for real_world simple_* demos.

Usage in a script:
    from real_world.cli import add_standard_args, setup_logging

    parser = argparse.ArgumentParser()
    add_standard_args(parser, default_save_prefix="simple_gepa_basic")
    # add script-specific args here
    args = parser.parse_args()

    setup_logging(args.log_level)
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger


def add_standard_args(
    parser: argparse.ArgumentParser,
    *,
    default_save_prefix: str,
    include_dummy: bool = True,
    include_logging: bool = True,
    include_save: bool = True,
) -> None:
    """Attach common CLI args used by simple_* demos.

    - --dummy: run with DummyLM (no external calls)
    - --log-level: loguru logging level
    - --save-dir / --save-prefix: artifacts destination
    """
    if include_dummy:
        parser.add_argument("--dummy", action="store_true", help="Use DummyLM (no external calls)")
    if include_logging:
        parser.add_argument("--log-level", default="INFO", help="Log level (e.g., DEBUG, INFO, WARNING)")
    if include_save:
        parser.add_argument("--save-dir", default="real_world/exports", help="Directory to save artifacts (.json)")
        parser.add_argument("--save-prefix", default=default_save_prefix, help="Filename prefix for saved artifacts")


def setup_logging(level: str | int) -> None:
    """Configure loguru to stderr with colors preserved when possible.

    Using a lambda/print sink strips Loguru's formatting and colorization.
    Route directly to sys.stderr so Loguru can handle coloring and formatting.
    """
    logger.remove()
    # Preserve colors when running in a TTY; otherwise fall back to plain text.
    colorize = sys.stderr.isatty()
    logger.add(sys.stderr, level=str(level).upper(), colorize=colorize)
