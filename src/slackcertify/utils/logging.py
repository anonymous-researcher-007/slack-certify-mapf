"""Rich-formatted logger factory for slackcertify.

The default level is ``INFO``; override at runtime by setting the
``SLACKCERTIFY_LOG_LEVEL`` environment variable to one of
``DEBUG``/``INFO``/``WARNING``/``ERROR``/``CRITICAL``.
"""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

__all__ = ["DEFAULT_LEVEL", "get_logger"]


DEFAULT_LEVEL: str = "INFO"

_HANDLER_INSTALLED: set[str] = set()


def get_logger(name: str = "slackcertify") -> logging.Logger:
    """Return a logger with a single :class:`rich.logging.RichHandler` attached.

    Subsequent calls with the same ``name`` reuse the existing handler
    so duplicate messages are not produced.

    Examples
    --------
    >>> log = get_logger("slackcertify.test")
    >>> log.name
    'slackcertify.test'
    """
    level_name = os.environ.get("SLACKCERTIFY_LOG_LEVEL", DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger(name)
    if name not in _HANDLER_INSTALLED:
        handler = RichHandler(
            level=level,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        _HANDLER_INSTALLED.add(name)
    logger.setLevel(level)
    return logger
