"""Structured logging utilities for Sherman.

This module provides the StructuredFormatter and default logging configuration
used by the Sherman daemon.

Logging Configuration:
    The `logging` key in agent.yaml is passed to `logging.config.dictConfig()`.
    If omitted, built-in defaults apply: INFO level to stderr, standard format.
    See `_DEFAULT_LOGGING` for the default configuration schema.
"""

import logging


_BUILTIN_LOG_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "process", "processName", "message", "asctime", "taskName",
}


class StructuredFormatter(logging.Formatter):
    """Formatter that appends extra log record fields as key=value pairs.

    Any field passed via ``extra=`` that is not a standard LogRecord attribute
    is appended to the formatted message, e.g.::

        logger.debug("tool call arguments", extra={"tool": "shell", "arguments": "ls"})
        # → "2026-04-23 12:00:00 DEBUG    sherman.agent_loop: tool call arguments  tool='shell' arguments='ls'"

    Reference this class in a YAML logging config with::

        formatters:
          structured:
            (): sherman.logging.StructuredFormatter
            format: "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    """

    def format(self, record: logging.LogRecord) -> str:
        # Snapshot extras before super().format() mutates the record
        # (it sets record.message, record.asctime, record.exc_text).
        # Skip private attrs (e.g. _msecs backing fields in CPython 3.13+).
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _BUILTIN_LOG_ATTRS and not k.startswith("_")
        }
        s = super().format(record)
        if extras:
            suffix = "  " + " ".join(f"{k}={v!r}" for k, v in extras.items())
            # Insert before the traceback block (first newline) so extras
            # appear on the message line, not after the traceback.
            first_nl = s.find("\n")
            if first_nl == -1:
                s += suffix
            else:
                s = s[:first_nl] + suffix + s[first_nl:]
        return s


# Default logging configuration applied when no `logging` section exists in
# agent.yaml. Passed directly to logging.config.dictConfig(). Key choices:
#   - Output to stderr (stdout reserved for structured output if needed)
#   - Sherman loggers at INFO (production operational level)
#   - Root logger at WARNING (suppresses noisy library debug output)
#   - disable_existing_loggers: False preserves loggers created at import time
#   - propagate: False on sherman logger prevents double-output through root
#
# Users can override by providing a `logging` section in agent.yaml that
# follows the same schema (log levels, file handlers, JSON formatters, etc.).
_DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "()": StructuredFormatter,
            "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "sherman": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}
