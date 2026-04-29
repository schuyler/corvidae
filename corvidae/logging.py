"""Structured logging utilities for Corvidae.

This module provides the StructuredFormatter and configure_logging() function
used by the Corvidae daemon.

Logging Configuration:
    Call `configure_logging()` to apply logging configuration from simplified
    options (level, file). Used by `main.py` during startup.
"""

import logging
import logging.config
import logging.handlers


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
        # → "2026-04-23 12:00:00 DEBUG    corvidae.agent_loop: tool call arguments  tool='shell' arguments='ls'"

    This formatter is applied automatically by ``configure_logging()``. It does
    not need to be referenced in YAML configuration.
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


_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def configure_logging(
    *,
    level: str = "INFO",
    file: str | None = None,
) -> None:
    """Build and apply logging configuration from simplified options.

    Args:
        level: Minimum log level for the corvidae logger.
               One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
               Case-insensitive.
        file: Path to the rotating log file. If set, logs go to that file
              via RotatingFileHandler. If None, logs go to stderr via
              StreamHandler. Relative paths resolve from the process working
              directory.

    Raises:
        ValueError: If level is not a recognized value.
    """
    level = level.upper()
    if level not in _VALID_LEVELS:
        raise ValueError(f"Invalid log level {level!r}. Must be one of {sorted(_VALID_LEVELS)}")

    if file is not None:
        handler_cfg = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "structured",
            "filename": file,
            "maxBytes": 10_485_760,   # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
    else:
        handler_cfg = {
            "class": "logging.StreamHandler",
            "formatter": "structured",
            "stream": "ext://sys.stderr",
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
                "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {"main": handler_cfg},
        "loggers": {
            "corvidae": {
                "level": level,
                "handlers": ["main"],
                "propagate": False,
            },
        },
        "root": {
            "level": "WARNING",
            "handlers": ["main"],
        },
    }
    logging.config.dictConfig(config)
