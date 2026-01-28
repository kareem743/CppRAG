import logging
import time
from functools import wraps
from typing import Callable, Optional


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            )
        }
        parts = [f"{key}={value}" for key, value in {**base, **extra}.items()]
        if record.exc_info:
            parts.append("exc_info=true")
        return " ".join(parts)


def setup_logging(level: str, verbose: bool = False) -> None:
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else level.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter())
    logger.addHandler(handler)


def timed(name: Optional[str] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        label = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "timing",
                    extra={"event": "timing", "operation": label, "duration_ms": f"{duration_ms:.2f}"},
                )

        return wrapper

    return decorator
