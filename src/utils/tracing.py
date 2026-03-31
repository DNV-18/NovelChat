"""Tracing helpers with graceful fallback when LangSmith is unavailable."""

from typing import Any, Callable


try:
    from langsmith import traceable as _langsmith_traceable
except Exception:  # pragma: no cover
    _langsmith_traceable = None


def traceable(*args: Any, **kwargs: Any) -> Callable:
    """Return LangSmith traceable decorator when available, otherwise no-op."""

    def _decorator(func: Callable) -> Callable:
        return func

    if _langsmith_traceable is None:
        return _decorator
    return _langsmith_traceable(*args, **kwargs)
