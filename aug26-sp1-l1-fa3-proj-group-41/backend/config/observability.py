"""No-op stand-ins for the old LangFuse hooks.

LangFuse is not part of this stack, but some agents still import these names.
They do nothing, so the agents run unchanged.
"""
from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler


def observe(*args, **kwargs):
    """Works both as @observe and @observe(name=...)."""
    if args and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(func):
        return func
    return decorator


class _NoOpContext:
    def update_current_trace(self, **_kwargs) -> None:
        pass

    def update_current_observation(self, **_kwargs) -> None:
        pass


langfuse_context = _NoOpContext()


def get_callback_handler(**_kwargs) -> BaseCallbackHandler:
    return BaseCallbackHandler()


def flush_langfuse() -> None:
    pass
