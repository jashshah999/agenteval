"""Record agent execution traces from various frameworks."""

from __future__ import annotations

import time
import functools
from contextlib import contextmanager
from typing import Any, Callable

from .trace import Trace, ToolCall, LLMResponse


class AgentRecorder:
    """Records agent execution into a Trace."""

    def __init__(self, name: str = "", input_text: str = ""):
        self._trace = Trace(name=name, input=input_text)
        self._start_time: float | None = None

    def start(self) -> None:
        self._start_time = time.time()

    def stop(self, output: str = "") -> Trace:
        if self._start_time:
            self._trace.duration_ms = (time.time() - self._start_time) * 1000
        self._trace.output = output
        return self._trace

    def record_tool_call(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        self._trace.add_tool_call(ToolCall(
            name=name,
            args=args or {},
            result=result,
            error=error,
            duration_ms=duration_ms,
        ))

    def record_llm_response(
        self,
        model: str = "",
        prompt: str = "",
        response: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        self._trace.add_llm_response(LLMResponse(
            model=model,
            prompt=prompt,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        ))

    def record_error(self, error: str) -> None:
        self._trace.error = error

    def set_metadata(self, key: str, value: Any) -> None:
        self._trace.metadata[key] = value

    @property
    def trace(self) -> Trace:
        return self._trace

    def wrap_tool(self, name: str) -> Callable:
        """Decorator to auto-record a function as a tool call."""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.time()
                try:
                    result = fn(*args, **kwargs)
                    self.record_tool_call(
                        name=name,
                        args=kwargs if kwargs else {"args": list(args)},
                        result=result,
                        duration_ms=(time.time() - t0) * 1000,
                    )
                    return result
                except Exception as e:
                    self.record_tool_call(
                        name=name,
                        args=kwargs if kwargs else {"args": list(args)},
                        error=str(e),
                        duration_ms=(time.time() - t0) * 1000,
                    )
                    raise
            return wrapper
        return decorator


@contextmanager
def record(name: str = "", input_text: str = ""):
    """Context manager that records an agent execution trace.

    Usage:
        with record("my_agent", input_text="query") as rec:
            # run your agent
            rec.record_tool_call("search", args={"q": "hello"}, result="world")
        trace = rec.trace
    """
    rec = AgentRecorder(name=name, input_text=input_text)
    rec.start()
    try:
        yield rec
    except Exception as e:
        rec.record_error(str(e))
        raise
    finally:
        rec.stop()
