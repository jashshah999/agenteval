"""Auto-instrumentation for OpenAI SDK.

Monkey-patches the OpenAI client to automatically capture all LLM calls
and tool calls into agenteval traces. Zero code changes needed.

Usage:
    from agenteval.integrations.openai_patch import patch_openai

    recorder = patch_openai()  # patches globally, returns recorder
    # ... use OpenAI as normal ...
    trace = recorder.trace
"""

from __future__ import annotations

import time
import json
from typing import Any

from ..recorder import AgentRecorder


_original_create = None
_original_async_create = None
_active_recorder: AgentRecorder | None = None


def patch_openai(recorder: AgentRecorder | None = None) -> AgentRecorder:
    """Patch the OpenAI SDK to auto-record all calls.

    Args:
        recorder: Existing recorder to use. If None, creates a new one.

    Returns:
        The active AgentRecorder capturing all OpenAI calls.
    """
    global _original_create, _active_recorder

    if recorder is None:
        recorder = AgentRecorder(name="openai_session")
        recorder.start()
    _active_recorder = recorder

    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    if _original_create is None:
        _original_create = Completions.create

    def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        response = _original_create(self, *args, **kwargs)
        duration = (time.time() - t0) * 1000

        if _active_recorder is not None:
            _record_chat_completion(kwargs, response, duration)

        return response

    Completions.create = patched_create
    return recorder


def unpatch_openai() -> None:
    """Restore the original OpenAI SDK methods."""
    global _original_create, _active_recorder

    if _original_create is not None:
        from openai.resources.chat.completions import Completions
        Completions.create = _original_create
        _original_create = None

    _active_recorder = None


def _record_chat_completion(kwargs: dict, response: Any, duration_ms: float) -> None:
    rec = _active_recorder
    if rec is None:
        return

    model = getattr(response, "model", kwargs.get("model", ""))
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    choices = getattr(response, "choices", [])
    if not choices:
        return

    message = choices[0].message
    response_text = getattr(message, "content", "") or ""

    # Extract prompt from messages
    messages = kwargs.get("messages", [])
    prompt = ""
    if messages:
        last_user = [m for m in messages if m.get("role") == "user"]
        if last_user:
            content = last_user[-1].get("content", "")
            prompt = content if isinstance(content, str) else str(content)

    rec.record_llm_response(
        model=model,
        prompt=prompt[:2000],
        response=response_text[:2000],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
    )

    # Record tool calls if present
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn:
                name = getattr(fn, "name", "unknown")
                try:
                    args = json.loads(getattr(fn, "arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": getattr(fn, "arguments", "")}
                rec.record_tool_call(name=name, args=args, duration_ms=duration_ms)
