"""Auto-instrumentation for Anthropic SDK.

Monkey-patches the Anthropic client to automatically capture all LLM calls
and tool use into agenteval traces.

Usage:
    from agenteval.integrations.anthropic_patch import patch_anthropic

    recorder = patch_anthropic()
    # ... use Anthropic as normal ...
    trace = recorder.trace
"""

from __future__ import annotations

import time
from typing import Any

from ..recorder import AgentRecorder


_original_create = None
_active_recorder: AgentRecorder | None = None


def patch_anthropic(recorder: AgentRecorder | None = None) -> AgentRecorder:
    """Patch the Anthropic SDK to auto-record all calls."""
    global _original_create, _active_recorder

    if recorder is None:
        recorder = AgentRecorder(name="anthropic_session")
        recorder.start()
    _active_recorder = recorder

    try:
        from anthropic.resources.messages import Messages
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    if _original_create is None:
        _original_create = Messages.create

    def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        response = _original_create(self, *args, **kwargs)
        duration = (time.time() - t0) * 1000

        if _active_recorder is not None:
            _record_message(kwargs, response, duration)

        return response

    Messages.create = patched_create
    return recorder


def unpatch_anthropic() -> None:
    """Restore the original Anthropic SDK methods."""
    global _original_create, _active_recorder

    if _original_create is not None:
        from anthropic.resources.messages import Messages
        Messages.create = _original_create
        _original_create = None

    _active_recorder = None


def _record_message(kwargs: dict, response: Any, duration_ms: float) -> None:
    rec = _active_recorder
    if rec is None:
        return

    model = getattr(response, "model", kwargs.get("model", ""))
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    # Extract response text and tool use
    content_blocks = getattr(response, "content", [])
    response_text = ""
    for block in content_blocks:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            response_text += getattr(block, "text", "")
        elif block_type == "tool_use":
            name = getattr(block, "name", "unknown")
            tool_input = getattr(block, "input", {})
            rec.record_tool_call(name=name, args=tool_input, duration_ms=duration_ms)

    # Extract prompt
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
