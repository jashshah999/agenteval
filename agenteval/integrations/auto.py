"""One-line auto-instrumentation for all supported providers.

Usage:
    from agenteval.integrations.auto import watch

    recorder = watch()
    # ... use OpenAI, Anthropic, LangChain as normal ...
    trace = recorder.trace

    # Assert on the auto-captured trace
    assert_tool_called(trace, "search")
"""

from __future__ import annotations

from ..recorder import AgentRecorder


def watch(name: str = "agent", providers: list[str] | None = None) -> AgentRecorder:
    """Auto-instrument all detected LLM providers. Zero config.

    Args:
        name: Name for the trace.
        providers: List of providers to patch. None = auto-detect.
            Options: "openai", "anthropic"

    Returns:
        AgentRecorder that's capturing all LLM calls.
    """
    recorder = AgentRecorder(name=name)
    recorder.start()
    patched = []

    if providers is None:
        providers = _detect_providers()

    for provider in providers:
        if provider == "openai":
            try:
                from .openai_patch import patch_openai
                patch_openai(recorder)
                patched.append("openai")
            except ImportError:
                pass
        elif provider == "anthropic":
            try:
                from .anthropic_patch import patch_anthropic
                patch_anthropic(recorder)
                patched.append("anthropic")
            except ImportError:
                pass

    if patched:
        print(f"agenteval: watching {', '.join(patched)}")
    else:
        print("agenteval: no providers detected. Use recorder.record_tool_call() manually.")

    return recorder


def unwatch() -> None:
    """Remove all patches."""
    try:
        from .openai_patch import unpatch_openai
        unpatch_openai()
    except (ImportError, Exception):
        pass
    try:
        from .anthropic_patch import unpatch_anthropic
        unpatch_anthropic()
    except (ImportError, Exception):
        pass


def _detect_providers() -> list[str]:
    providers = []
    try:
        import openai  # noqa: F401
        providers.append("openai")
    except ImportError:
        pass
    try:
        import anthropic  # noqa: F401
        providers.append("anthropic")
    except ImportError:
        pass
    return providers
