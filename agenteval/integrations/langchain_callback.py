"""LangChain callback handler for auto-recording traces.

Drop-in callback that captures all tool calls and LLM responses
from any LangChain agent/chain into an agenteval trace.

Usage:
    from agenteval.integrations.langchain_callback import AgentEvalCallback

    cb = AgentEvalCallback()
    agent.invoke({"input": "query"}, config={"callbacks": [cb]})
    trace = cb.trace
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from ..recorder import AgentRecorder


class AgentEvalCallback:
    """LangChain callback handler that records to an agenteval trace.

    Compatible with both LangChain and LangGraph. Pass as a callback
    to any chain, agent, or graph invocation.
    """

    def __init__(self, name: str = "langchain_agent"):
        self._recorder = AgentRecorder(name=name)
        self._recorder.start()
        self._tool_starts: dict[str, float] = {}
        self._llm_starts: dict[str, tuple[float, str]] = {}

    @property
    def trace(self):
        return self._recorder.trace

    def stop(self, output: str = ""):
        return self._recorder.stop(output=output)

    # --- LLM callbacks ---

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *,
        run_id: UUID, **kwargs: Any,
    ) -> None:
        prompt = prompts[0] if prompts else ""
        self._llm_starts[str(run_id)] = (time.time(), prompt)

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list, *,
        run_id: UUID, **kwargs: Any,
    ) -> None:
        prompt = ""
        if messages and messages[0]:
            last = messages[0][-1] if isinstance(messages[0], list) else messages[0]
            prompt = str(getattr(last, "content", last))
        self._llm_starts[str(run_id)] = (time.time(), prompt[:2000])

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run_key = str(run_id)
        t0, prompt = self._llm_starts.pop(run_key, (time.time(), ""))
        duration = (time.time() - t0) * 1000

        model = ""
        response_text = ""
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, "llm_output") and response.llm_output:
            model = response.llm_output.get("model_name", "")
            token_usage = response.llm_output.get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

        generations = getattr(response, "generations", [])
        if generations and generations[0]:
            gen = generations[0][0]
            response_text = getattr(gen, "text", "")
            if not response_text:
                msg = getattr(gen, "message", None)
                if msg:
                    response_text = str(getattr(msg, "content", ""))

                    # Check for tool calls in the message
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                            args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                            self._recorder.record_tool_call(name=name, args=args)

        self._recorder.record_llm_response(
            model=model,
            prompt=prompt[:2000],
            response=response_text[:2000],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._llm_starts.pop(str(run_id), None)

    # --- Tool callbacks ---

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *,
        run_id: UUID, **kwargs: Any,
    ) -> None:
        self._tool_starts[str(run_id)] = time.time()

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        run_key = str(run_id)
        t0 = self._tool_starts.pop(run_key, time.time())
        duration = (time.time() - t0) * 1000

        name = kwargs.get("name", "unknown_tool")
        self._recorder.record_tool_call(
            name=name,
            result=output[:2000] if isinstance(output, str) else str(output)[:2000],
            duration_ms=duration,
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        run_key = str(run_id)
        t0 = self._tool_starts.pop(run_key, time.time())
        duration = (time.time() - t0) * 1000

        name = kwargs.get("name", "unknown_tool")
        self._recorder.record_tool_call(
            name=name,
            error=str(error),
            duration_ms=duration,
        )

    # --- Chain callbacks (for capturing final output) ---

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        if "input" in inputs:
            self._recorder._trace.input = str(inputs["input"])[:2000]

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        if "output" in outputs:
            self._recorder._trace.output = str(outputs["output"])[:2000]

    # no-ops for other callbacks
    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self._recorder.record_error(str(error))

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        pass

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        output = getattr(finish, "return_values", {}).get("output", "")
        if output:
            self._recorder._trace.output = str(output)[:2000]
