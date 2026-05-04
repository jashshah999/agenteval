"""Trace data model for agent executions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": "tool_call",
            "name": self.name,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ToolCall:
        return cls(
            name=d["name"],
            args=d.get("args", {}),
            result=d.get("result"),
            error=d.get("error"),
            duration_ms=d.get("duration_ms", 0),
            timestamp=d.get("timestamp", 0),
        )


@dataclass
class LLMResponse:
    model: str = ""
    prompt: str = ""
    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": "llm_response",
            "model": self.model,
            "prompt": self.prompt,
            "response": self.response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LLMResponse:
        return cls(
            model=d.get("model", ""),
            prompt=d.get("prompt", ""),
            response=d.get("response", ""),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            duration_ms=d.get("duration_ms", 0),
            timestamp=d.get("timestamp", 0),
        )


@dataclass
class TraceStep:
    step_type: str  # "tool_call", "llm_response", "custom"
    data: ToolCall | LLMResponse | dict = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        if isinstance(self.data, (ToolCall, LLMResponse)):
            data_dict = self.data.to_dict()
        else:
            data_dict = self.data
        return {
            "step_type": self.step_type,
            "data": data_dict,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TraceStep:
        step_type = d["step_type"]
        raw = d.get("data", {})
        if step_type == "tool_call":
            data = ToolCall.from_dict(raw)
        elif step_type == "llm_response":
            data = LLMResponse.from_dict(raw)
        else:
            data = raw
        return cls(step_type=step_type, data=data, metadata=d.get("metadata", {}))


@dataclass
class Trace:
    name: str = ""
    input: str = ""
    output: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    def add_tool_call(self, tool_call: ToolCall) -> None:
        self.steps.append(TraceStep(step_type="tool_call", data=tool_call))

    def add_llm_response(self, llm_response: LLMResponse) -> None:
        self.steps.append(TraceStep(step_type="llm_response", data=llm_response))

    def add_custom(self, data: dict, metadata: dict | None = None) -> None:
        self.steps.append(TraceStep(step_type="custom", data=data, metadata=metadata or {}))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.data for s in self.steps if s.step_type == "tool_call"]

    @property
    def llm_responses(self) -> list[LLMResponse]:
        return [s.data for s in self.steps if s.step_type == "llm_response"]

    @property
    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.llm_responses)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> Trace:
        return cls(
            name=d.get("name", ""),
            input=d.get("input", ""),
            output=d.get("output", ""),
            steps=[TraceStep.from_dict(s) for s in d.get("steps", [])],
            metadata=d.get("metadata", {}),
            duration_ms=d.get("duration_ms", 0),
            error=d.get("error"),
            timestamp=d.get("timestamp", 0),
        )

    @classmethod
    def from_json(cls, s: str) -> Trace:
        return cls.from_dict(json.loads(s))
