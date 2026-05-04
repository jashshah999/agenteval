"""Assertions for agent traces -- deterministic and LLM-as-judge."""

from __future__ import annotations

import re
from typing import Any, Callable

from .trace import Trace


class AssertionResult:
    def __init__(self, passed: bool, message: str = "", details: dict | None = None):
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"AssertionResult({status}: {self.message})"


class AssertionError(Exception):
    def __init__(self, result: AssertionResult):
        self.result = result
        super().__init__(result.message)


def _fail(msg: str, **details: Any) -> AssertionResult:
    result = AssertionResult(False, msg, details)
    raise AssertionError(result)


def _pass(msg: str = "", **details: Any) -> AssertionResult:
    return AssertionResult(True, msg, details)


# === Deterministic assertions ===

def assert_tool_called(trace: Trace, tool_name: str, min_times: int = 1) -> AssertionResult:
    """Assert a tool was called at least min_times."""
    count = sum(1 for tc in trace.tool_calls if tc.name == tool_name)
    if count >= min_times:
        return _pass(f"Tool '{tool_name}' called {count} times (>= {min_times})")
    return _fail(
        f"Tool '{tool_name}' called {count} times, expected >= {min_times}. "
        f"Tools called: {trace.tool_names}",
        tool_name=tool_name,
        actual_count=count,
        expected_min=min_times,
    )


def assert_tool_not_called(trace: Trace, tool_name: str) -> AssertionResult:
    """Assert a tool was never called."""
    count = sum(1 for tc in trace.tool_calls if tc.name == tool_name)
    if count == 0:
        return _pass(f"Tool '{tool_name}' was not called")
    return _fail(
        f"Tool '{tool_name}' was called {count} times, expected 0",
        tool_name=tool_name,
        actual_count=count,
    )


def assert_tool_called_with(
    trace: Trace, tool_name: str, args: dict[str, Any] | None = None, result: Any = None
) -> AssertionResult:
    """Assert a tool was called with specific args and/or returned a specific result."""
    matching = [tc for tc in trace.tool_calls if tc.name == tool_name]
    if not matching:
        return _fail(f"Tool '{tool_name}' was never called. Tools called: {trace.tool_names}")

    for tc in matching:
        args_match = True
        result_match = True

        if args is not None:
            for k, v in args.items():
                if tc.args.get(k) != v:
                    args_match = False
                    break

        if result is not None:
            result_match = tc.result == result

        if args_match and result_match:
            return _pass(f"Tool '{tool_name}' called with matching args/result")

    return _fail(
        f"Tool '{tool_name}' was called but no invocation matched. "
        f"Expected args={args}, result={result}. "
        f"Got: {[{'args': tc.args, 'result': tc.result} for tc in matching]}",
        tool_name=tool_name,
        expected_args=args,
        expected_result=result,
    )


def assert_tool_call_order(trace: Trace, expected_order: list[str]) -> AssertionResult:
    """Assert tools were called in a specific order (subsequence match)."""
    actual = trace.tool_names
    idx = 0
    for tool in actual:
        if idx < len(expected_order) and tool == expected_order[idx]:
            idx += 1
    if idx == len(expected_order):
        return _pass(f"Tool call order matches: {expected_order}")
    return _fail(
        f"Tool call order mismatch. Expected subsequence {expected_order}, "
        f"got {actual}. Matched up to index {idx} ({expected_order[:idx]})",
        expected=expected_order,
        actual=actual,
        matched_up_to=idx,
    )


def assert_output_contains(trace: Trace, substring: str) -> AssertionResult:
    """Assert the final output contains a substring."""
    if substring.lower() in trace.output.lower():
        return _pass(f"Output contains '{substring}'")
    return _fail(
        f"Output does not contain '{substring}'. "
        f"Output (first 500 chars): {trace.output[:500]}",
        expected_substring=substring,
    )


def assert_output_matches(trace: Trace, pattern: str) -> AssertionResult:
    """Assert the final output matches a regex pattern."""
    if re.search(pattern, trace.output, re.DOTALL):
        return _pass(f"Output matches pattern '{pattern}'")
    return _fail(
        f"Output does not match pattern '{pattern}'. "
        f"Output (first 500 chars): {trace.output[:500]}",
        pattern=pattern,
    )


def assert_steps_between(trace: Trace, min_steps: int = 1, max_steps: int = 100) -> AssertionResult:
    """Assert the number of steps is within bounds."""
    n = len(trace.steps)
    if min_steps <= n <= max_steps:
        return _pass(f"Step count {n} is within [{min_steps}, {max_steps}]")
    return _fail(
        f"Step count {n} is outside [{min_steps}, {max_steps}]",
        actual_steps=n,
        min_steps=min_steps,
        max_steps=max_steps,
    )


def assert_no_hallucination(
    trace: Trace,
    ground_truth: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> AssertionResult:
    """Assert the output doesn't hallucinate relative to ground truth (LLM judge)."""
    return assert_llm_judge(
        trace,
        criterion="The agent's output is factually consistent with the ground truth. "
        "It does not fabricate information, invent details, or contradict the ground truth. "
        "Minor omissions are OK, but additions that aren't in the ground truth are not.",
        context=f"Ground truth:\n{ground_truth}",
        model=model,
        provider=provider,
    )


def assert_llm_judge(
    trace: Trace,
    criterion: str,
    context: str = "",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    threshold: float = 0.7,
) -> AssertionResult:
    """Use an LLM to judge whether a trace meets a criterion.

    Args:
        trace: The agent execution trace.
        criterion: Natural language description of what to check.
        context: Additional context (ground truth, expected behavior, etc).
        model: LLM model to use for judging.
        provider: "openai" or "anthropic".
        threshold: Minimum score (0-1) to pass.
    """
    prompt = _build_judge_prompt(trace, criterion, context)

    try:
        score, reasoning = _call_judge(prompt, model, provider)
    except Exception as e:
        return _fail(f"LLM judge failed: {e}")

    if score >= threshold:
        return _pass(
            f"LLM judge score: {score:.2f} (>= {threshold}). {reasoning}",
            score=score,
            reasoning=reasoning,
            criterion=criterion,
        )
    return _fail(
        f"LLM judge score: {score:.2f} (< {threshold}). {reasoning}",
        score=score,
        reasoning=reasoning,
        criterion=criterion,
    )


def assert_custom(trace: Trace, fn: Callable[[Trace], bool], message: str = "") -> AssertionResult:
    """Assert using a custom function."""
    try:
        result = fn(trace)
        if result:
            return _pass(message or "Custom assertion passed")
        return _fail(message or "Custom assertion failed")
    except Exception as e:
        return _fail(f"Custom assertion raised: {e}")


# === LLM Judge internals ===

def _build_judge_prompt(trace: Trace, criterion: str, context: str) -> str:
    tool_calls_str = ""
    for tc in trace.tool_calls:
        tool_calls_str += f"  - {tc.name}({tc.args}) -> {tc.result}\n"
        if tc.error:
            tool_calls_str += f"    ERROR: {tc.error}\n"

    return f"""You are an AI agent evaluator. Judge whether the agent's execution meets the given criterion.

## Agent Input
{trace.input}

## Agent Output
{trace.output}

## Tool Calls
{tool_calls_str or "  (none)"}

## Additional Context
{context or "(none)"}

## Criterion
{criterion}

## Instructions
Rate how well the agent meets the criterion on a scale of 0.0 to 1.0:
- 1.0 = perfectly meets the criterion
- 0.5 = partially meets the criterion
- 0.0 = completely fails the criterion

Respond in EXACTLY this format (no other text):
SCORE: <float between 0.0 and 1.0>
REASONING: <one sentence explanation>"""


def _call_judge(prompt: str, model: str, provider: str) -> tuple[float, str]:
    if provider == "openai":
        return _call_openai_judge(prompt, model)
    elif provider == "anthropic":
        return _call_anthropic_judge(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")


def _call_openai_judge(prompt: str, model: str) -> tuple[float, str]:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    return _parse_judge_response(response.choices[0].message.content)


def _call_anthropic_judge(prompt: str, model: str) -> tuple[float, str]:
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_judge_response(response.content[0].text)


def _parse_judge_response(text: str) -> tuple[float, str]:
    score = 0.0
    reasoning = ""

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return score, reasoning
