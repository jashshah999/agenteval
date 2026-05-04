"""Test suite runner for agent evaluations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console
from rich.table import Table

from .assertions import AssertionResult, AssertionError


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float = 0.0
    assertions: list[AssertionResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class SuiteResult:
    name: str
    results: list[TestResult] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    def print_summary(self) -> None:
        console = Console()
        table = Table(title=f"agenteval: {self.name}")
        table.add_column("Test", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Time", justify="right")
        table.add_column("Details")

        for r in self.results:
            status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            time_str = f"{r.duration_ms:.0f}ms"
            details = ""
            if r.error:
                details = f"[red]{r.error[:80]}[/red]"
            elif not r.passed:
                failed = [a for a in r.assertions if not a.passed]
                if failed:
                    details = f"[red]{failed[0].message[:80]}[/red]"
            table.add_row(r.name, status, time_str, details)

        console.print(table)
        console.print(
            f"\n[bold]{self.passed}/{self.total} passed[/bold] "
            f"in {self.duration_ms:.0f}ms"
        )


_registered_tests: list[tuple[str, Callable]] = []


def agent_test(name: str | None = None) -> Callable:
    """Decorator to register a function as an agent test.

    The function should accept no arguments and return a Trace.
    Assertions should be called inside the function body.
    """
    def decorator(fn: Callable) -> Callable:
        test_name = name or fn.__name__
        _registered_tests.append((test_name, fn))
        return fn
    return decorator


class AgentTestSuite:
    """Collects and runs agent tests."""

    def __init__(self, name: str = "Agent Tests"):
        self.name = name
        self.tests: list[tuple[str, Callable]] = []

    def add_test(self, name: str, fn: Callable) -> None:
        self.tests.append((name, fn))

    def add_tests_from_registry(self) -> None:
        self.tests.extend(_registered_tests)

    def run(self, verbose: bool = True) -> SuiteResult:
        suite_start = time.time()
        results = []

        for test_name, test_fn in self.tests:
            t0 = time.time()
            assertions = []
            error = None
            passed = True

            try:
                test_fn()
            except AssertionError as e:
                passed = False
                assertions.append(e.result)
            except Exception as e:
                passed = False
                error = str(e)

            duration = (time.time() - t0) * 1000
            results.append(TestResult(
                name=test_name,
                passed=passed,
                duration_ms=duration,
                assertions=assertions,
                error=error,
            ))

        suite_result = SuiteResult(
            name=self.name,
            results=results,
            duration_ms=(time.time() - suite_start) * 1000,
        )

        if verbose:
            suite_result.print_summary()

        return suite_result
