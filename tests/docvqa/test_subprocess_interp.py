"""Tests for the vendored subprocess interpreter."""
import pytest

from docvqa.subprocess_interp import (
    SubprocessInterpreter, FinalOutput, CodeInterpreterError,
)


def test_state_persists_across_executes():
    interp = SubprocessInterpreter()
    try:
        interp.execute("x = 7")
        out = interp.execute("print(x + 3)")
        assert "10" in out
    finally:
        interp.shutdown()


def test_submit_returns_final_output():
    interp = SubprocessInterpreter(output_fields=[{"name": "answer", "type": "str"}])
    try:
        result = interp.execute("SUBMIT(answer='42')")
        # SubprocessInterpreter.execute returns (FinalOutput, captured_stdout) on SUBMIT
        assert isinstance(result, tuple)
        final, _captured = result
        assert isinstance(final, FinalOutput)
        assert final.output == {"answer": "42"}
    finally:
        interp.shutdown()


def test_tool_call_round_trip():
    """Host-registered tool should be callable from subprocess code via IPC."""
    calls = []

    def echo_tool(s: str) -> str:
        calls.append(s)
        return s.upper()

    interp = SubprocessInterpreter(tools={"echo_tool": echo_tool})
    try:
        out = interp.execute("print(echo_tool('hello'))")
        assert "HELLO" in out
        assert calls == ["hello"]
    finally:
        interp.shutdown()


def test_runtime_error_returns_error_marker():
    interp = SubprocessInterpreter()
    try:
        try:
            out = interp.execute("1 / 0")
            # Either form is acceptable: returns string with marker, or raises.
            assert "ZeroDivisionError" in str(out)
        except CodeInterpreterError as e:
            assert "ZeroDivisionError" in str(e)
    finally:
        interp.shutdown()


def test_sandbox_code_runs_at_startup():
    interp = SubprocessInterpreter(sandbox_code="GREETING = 'hi'")
    try:
        out = interp.execute("print(GREETING)")
        assert "hi" in out
    finally:
        interp.shutdown()


def test_shutdown_is_idempotent():
    interp = SubprocessInterpreter()
    interp.start()
    interp.shutdown()
    interp.shutdown()  # must not raise


def test_extra_env_is_passed_to_subprocess():
    """The extra_env dict added in vendoring must reach the subprocess's os.environ."""
    interp = SubprocessInterpreter(extra_env={"DOCVQA_TEST_VAR": "hello-from-host"})
    try:
        out = interp.execute("import os; print(os.environ.get('DOCVQA_TEST_VAR', 'MISSING'))")
        assert "hello-from-host" in out
    finally:
        interp.shutdown()
