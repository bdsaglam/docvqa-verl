"""Subprocess-based Python interpreter (vendored from docvqa/rlm).

Replaces the Deno/Pyodide-based PythonInterpreter with a persistent CPython
subprocess. Enables native usage of numpy/scipy and all installed packages
without WASM limitations.

Communication uses line-delimited JSON over stdin/stdout.

DSPy-specific bits (display(), RESET_HISTORY(), dspy_lm config) have been
stripped; this module has no dspy dependency.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import select
import subprocess
import sys
from typing import Any, Callable


class CodeInterpreterError(RuntimeError):
    """Raised by the interpreter when subprocess code fails or IPC misbehaves."""


class FinalOutput:
    """Marker for a successful SUBMIT() call.

    Attributes:
        output: dict[str, Any] — the kwargs passed to SUBMIT
                (e.g. {"answer": "42"}).
    """
    def __init__(self, output):
        self.output = output


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# REPL loop script (runs inside the subprocess)
# ---------------------------------------------------------------------------

_REPL_SCRIPT = r'''
import json
import sys
import io
import traceback

# ---- Signal exceptions ----

class _FinalOutputSignal(Exception):
    def __init__(self, data):
        self.data = data

def _make_json_serializable(obj):
    """Recursively convert numpy arrays and other non-JSON types to serializable forms."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj

# ---- IPC functions (use _pipe_stdout to bypass print capture) ----

_pipe_stdout = sys.stdout  # Save real pipe before any capture redirects it

def _ipc_send(msg):
    """Send JSON to host via the real pipe stdout, bypassing print capture."""
    _pipe_stdout.write(json.dumps(msg) + "\n")
    _pipe_stdout.flush()

def _ipc_recv():
    """Read JSON from host via stdin."""
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line.strip())

# ---- Tool proxy with inline IPC (returns values, supports multiple calls) ----

_tool_call_id = 0

def _make_tool_proxy(tool_name):
    """Create a proxy that calls a host-side tool via inline IPC.

    Unlike the old exception-based approach, this:
    - Returns the tool result as a value (can be assigned to variables)
    - Supports multiple tool calls per code block
    - Works with print(tool_call(...)) patterns
    """
    def proxy(*args, **kwargs):
        global _tool_call_id
        _tool_call_id += 1
        call_id = _tool_call_id
        # Send tool call request via pipe stdout (bypasses print capture)
        _ipc_send({"type": "tool_call", "id": call_id, "name": tool_name,
                    "args": {"args": list(args), "kwargs": kwargs}})
        # Block until host responds
        response = _ipc_recv()
        if response.get("error"):
            raise RuntimeError(f"Tool error ({tool_name}): {response['error']}")
        return response.get("result", "")
    proxy.__name__ = tool_name
    proxy.__qualname__ = tool_name
    return proxy

# ---- Sandbox globals ----

def SUBMIT(*args, **kwargs):
    """Submit final output and end the REPL session.
    Usage: SUBMIT(answer="your answer") or SUBMIT("your answer") if single output."""
    if args and not kwargs:
        # Single positional arg — map to the first (or only) output field
        if len(_output_field_names) == 1:
            kwargs = {_output_field_names[0]: args[0]}
        else:
            raise TypeError(
                f"SUBMIT() with positional args requires exactly 1 output field, "
                f"but there are {len(_output_field_names)}: {_output_field_names}. "
                f"Use SUBMIT({', '.join(n + '=...' for n in _output_field_names)})"
            )
    if not kwargs and _output_field_names:
        raise TypeError(
            f"SUBMIT() requires output values. "
            f"Use SUBMIT({', '.join(n + '=...' for n in _output_field_names)})"
        )
    raise _FinalOutputSignal(kwargs)

_output_field_names = []  # Populated from config at startup

namespace = {"SUBMIT": SUBMIT, "__builtins__": __builtins__}

# ---- Initialization ----

config = _ipc_recv()

# Set output field names for SUBMIT validation
for field_info in config.get("output_fields", []):
    _output_field_names.append(field_info["name"])

# Register tools as inline IPC proxies
for tool_info in config.get("tools", []):
    name = tool_info["name"]
    namespace[name] = _make_tool_proxy(name)

# Execute sandbox code if provided
if config.get("sandbox_code"):
    try:
        exec(config["sandbox_code"], namespace)
    except Exception as _sandbox_err:
        import traceback as _tb
        print(f"[sandbox_code error] {_sandbox_err}\n{''.join(_tb.format_exception(type(_sandbox_err), _sandbox_err, _sandbox_err.__traceback__))}", flush=True)

_ipc_send({"ready": True})

# ---- Main REPL loop ----

while True:
    try:
        msg = _ipc_recv()
    except (json.JSONDecodeError, EOFError):
        break

    if msg.get("shutdown"):
        break

    code = msg.get("code", "")
    variables = msg.get("variables", {})

    # Inject variables
    for k, v in variables.items():
        namespace[k] = v

    # Redirect print() to capture buffer; tool proxies use _pipe_stdout directly
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured

    try:
        exec(code, namespace)
        sys.stdout = old_stdout
        output = captured.getvalue()
        _ipc_send({"output": output})

    except _FinalOutputSignal as e:
        sys.stdout = old_stdout
        captured_output = captured.getvalue()
        serializable_data = _make_json_serializable(e.data)
        _ipc_send({"error": "FinalOutput", "errorType": "FinalOutput",
                    "errorArgs": [serializable_data], "output": captured_output})

    except SyntaxError as e:
        sys.stdout = old_stdout
        _ipc_send({"error": str(e), "errorType": "SyntaxError"})

    except Exception as e:
        sys.stdout = old_stdout
        tb = traceback.format_exc()
        _ipc_send({"error": tb, "errorType": type(e).__name__})
'''


# ---------------------------------------------------------------------------
# SubprocessInterpreter
# ---------------------------------------------------------------------------


class SubprocessInterpreter:
    """CodeInterpreter that runs a persistent CPython subprocess.

    Enables native usage of numpy/scipy and all installed packages.
    State persists across execute() calls. Communication is JSON-RPC
    over stdin/stdout.
    """

    def __init__(
        self,
        sandbox_code: str | None = None,
        tools: dict[str, Callable] | None = None,
        output_fields: list[dict] | None = None,
        allowed_modules: list[str] | None = None,
        timeout: float = 120.0,
        extra_env: dict[str, str] | None = None,
    ):
        """
        Args:
            sandbox_code: Python code to inject into the subprocess namespace at startup.
            tools: Dictionary of tool name -> callable functions available via IPC.
            output_fields: Output field definitions for typed SUBMIT signature.
            allowed_modules: Additional modules the subprocess is allowed to import.
            timeout: Per-execution timeout in seconds.
            extra_env: Additional environment variables to pass to the subprocess.
        """
        self._sandbox_code = sandbox_code
        self._tools: dict[str, Callable] = dict(tools or {})
        self._output_fields = output_fields
        self._allowed_modules = allowed_modules
        self._timeout = timeout
        self._extra_env = extra_env
        self._process: subprocess.Popen | None = None
        self._started = False
        # For compatibility with RLM's injection logic
        self._tools_registered = False

    @property
    def tools(self) -> dict[str, Callable]:
        return self._tools

    @tools.setter
    def tools(self, value: dict[str, Callable]) -> None:
        self._tools = value

    @property
    def output_fields(self) -> list[dict] | None:
        return self._output_fields

    @output_fields.setter
    def output_fields(self, value: list[dict] | None) -> None:
        self._output_fields = value

    def start(self) -> None:
        """Start the subprocess and configure it."""
        if self._started and self._process and self._process.poll() is None:
            return

        # Build environment - inherit current env plus any API keys
        env = os.environ.copy()
        if self._extra_env:
            env.update(self._extra_env)

        # Use a temp directory as cwd to avoid polluting the project root with
        # files created by agent code (e.g. image crops saved via PIL).
        import tempfile
        self._sandbox_tmpdir = tempfile.mkdtemp(prefix="rlm_sandbox_")

        self._process = subprocess.Popen(
            [sys.executable, "-c", _REPL_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=self._sandbox_tmpdir,
            bufsize=1,  # Line-buffered
        )

        # Send initial config
        config: dict[str, Any] = {}

        # Tools info (names + parameter info for proxy creation)
        if self._tools:
            tools_info = []
            for name, fn in self._tools.items():
                tools_info.append(
                    {
                        "name": name,
                        "parameters": self._extract_parameters(fn),
                    }
                )
            config["tools"] = tools_info

        # Output fields (for SUBMIT validation)
        if self._output_fields:
            config["output_fields"] = self._output_fields

        # Sandbox code
        if self._sandbox_code:
            config["sandbox_code"] = self._sandbox_code

        self._send(config)

        # Wait for ready signal
        response = self._recv()
        if not response.get("ready"):
            raise CodeInterpreterError(f"Subprocess failed to start: {response}")

        self._started = True
        self._tools_registered = True

    def shutdown(self) -> None:
        """Terminate the subprocess."""
        if self._process and self._process.poll() is None:
            try:
                self._send({"shutdown": True})
                assert self._process.stdin is not None
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
                self._process.wait()
        self._process = None
        self._started = False
        self._tools_registered = False
        # Clean up sandbox temp directory
        if hasattr(self, "_sandbox_tmpdir"):
            import shutil
            shutil.rmtree(self._sandbox_tmpdir, ignore_errors=True)

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code in the subprocess.

        Returns:
            - (FinalOutput, captured_stdout) tuple if SUBMIT() was called
            - str with captured stdout otherwise

        Raises:
            CodeInterpreterError: On runtime errors
            SyntaxError: On syntax errors
        """
        if not self._started:
            self.start()

        variables = variables or {}

        # Serialize variables (only simple JSON-compatible types)
        serialized_vars = {}
        for k, v in variables.items():
            serialized_vars[k] = self._serialize_value(k, v)

        self._send({"code": code, "variables": serialized_vars})

        # Read and handle messages until we get output
        while True:
            result = self._recv()

            # Defensive: a non-dict line is stray stdout that leaked onto the IPC pipe
            # (e.g. a bare float that escaped the print-capture redirect). It is NOT a
            # protocol message — skip it and read the next line rather than crash on
            # `.get()`. A single leaked line must never kill a multi-hour RL run.
            if not isinstance(result, dict):
                continue

            # Handle tool call requests
            if result.get("type") == "tool_call":
                self._handle_tool_call(result)
                continue

            # Handle errors
            if "error" in result:
                error_msg = result["error"]
                error_type = result.get("errorType", "RuntimeError")
                error_args = result.get("errorArgs", [])

                if error_type == "FinalOutput":
                    final_data = error_args[0] if error_args else None
                    captured_output = result.get("output", "")
                    return FinalOutput(final_data), captured_output
                elif error_type == "SyntaxError":
                    raise SyntaxError(f"Invalid Python syntax: {error_msg}")
                else:
                    raise CodeInterpreterError(f"{error_type}: {error_msg}")

            # Normal output
            return result.get("output", "")

    # ---- Internal helpers ----

    def _send(self, msg: dict) -> None:
        """Send JSON message to subprocess stdin."""
        if self._process is None or self._process.poll() is not None:
            stderr = ""
            if self._process and self._process.stderr:
                try:
                    stderr = self._process.stderr.read()
                except Exception:
                    pass
            exit_code = self._process.returncode if self._process else None
            raise CodeInterpreterError(
                f"Subprocess is not running (exit code: {exit_code})" + (f". Stderr: {stderr}" if stderr else "")
            )
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(msg) + "\n")
        self._process.stdin.flush()

    def _recv(self) -> dict:
        """Read JSON message from subprocess stdout with timeout."""
        if self._process is None or self._process.poll() is not None:
            stderr = ""
            if self._process and self._process.stderr:
                stderr = self._process.stderr.read()
            raise CodeInterpreterError(f"Subprocess is not running. Stderr: {stderr}")

        assert self._process.stdout is not None

        # Use select to implement timeout on readline
        # This prevents blocking forever if subprocess hangs
        fd = self._process.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], self._timeout)

        if not ready:
            # Timeout - kill the subprocess and raise
            self._process.kill()
            self._process.wait()
            raise CodeInterpreterError(f"Subprocess read timeout after {self._timeout}s")

        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise CodeInterpreterError(f"No output from subprocess. Stderr: {stderr}")

        try:
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            raise CodeInterpreterError(f"Invalid JSON from subprocess: {line.strip()!r}") from e

    def _handle_tool_call(self, msg: dict) -> None:
        """Handle a tool call request from the subprocess."""
        tool_name = msg.get("name", "")
        call_id = msg.get("id", 0)
        args_info = msg.get("args", {})

        tool_fn = self._tools.get(tool_name)
        if tool_fn is None:
            self._send(
                {
                    "type": "tool_response",
                    "id": call_id,
                    "result": None,
                    "error": f"Unknown tool: {tool_name}",
                }
            )
            return

        try:
            # Call the tool with positional and keyword args
            positional = args_info.get("args", [])
            keyword = args_info.get("kwargs", {})
            result = tool_fn(*positional, **keyword)
            # Preserve structured results (list, dict) as JSON-serializable;
            # fall back to str() for other types
            if isinstance(result, (list, dict, tuple)):
                serialized = result
            elif result is not None:
                serialized = str(result)
            else:
                serialized = ""
            self._send(
                {
                    "type": "tool_response",
                    "id": call_id,
                    "result": serialized,
                    "error": None,
                }
            )
        except Exception as e:
            self._send(
                {
                    "type": "tool_response",
                    "id": call_id,
                    "result": None,
                    "error": str(e),
                }
            )

    @staticmethod
    def _serialize_value(name: str, value: Any) -> Any:
        """Serialize a value for JSON transport to subprocess."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, dict, tuple)):
            # Let json.dumps handle the validation
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                raise CodeInterpreterError(f"Variable '{name}' contains non-serializable types")
        if hasattr(value, "tolist"):
            # numpy arrays
            return value.tolist()
        raise CodeInterpreterError(f"Cannot serialize variable '{name}' of type {type(value).__name__}")

    @staticmethod
    def _extract_parameters(fn: Callable) -> list[dict]:
        """Extract parameter info from a callable for tool registration."""
        params = []
        try:
            sig = inspect.signature(fn)
            for p_name, p in sig.parameters.items():
                param_info: dict[str, Any] = {"name": p_name}
                if p.annotation != inspect.Parameter.empty:
                    type_name = getattr(p.annotation, "__name__", str(p.annotation))
                    param_info["type"] = type_name
                if p.default != inspect.Parameter.empty:
                    param_info["default"] = p.default
                params.append(param_info)
        except (ValueError, TypeError):
            pass
        return params

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
