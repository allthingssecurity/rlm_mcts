"""
REPL Environment: Executes LLM-generated code with `context` variable.

The full transcript is loaded into `context` (a string).
The LLM writes Python code in ```repl blocks to parse, search, analyze it.
Variables persist across executions within the same REPL session.
"""

import io
import sys
import os
import time
import traceback
import tempfile
import asyncio
from dataclasses import dataclass
from typing import Any

from openai import OpenAI  # sync client for use inside REPL threads
from dotenv import load_dotenv

load_dotenv()


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals_snapshot: dict[str, str]  # var_name -> repr(value), for display
    execution_time_ms: float
    success: bool


class REPLEnv:
    """
    Persistent REPL environment. The `context` variable holds the full transcript.
    LLM-generated code runs here with access to all previous variables.
    """

    def __init__(self, context: str, sub_llm_model: str = "gpt-4o-mini"):
        self._globals: dict[str, Any] = {}
        self._locals: dict[str, Any] = {}
        self._temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")

        # Sync OpenAI client for use inside generated code (runs in thread)
        api_key = os.getenv("OPENAI_API_KEY")
        self._sync_client = OpenAI(api_key=api_key)
        self._sub_model = sub_llm_model

        # --- Built-in functions available to generated code ---

        sync_client = self._sync_client
        sub_model = self._sub_model
        self._llm_call_count = 0
        max_llm_calls = 3  # prevent runaway loops

        def llm_query(prompt: str) -> str:
            """Call a sub-LLM (max 3 calls per execution to prevent timeouts)."""
            self._llm_call_count += 1
            if self._llm_call_count > max_llm_calls:
                return "[llm_query limit reached — use Python string operations instead]"
            if len(prompt) > 100_000:
                prompt = prompt[:100_000] + "\n...[truncated]"
            response = sync_client.chat.completions.create(
                model=sub_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        def final_var(variable_name: str) -> str:
            """Mark a variable as the final answer."""
            variable_name = variable_name.strip().strip('"').strip("'").strip()
            if variable_name in self._locals:
                return str(self._locals[variable_name])
            elif variable_name in self._globals:
                return str(self._globals[variable_name])
            return f"Error: Variable '{variable_name}' not found"

        # Inject into namespace
        self._globals["llm_query"] = llm_query
        self._globals["FINAL_VAR"] = final_var
        self._globals["__builtins__"] = __builtins__

        # Load transcript as `context`
        self._locals["context"] = context

        # Also write to temp file
        context_path = os.path.join(self._temp_dir, "context.txt")
        with open(context_path, "w") as f:
            f.write(context)
        self._locals["context_path"] = context_path

    def execute(self, code: str) -> REPLResult:
        """Execute Python code in the persistent REPL. Thread-safe."""
        self._llm_call_count = 0  # reset per execution
        start = time.time()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        success = True
        try:
            # Separate imports
            lines = code.strip().split("\n")
            import_lines = []
            other_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    import_lines.append(line)
                else:
                    other_lines.append(line)

            if import_lines:
                exec("\n".join(import_lines), self._globals, self._globals)

            if other_lines:
                combined = {**self._globals, **self._locals}
                exec("\n".join(other_lines), combined, combined)
                # Persist new variables
                for key, value in combined.items():
                    if key not in self._globals and not key.startswith("_"):
                        self._locals[key] = value

        except Exception:
            stderr_buf.write(traceback.format_exc())
            success = False
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        elapsed = (time.time() - start) * 1000

        # Safe snapshot
        locals_snap = {}
        for k, v in self._locals.items():
            if k.startswith("_") or k == "context":
                continue
            try:
                r = repr(v)
                locals_snap[k] = r[:200] if len(r) > 200 else r
            except Exception:
                locals_snap[k] = "<unrepresentable>"

        return REPLResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            locals_snapshot=locals_snap,
            execution_time_ms=round(elapsed, 1),
            success=success,
        )

    async def execute_async(self, code: str, timeout_sec: float = 30.0) -> REPLResult:
        """Execute code in a thread with a timeout to avoid blocking."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self.execute, code),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            return REPLResult(
                stdout="",
                stderr=f"Execution timed out after {timeout_sec}s",
                locals_snapshot={},
                execution_time_ms=timeout_sec * 1000,
                success=False,
            )

    def get_variable(self, name: str) -> Any | None:
        """Retrieve a variable from the REPL environment."""
        name = name.strip().strip('"').strip("'").strip()
        if name in self._locals:
            return self._locals[name]
        if name in self._globals:
            return self._globals[name]
        return None

    @property
    def context_length(self) -> int:
        ctx = self._locals.get("context", "")
        return len(ctx) if isinstance(ctx, str) else 0
