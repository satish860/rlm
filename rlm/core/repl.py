"""
RLM REPL Environment - Execute Python code with persistent namespace.

The REPL provides:
- Persistent namespace across executions
- Stdout capture for output
- Error handling with line number context
- Built-in helpers (env, progress, save_output)
"""

import io
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable, TextIO


class REPLEnvironment:
    """
    Python REPL environment with persistent namespace.

    Provides code execution with:
    - Persistent variables across calls
    - Stdout capture
    - Error context with line numbers
    - Built-in helper functions

    Example:
        repl = REPLEnvironment()
        repl.namespace["x"] = 10
        result = repl.execute("y = x * 2; print(y)")
        # result = "20"
        # repl.namespace["y"] == 20
    """

    def __init__(
        self,
        output_dir: Path = None,
        progress_callback: Callable[[str], None] = None
    ):
        """
        Initialize REPL environment.

        Args:
            output_dir: Directory for save_output() files
            progress_callback: Function called for progress messages
        """
        self.output_dir = output_dir or Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._progress_callback = progress_callback
        self._real_stdout = sys.stdout

        # Initialize namespace with builtins
        self.namespace: Dict[str, Any] = {}
        self._setup_builtins()

    def _setup_builtins(self):
        """Add built-in helpers to namespace."""
        from typing import List, Optional
        from pydantic import BaseModel, Field

        self.namespace.update({
            # Python builtins
            "List": List,
            "Optional": Optional,
            "BaseModel": BaseModel,
            "Field": Field,
            # Storage
            "records": [],
            "extracted_data": {},
            # Helpers
            "env": self._env,
            "save_output": self._save_output,
            "progress": self._progress,
        })

    def _env(self) -> str:
        """Show current environment - what variables exist and their sizes."""
        info = []
        for key, val in self.namespace.items():
            if callable(val):
                info.append(f"  {key}: <function>")
            elif isinstance(val, list):
                info.append(f"  {key}: list with {len(val)} items")
            elif isinstance(val, dict):
                info.append(f"  {key}: dict with {len(val)} keys")
            elif isinstance(val, str):
                info.append(f"  {key}: str ({len(val)} chars)")
            elif isinstance(val, (int, float)):
                info.append(f"  {key}: {type(val).__name__} = {val}")
            else:
                info.append(f"  {key}: {type(val).__name__}")
        return "Environment:\n" + "\n".join(info)

    def _save_output(self, filename: str, data: Any) -> str:
        """Save data to output directory and return confirmation."""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                f.write(str(data))

        size = len(json.dumps(data, default=str)) if isinstance(data, (dict, list)) else len(str(data))
        return f"Saved to {filepath} ({size} chars)"

    def _progress(self, msg: str):
        """Print progress message to real stdout."""
        # Remove non-ASCII for Windows compatibility
        clean_msg = msg.encode('ascii', 'replace').decode('ascii')
        if self._progress_callback:
            self._progress_callback(clean_msg)
        else:
            self._real_stdout.write(f"  >> {clean_msg}\n")
            self._real_stdout.flush()

    def execute(self, code: str) -> str:
        """
        Execute Python code and return output.

        Args:
            code: Python code to execute

        Returns:
            Captured stdout output, or error message with context
        """
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            exec(code, self.namespace)
            result = sys.stdout.getvalue()
            return result or "(no output)"

        except Exception as e:
            # Get the traceback
            tb_lines = traceback.format_exc().split('\n')

            # Try to extract line number from traceback
            line_no = None
            for tb_line in tb_lines:
                if 'line ' in tb_line and '<string>' in tb_line:
                    try:
                        line_no = int(tb_line.split('line ')[1].split(',')[0].split()[0])
                    except (ValueError, IndexError):
                        pass

            # Build helpful error message
            code_lines = code.split('\n')
            error_msg = f"Error: {type(e).__name__}: {e}\n"

            if line_no and 1 <= line_no <= len(code_lines):
                error_msg += f"\nAt line {line_no}:\n"
                # Show context: 2 lines before, error line, 2 lines after
                start = max(0, line_no - 3)
                end = min(len(code_lines), line_no + 2)
                for i in range(start, end):
                    marker = ">>> " if i == line_no - 1 else "    "
                    error_msg += f"{marker}{i+1:3d} | {code_lines[i]}\n"

            return sys.stdout.getvalue() + error_msg

        finally:
            sys.stdout = old_stdout

    def reset(self):
        """Reset namespace to initial state."""
        self.namespace.clear()
        self._setup_builtins()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from namespace."""
        return self.namespace.get(key, default)

    def set(self, key: str, value: Any):
        """Set value in namespace."""
        self.namespace[key] = value

    def update(self, values: Dict[str, Any]):
        """Update namespace with multiple values."""
        self.namespace.update(values)
