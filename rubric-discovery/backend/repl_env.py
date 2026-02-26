"""REPL environment for executing rubric hypotheses.

Provides `training_examples`, `sample_examples`, and `test_rubric()` helper.
Eval data is NEVER exposed to the REPL namespace.
"""
from __future__ import annotations

import io
import traceback
import random
from typing import Any


class REPLEnvironment:
    """Sandboxed execution environment for rubric code."""

    def __init__(self, train_examples: list[dict], eval_examples: list[dict]):
        self._train = train_examples
        self._eval = eval_examples  # hidden from namespace

        # Sample subset for quick testing
        # Stratified sample: equal representation from each score tier
        self._sample = _stratified_sample(train_examples, n=20, seed=123)

    def execute_rubric(self, rubric_code: str) -> dict[str, Any]:
        """Execute rubric code and test it against training data.

        Returns:
            {
                "success": bool,
                "train_results": [{"input": ..., "response": ..., "predicted": float, "actual": float}],
                "eval_results": [...],  # computed internally, never exposed to agent
                "stdout": str,
                "stderr": str,
            }
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Build namespace with helpers
        namespace: dict[str, Any] = {
            "training_examples": [
                {
                    "input": e["input"],
                    "response": e["response"],
                    "score": e["score"],
                    "spec": e.get("spec", {}),
                }
                for e in self._train
            ],
            "sample_examples": [
                {
                    "input": e["input"],
                    "response": e["response"],
                    "score": e["score"],
                    "spec": e.get("spec", {}),
                }
                for e in self._sample
            ],
            "__builtins__": _safe_builtins(),
            "print": lambda *args, **kw: stdout_buf.write(" ".join(str(a) for a in args) + kw.get("end", "\n")),
        }

        # Add commonly needed imports to namespace
        import re as _re
        import json as _json
        import math as _math
        namespace["re"] = _re
        namespace["json"] = _json
        namespace["math"] = _math

        # Inject test_rubric helper
        train_data = self._train
        eval_data = self._eval

        def test_rubric(rubric_fn) -> dict:
            """Test a rubric function against sample data. Returns MAE and per-example results."""
            results = []
            errors = []
            for ex in namespace["sample_examples"]:
                try:
                    predicted = float(rubric_fn(ex["response"]))
                    predicted = max(0.0, min(1.0, predicted))
                except Exception as exc:
                    predicted = 0.0
                    errors.append(str(exc))
                results.append({
                    "predicted": round(predicted, 4),
                    "actual": ex["score"],
                    "error": round(abs(predicted - ex["score"]), 4),
                })

            mae = sum(r["error"] for r in results) / max(len(results), 1)
            stdout_buf.write(f"test_rubric: MAE={mae:.4f} on {len(results)} samples\n")
            if errors:
                stdout_buf.write(f"  {len(errors)} execution errors\n")
            return {"mae": round(mae, 4), "results": results, "errors": errors}

        namespace["test_rubric"] = test_rubric

        # Execute the rubric code
        try:
            exec(rubric_code, namespace)
            success = True
        except Exception:
            stderr_buf.write(traceback.format_exc())
            success = False

        # Try to find rubric_fn in namespace
        rubric_fn = namespace.get("rubric_fn")
        train_results = []
        eval_results = []

        if success and callable(rubric_fn):
            # Run against all training data
            train_results = _run_rubric(rubric_fn, self._train)
            # Run against eval data (hidden from agent)
            eval_results = _run_rubric(rubric_fn, self._eval)
        elif success:
            stderr_buf.write("Warning: No `rubric_fn` function found in code.\n")
            # Check if test_rubric was called (results might be in stdout)

        return {
            "success": success,
            "train_results": train_results,
            "eval_results": eval_results,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "rubric_fn_found": callable(rubric_fn),
        }


def _stratified_sample(examples: list[dict], n: int = 20, seed: int = 123) -> list[dict]:
    """Pick equal examples from each score tier so the LLM sees full diversity."""
    rng = random.Random(seed)
    tiers: dict[str, list[dict]] = {
        "low": [],    # 0.0-0.3
        "mid": [],    # 0.3-0.7
        "high": [],   # 0.7-1.0
    }
    for ex in examples:
        s = ex["score"]
        if s < 0.3:
            tiers["low"].append(ex)
        elif s < 0.7:
            tiers["mid"].append(ex)
        else:
            tiers["high"].append(ex)

    per_tier = max(n // 3, 2)
    result = []
    for tier_name in ["low", "mid", "high"]:
        pool = tiers[tier_name]
        if pool:
            k = min(per_tier, len(pool))
            result.extend(rng.sample(pool, k))

    # Fill remainder if any tier was short
    remaining = n - len(result)
    if remaining > 0:
        used = {id(e) for e in result}
        leftover = [e for e in examples if id(e) not in used]
        if leftover:
            result.extend(rng.sample(leftover, min(remaining, len(leftover))))

    rng.shuffle(result)
    return result[:n]


def _run_rubric(rubric_fn, examples: list[dict]) -> list[dict]:
    """Run a rubric function against examples, catching errors."""
    results = []
    for ex in examples:
        try:
            predicted = float(rubric_fn(ex["response"]))
            predicted = max(0.0, min(1.0, predicted))
        except Exception:
            predicted = 0.0
        results.append({
            "predicted": round(predicted, 4),
            "actual": ex["score"],
        })
    return results


def _safe_builtins() -> dict:
    """Return a restricted set of builtins for the REPL."""
    import builtins
    allowed = [
        "abs", "all", "any", "bool", "chr", "dict", "enumerate", "filter",
        "float", "frozenset", "getattr", "hasattr", "hash", "hex", "id",
        "int", "isinstance", "issubclass", "iter", "len", "list", "map",
        "max", "min", "next", "object", "oct", "ord", "pow", "print",
        "range", "repr", "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "type", "vars", "zip",
        "True", "False", "None",
        "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
        "Exception", "RuntimeError", "StopIteration",
    ]
    safe = {}
    for name in allowed:
        val = getattr(builtins, name, None)
        if val is not None:
            safe[name] = val

    # Allow imports of safe modules only
    _ALLOWED_MODULES = {"re", "json", "math", "string", "collections", "functools", "itertools"}
    _real_import = builtins.__import__

    def _restricted_import(name, *args, **kwargs):
        if name not in _ALLOWED_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed. Allowed: {_ALLOWED_MODULES}")
        return _real_import(name, *args, **kwargs)

    safe["__import__"] = _restricted_import
    return safe
