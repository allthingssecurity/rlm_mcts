"""Value network: purely algorithmic evaluation of rubric hypotheses.

Thin wrapper around reward_signals.compute_rewards(). No LLM calls.
"""
from __future__ import annotations

from .reward_signals import compute_rewards


async def evaluate_rubric(
    rubric_code: str,
    train_results: list[dict],
    eval_results: list[dict],
    execution_success: bool,
    stdout: str,
    stderr: str,
    parent_mae: float | None = None,
) -> dict:
    """Evaluate a rubric hypothesis using 5 algorithmic reward signals.

    This is intentionally async to match the interface pattern, but does no I/O.
    """
    return compute_rewards(
        rubric_code=rubric_code,
        train_results=train_results,
        eval_results=eval_results,
        execution_success=execution_success,
        stdout=stdout,
        stderr=stderr,
        parent_mae=parent_mae,
    )
