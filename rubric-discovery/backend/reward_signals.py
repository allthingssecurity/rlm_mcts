"""5 algorithmic reward signals for evaluating rubric hypotheses. No LLM calls."""
from __future__ import annotations

import math
from typing import Any


def generalization_reward(
    train_results: list[dict],
    eval_results: list[dict],
) -> float:
    """Reward for how well the rubric generalizes to unseen data.

    Compares MAE on train vs eval. If eval MAE is close to train MAE,
    the rubric generalizes well. Weight: 1.0
    """
    if not train_results or not eval_results:
        return 0.0

    train_mae = _mae(train_results)
    eval_mae = _mae(eval_results)

    if train_mae == 0 and eval_mae == 0:
        return 1.0

    # Penalize overfitting: eval_mae >> train_mae means memorization
    gap = max(0, eval_mae - train_mae)
    # Also reward low absolute eval error
    eval_accuracy = max(0, 1.0 - eval_mae)
    gen_score = eval_accuracy * (1.0 - min(gap, 1.0))
    return round(max(0, min(1.0, gen_score)), 4)


def calibration_reward(results: list[dict]) -> float:
    """Reward for how well predicted scores match actual score distribution.

    Measures if the rubric produces scores in similar ranges as ground truth.
    Weight: 0.4
    """
    if not results:
        return 0.0

    actuals = [r["actual"] for r in results]
    preds = [r["predicted"] for r in results]

    actual_mean = sum(actuals) / len(actuals)
    pred_mean = sum(preds) / len(preds)
    mean_diff = abs(actual_mean - pred_mean)

    actual_std = _std(actuals)
    pred_std = _std(preds)
    std_ratio = min(pred_std, actual_std) / max(pred_std, actual_std, 1e-6)

    # Combine: close means + similar spread
    calibration = (1.0 - min(mean_diff, 1.0)) * 0.6 + std_ratio * 0.4
    return round(max(0, min(1.0, calibration)), 4)


def discrimination_reward(results: list[dict]) -> float:
    """Reward for how well the rubric separates high-scoring from low-scoring responses.

    Uses rank correlation: do higher-scoring responses get higher rubric scores?
    Weight: 0.3
    """
    if len(results) < 3:
        return 0.0

    # Spearman rank correlation (simplified)
    n = len(results)
    actuals = [r["actual"] for r in results]
    preds = [r["predicted"] for r in results]

    ranked_a = _rank(actuals)
    ranked_p = _rank(preds)

    d_sq = sum((ra - rp) ** 2 for ra, rp in zip(ranked_a, ranked_p))
    rho = 1.0 - (6 * d_sq) / (n * (n * n - 1))

    # Map from [-1, 1] to [0, 1]
    return round(max(0, min(1.0, (rho + 1) / 2)), 4)


def validity_reward(
    rubric_code: str,
    execution_success: bool,
    stdout: str,
    stderr: str,
) -> float:
    """Reward for code quality and successful execution.

    Penalizes syntax errors, runtime crashes, trivial rubrics.
    Weight: 0.2
    """
    if not execution_success:
        return 0.0

    score = 0.6  # base for successful execution

    # Penalize trivial rubrics (always return constant)
    if "return 0" in rubric_code and rubric_code.count("return") == 1:
        score -= 0.3
    if "return 1" in rubric_code and rubric_code.count("return") == 1:
        score -= 0.3

    # Bonus for non-trivial logic
    logic_keywords = ["if ", "for ", "len(", "re.", "split", "lower", "count"]
    logic_count = sum(1 for kw in logic_keywords if kw in rubric_code)
    score += min(logic_count * 0.05, 0.3)

    # Bonus for using the response text
    if "response" in rubric_code or "text" in rubric_code:
        score += 0.1

    return round(max(0, min(1.0, score)), 4)


def iteration_reward(
    current_mae: float,
    parent_mae: float | None,
) -> float:
    """Reward for improvement over parent hypothesis.

    Encourages refinements that reduce error.
    Weight: 0.2
    """
    if parent_mae is None:
        # Root node, reward based on absolute quality
        return round(max(0, min(1.0, 1.0 - current_mae)), 4)

    if parent_mae == 0:
        return 1.0 if current_mae == 0 else 0.0

    improvement = (parent_mae - current_mae) / parent_mae
    # Map: big improvement → 1.0, no change → 0.3, regression → 0.0
    if improvement > 0:
        return round(min(1.0, 0.3 + improvement * 0.7), 4)
    else:
        return round(max(0, 0.3 + improvement), 4)


def compute_rewards(
    rubric_code: str,
    train_results: list[dict],
    eval_results: list[dict],
    execution_success: bool,
    stdout: str,
    stderr: str,
    parent_mae: float | None,
) -> dict:
    """Compute all 5 reward signals and the weighted composite."""
    weights = {
        "generalization": 1.0,
        "calibration": 0.4,
        "discrimination": 0.3,
        "validity": 0.2,
        "iteration": 0.2,
    }

    train_mae = _mae(train_results) if train_results else 1.0

    signals = {
        "generalization": generalization_reward(train_results, eval_results),
        "calibration": calibration_reward(train_results),
        "discrimination": discrimination_reward(train_results),
        "validity": validity_reward(rubric_code, execution_success, stdout, stderr),
        "iteration": iteration_reward(train_mae, parent_mae),
    }

    total_w = sum(weights.values())
    composite = sum(signals[k] * weights[k] for k in weights) / total_w
    signals["composite"] = round(composite, 4)
    signals["weights"] = weights

    return signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mae(results: list[dict]) -> float:
    if not results:
        return 1.0
    errors = [abs(r["predicted"] - r["actual"]) for r in results]
    return sum(errors) / len(errors)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _rank(values: list[float]) -> list[float]:
    """Assign ranks to values (1-based, average ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # average rank for ties
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks
