"""MCTS engine for rubric discovery.

RubricNode dataclass + MCTSRubricTree with select/expand/evaluate/backpropagate.
"""
from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from .repl_env import REPLEnvironment
from .policy_network import expand_root, expand_refinement
from .value_network import evaluate_rubric
from .reward_signals import _mae


@dataclass
class RubricNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    rubric_code: str = ""
    node_type: str = "root"  # root, hypothesis, refinement, final
    depth: int = 0
    visits: int = 0

    # Reward signals
    reward_generalization: float = 0.0
    reward_calibration: float = 0.0
    reward_discrimination: float = 0.0
    reward_validity: float = 0.0
    reward_iteration: float = 0.0
    reward_composite: float = 0.0

    # Test results
    train_results: list[dict] = field(default_factory=list)
    eval_results: list[dict] = field(default_factory=list)
    train_mae: float = 1.0
    eval_mae: float = 1.0
    stdout: str = ""
    stderr: str = ""
    execution_success: bool = False

    # Tree links
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    # UCB
    total_reward: float = 0.0

    def ucb_score(self, parent_visits: int, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.total_reward / self.visits
        explore = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rubric_code": self.rubric_code,
            "node_type": self.node_type,
            "depth": self.depth,
            "visits": self.visits,
            "reward_generalization": self.reward_generalization,
            "reward_calibration": self.reward_calibration,
            "reward_discrimination": self.reward_discrimination,
            "reward_validity": self.reward_validity,
            "reward_iteration": self.reward_iteration,
            "reward_composite": self.reward_composite,
            "train_mae": self.train_mae,
            "eval_mae": self.eval_mae,
            "stdout": self.stdout[:500],
            "stderr": self.stderr[:500],
            "execution_success": self.execution_success,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "train_results": self.train_results[:20],  # limit for WS
            "eval_results": self.eval_results[:20],
        }


class MCTSRubricTree:
    """Monte Carlo Tree Search for rubric discovery."""

    def __init__(
        self,
        repl: REPLEnvironment,
        max_iterations: int = 15,
        max_depth: int = 4,
        on_node_update: Callable[[RubricNode, int], Awaitable[None]] | None = None,
    ):
        self.repl = repl
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.on_node_update = on_node_update

        self.root = RubricNode(node_type="root")
        self.nodes: dict[str, RubricNode] = {self.root.id: self.root}
        self.best_node: RubricNode | None = None

    async def run(self) -> RubricNode:
        """Run the full MCTS loop."""
        for iteration in range(self.max_iterations):
            # SELECT
            leaf = self._select(self.root)

            # EXPAND
            children = await self._expand(leaf, iteration)

            # EVALUATE + BACKPROPAGATE for each child
            for child in children:
                self._backpropagate(child, child.reward_composite)

                # Track best node
                if self.best_node is None or child.reward_composite > self.best_node.reward_composite:
                    self.best_node = child

                # Notify
                if self.on_node_update:
                    await self.on_node_update(child, iteration)

        # Mark best as final
        if self.best_node:
            self.best_node.node_type = "final"

        return self.best_node or self.root

    def _select(self, node: RubricNode) -> RubricNode:
        """UCB-based selection down the tree."""
        current = node
        while current.children_ids and current.depth < self.max_depth:
            children = [self.nodes[cid] for cid in current.children_ids]
            # Pick child with highest UCB score
            best_child = max(children, key=lambda c: c.ucb_score(current.visits or 1))
            if best_child.visits == 0:
                return best_child
            current = best_child
        return current

    async def _expand(self, node: RubricNode, iteration: int) -> list[RubricNode]:
        """Generate child hypotheses and evaluate them."""
        # Full sample examples for the LLM (longer previews for feature visibility)
        sample = [
            {
                "input": e["input"],
                "response": e["response"],
                "score": e["score"],
                "spec": e.get("spec", {}),
            }
            for e in self.repl._sample
        ]

        if node.node_type == "root" and not node.children_ids:
            # Initial expansion: generate diverse hypotheses
            code_candidates = await expand_root(sample)
        else:
            # For refinement, run parent rubric on sample to get aligned results
            sample_results = _run_rubric_on_sample(node.rubric_code, self.repl._sample)
            code_candidates = await expand_refinement(
                parent_code=node.rubric_code,
                test_results=sample_results,
                reward_signals={
                    "generalization": node.reward_generalization,
                    "calibration": node.reward_calibration,
                    "discrimination": node.reward_discrimination,
                    "validity": node.reward_validity,
                    "iteration": node.reward_iteration,
                    "composite": node.reward_composite,
                },
                sample_examples=sample,
            )

        children = []
        for code in code_candidates:
            child = await self._create_and_evaluate(
                code=code,
                parent=node,
                node_type="hypothesis" if node.node_type == "root" else "refinement",
            )
            children.append(child)

        return children

    async def _create_and_evaluate(
        self, code: str, parent: RubricNode, node_type: str
    ) -> RubricNode:
        """Create a node, execute its rubric, and compute rewards."""
        child = RubricNode(
            rubric_code=code,
            node_type=node_type,
            depth=parent.depth + 1,
            parent_id=parent.id,
        )

        # Execute in REPL
        result = self.repl.execute_rubric(code)
        child.execution_success = result["success"]
        child.train_results = result["train_results"]
        child.eval_results = result["eval_results"]
        child.stdout = result["stdout"]
        child.stderr = result["stderr"]

        # Compute MAE
        if child.train_results:
            child.train_mae = _mae(child.train_results)
        if child.eval_results:
            child.eval_mae = _mae(child.eval_results)

        # Compute rewards
        parent_mae = parent.train_mae if parent.node_type != "root" else None
        rewards = await evaluate_rubric(
            rubric_code=code,
            train_results=child.train_results,
            eval_results=child.eval_results,
            execution_success=child.execution_success,
            stdout=child.stdout,
            stderr=child.stderr,
            parent_mae=parent_mae,
        )

        child.reward_generalization = rewards["generalization"]
        child.reward_calibration = rewards["calibration"]
        child.reward_discrimination = rewards["discrimination"]
        child.reward_validity = rewards["validity"]
        child.reward_iteration = rewards["iteration"]
        child.reward_composite = rewards["composite"]

        # Register in tree
        self.nodes[child.id] = child
        parent.children_ids.append(child.id)

        return child

    def _backpropagate(self, node: RubricNode, reward: float):
        """Propagate reward up to root."""
        current_id: str | None = node.id
        while current_id is not None:
            n = self.nodes[current_id]
            n.visits += 1
            n.total_reward += reward
            current_id = n.parent_id

    def get_tree_snapshot(self) -> dict:
        """Return the full tree as a dict for the frontend."""
        return {
            "root_id": self.root.id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "best_node_id": self.best_node.id if self.best_node else None,
        }

    def get_eval_results(self) -> dict:
        """Run final evaluation on best rubric."""
        if not self.best_node or not self.best_node.eval_results:
            return {"error": "No valid rubric found"}

        eval_res = self.best_node.eval_results
        mae = _mae(eval_res)

        # Generalization accuracy (within 0.15 tolerance)
        correct = sum(1 for r in eval_res if abs(r["predicted"] - r["actual"]) < 0.15)
        accuracy = correct / max(len(eval_res), 1)

        return {
            "best_rubric_code": self.best_node.rubric_code,
            "eval_mae": round(mae, 4),
            "eval_accuracy": round(accuracy, 4),
            "eval_count": len(eval_res),
            "eval_results": eval_res,
            "best_composite": self.best_node.reward_composite,
        }


def _run_rubric_on_sample(rubric_code: str, sample_examples: list[dict]) -> list[dict]:
    """Run a rubric against sample examples to get aligned (index-matched) results.

    Used by refinement so the LLM can see which specific examples were mispredicted.
    """
    import re as _re
    import json as _json
    import math as _math

    namespace: dict = {"re": _re, "json": _json, "math": _math}
    try:
        exec(rubric_code, namespace)
    except Exception:
        return []

    rubric_fn = namespace.get("rubric_fn")
    if not callable(rubric_fn):
        return []

    results = []
    for ex in sample_examples:
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
