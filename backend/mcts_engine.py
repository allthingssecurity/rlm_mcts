"""
MCTS Engine for RLM (Recursive Language Model) reasoning.

Each MCTS node represents a REPL code strategy for analyzing the transcript.
The LLM generates ```repl code blocks that execute against the `context` variable.
MCTS explores multiple code strategies in parallel, backpropagating quality scores.

Node types in the tree:
  - "question": root (the user's question)
  - "strategy": a plan/approach for answering (LLM text, no code yet)
  - "code": executable Python code that runs in the REPL
  - "result": the output of code execution (stdout + variables)
  - "answer": a final synthesized answer (FINAL_VAR)
"""

import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from repl_env import REPLEnv, REPLResult


@dataclass
class ReasoningNode:
    id: str
    content: str
    node_type: str  # "question", "strategy", "code", "result", "answer"
    parent_id: str | None
    children: list[str] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    depth: int = 0

    # Code execution metadata
    code: str = ""              # the actual Python code (for "code" nodes)
    repl_stdout: str = ""       # stdout from execution
    repl_stderr: str = ""       # stderr from execution
    repl_vars: dict = field(default_factory=dict)  # variable snapshot
    execution_ms: float = 0.0

    @property
    def avg_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, parent_visits: int, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_value / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content[:300],
            "node_type": self.node_type,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "visits": self.visits,
            "total_value": round(self.total_value, 4),
            "avg_value": round(self.avg_value, 4),
            "depth": self.depth,
            "code": self.code[:500] if self.code else "",
            "repl_stdout": self.repl_stdout[:300] if self.repl_stdout else "",
            "repl_stderr": self.repl_stderr[:200] if self.repl_stderr else "",
            "repl_vars": self.repl_vars,
            "execution_ms": self.execution_ms,
        }


# Callback types
OnNodeEvent = Callable[[ReasoningNode, dict], Awaitable[None]]
OnAnswerReady = Callable[[str, float], Awaitable[None]]


class MCTSReasoningTree:
    """
    MCTS + RLM engine.

    Each iteration:
    1. SELECT: UCB to find a promising leaf
    2. EXPAND: LLM generates code strategies → code runs in REPL → results captured
    3. EVALUATE: LLM-as-judge scores how useful the result is
    4. BACKPROPAGATE: update values up to root
    """

    def __init__(
        self,
        repl_env: REPLEnv,
        policy_fn: Callable,   # async (node, messages, question, repl_env) -> list[ReasoningNode]
        value_fn: Callable,    # async (node, question) -> float
        synthesize_fn: Callable,  # async (question, results, context_len) -> str
        max_iterations: int = 20,
        max_depth: int = 5,
        exploration_constant: float = 1.414,
    ):
        self.repl = repl_env
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.synthesize_fn = synthesize_fn
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.c = exploration_constant
        self.nodes: dict[str, ReasoningNode] = {}

        # Accumulate message history per branch (for multi-turn REPL)
        self.branch_messages: dict[str, list[dict]] = {}

    def _make_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def tree_snapshot(self) -> dict:
        return {nid: n.to_dict() for nid, n in self.nodes.items()}

    async def search(
        self,
        question: str,
        on_node: OnNodeEvent | None = None,
        on_answer: OnAnswerReady | None = None,
    ) -> tuple[str, float]:
        """Run MCTS search with RLM REPL execution."""

        # Create root
        root_id = self._make_id()
        root = ReasoningNode(
            id=root_id,
            content=question,
            node_type="question",
            parent_id=None,
            depth=0,
        )
        self.nodes[root_id] = root

        # Initialize root branch messages
        self.branch_messages[root_id] = []

        if on_node:
            await on_node(root, self.tree_snapshot())

        for _ in range(self.max_iterations):
            # 1. SELECT
            leaf_id = self._select(root_id)

            # 2. EXPAND: generate code children, execute them
            if self.nodes[leaf_id].depth < self.max_depth and not self.nodes[leaf_id].children:
                children = await self._expand(leaf_id, question)
                if children:
                    leaf_id = children[0].id

            # 3. EVALUATE
            value = await self._evaluate(leaf_id, question)

            # 4. BACKPROPAGATE
            self._backpropagate(leaf_id, value)

            # 5. STREAM: send full tree snapshot AFTER backprop (visits/values are current)
            if on_node:
                await on_node(self.nodes[leaf_id], self.tree_snapshot())

        # Synthesize final answer from all results
        answer, confidence = await self._synthesize_answer(root_id, question)
        if on_answer:
            await on_answer(answer, confidence)

        return answer, confidence

    def _select(self, root_id: str) -> str:
        node_id = root_id
        while True:
            node = self.nodes[node_id]
            if not node.children:
                return node_id
            parent_visits = max(node.visits, 1)
            best = max(
                node.children,
                key=lambda cid: self.nodes[cid].ucb_score(parent_visits, self.c),
            )
            node_id = best

    async def _expand(self, node_id: str, question: str) -> list[ReasoningNode]:
        """Expand a node: LLM generates code → REPL executes → results captured."""
        node = self.nodes[node_id]

        # Build message history for this branch
        messages = self._get_branch_messages(node_id)

        # Call policy to get children (strategies or code)
        children = await self.policy_fn(node, messages, question, self.repl)

        registered = []
        for child in children:
            child.id = self._make_id()
            child.parent_id = node_id
            child.depth = node.depth + 1

            # If it's a code node, execute it in the REPL (async to not block event loop)
            if child.node_type == "code" and child.code:
                result = await self.repl.execute_async(child.code)
                child.repl_stdout = result.stdout[:2000]
                child.repl_stderr = result.stderr[:1000]
                child.repl_vars = result.locals_snapshot
                child.execution_ms = result.execution_time_ms

                # Update content with execution summary
                if result.success and result.stdout.strip():
                    child.content = f"Code executed → {result.stdout[:200].strip()}"
                elif not result.success:
                    child.content = f"Code error → {result.stderr[:200].strip()}"
                else:
                    child.content = f"Code executed (no output), vars: {list(result.locals_snapshot.keys())}"

                # Check if code produced a final answer via FINAL_VAR
                final = self._check_final_var(child)
                if final:
                    ans_node = ReasoningNode(
                        id=self._make_id(),
                        content=final,
                        node_type="answer",
                        parent_id=child.id,
                        depth=child.depth + 1,
                    )
                    self.nodes[ans_node.id] = ans_node
                    child.children.append(ans_node.id)

            self.nodes[child.id] = child
            node.children.append(child.id)

            # Store branch messages for this child
            child_msgs = list(messages)
            if child.code:
                child_msgs.append({
                    "role": "assistant",
                    "content": f"```repl\n{child.code}\n```",
                })
                child_msgs.append({
                    "role": "user",
                    "content": (
                        f"REPL output:\n{child.repl_stdout[:3000]}\n"
                        + (f"Errors:\n{child.repl_stderr[:500]}" if child.repl_stderr else "")
                    ),
                })
            self.branch_messages[child.id] = child_msgs

            registered.append(child)

        return registered

    async def _evaluate(self, node_id: str, question: str) -> float:
        node = self.nodes[node_id]
        return await self.value_fn(node, question)

    def _backpropagate(self, node_id: str, value: float):
        current_id = node_id
        while current_id is not None:
            node = self.nodes[current_id]
            node.visits += 1
            node.total_value += value
            current_id = node.parent_id

    def _get_branch_messages(self, node_id: str) -> list[dict]:
        """Get the full message history for a branch."""
        if node_id in self.branch_messages:
            return list(self.branch_messages[node_id])
        # Trace back to root to build history
        path = []
        cid = node_id
        while cid is not None:
            path.append(cid)
            cid = self.nodes[cid].parent_id
        path.reverse()
        # Use the deepest ancestor that has messages
        for nid in reversed(path):
            if nid in self.branch_messages:
                return list(self.branch_messages[nid])
        return []

    def _check_final_var(self, node: ReasoningNode) -> str | None:
        """Check if a code node's output contains FINAL_VAR."""
        combined = node.code + "\n" + node.repl_stdout
        match = re.search(r"FINAL_VAR\(([^)]+)\)", combined)
        if match:
            var_name = match.group(1).strip().strip('"').strip("'")
            val = self.repl.get_variable(var_name)
            if val is not None:
                return str(val)
        return None

    async def _synthesize_answer(self, root_id: str, question: str) -> tuple[str, float]:
        """Gather all results from the tree and synthesize a comprehensive answer."""
        # Collect all code execution results and answers
        results = []
        for n in self.nodes.values():
            if n.node_type == "answer" and n.visits > 0:
                results.append({
                    "content": n.content,
                    "score": round(n.avg_value, 3),
                    "type": "answer",
                })
            elif n.node_type == "code" and n.visits > 0 and n.repl_stdout.strip():
                results.append({
                    "content": n.repl_stdout[:500],
                    "score": round(n.avg_value, 3),
                    "type": "code_result",
                    "code": n.code[:300],
                })

        results.sort(key=lambda r: r["score"], reverse=True)

        if not results:
            return "Could not determine an answer.", 0.0

        answer = await self.synthesize_fn(question, results[:10], self.repl.context_length)
        best_score = results[0]["score"] if results else 0.5
        return answer, min(best_score, 1.0)
