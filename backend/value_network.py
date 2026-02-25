"""
Value Network: LLM-as-Judge scoring for MCTS nodes.
Evaluates how useful a code execution result is for answering the question.
"""

import os
import re

from openai import AsyncOpenAI
from dotenv import load_dotenv

from mcts_engine import ReasoningNode

load_dotenv()

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


async def evaluate_node(node: ReasoningNode, question: str) -> float:
    """Score how useful this node is for answering the question. Returns 0.0-1.0."""
    client = _get_client()

    # Build description based on node type
    if node.node_type == "question":
        return 0.5  # Root always gets neutral score

    if node.node_type == "answer":
        desc = f"Final answer: {node.content}"
    elif node.node_type == "code":
        desc = f"Code:\n{node.code[:500]}\n\nOutput:\n{node.repl_stdout[:500]}"
        if node.repl_stderr:
            desc += f"\nErrors:\n{node.repl_stderr[:200]}"
    elif node.node_type == "strategy":
        desc = f"Strategy: {node.content}"
    else:
        desc = node.content

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You evaluate reasoning steps in a video transcript Q&A system. "
                    "Score how useful this step is for answering the question. "
                    "Consider: Does the code run successfully? Does it extract relevant info? "
                    "Does it move toward a complete answer? "
                    "Respond with ONLY a number between 0.0 and 1.0.\n"
                    "- 0.0-0.2: Error, irrelevant, or no useful output\n"
                    "- 0.3-0.5: Partially useful, some relevant info\n"
                    "- 0.6-0.8: Good result, relevant information extracted\n"
                    "- 0.9-1.0: Excellent, directly answers the question with evidence"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nReasoning step:\n{desc}\n\nScore (0.0-1.0):",
            },
        ],
        max_tokens=10,
        temperature=0.0,
    )

    return _extract_score(response.choices[0].message.content)


def _extract_score(text: str) -> float:
    text = text.strip()
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))
    return 0.5
