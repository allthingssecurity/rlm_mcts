"""
Plain RLM Engine: Single-pass code generation + execution (no MCTS tree search).

Used for comparison mode to demonstrate the quality difference
between a single-pass approach and full MCTS exploration.
"""

import time
import logging
from typing import Callable, Awaitable
from dataclasses import dataclass

from repl_env import REPLEnv
from policy_network import _get_client, SYSTEM_PROMPT, _extract_code_blocks
from value_network import evaluate_node
from mcts_engine import ReasoningNode

logger = logging.getLogger("rlm-qa")


@dataclass
class PlainRLMStep:
    """A single step in the plain RLM pipeline."""
    step_number: int
    code: str
    stdout: str
    stderr: str
    execution_ms: float
    success: bool

    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "code": self.code[:500],
            "stdout": self.stdout[:1000],
            "stderr": self.stderr[:500],
            "execution_ms": self.execution_ms,
            "success": self.success,
        }


@dataclass
class PlainRLMResult:
    """Result from a plain RLM search."""
    answer: str
    confidence: float
    metrics: dict
    steps: list[PlainRLMStep]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "metrics": self.metrics,
            "steps": [s.to_dict() for s in self.steps],
        }


OnPlainStep = Callable[[PlainRLMStep], Awaitable[None]]


async def plain_rlm_search(
    question: str,
    repl_env: REPLEnv,
    on_step: OnPlainStep | None = None,
) -> PlainRLMResult:
    """
    Single-pass RLM: one code generation -> one execution -> answer.
    With one follow-up attempt if the first fails or is insufficient.
    """
    client = _get_client()
    ctx_len = repl_env.context_length

    start_time = time.time()
    llm_calls = 0
    steps: list[PlainRLMStep] = []

    # Step 1: Generate ONE code strategy
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"The context is {ctx_len:,} characters long and may contain MULTIPLE video transcripts "
                    f"separated by '=== Title ===' headers.\n\n"
                    "Write a SINGLE ```repl code block to answer this question. "
                    "Use fast Python (regex, string slicing, etc.) to extract relevant information "
                    "from the `context` variable, then print your findings. "
                    "Use FINAL_VAR(variable_name) when you have a definitive answer.\n"
                    "Keep the code block fast (< 10 seconds). Use print() to show results."
                ),
            },
        ],
        max_tokens=2000,
        temperature=0.5,
    )
    llm_calls += 1

    text = response.choices[0].message.content.strip()
    code_blocks = _extract_code_blocks(text)

    if not code_blocks:
        # No code generated — return the raw text as answer
        elapsed = (time.time() - start_time) * 1000
        return PlainRLMResult(
            answer=text,
            confidence=0.3,
            metrics={
                "total_time_ms": round(elapsed),
                "llm_calls": llm_calls,
                "code_executions": 0,
                "successful_code_blocks": 0,
                "answer_length": len(text),
                "confidence": 0.3,
            },
            steps=[],
        )

    # Step 2: Execute the first code block
    code = code_blocks[0]
    result = await repl_env.execute_async(code)
    step1 = PlainRLMStep(
        step_number=1,
        code=code,
        stdout=result.stdout,
        stderr=result.stderr,
        execution_ms=result.execution_time_ms,
        success=result.success,
    )
    steps.append(step1)
    if on_step:
        await on_step(step1)

    # Step 3: If code failed or produced no output, try ONE follow-up
    needs_followup = not result.success or not result.stdout.strip()
    if needs_followup:
        followup_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Context length: {ctx_len:,} characters.\n"
                        "Use the `context` variable to answer."
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"```repl\n{code}\n```",
                },
                {
                    "role": "user",
                    "content": (
                        f"Previous code output:\n{result.stdout[:2000]}\n"
                        + (f"Errors:\n{result.stderr[:500]}\n" if result.stderr else "")
                        + "\nThe previous attempt "
                        + ("had errors. " if not result.success else "produced no output. ")
                        + "Write a FIXED ```repl code block. Use FINAL_VAR(variable_name) when ready."
                    ),
                },
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        llm_calls += 1

        followup_text = followup_response.choices[0].message.content.strip()
        followup_blocks = _extract_code_blocks(followup_text)

        if followup_blocks:
            followup_code = followup_blocks[0]
            followup_result = await repl_env.execute_async(followup_code)
            step2 = PlainRLMStep(
                step_number=2,
                code=followup_code,
                stdout=followup_result.stdout,
                stderr=followup_result.stderr,
                execution_ms=followup_result.execution_time_ms,
                success=followup_result.success,
            )
            steps.append(step2)
            if on_step:
                await on_step(step2)
            # Use the followup result for synthesis
            result = followup_result
            code = followup_code

    # Step 4: Synthesize an answer from the output
    best_output = result.stdout[:3000] if result.stdout.strip() else "(no output)"
    synth_response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You synthesize answers from REPL analysis results of video transcripts. "
                    "Be concise but thorough. Include evidence from the output."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Code executed:\n```python\n{code[:1000]}\n```\n\n"
                    f"Output:\n{best_output}\n\n"
                    "Synthesize a clear answer based on this output:"
                ),
            },
        ],
        max_tokens=2000,
        temperature=0.3,
    )
    llm_calls += 1
    answer = synth_response.choices[0].message.content.strip()

    # Step 5: Score the result using the value network for fair comparison
    score_node = ReasoningNode(
        id="plain_result",
        content=answer,
        node_type="answer",
        parent_id=None,
    )
    confidence = await evaluate_node(score_node, question)
    llm_calls += 1  # value network call

    elapsed = (time.time() - start_time) * 1000
    successful = sum(1 for s in steps if s.success)

    return PlainRLMResult(
        answer=answer,
        confidence=confidence,
        metrics={
            "total_time_ms": round(elapsed),
            "llm_calls": llm_calls,
            "code_executions": len(steps),
            "successful_code_blocks": successful,
            "answer_length": len(answer),
            "confidence": round(confidence, 4),
        },
        steps=steps,
    )
