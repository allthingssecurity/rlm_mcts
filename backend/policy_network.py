"""
Policy Network for RLM + MCTS.

The LLM generates ```repl code blocks that execute against the `context` variable.
Different node types get different prompts:
  - question → generate 2-3 code strategies
  - strategy → generate executable code
  - code/result → generate follow-up code or FINAL_VAR
"""

import os
import re

from openai import AsyncOpenAI
from dotenv import load_dotenv

from mcts_engine import ReasoningNode
from repl_env import REPLEnv

load_dotenv()

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


SYSTEM_PROMPT = """\
You are an expert Python programmer solving questions about video transcripts.

## REPL Environment
- `context` — string variable with the FULL video transcript (with [MM:SS] timestamps)
- `llm_query(prompt)` — call a sub-LLM (LIMITED to 3 calls per code block! Do NOT loop over chunks)
- `print()` — output results (ALWAYS print your findings)
- `FINAL_VAR(variable_name)` — mark a variable as the final answer

Variables persist between code blocks.

## CRITICAL RULES
1. **NEVER** loop `llm_query()` over chunks — you only get 3 calls per block, and each takes ~5s.
2. **DO** use fast Python: regex, string slicing, collections.Counter, split, etc.
3. For summaries: extract key sentences with Python, then call `llm_query()` ONCE on the extracted text.
4. For specific questions: use `re.findall()` or `context.find()` to locate relevant sections, print them.
5. **ALWAYS** `print()` your results so the output is captured.
6. Each code block must complete in under 20 seconds.

## Example
```repl
import re
from collections import Counter
# Extract all timestamped lines
lines = context.split('\\n')
print(f"Transcript: {len(lines)} lines, {len(context)} chars")
# Find key topics by word frequency
words = re.findall(r'[a-z]{4,}', context.lower())
top = Counter(words).most_common(20)
print("Top words:", top)
# Extract first and last sections
print("\\nOpening:", '\\n'.join(lines[:5]))
print("\\nClosing:", '\\n'.join(lines[-5:]))
```
"""


async def expand_node(
    node: ReasoningNode,
    messages: list[dict],
    question: str,
    repl_env: REPLEnv,
) -> list[ReasoningNode]:
    """Generate child nodes. Returns code nodes that the engine will execute."""
    client = _get_client()

    if node.node_type == "question":
        return await _expand_question(client, node, question, repl_env)
    elif node.node_type in ("strategy", "code", "result"):
        return await _expand_with_history(client, node, messages, question, repl_env)
    else:
        return []


async def _expand_question(
    client: AsyncOpenAI,
    node: ReasoningNode,
    question: str,
    repl_env: REPLEnv,
) -> list[ReasoningNode]:
    """From root question: generate 2-3 different code strategies."""
    ctx_len = repl_env.context_length

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"The transcript is {ctx_len:,} characters long.\n\n"
                    "Generate 2-3 DIFFERENT code strategies to answer this question. "
                    "Each strategy should be a separate ```repl block with a different approach.\n\n"
                    "IMPORTANT: For the FIRST round, do NOT use llm_query() — use fast Python only:\n"
                    "- Strategy 1: Direct regex/string search for key terms\n"
                    "- Strategy 2: Structural analysis (split by timestamps, count sections, extract headings)\n"
                    "- Strategy 3: Statistical analysis (word frequency, key phrase extraction)\n\n"
                    "Keep each code block fast (< 5 seconds). Use print() to show results.\n"
                    "Make sure each code block is self-contained and uses the `context` variable."
                ),
            },
        ],
        max_tokens=2000,
        temperature=0.8,
    )

    text = response.choices[0].message.content.strip()
    code_blocks = _extract_code_blocks(text)

    children = []
    for i, code in enumerate(code_blocks[:3]):
        children.append(ReasoningNode(
            id="",  # assigned by engine
            content=f"Strategy {i+1}",
            node_type="code",
            parent_id=None,
            code=code,
        ))

    # If no code blocks found, create a single strategy node
    if not children:
        children.append(ReasoningNode(
            id="",
            content=text[:300],
            node_type="strategy",
            parent_id=None,
        ))

    return children


async def _expand_with_history(
    client: AsyncOpenAI,
    node: ReasoningNode,
    messages: list[dict],
    question: str,
    repl_env: REPLEnv,
) -> list[ReasoningNode]:
    """Continue the REPL conversation — generate next code based on previous results."""
    ctx_len = repl_env.context_length

    # Build the conversation
    conv = [{"role": "system", "content": SYSTEM_PROMPT}]
    conv.append({
        "role": "user",
        "content": (
            f"Question: {question}\n"
            f"Transcript length: {ctx_len:,} characters.\n"
            "Use the `context` variable to answer."
        ),
    })
    # Add previous REPL history
    conv.extend(messages[-10:])  # keep last 10 messages to stay in context

    # Add the current node's info
    if node.node_type == "code" and node.repl_stdout:
        conv.append({
            "role": "user",
            "content": (
                f"Previous code output:\n{node.repl_stdout[:3000]}\n"
                + (f"Errors:\n{node.repl_stderr[:500]}\n" if node.repl_stderr else "")
                + "\nNow write the next code block to continue analyzing or "
                "produce a final answer. Use FINAL_VAR(variable_name) when ready.\n"
                "If the previous code had errors, fix them."
            ),
        })
    elif node.node_type == "strategy":
        conv.append({
            "role": "user",
            "content": (
                f"Implement this strategy: {node.content}\n\n"
                "Write a ```repl code block that uses the `context` variable."
            ),
        })
    else:
        conv.append({
            "role": "user",
            "content": "Write the next ```repl code block to continue the analysis.",
        })

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=conv,
        max_tokens=2000,
        temperature=0.5,
    )

    text = response.choices[0].message.content.strip()
    code_blocks = _extract_code_blocks(text)

    children = []
    for code in code_blocks[:2]:
        children.append(ReasoningNode(
            id="",
            content=f"Follow-up code",
            node_type="code",
            parent_id=None,
            code=code,
        ))

    if not children:
        # No code block — treat as a text strategy
        children.append(ReasoningNode(
            id="",
            content=text[:300],
            node_type="strategy",
            parent_id=None,
        ))

    return children


async def synthesize_answer(
    question: str,
    results: list[dict],
    context_length: int,
) -> str:
    """Synthesize a comprehensive answer from MCTS REPL results."""
    client = _get_client()

    results_text = ""
    for i, r in enumerate(results):
        results_text += f"\n--- Result {i+1} (score={r['score']}, type={r['type']}) ---\n"
        results_text += r["content"] + "\n"
        if r.get("code"):
            results_text += f"Code used: {r['code']}\n"

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You synthesize comprehensive answers from multiple REPL analysis results. "
                    "The results come from different code strategies that analyzed a video transcript.\n\n"
                    "Guidelines:\n"
                    "- Combine insights from ALL results, prioritizing higher-scored ones\n"
                    "- For summaries: be thorough, cover all major topics proportional to source length\n"
                    "- For specific questions: give a precise, evidence-backed answer\n"
                    "- Structure with sections/bullets for long answers\n"
                    "- Include timestamps or quotes where available"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Source transcript was {context_length:,} characters long.\n\n"
                    f"Analysis results from multiple code strategies:\n{results_text}\n\n"
                    "Synthesize a comprehensive answer:"
                ),
            },
        ],
        max_tokens=3000,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def _extract_code_blocks(text: str) -> list[str]:
    """Extract ```repl or ```python code blocks from LLM output."""
    patterns = [
        r"```repl\s*\n(.*?)```",
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    blocks = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            code = match.group(1).strip()
            if code and len(code) > 5:
                blocks.append(code)

    # Deduplicate
    seen = set()
    unique = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique.append(b)

    return unique
