"""Policy network: LLM generates rubric hypotheses and refinements."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _client


SYSTEM_PROMPT = """You are an expert at reverse-engineering scoring rubrics from labeled data.

Your task: Given (input, response, score) examples, discover the hidden scoring function
that maps responses to scores in [0.0, 1.0].

The examples come from SAP copilot evaluation data. Responses are scored on features like:
- Presence of structured plans (numbered/bulleted steps)
- Assumptions section
- SAP module/T-code/Fiori app mentions
- Tool call JSON blocks
- Safety considerations
- Response quality and completeness

You must write a Python function called `rubric_fn(response: str) -> float` that takes
a response string and returns a score between 0.0 and 1.0.

IMPORTANT RULES:
1. Your function MUST be named `rubric_fn`
2. It takes a single string argument `response`
3. It MUST return a float in [0.0, 1.0] — NEVER exceed 1.0
4. You can use `re`, `json`, `math` modules (already imported)
5. After defining rubric_fn, call `test_rubric(rubric_fn)` to see results
6. The variable `sample_examples` has 20 labeled examples you can inspect
7. Each example has keys: input, response, score, spec

SCORING PATTERN — use weighted normalized sums:
```
score = 0.0
total_weight = 0.0

# ALWAYS add to total_weight, only add to score if condition is met
total_weight += 1.0          # <-- always increment total
if has_plan:
    score += 1.0             # <-- only increment score when feature present

total_weight += 0.8          # <-- always increment total
if has_assumptions:
    score += 0.8             # <-- only increment score when feature present

# ... more features ...

return score / max(total_weight, 1e-6)  # Always normalize!
```
CRITICAL: `total_weight +=` must be OUTSIDE the if-block (unconditional).
           `score +=` must be INSIDE the if-block (conditional).
This ensures the output is always in [0, 1]. Do NOT use additive bonuses or
multiplicative modifiers that can push the score above 1.0.

Focus on patterns that distinguish high-scoring from low-scoring responses.
"""


async def expand_root(sample_examples: list[dict]) -> list[str]:
    """Generate 2-3 initial rubric hypotheses from sample data."""
    client = _get_client()

    # Sort examples by score to show clear low/high contrast
    sorted_ex = sorted(sample_examples, key=lambda e: e["score"])

    # Pick 3 low-scoring + 3 high-scoring + 2 mid-scoring for contrast
    low = [e for e in sorted_ex if e["score"] < 0.35][:3]
    high = [e for e in sorted_ex if e["score"] > 0.7][:3]
    mid = [e for e in sorted_ex if 0.35 <= e["score"] <= 0.7][:2]
    selected = low + mid + high

    examples_str = ""
    for i, ex in enumerate(selected):
        # Longer previews (500 chars) so tool calls and SAP terms are visible
        resp_preview = ex["response"][:500].replace("\n", "\\n")
        examples_str += f"\nExample {i+1} (score={ex['score']:.4f}):\n  Response: {resp_preview}\n"

    prompt = f"""Analyze these labeled examples carefully. Notice the CONTRAST between LOW-scoring and HIGH-scoring responses.

LOW-SCORING EXAMPLES (first {len(low)}) vs HIGH-SCORING EXAMPLES (last {len(high)}):
{examples_str}

KEY PATTERNS TO LOOK FOR:
- Do high-scoring responses have "Assumptions:" sections? Do low-scoring ones lack it?
- Do high-scoring responses contain JSON tool calls like {{"tool": "...", "args": {{...}}}}?
- Do high-scoring responses mention SAP module names, T-Codes, Fiori app names?
- Do high-scoring responses have numbered/bulleted step plans?
- Are high-scoring responses longer?
- Any safety or quality markers?

Generate exactly 3 SEPARATE code blocks, each with a different scoring strategy:
- Hypothesis 1: Weighted checklist (plan, assumptions, tool call, length, step count)
- Hypothesis 2: SAP domain features (module mentions, T-code patterns, Fiori app, tool call JSON detection)
- Hypothesis 3: Combined weighted approach using all discovered features

Each hypothesis must:
1. Define `rubric_fn(response: str) -> float` returning a score in [0, 1]
2. Call `test_rubric(rubric_fn)` to evaluate
3. Use `re` module for pattern matching (already imported)

Format each as:
```python
# Hypothesis N: [description]
def rubric_fn(response):
    ...
    return score

test_rubric(rubric_fn)
```

Separate each hypothesis with "---HYPOTHESIS---"
"""

    model = os.getenv("POLICY_MODEL", "gpt-4o-mini")
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content or ""
    return _parse_hypotheses(raw)


async def expand_refinement(
    parent_code: str,
    test_results: list[dict],
    reward_signals: dict[str, float],
    sample_examples: list[dict],
) -> list[str]:
    """Generate 1-2 refined rubric hypotheses based on parent's performance."""
    client = _get_client()

    # Sort test results by error magnitude and pair with sample examples
    # test_results come from sample_examples (same indices)
    paired = []
    for i, r in enumerate(test_results):
        entry = dict(r)
        entry["error"] = abs(r["predicted"] - r["actual"])
        if i < len(sample_examples):
            entry["response_preview"] = sample_examples[i].get("response", "")[:400]
        paired.append(entry)

    paired.sort(key=lambda x: x["error"], reverse=True)

    # Show worst errors with actual response text
    error_analysis = "\nWORST PREDICTIONS (biggest errors):\n"
    for i, e in enumerate(paired[:5]):
        error_analysis += (
            f"\n  Error {i+1}: predicted={e['predicted']:.3f}, actual={e['actual']:.3f}, diff={e['error']:.3f}\n"
        )
        if "response_preview" in e:
            resp = e["response_preview"].replace("\n", "\\n")
            error_analysis += f"    Response: {resp}\n"

    # Identify the weakest reward signal
    signal_names = ["generalization", "calibration", "discrimination", "validity", "iteration"]
    weakest = min(signal_names, key=lambda k: reward_signals.get(k, 0))
    weakest_val = reward_signals.get(weakest, 0)

    rewards_str = "\n".join(f"  {k}: {v:.3f}" for k, v in reward_signals.items() if k not in ("weights", "composite"))

    prompt = f"""Improve this rubric function based on its errors.

CURRENT RUBRIC:
```python
{parent_code}
```

REWARD SIGNALS:
{rewards_str}
  composite: {reward_signals.get('composite', 0):.3f}

WEAKEST SIGNAL: {weakest} = {weakest_val:.3f}

{error_analysis}

ANALYSIS HINTS:
- Look at the worst-predicted responses above. What features does your rubric miss?
- High-scoring SAP responses typically have: Assumptions section, Plan with 3+ steps,
  SAP module/T-code/Fiori app mentions, JSON tool calls, safety awareness, length > 120 chars
- Low-scoring responses are usually short, vague, missing structured plans and tool calls
- The score formula is likely: weighted_feature_checks * safety_modifier + quality_bonus

Generate 1-2 improved versions that:
1. Fix the largest errors shown above
2. Improve the weakest reward signal ({weakest})
3. Add any missing feature checks

Format each as:
```python
def rubric_fn(response):
    ...
    return score

test_rubric(rubric_fn)
```

Separate with "---HYPOTHESIS---"
"""

    model = os.getenv("POLICY_MODEL", "gpt-4o-mini")
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content or ""
    return _parse_hypotheses(raw)


def _parse_hypotheses(raw: str) -> list[str]:
    """Extract Python code blocks from LLM response."""
    # Split by hypothesis separator
    parts = re.split(r"---HYPOTHESIS---", raw)

    hypotheses = []
    for part in parts:
        # Extract code from markdown code blocks
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", part, re.S)
        if code_blocks:
            # Join all code blocks in this hypothesis
            code = "\n".join(code_blocks)
            hypotheses.append(code.strip())
        elif "def rubric_fn" in part:
            # Try to extract function directly
            lines = part.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith("def rubric_fn") or line.strip().startswith("# Hypothesis"):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            if code_lines:
                hypotheses.append("\n".join(code_lines).strip())

    # Fallback: if no hypotheses found, try one big code block
    if not hypotheses:
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.S)
        hypotheses = [b.strip() for b in code_blocks if "rubric_fn" in b]

    return hypotheses if hypotheses else [_fallback_rubric()]


def _fallback_rubric() -> str:
    """Simple fallback rubric in case LLM output parsing fails."""
    return '''
def rubric_fn(response):
    """Baseline rubric: structural features."""
    score = 0.0
    total = 0.0

    # Plan presence
    total += 1.0
    if "plan" in response.lower() or re.search(r"^\\s*\\d+\\.", response, re.M):
        score += 1.0

    # Step count
    total += 0.8
    steps = len(re.findall(r"^\\s*[-*\\d]", response, re.M))
    if steps >= 3:
        score += 0.8

    # Length
    total += 0.4
    if len(response.strip()) >= 120:
        score += 0.4

    # Tool call
    total += 0.8
    if '"tool"' in response or "tool_call" in response:
        score += 0.8

    return score / max(total, 1e-6)

test_rubric(rubric_fn)
'''
