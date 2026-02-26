"""Load DPO + RFT data, generate synthetic responses, score with hidden grader, split train/eval."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Hidden grader (imported from codexgame3)
# ---------------------------------------------------------------------------
_GRADER_DIR = Path(__file__).resolve().parent.parent.parent / "codexgame3" / "sapft" / "scripts"
sys.path.insert(0, str(_GRADER_DIR))
from rft_grader import _score  # noqa: E402

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "codexgame3" / "sapft" / "data"
DPO_TRAIN = _DATA_ROOT / "dpo" / "train.jsonl"
DPO_EVAL = _DATA_ROOT / "dpo" / "eval.jsonl"
RFT_TRAIN = _DATA_ROOT / "rft" / "train.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _extract_text(output_field: list[dict] | str) -> str:
    """Extract text content from DPO preferred/non_preferred output."""
    if isinstance(output_field, str):
        return output_field
    if isinstance(output_field, list):
        parts = []
        for msg in output_field:
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(msg["content"])
        return "\n".join(parts)
    return str(output_field)


def _build_spec_from_dpo(input_field: dict) -> dict:
    """Build a minimal spec from DPO input (no solution field in DPO)."""
    return {
        "require_plan": True,
        "require_assumptions": True,
        "min_steps": 3,
    }


def _generate_synthetic_responses(rft_record: dict) -> list[dict[str, Any]]:
    """Generate 3 synthetic responses (high/medium/low quality) from an RFT record."""
    solution_raw = rft_record.get("solution", "{}")
    meta = rft_record.get("meta", {})
    tools = rft_record.get("tools", [])

    if isinstance(solution_raw, str):
        try:
            spec = json.loads(solution_raw)
        except json.JSONDecodeError:
            spec = {}
    else:
        spec = solution_raw or {}

    module = spec.get("require_module") or meta.get("module", "FI")
    tcode = spec.get("require_tcode") or meta.get("tcode", "FB01")
    app = spec.get("require_app") or meta.get("fiori_app", "Manage Journal Entries")
    tool_name = spec.get("require_tool_call", "")
    if not tool_name and tools:
        tool_name = tools[0].get("function", {}).get("name", "execute_action")
    min_steps = spec.get("min_steps", 3)

    results = []

    # HIGH quality response
    steps = "\n".join([f"{i+1}. Step {i+1}: Detailed action for {module}" for i in range(max(min_steps, 4))])
    high = (
        f"Assumptions:\n- User has correct authorization roles in {module}\n"
        f"- System is available and master data is maintained\n"
        f"- Test environment validated before production\n\n"
        f"Plan:\n{steps}\n\n"
        f"Module: {module}, T-Code: {tcode}, Fiori App: {app}\n\n"
        f'Tool Call:\n{{"tool": "{tool_name}", "args": {{"item_id": "demo", "quantity": 1}}}}\n\n'
        f"Created successfully. Check authorization before proceeding."
    )
    results.append({"text": high, "spec": spec, "quality": "high"})

    # MEDIUM quality response (missing assumptions, fewer steps)
    med_steps = "\n".join([f"- Step {i+1}: Action item" for i in range(2)])
    medium = (
        f"Plan:\n{med_steps}\n\n"
        f"Use {module} module with {tcode}.\n"
        f'{{"tool": "{tool_name}", "args": {{"id": "test"}}}}'
    )
    results.append({"text": medium, "spec": spec, "quality": "medium"})

    # LOW quality response
    low = f"Plan:\n- Review the request\n- Proceed as usual"
    results.append({"text": low, "spec": spec, "quality": "low"})

    return results


def _score_example(text: str, spec: dict | str) -> float:
    """Score using the hidden grader."""
    if isinstance(spec, dict):
        spec_str = json.dumps(spec)
    else:
        spec_str = spec
    return _score(text, spec_str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DatasetInfo:
    def __init__(self, train: list[dict], eval_: list[dict]):
        self.train = train
        self.eval = eval_

    def to_dict(self) -> dict:
        train_scores = [e["score"] for e in self.train]
        eval_scores = [e["score"] for e in self.eval]
        return {
            "num_training": len(self.train),
            "num_eval": len(self.eval),
            "train_score_mean": round(sum(train_scores) / max(len(train_scores), 1), 4),
            "train_score_min": round(min(train_scores, default=0), 4),
            "train_score_max": round(max(train_scores, default=0), 4),
            "eval_score_mean": round(sum(eval_scores) / max(len(eval_scores), 1), 4),
            "score_distribution": _score_distribution(train_scores),
        }


def _score_distribution(scores: list[float]) -> dict[str, int]:
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for s in scores:
        if s < 0.2:
            buckets["0.0-0.2"] += 1
        elif s < 0.4:
            buckets["0.2-0.4"] += 1
        elif s < 0.6:
            buckets["0.4-0.6"] += 1
        elif s < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    return buckets


def load_dataset(seed: int = 42) -> DatasetInfo:
    """Load all data sources, score them, and split into train/eval."""
    rng = random.Random(seed)
    all_examples: list[dict] = []

    # --- DPO train ---
    for rec in _read_jsonl(DPO_TRAIN):
        spec = _build_spec_from_dpo(rec["input"])
        pref_text = _extract_text(rec["preferred_output"])
        nonpref_text = _extract_text(rec["non_preferred_output"])
        all_examples.append({
            "input": rec["input"],
            "response": pref_text,
            "score": _score_example(pref_text, spec),
            "source": "dpo_train_preferred",
            "spec": spec,
        })
        all_examples.append({
            "input": rec["input"],
            "response": nonpref_text,
            "score": _score_example(nonpref_text, spec),
            "source": "dpo_train_nonpreferred",
            "spec": spec,
        })

    # --- DPO eval ---
    for rec in _read_jsonl(DPO_EVAL):
        spec = _build_spec_from_dpo(rec["input"])
        pref_text = _extract_text(rec["preferred_output"])
        nonpref_text = _extract_text(rec["non_preferred_output"])
        all_examples.append({
            "input": rec["input"],
            "response": pref_text,
            "score": _score_example(pref_text, spec),
            "source": "dpo_eval_preferred",
            "spec": spec,
        })
        all_examples.append({
            "input": rec["input"],
            "response": nonpref_text,
            "score": _score_example(nonpref_text, spec),
            "source": "dpo_eval_nonpreferred",
            "spec": spec,
        })

    # --- RFT synthetic ---
    for rec in _read_jsonl(RFT_TRAIN):
        solution_raw = rec.get("solution", "{}")
        if isinstance(solution_raw, str):
            try:
                spec = json.loads(solution_raw)
            except json.JSONDecodeError:
                spec = {}
        else:
            spec = solution_raw or {}

        synthetics = _generate_synthetic_responses(rec)
        for s in synthetics:
            all_examples.append({
                "input": {"messages": rec.get("messages", [])[:2]},  # user turns only
                "response": s["text"],
                "score": _score_example(s["text"], s["spec"]),
                "source": f"rft_synthetic_{s['quality']}",
                "spec": s["spec"],
            })

    # Shuffle and split: ~80% train, ~20% eval
    rng.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.8)
    train = all_examples[:split_idx]
    eval_ = all_examples[split_idx:]

    return DatasetInfo(train, eval_)
