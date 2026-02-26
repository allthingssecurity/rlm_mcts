"""FastAPI server for Rubric Discovery MCTS agent."""
from __future__ import annotations

import asyncio
import json
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .data_loader import load_dataset, DatasetInfo
from .repl_env import REPLEnvironment
from .mcts_engine import MCTSRubricTree, RubricNode

app = FastAPI(title="Rubric Discovery MCTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_dataset: DatasetInfo | None = None


@app.post("/load-dataset")
async def load_dataset_endpoint():
    global _dataset
    _dataset = load_dataset()
    return _dataset.to_dict()


@app.get("/dataset-info")
async def dataset_info():
    if _dataset is None:
        return {"error": "Dataset not loaded. Call POST /load-dataset first."}
    return _dataset.to_dict()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "ping":
                await ws.send_json({"event": "pong"})
                continue

            if msg.get("type") == "discover":
                await _handle_discover(ws, msg)
                continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass


async def _handle_discover(ws: WebSocket, msg: dict):
    global _dataset

    if _dataset is None:
        _dataset = load_dataset()

    max_iterations = msg.get("max_iterations", 15)
    max_depth = msg.get("max_depth", 4)

    repl = REPLEnvironment(_dataset.train, _dataset.eval)

    # Send start event
    await ws.send_json({
        "event": "discovery_started",
        "num_training": len(_dataset.train),
        "num_eval": len(_dataset.eval),
    })

    iteration_counter = {"current": 0}

    async def on_node_update(node: RubricNode, iteration: int):
        iteration_counter["current"] = iteration
        tree_snap = mcts.get_tree_snapshot()
        await ws.send_json({
            "event": "node_update",
            "node": node.to_dict(),
            "tree_snapshot": tree_snap,
            "iteration": iteration + 1,
            "total_iterations": max_iterations,
        })

    mcts = MCTSRubricTree(
        repl=repl,
        max_iterations=max_iterations,
        max_depth=max_depth,
        on_node_update=on_node_update,
    )

    try:
        best = await mcts.run()
        eval_results = mcts.get_eval_results()

        await ws.send_json({
            "event": "discovery_complete",
            "best_rubric_code": best.rubric_code,
            "best_score": best.reward_composite,
            "eval_results": eval_results,
            "tree_snapshot": mcts.get_tree_snapshot(),
        })
    except Exception as e:
        await ws.send_json({
            "event": "error",
            "message": f"Discovery failed: {traceback.format_exc()}",
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)
