"""
FastAPI server: REST endpoints + WebSocket for real-time RLM+MCTS tree streaming.
"""

import asyncio
import logging
import uuid
import traceback
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rlm-qa")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from transcriber import transcribe, get_video_info
from repl_env import REPLEnv
from mcts_engine import MCTSReasoningTree, ReasoningNode
from policy_network import expand_node, synthesize_answer
from value_network import evaluate_node
from plain_rlm import plain_rlm_search, PlainRLMStep

load_dotenv()


# --- In-memory stores ---

videos: dict[str, dict] = {}           # video_id -> {info, segments, full_text}
active_connections: dict[str, WebSocket] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    active_connections.clear()


app = FastAPI(title="RLM Video Q&A", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---

class TranscribeRequest(BaseModel):
    urls: list[str]

class TranscribeResponse(BaseModel):
    videos: list[dict]

class AskRequest(BaseModel):
    question: str
    video_ids: list[str]
    max_iterations: int = 12

class AskResponse(BaseModel):
    answer: str
    confidence: float
    tree: dict


# --- Helpers ---

def _build_full_text(video_ids: list[str]) -> str:
    """Build the full transcript text from video IDs (this becomes `context`)."""
    texts = []
    ids = video_ids if video_ids else list(videos.keys())
    matched = 0
    for vid in ids:
        if vid in videos:
            v = videos[vid]
            texts.append(f"=== {v['info']['title']} ===\n")
            texts.append(v["full_text"])
            texts.append("\n\n")
            matched += 1
    combined = "".join(texts)
    logger.info(
        f"Built context from {matched}/{len(ids)} videos, "
        f"requested IDs={ids}, stored IDs={list(videos.keys())}, "
        f"total chars={len(combined)}"
    )
    return combined


def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


# --- REST Endpoints ---

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_videos(req: TranscribeRequest):
    """Transcribe one or more YouTube videos."""
    results = []
    for url in req.urls:
        url = url.strip()
        if not url:
            continue
        try:
            info = get_video_info(url)
            video_id = info["id"] or uuid.uuid4().hex[:8]

            segments = await transcribe(url)

            # Build full text with timestamps (this is what goes into `context`)
            full_text = "\n".join(
                f"[{_fmt_time(s.start)}] {s.text}" for s in segments
            )

            videos[video_id] = {
                "info": info,
                "segments": segments,
                "full_text": full_text,
            }

            results.append({
                "video_id": video_id,
                "title": info["title"],
                "duration": info["duration"],
                "channel": info["channel"],
                "segment_count": len(segments),
                "transcript_chars": len(full_text),
                "transcript_preview": full_text[:500],
            })
        except Exception as e:
            results.append({
                "video_id": "",
                "title": url,
                "error": str(e),
            })

    return TranscribeResponse(videos=results)


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Ask a question about transcribed videos (non-streaming REST fallback)."""
    full_text = _build_full_text(req.video_ids)
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="No transcripts found.")

    repl = REPLEnv(context=full_text)
    tree = MCTSReasoningTree(
        repl_env=repl,
        policy_fn=expand_node,
        value_fn=evaluate_node,
        synthesize_fn=synthesize_answer,
        max_iterations=req.max_iterations,
    )

    try:
        answer, confidence = await tree.search(req.question)
    except Exception as e:
        logger.exception("MCTS search failed")
        raise HTTPException(status_code=500, detail=str(e))
    return AskResponse(answer=answer, confidence=confidence, tree=tree.tree_snapshot())


# --- WebSocket endpoint ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = uuid.uuid4().hex[:8]
    active_connections[session_id] = ws

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "ask":
                question = data.get("question", "")
                video_ids = data.get("video_ids", [])
                max_iter = data.get("max_iterations", 12)

                if not question:
                    await ws.send_json({"event": "error", "message": "No question provided."})
                    continue

                full_text = _build_full_text(video_ids)
                if not full_text.strip():
                    await ws.send_json({"event": "error", "message": "No transcripts loaded."})
                    continue

                repl = REPLEnv(context=full_text)
                tree = MCTSReasoningTree(
                    repl_env=repl,
                    policy_fn=expand_node,
                    value_fn=evaluate_node,
                    synthesize_fn=synthesize_answer,
                    max_iterations=max_iter,
                )

                async def on_node(node: ReasoningNode, snapshot: dict):
                    logger.info(f"WS sending node_update: {node.node_type} id={node.id} tree_size={len(snapshot)}")
                    await ws.send_json({
                        "event": "node_update",
                        "node": node.to_dict(),
                        "tree_snapshot": snapshot,
                    })

                async def on_answer(answer: str, confidence: float):
                    await ws.send_json({
                        "event": "answer_ready",
                        "answer": answer,
                        "confidence": round(confidence, 4),
                    })

                await ws.send_json({
                    "event": "search_started",
                    "question": question,
                    "context_chars": len(full_text),
                })

                try:
                    answer, confidence = await tree.search(
                        question,
                        on_node=on_node,
                        on_answer=on_answer,
                    )

                    await ws.send_json({
                        "event": "search_complete",
                        "answer": answer,
                        "confidence": round(confidence, 4),
                        "tree": tree.tree_snapshot(),
                    })
                except Exception as e:
                    logger.exception("MCTS search failed in WS")
                    await ws.send_json({
                        "event": "error",
                        "message": f"Search failed: {e}",
                    })

            elif msg_type == "compare":
                question = data.get("question", "")
                video_ids = data.get("video_ids", [])
                max_iter = data.get("max_iterations", 12)

                if not question:
                    await ws.send_json({"event": "error", "message": "No question provided."})
                    continue

                full_text = _build_full_text(video_ids)
                if not full_text.strip():
                    await ws.send_json({"event": "error", "message": "No transcripts loaded."})
                    continue

                await ws.send_json({
                    "event": "search_started",
                    "question": question,
                    "context_chars": len(full_text),
                })

                # Independent REPL environments for each mode
                plain_repl = REPLEnv(context=full_text)
                mcts_repl = REPLEnv(context=full_text)

                mcts_tree = MCTSReasoningTree(
                    repl_env=mcts_repl,
                    policy_fn=expand_node,
                    value_fn=evaluate_node,
                    synthesize_fn=synthesize_answer,
                    max_iterations=max_iter,
                )

                # Callbacks for streaming progress
                async def on_plain_step(step: PlainRLMStep):
                    await ws.send_json({
                        "event": "plain_step",
                        "step": step.to_dict(),
                    })

                async def on_mcts_node(node: ReasoningNode, snapshot: dict):
                    await ws.send_json({
                        "event": "node_update",
                        "node": node.to_dict(),
                        "tree_snapshot": snapshot,
                    })

                async def on_mcts_answer(answer: str, confidence: float):
                    await ws.send_json({
                        "event": "answer_ready",
                        "answer": answer,
                        "confidence": round(confidence, 4),
                    })

                try:
                    import time as _time
                    mcts_start = _time.time()

                    # Run BOTH modes concurrently
                    plain_result, (mcts_answer, mcts_confidence) = await asyncio.gather(
                        plain_rlm_search(question, plain_repl, on_step=on_plain_step),
                        mcts_tree.search(question, on_node=on_mcts_node, on_answer=on_mcts_answer),
                    )

                    mcts_elapsed = (_time.time() - mcts_start) * 1000

                    # Collect MCTS metrics from the tree
                    all_nodes = list(mcts_tree.nodes.values())
                    code_nodes = [n for n in all_nodes if n.node_type == "code"]
                    successful_code = [n for n in code_nodes if n.repl_stdout and not n.repl_stderr]
                    root_id = next((n.id for n in all_nodes if n.node_type == "question"), None)
                    root_children = [n for n in all_nodes if n.parent_id == root_id] if root_id else []
                    max_depth = max((n.depth for n in all_nodes), default=0)
                    visited_values = [n.avg_value for n in all_nodes if n.visits > 0 and n.node_type != "question"]
                    avg_node_value = sum(visited_values) / len(visited_values) if visited_values else 0.0

                    mcts_metrics = {
                        "total_time_ms": round(mcts_elapsed),
                        "llm_calls": max_iter * 2 + 1,  # ~2 per iteration (policy + value) + 1 synthesis
                        "code_executions": len(code_nodes),
                        "successful_code_blocks": len(successful_code),
                        "unique_strategies": len(root_children),
                        "max_depth_reached": max_depth,
                        "avg_node_value": round(avg_node_value, 4),
                        "answer_length": len(mcts_answer),
                        "confidence": round(mcts_confidence, 4),
                    }

                    await ws.send_json({
                        "event": "comparison_complete",
                        "plain": plain_result.to_dict(),
                        "mcts": {
                            "answer": mcts_answer,
                            "confidence": round(mcts_confidence, 4),
                            "metrics": mcts_metrics,
                            "tree": mcts_tree.tree_snapshot(),
                        },
                    })
                except Exception as e:
                    logger.exception("Comparison search failed")
                    await ws.send_json({
                        "event": "error",
                        "message": f"Comparison failed: {e}",
                    })

            elif msg_type == "ping":
                await ws.send_json({"event": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        active_connections.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
