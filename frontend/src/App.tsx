import React, { useState, useCallback } from "react";
import VideoInput from "./components/VideoInput";
import ChatPanel from "./components/ChatPanel";
import MCTSTreeViz from "./components/MCTSTreeViz";
import NodeDetail from "./components/NodeDetail";
import ComparisonView from "./components/ComparisonView";
import { useWebSocket } from "./hooks/useWebSocket";
import { VideoInfo, ReasoningNode } from "./types";

export default function App() {
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [selectedNode, setSelectedNode] = useState<ReasoningNode | null>(null);

  const {
    connected,
    tree,
    answer,
    confidence,
    searching,
    contextChars,
    sendQuestion,
    sendCompare,
    comparing,
    comparisonResult,
    plainSteps,
  } = useWebSocket();

  const handleTranscribed = useCallback((newVideos: VideoInfo[]) => {
    setVideos((prev) => [...prev, ...newVideos]);
  }, []);

  const videoIds = videos
    .filter((v) => v.video_id && !v.error)
    .map((v) => v.video_id);

  const handleSendQuestion = useCallback(
    (question: string) => {
      sendQuestion(question, videoIds);
      setSelectedNode(null);
    },
    [videoIds, sendQuestion]
  );

  const handleSendCompare = useCallback(
    (question: string) => {
      sendCompare(question, videoIds);
      setSelectedNode(null);
    },
    [videoIds, sendCompare]
  );

  const handleNodeClick = useCallback((node: ReasoningNode) => {
    setSelectedNode(node);
  }, []);

  const showComparison = comparisonResult != null;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-3">
        <div className="flex items-center justify-between max-w-[1400px] mx-auto">
          <div>
            <h1 className="text-lg font-bold tracking-tight">
              RLM Video Q&A
            </h1>
            <p className="text-xs text-gray-500">
              Recursive Language Model + MCTS — LLM writes code to parse transcript via REPL
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                connected ? "bg-green-500" : "bg-red-500 animate-pulse-slow"
              }`}
            />
            <span className="text-xs text-gray-500">
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto p-4 grid grid-cols-12 gap-4 h-[calc(100vh-60px)]">
        {/* Left: Video input + Chat */}
        <div className="col-span-3 flex flex-col gap-4 overflow-y-auto">
          <VideoInput videos={videos} onTranscribed={handleTranscribed} />
          <div className="flex-1 min-h-0">
            <ChatPanel
              answer={showComparison ? null : answer}
              confidence={showComparison ? null : confidence}
              searching={searching}
              tree={tree}
              contextChars={contextChars}
              comparing={comparing}
              plainSteps={plainSteps}
              onSendQuestion={handleSendQuestion}
              onSendCompare={handleSendCompare}
            />
          </div>
        </div>

        {/* Center + Right: either Comparison view or normal Tree + NodeDetail */}
        {showComparison ? (
          <div className="col-span-9 overflow-y-auto">
            <ComparisonView
              result={comparisonResult}
              plainSteps={plainSteps}
              onNodeClick={handleNodeClick}
            />
          </div>
        ) : (
          <>
            {/* Center: Tree viz */}
            <div className="col-span-6 overflow-auto">
              <MCTSTreeViz tree={tree} onNodeClick={handleNodeClick} />

              {Object.keys(tree).length > 0 && (
                <div className="mt-2 grid grid-cols-5 gap-2">
                  {(["question", "strategy", "code", "result", "answer"] as const).map(
                    (type) => {
                      const count = Object.values(tree).filter(
                        (n) => n.node_type === type
                      ).length;
                      return (
                        <div
                          key={type}
                          className="bg-gray-900 border border-gray-800 rounded p-2 text-center"
                        >
                          <p className="text-xs text-gray-500">{type}</p>
                          <p className="text-lg font-mono text-gray-300">{count}</p>
                        </div>
                      );
                    }
                  )}
                </div>
              )}
            </div>

            {/* Right: Node detail */}
            <div className="col-span-3 overflow-y-auto">
              <NodeDetail node={selectedNode} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}
