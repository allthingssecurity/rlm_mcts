import React, { useState } from "react";
import { TreeSnapshot, ReasoningNode, PlainRLMStep } from "../types";

interface Props {
  answer: string | null;
  confidence: number | null;
  searching: boolean;
  tree: TreeSnapshot;
  contextChars: number;
  comparing: boolean;
  plainSteps: PlainRLMStep[];
  onSendQuestion: (question: string) => void;
  onSendCompare: (question: string) => void;
}

export default function ChatPanel({
  answer,
  confidence,
  searching,
  tree,
  contextChars,
  comparing,
  plainSteps,
  onSendQuestion,
  onSendCompare,
}: Props) {
  const [question, setQuestion] = useState("");
  const [compareMode, setCompareMode] = useState(false);

  const handleSend = () => {
    if (question.trim() && !searching) {
      if (compareMode) {
        onSendCompare(question.trim());
      } else {
        onSendQuestion(question.trim());
      }
      setQuestion("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const codeNodes = Object.values(tree).filter((n) => n.node_type === "code");
  const successfulCode = codeNodes.filter((n) => n.repl_stdout && !n.repl_stderr);

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800 flex flex-col">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Q&A
      </h2>

      {/* Answer display */}
      <div className="flex-1 min-h-0 overflow-y-auto mb-3">
        {searching && !answer && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              {comparing ? "Running Plain RLM + MCTS comparison..." : "RLM REPL + MCTS reasoning..."}
            </div>
            {contextChars > 0 && (
              <p className="text-xs text-gray-600">
                Context: {(contextChars / 1000).toFixed(0)}K chars loaded into REPL
              </p>
            )}
            {comparing && plainSteps.length > 0 && (
              <p className="text-xs text-gray-600">
                Plain RLM: {plainSteps.length} step{plainSteps.length !== 1 ? "s" : ""} complete
              </p>
            )}
            {codeNodes.length > 0 && (
              <p className="text-xs text-gray-600">
                MCTS: {codeNodes.length} code strategies explored, {successfulCode.length} successful
              </p>
            )}
          </div>
        )}

        {answer && (
          <div className="space-y-3">
            <div className="bg-gray-800 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-gray-400">Answer</span>
                {confidence != null && (
                  <span
                    className={`text-xs font-mono px-2 py-0.5 rounded ${
                      confidence >= 0.7
                        ? "bg-green-900/50 text-green-400"
                        : confidence >= 0.4
                        ? "bg-yellow-900/50 text-yellow-400"
                        : "bg-red-900/50 text-red-400"
                    }`}
                  >
                    {(confidence * 100).toFixed(0)}% confidence
                  </span>
                )}
              </div>
              <div className="text-gray-200 text-sm leading-relaxed whitespace-pre-wrap">
                {answer}
              </div>
            </div>

            {/* Code execution trace */}
            {codeNodes.length > 0 && (
              <details className="group">
                <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                  REPL execution trace ({codeNodes.length} code blocks)
                </summary>
                <div className="mt-2 space-y-2">
                  {codeNodes.map((node) => (
                    <div key={node.id} className="bg-gray-950 rounded p-2 text-xs">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`w-1.5 h-1.5 rounded-full ${node.repl_stderr ? "bg-red-500" : "bg-green-500"}`} />
                        <span className="text-gray-500 font-mono">
                          {node.execution_ms.toFixed(0)}ms
                        </span>
                        <span className="text-gray-600 font-mono">
                          v:{node.avg_value.toFixed(2)}
                        </span>
                      </div>
                      <pre className="text-cyan-400 font-mono overflow-x-auto">
                        {node.code.slice(0, 200)}
                        {node.code.length > 200 ? "..." : ""}
                      </pre>
                      {node.repl_stdout && (
                        <pre className="text-gray-400 font-mono mt-1 overflow-x-auto">
                          → {node.repl_stdout.slice(0, 150)}
                        </pre>
                      )}
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}
      </div>

      {/* Compare toggle */}
      <div className="flex items-center gap-2 mb-2">
        <button
          className={`text-xs px-2 py-1 rounded border transition-colors ${
            compareMode
              ? "bg-purple-900/50 border-purple-600 text-purple-300"
              : "bg-gray-800 border-gray-700 text-gray-500 hover:text-gray-400"
          }`}
          onClick={() => setCompareMode(!compareMode)}
          disabled={searching}
        >
          {compareMode ? "Compare: ON" : "Compare: OFF"}
        </button>
        {compareMode && (
          <span className="text-xs text-gray-600">
            Runs Plain RLM vs MCTS side-by-side
          </span>
        )}
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          className="flex-1 bg-gray-800 text-gray-200 rounded px-3 py-2 text-sm
                     placeholder-gray-600 border border-gray-700 focus:border-blue-500
                     focus:outline-none"
          placeholder={compareMode ? "Ask to compare Plain vs MCTS..." : "Ask about the video..."}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={searching}
        />
        <button
          className={`font-medium py-2 px-4 rounded text-sm transition-colors disabled:bg-gray-700
                     disabled:text-gray-500 text-white ${
                       compareMode
                         ? "bg-purple-600 hover:bg-purple-500"
                         : "bg-blue-600 hover:bg-blue-500"
                     }`}
          onClick={handleSend}
          disabled={searching || !question.trim()}
        >
          {compareMode ? "Compare" : "Ask"}
        </button>
      </div>
    </div>
  );
}
