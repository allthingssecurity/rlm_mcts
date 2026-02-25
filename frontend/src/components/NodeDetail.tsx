import React from "react";
import { ReasoningNode } from "../types";

interface Props {
  node: ReasoningNode | null;
}

const TYPE_STYLES: Record<string, string> = {
  question: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  strategy: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  code: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  result: "bg-green-500/20 text-green-400 border-green-500/30",
  answer: "bg-orange-500/20 text-orange-400 border-orange-500/30",
};

export default function NodeDetail({ node }: Props) {
  if (!node) {
    return (
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Node Detail
        </h2>
        <p className="text-gray-600 text-xs">Click a node in the tree to inspect it.</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Node Detail
      </h2>

      <div className="space-y-3">
        {/* Type badge */}
        <div className="flex items-center gap-2">
          <span
            className={`text-xs px-2 py-0.5 rounded border ${
              TYPE_STYLES[node.node_type] || "bg-gray-800 text-gray-400"
            }`}
          >
            {node.node_type}
          </span>
          <span className="text-xs text-gray-600 font-mono">depth {node.depth}</span>
          {node.execution_ms > 0 && (
            <span className="text-xs text-gray-600 font-mono">
              {node.execution_ms.toFixed(0)}ms
            </span>
          )}
        </div>

        {/* Content */}
        <p className="text-sm text-gray-200 leading-relaxed">{node.content}</p>

        {/* Code block */}
        {node.code && (
          <div>
            <p className="text-xs text-gray-500 mb-1">Code:</p>
            <pre className="bg-gray-950 text-green-400 text-xs p-2 rounded overflow-x-auto max-h-40 overflow-y-auto font-mono">
              {node.code}
            </pre>
          </div>
        )}

        {/* REPL stdout */}
        {node.repl_stdout && (
          <div>
            <p className="text-xs text-gray-500 mb-1">REPL Output:</p>
            <pre className="bg-gray-950 text-gray-300 text-xs p-2 rounded overflow-x-auto max-h-32 overflow-y-auto font-mono">
              {node.repl_stdout}
            </pre>
          </div>
        )}

        {/* REPL stderr */}
        {node.repl_stderr && (
          <div>
            <p className="text-xs text-red-500 mb-1">Errors:</p>
            <pre className="bg-gray-950 text-red-400 text-xs p-2 rounded overflow-x-auto max-h-24 overflow-y-auto font-mono">
              {node.repl_stderr}
            </pre>
          </div>
        )}

        {/* Variables */}
        {Object.keys(node.repl_vars).length > 0 && (
          <div>
            <p className="text-xs text-gray-500 mb-1">Variables:</p>
            <div className="space-y-1">
              {Object.entries(node.repl_vars).map(([k, v]) => (
                <div key={k} className="flex gap-2 text-xs font-mono">
                  <span className="text-cyan-400">{k}</span>
                  <span className="text-gray-600">=</span>
                  <span className="text-gray-400 truncate">{v}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-gray-800 rounded p-2">
            <p className="text-xs text-gray-500">Visits</p>
            <p className="text-sm font-mono text-gray-300">{node.visits}</p>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <p className="text-xs text-gray-500">Avg Value</p>
            <p className="text-sm font-mono text-gray-300">
              {node.avg_value.toFixed(3)}
            </p>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <p className="text-xs text-gray-500">Children</p>
            <p className="text-sm font-mono text-gray-300">{node.children.length}</p>
          </div>
        </div>

        <p className="text-xs text-gray-600 font-mono">ID: {node.id}</p>
      </div>
    </div>
  );
}
