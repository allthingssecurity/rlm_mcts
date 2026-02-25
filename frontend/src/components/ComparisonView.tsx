import React from "react";
import { ComparisonResult, PlainRLMStep, TreeSnapshot } from "../types";
import MCTSTreeViz from "./MCTSTreeViz";
import { ReasoningNode } from "../types";

interface Props {
  result: ComparisonResult;
  plainSteps: PlainRLMStep[];
  onNodeClick: (node: ReasoningNode) => void;
}

interface MetricRow {
  label: string;
  plainValue: string;
  mctsValue: string;
  winner: "plain" | "mcts" | "tie";
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function buildMetrics(result: ComparisonResult): MetricRow[] {
  const pm = result.plain.metrics;
  const mm = result.mcts.metrics;

  return [
    {
      label: "Time",
      plainValue: formatMs(pm.total_time_ms),
      mctsValue: formatMs(mm.total_time_ms),
      winner: pm.total_time_ms < mm.total_time_ms ? "plain" : pm.total_time_ms > mm.total_time_ms ? "mcts" : "tie",
    },
    {
      label: "LLM Calls",
      plainValue: String(pm.llm_calls),
      mctsValue: String(mm.llm_calls),
      winner: pm.llm_calls < mm.llm_calls ? "plain" : pm.llm_calls > mm.llm_calls ? "mcts" : "tie",
    },
    {
      label: "Code Executions",
      plainValue: String(pm.code_executions),
      mctsValue: String(mm.code_executions),
      winner: "tie",
    },
    {
      label: "Successful Code",
      plainValue: String(pm.successful_code_blocks),
      mctsValue: String(mm.successful_code_blocks),
      winner:
        pm.successful_code_blocks > mm.successful_code_blocks
          ? "plain"
          : pm.successful_code_blocks < mm.successful_code_blocks
          ? "mcts"
          : "tie",
    },
    {
      label: "Confidence",
      plainValue: `${(pm.confidence * 100).toFixed(0)}%`,
      mctsValue: `${(mm.confidence * 100).toFixed(0)}%`,
      winner: pm.confidence > mm.confidence ? "plain" : pm.confidence < mm.confidence ? "mcts" : "tie",
    },
    {
      label: "Answer Length",
      plainValue: `${pm.answer_length} chars`,
      mctsValue: `${mm.answer_length} chars`,
      winner: pm.answer_length > mm.answer_length ? "plain" : pm.answer_length < mm.answer_length ? "mcts" : "tie",
    },
    {
      label: "Strategies Tried",
      plainValue: "1",
      mctsValue: String(mm.unique_strategies ?? "-"),
      winner: (mm.unique_strategies ?? 1) > 1 ? "mcts" : "tie",
    },
    {
      label: "Max Depth",
      plainValue: String(pm.code_executions),
      mctsValue: String(mm.max_depth_reached ?? "-"),
      winner: "tie",
    },
  ];
}

function winnerBadge(winner: "plain" | "mcts" | "tie"): React.ReactNode {
  if (winner === "plain")
    return <span className="text-xs font-medium text-blue-400">Plain</span>;
  if (winner === "mcts")
    return <span className="text-xs font-medium text-green-400">MCTS</span>;
  return <span className="text-xs text-gray-600">-</span>;
}

export default function ComparisonView({ result, plainSteps, onNodeClick }: Props) {
  const metrics = buildMetrics(result);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Plain RLM vs RLM + MCTS Comparison
        </h2>
      </div>

      {/* Side-by-side answers */}
      <div className="grid grid-cols-2 gap-4">
        {/* Plain RLM */}
        <div className="bg-gray-900 border border-blue-900/50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-blue-400 uppercase tracking-wider">
              Plain RLM
            </h3>
            <span
              className={`text-xs font-mono px-2 py-0.5 rounded ${
                result.plain.confidence >= 0.7
                  ? "bg-green-900/50 text-green-400"
                  : result.plain.confidence >= 0.4
                  ? "bg-yellow-900/50 text-yellow-400"
                  : "bg-red-900/50 text-red-400"
              }`}
            >
              {(result.plain.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div className="text-gray-200 text-sm leading-relaxed whitespace-pre-wrap mb-3 max-h-60 overflow-y-auto">
            {result.plain.answer}
          </div>

          {/* Code steps */}
          {result.plain.steps.length > 0 && (
            <details className="group">
              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                Code executed ({result.plain.steps.length} block{result.plain.steps.length !== 1 ? "s" : ""})
              </summary>
              <div className="mt-2 space-y-2">
                {result.plain.steps.map((step) => (
                  <div key={step.step_number} className="bg-gray-950 rounded p-2 text-xs">
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          step.success ? "bg-green-500" : "bg-red-500"
                        }`}
                      />
                      <span className="text-gray-500 font-mono">
                        Step {step.step_number} - {step.execution_ms.toFixed(0)}ms
                      </span>
                    </div>
                    <pre className="text-cyan-400 font-mono overflow-x-auto whitespace-pre-wrap">
                      {step.code.slice(0, 300)}
                      {step.code.length > 300 ? "..." : ""}
                    </pre>
                    {step.stdout && (
                      <pre className="text-gray-400 font-mono mt-1 overflow-x-auto whitespace-pre-wrap">
                        {step.stdout.slice(0, 300)}
                      </pre>
                    )}
                    {step.stderr && (
                      <pre className="text-red-400 font-mono mt-1 overflow-x-auto whitespace-pre-wrap">
                        {step.stderr.slice(0, 200)}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>

        {/* MCTS */}
        <div className="bg-gray-900 border border-green-900/50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-green-400 uppercase tracking-wider">
              RLM + MCTS
            </h3>
            <span
              className={`text-xs font-mono px-2 py-0.5 rounded ${
                result.mcts.confidence >= 0.7
                  ? "bg-green-900/50 text-green-400"
                  : result.mcts.confidence >= 0.4
                  ? "bg-yellow-900/50 text-yellow-400"
                  : "bg-red-900/50 text-red-400"
              }`}
            >
              {(result.mcts.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div className="text-gray-200 text-sm leading-relaxed whitespace-pre-wrap mb-3 max-h-60 overflow-y-auto">
            {result.mcts.answer}
          </div>

          {/* Mini MCTS tree */}
          <details className="group">
            <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
              MCTS Tree ({Object.keys(result.mcts.tree).length} nodes)
            </summary>
            <div className="mt-2 h-48 bg-gray-950 rounded overflow-hidden">
              <MCTSTreeViz tree={result.mcts.tree} onNodeClick={onNodeClick} />
            </div>
          </details>
        </div>
      </div>

      {/* Metrics comparison table */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="text-left px-4 py-2 text-gray-400 font-medium">Metric</th>
              <th className="text-center px-4 py-2 text-blue-400 font-medium">Plain RLM</th>
              <th className="text-center px-4 py-2 text-green-400 font-medium">RLM + MCTS</th>
              <th className="text-center px-4 py-2 text-gray-400 font-medium">Winner</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((row) => (
              <tr key={row.label} className="border-b border-gray-800/50">
                <td className="px-4 py-2 text-gray-300">{row.label}</td>
                <td
                  className={`text-center px-4 py-2 font-mono ${
                    row.winner === "plain" ? "text-blue-300" : "text-gray-400"
                  }`}
                >
                  {row.plainValue}
                </td>
                <td
                  className={`text-center px-4 py-2 font-mono ${
                    row.winner === "mcts" ? "text-green-300" : "text-gray-400"
                  }`}
                >
                  {row.mctsValue}
                </td>
                <td className="text-center px-4 py-2">{winnerBadge(row.winner)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
