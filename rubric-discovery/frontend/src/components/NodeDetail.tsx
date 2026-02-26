import type { RubricNode } from '../types'

interface Props {
  node: RubricNode | null
}

const REWARD_LABELS: Record<string, { label: string; weight: number; color: string }> = {
  reward_generalization: { label: 'Generalization', weight: 1.0, color: '#22c55e' },
  reward_calibration: { label: 'Calibration', weight: 0.4, color: '#eab308' },
  reward_discrimination: { label: 'Discrimination', weight: 0.3, color: '#3b82f6' },
  reward_validity: { label: 'Validity', weight: 0.2, color: '#a855f7' },
  reward_iteration: { label: 'Iteration', weight: 0.2, color: '#06b6d4' },
}

export default function NodeDetail({ node }: Props) {
  if (!node) {
    return (
      <div className="h-full flex items-center justify-center text-gray-600 text-sm p-4">
        Click a node in the tree to view its details
      </div>
    )
  }

  return (
    <div className="p-4 space-y-4 text-sm">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
          Node Detail
        </h2>
        <span className={`text-xs px-2 py-0.5 rounded-full ${
          node.node_type === 'final' ? 'bg-orange-900 text-orange-300' :
          node.node_type === 'hypothesis' ? 'bg-purple-900 text-purple-300' :
          node.node_type === 'refinement' ? 'bg-cyan-900 text-cyan-300' :
          'bg-blue-900 text-blue-300'
        }`}>
          {node.node_type}
        </span>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="bg-gray-800 rounded p-2 text-center">
          <div className="text-gray-500">Composite</div>
          <div className="text-lg font-mono text-green-400">{node.reward_composite.toFixed(3)}</div>
        </div>
        <div className="bg-gray-800 rounded p-2 text-center">
          <div className="text-gray-500">Train MAE</div>
          <div className="text-lg font-mono text-yellow-400">{node.train_mae.toFixed(3)}</div>
        </div>
        <div className="bg-gray-800 rounded p-2 text-center">
          <div className="text-gray-500">Visits</div>
          <div className="text-lg font-mono text-blue-400">{node.visits}</div>
        </div>
      </div>

      {/* Reward signals mini-bars */}
      <div className="space-y-1.5">
        <h3 className="text-xs text-gray-500 uppercase">Reward Signals</h3>
        {Object.entries(REWARD_LABELS).map(([key, { label, weight, color }]) => {
          const value = (node as any)[key] as number
          return (
            <div key={key} className="flex items-center gap-2 text-xs">
              <span className="w-24 text-gray-400 truncate">{label}</span>
              <div className="flex-1 bg-gray-800 rounded-full h-2.5">
                <div
                  className="h-2.5 rounded-full transition-all"
                  style={{ width: `${value * 100}%`, backgroundColor: color }}
                />
              </div>
              <span className="w-10 text-right font-mono text-gray-300">{value.toFixed(2)}</span>
              <span className="w-8 text-right text-gray-600">w={weight}</span>
            </div>
          )
        })}
      </div>

      {/* Rubric code */}
      {node.rubric_code && (
        <div>
          <h3 className="text-xs text-gray-500 uppercase mb-1">Rubric Code</h3>
          <pre className="bg-gray-950 border border-gray-800 rounded p-3 text-xs overflow-x-auto max-h-64 overflow-y-auto text-green-300 leading-relaxed">
            {node.rubric_code}
          </pre>
        </div>
      )}

      {/* Test results table */}
      {node.train_results.length > 0 && (
        <div>
          <h3 className="text-xs text-gray-500 uppercase mb-1">
            Test Results ({node.train_results.length} samples)
          </h3>
          <div className="max-h-48 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-gray-900">
                <tr className="text-gray-500">
                  <th className="text-left py-1 pr-2">#</th>
                  <th className="text-right py-1 pr-2">Predicted</th>
                  <th className="text-right py-1 pr-2">Actual</th>
                  <th className="text-right py-1">Error</th>
                </tr>
              </thead>
              <tbody>
                {node.train_results.map((r, i) => (
                  <tr key={i} className="border-t border-gray-800">
                    <td className="py-0.5 pr-2 text-gray-600">{i + 1}</td>
                    <td className="py-0.5 pr-2 text-right font-mono">{r.predicted.toFixed(3)}</td>
                    <td className="py-0.5 pr-2 text-right font-mono">{r.actual.toFixed(3)}</td>
                    <td className={`py-0.5 text-right font-mono ${
                      Math.abs(r.predicted - r.actual) > 0.3 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {Math.abs(r.predicted - r.actual).toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* REPL output */}
      {(node.stdout || node.stderr) && (
        <div>
          <h3 className="text-xs text-gray-500 uppercase mb-1">REPL Output</h3>
          {node.stdout && (
            <pre className="bg-gray-950 border border-gray-800 rounded p-2 text-xs text-gray-300 max-h-32 overflow-y-auto whitespace-pre-wrap">
              {node.stdout}
            </pre>
          )}
          {node.stderr && (
            <pre className="bg-gray-950 border border-red-900 rounded p-2 text-xs text-red-400 max-h-32 overflow-y-auto whitespace-pre-wrap mt-1">
              {node.stderr}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}
