import { useState } from 'react'
import type { DatasetInfo } from '../types'

interface Props {
  datasetInfo: DatasetInfo | null
  discovering: boolean
  iteration: number
  totalIterations: number
  onLoadDataset: () => Promise<DatasetInfo | null>
  onStartDiscovery: (maxIterations: number, maxDepth: number) => void
}

export default function DatasetPanel({
  datasetInfo,
  discovering,
  iteration,
  totalIterations,
  onLoadDataset,
  onStartDiscovery,
}: Props) {
  const [loading, setLoading] = useState(false)
  const [maxIterations, setMaxIterations] = useState(15)
  const [maxDepth, setMaxDepth] = useState(4)

  const handleLoad = async () => {
    setLoading(true)
    await onLoadDataset()
    setLoading(false)
  }

  const progress = totalIterations > 0 ? (iteration / totalIterations) * 100 : 0

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Dataset</h2>

      {/* Load button */}
      <button
        onClick={handleLoad}
        disabled={loading || discovering}
        className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 rounded text-sm font-medium transition-colors"
      >
        {loading ? 'Loading...' : 'Load Dataset'}
      </button>

      {/* Dataset stats */}
      {datasetInfo && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <Stat label="Training" value={datasetInfo.num_training} />
            <Stat label="Eval" value={datasetInfo.num_eval} />
            <Stat label="Mean Score" value={datasetInfo.train_score_mean.toFixed(3)} />
            <Stat label="Score Range" value={`${datasetInfo.train_score_min.toFixed(2)}-${datasetInfo.train_score_max.toFixed(2)}`} />
          </div>

          {/* Score distribution */}
          <div>
            <h3 className="text-xs text-gray-500 mb-1">Score Distribution</h3>
            <div className="space-y-1">
              {Object.entries(datasetInfo.score_distribution).map(([range, count]) => {
                const maxCount = Math.max(...Object.values(datasetInfo.score_distribution))
                const pct = maxCount > 0 ? (count / maxCount) * 100 : 0
                return (
                  <div key={range} className="flex items-center gap-2 text-xs">
                    <span className="w-12 text-gray-500">{range}</span>
                    <div className="flex-1 bg-gray-800 rounded-full h-2">
                      <div
                        className="bg-purple-500 h-2 rounded-full transition-all"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="w-6 text-right text-gray-400">{count}</span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {/* Discovery controls */}
      <div className="border-t border-gray-800 pt-4 space-y-3">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Discovery</h2>

        <div className="space-y-2">
          <label className="flex items-center justify-between text-xs text-gray-400">
            <span>Iterations</span>
            <span className="text-gray-300">{maxIterations}</span>
          </label>
          <input
            type="range"
            min={3}
            max={30}
            value={maxIterations}
            onChange={(e) => setMaxIterations(Number(e.target.value))}
            disabled={discovering}
            className="w-full accent-purple-500"
          />

          <label className="flex items-center justify-between text-xs text-gray-400">
            <span>Max Depth</span>
            <span className="text-gray-300">{maxDepth}</span>
          </label>
          <input
            type="range"
            min={2}
            max={8}
            value={maxDepth}
            onChange={(e) => setMaxDepth(Number(e.target.value))}
            disabled={discovering}
            className="w-full accent-purple-500"
          />
        </div>

        <button
          onClick={() => onStartDiscovery(maxIterations, maxDepth)}
          disabled={discovering || !datasetInfo}
          className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 rounded text-sm font-medium transition-colors"
        >
          {discovering ? 'Discovering...' : 'Start Discovery'}
        </button>

        {/* Progress bar */}
        {discovering && (
          <div className="space-y-1">
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-purple-500 to-cyan-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 text-center">
              Iteration {iteration} of {totalIterations}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-800 rounded p-2">
      <div className="text-gray-500 text-[10px] uppercase">{label}</div>
      <div className="text-gray-200 font-mono text-sm">{value}</div>
    </div>
  )
}
